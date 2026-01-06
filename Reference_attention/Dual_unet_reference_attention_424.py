# 文件: models/ReferenceNet_attention_xca.py
# 版本: 最终修复版 v2，修复了 video_length 为 None 导致的 TypeError

import torch
import torch.nn.functional as F
import random
from einops import rearrange
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from diffusers.models.attention import BasicTransformerBlock, Attention

# 假设这个自定义的Block存在于同级目录的attention.py文件中
# from .attention import BasicTransformerBlock as _BasicTransformerBlock
# 为保证代码独立可运行，我们先定义一个占位符类
class _BasicTransformerBlock(torch.nn.Module):
    pass

def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

class ReferenceNetAttention():
    """
    通过劫持U-Net的注意力模块，实现参考图引导的生成。
    这个修改后的版本实现了两种引导方式的结合：
    1. 拼接自注意力 (原始功能): 将当前特征与参考特征拼接后进行自注意力计算。
    2. 直接交叉注意力 (新增功能): 将当前特征作为Query，参考特征作为Key/Value进行交叉注意力计算。
    此版本经过修改，以兼容没有独立CrossAttention类的旧版diffusers，并修复了LoRA层导致的AttributeError和video_length为None的TypeError。
    """
    def __init__(self,
                 unet: torch.nn.Module,
                 mode: str = "write",
                 do_classifier_free_guidance: bool = False,
                 attention_auto_machine_weight: float = float('inf'),
                 gn_auto_machine_weight: float = 1.0,
                 style_fidelity: float = 1.0,
                 reference_attn: bool = True,
                 fusion_blocks: str = "full",
                 fusion_mode: str = "add",
                 fusion_alpha: float = 0.5,
                 batch_size: int = 1,
                 is_image: bool = False,
                 ) -> None:
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        assert fusion_mode in ["add"]

        self.reference_attn = reference_attn
        self.fusion_blocks = fusion_blocks
        self.fusion_mode = fusion_mode
        self.fusion_alpha = fusion_alpha

        self.register_reference_hooks(
            mode=mode,
            do_classifier_free_guidance=do_classifier_free_guidance,
            attention_auto_machine_weight=attention_auto_machine_weight,
            gn_auto_machine_weight=gn_auto_machine_weight,
            style_fidelity=style_fidelity,
            reference_attn=reference_attn,
            fusion_blocks=fusion_blocks,
            batch_size=batch_size,
            is_image=is_image,
        )

    def register_reference_hooks(
            self,
            mode: str,
            do_classifier_free_guidance: bool,
            attention_auto_machine_weight: float,
            gn_auto_machine_weight: float,
            style_fidelity: float,
            reference_attn: bool,
            fusion_blocks: str,
            dtype: torch.dtype = torch.float16,
            batch_size: int = 1,
            num_images_per_prompt: int = 1,
            device: torch.device = torch.device("cpu"),
            is_image: bool = False,
        ):
        MODE = mode
        FUSION_MODE = self.fusion_mode
        FUSION_ALPHA = self.fusion_alpha

        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            video_length: Optional[int] = None,
        ):
            if hasattr(self, 'use_ada_layer_norm') and self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif hasattr(self, 'use_ada_layer_norm_zero') and self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

            if MODE == "write":
                self.bank.append(norm_hidden_states.clone())
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            elif MODE == "read":
                ref_bank_features = self.bank[0]
                
                # ========================== 核心变更点: 修复TypeError ==========================
                # 仅在处理视频(not is_image)且video_length有效(不是None)时，才执行此代码块
                if not is_image and video_length is not None:
                    ref_bank_features = rearrange(ref_bank_features.unsqueeze(1).repeat(1, video_length, 1, 1), "b t l c -> (b t) l c")[:hidden_states.shape[0]]
                # ===============================================================================

                # --- 通路 A: 拼接自注意力 ---
                modify_norm_hidden_states = torch.cat([norm_hidden_states, ref_bank_features], dim=1)
                concat_self_attn_output = self.attn1(
                    modify_norm_hidden_states,
                    encoder_hidden_states=modify_norm_hidden_states,
                )[:, :norm_hidden_states.shape[1], :]

                # --- 通路 B: 纯目标自注意力 (新增，来自原版逻辑) ---
                target_self_attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=norm_hidden_states,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )

                # --- 通路 C: 目标-参考交叉注意力 ---
                direct_cross_attn_output = self.ref_cross_attn(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=ref_bank_features,
                    attention_mask=None,
                    **cross_attention_kwargs,
                )

                # --- 融合结果 (根据您的要求进行加权) ---
                fused_attn_output = 0.4 * target_self_attn_output + 0.2 * direct_cross_attn_output + 0.4 * concat_self_attn_output 

                attn_output = fused_attn_output

                if do_classifier_free_guidance and hidden_states.shape[0] % 2 == 0:
                    uncond_chunk, cond_chunk = attn_output.chunk(2)
                    uncond_hidden_states_norm = norm_hidden_states.chunk(2)[0]
                    uncond_attn_output = self.attn1(
                        uncond_hidden_states_norm,
                        encoder_hidden_states=uncond_hidden_states_norm,
                        attention_mask=None,
                    )
                    attn_output = torch.cat([uncond_attn_output, cond_chunk], dim=0)
            
            else:
                 attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )

            if hasattr(self, 'use_ada_layer_norm_zero') and self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = self.norm2(hidden_states, timestep) if hasattr(self, 'use_ada_layer_norm') and self.use_ada_layer_norm else self.norm2(hidden_states)
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            norm_hidden_states = self.norm3(hidden_states)
            if hasattr(self, 'use_ada_layer_norm_zero') and self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            ff_output = self.ff(norm_hidden_states)
            if hasattr(self, 'use_ada_layer_norm_zero') and self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            hidden_states = ff_output + hidden_states

            if not is_image and hasattr(self, 'attn_temp'):
                d = hidden_states.shape[1]
                hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
                norm_hidden_states = self.norm_temp(hidden_states)
                hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
                hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)
            
            return hidden_states

        if self.reference_attn:
            if self.fusion_blocks == "midup":
                attn_modules = [module for module in (torch_dfs(self.unet.mid_block)+torch_dfs(self.unet.up_blocks)) if isinstance(module, (BasicTransformerBlock, _BasicTransformerBlock))]
            elif self.fusion_blocks == "full":
                attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, (BasicTransformerBlock, _BasicTransformerBlock))]
            attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                module.bank = []
                
                q_dim = module.norm1.normalized_shape[0]
                
                module.ref_cross_attn = Attention(
                    query_dim=q_dim,
                    heads=module.attn1.heads,
                    dim_head = q_dim // module.attn1.heads,
                    dropout=0.0,
                ).to(device, dtype=dtype)
    
    def update(self, writer: "ReferenceNetAttention", dtype: torch.dtype = torch.float16):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [module for module in (torch_dfs(self.unet.mid_block)+torch_dfs(self.unet.up_blocks)) if isinstance(module, (BasicTransformerBlock, _BasicTransformerBlock))]
                writer_attn_modules = [module for module in (torch_dfs(writer.unet.mid_block)+torch_dfs(writer.unet.up_blocks)) if isinstance(module, (BasicTransformerBlock, _BasicTransformerBlock))]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, (BasicTransformerBlock, _BasicTransformerBlock))]
                writer_attn_modules = [module for module in torch_dfs(writer.unet) if isinstance(module, (BasicTransformerBlock, _BasicTransformerBlock))]
            
            reader_attn_modules = sorted(reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0])    
            writer_attn_modules = sorted(writer_attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
        
            if len(reader_attn_modules) == 0 or len(writer_attn_modules) == 0:
                raise ValueError("Attention modules not found in reader or writer UNet.")
              
            for r, w in zip(reader_attn_modules, writer_attn_modules):
                r.bank = [v.clone().to(dtype) for v in w.bank]

    def clear(self):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                attn_modules = [module for module in (torch_dfs(self.unet.mid_block)+torch_dfs(self.unet.up_blocks)) if isinstance(module, (BasicTransformerBlock, _BasicTransformerBlock))]
            elif self.fusion_blocks == "full":
                attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, (BasicTransformerBlock, _BasicTransformerBlock))]
            
            if not attn_modules:
                return

            reader_attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
            for r in reader_attn_modules:
                if hasattr(r, 'bank'):
                    r.bank.clear()

class ReferenceNetAttentionXCA(ReferenceNetAttention):
    """兼容旧代码中导入 ReferenceNetAttentionXCA 的写法，功能完全继承自 ReferenceNetAttention。"""
    pass