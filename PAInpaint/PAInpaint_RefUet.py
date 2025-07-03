# Modified from https://github.com/tencent-ailab/IP-Adapter
import os
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from .utils import is_torch2_available
if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor
from .resampler import LinearResampler

# from models.ReferenceNet_attention import ReferenceNetAttention
from models.DSAM_424 import ReferenceNetAttentionXCA as ReferenceNetAttention


# 定义SpatialTransformer模块
# 这是一个简化的STN实现，用于处理图像latent（例如，4通道，64x64分辨率）
class SpatialTransformer(nn.Module):
    def __init__(self, input_channels):
        super(SpatialTransformer, self).__init__()
        # 定位网络：用于预测仿射变换参数
        # 假设输入latent是 (N, C, 64, 64)
        self.localization = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7), # (N, 32, 58, 58)
            nn.MaxPool2d(2, stride=2), # (N, 32, 29, 29)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5), # (N, 64, 25, 25)
            nn.MaxPool2d(2, stride=2), # (N, 64, 12, 12)
            nn.ReLU(True)
        )

        # 回归器：将定位网络的输出展平并回归2x3仿射矩阵
        # 展平后的尺寸取决于定位网络的输出尺寸
        # 对于 64x64 的输入，经过两次MaxPool2d(2)后，空间维度会变为 12x12
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 12 * 12, 32), # 64 * 12 * 12 = 9216
            nn.ReLU(True),
            nn.Linear(32, 3 * 2) # 输出2x3仿射矩阵的6个参数
        )

        # 初始化回归器的权重和偏置，使其最初表示一个恒等变换
        # 这有助于训练的稳定性，确保模型在学习新的变换之前不会立即扭曲图像
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # STN的前向传播
    def forward(self, x):
        # 1. 通过定位网络获取变换参数
        xs = self.localization(x)
        xs = xs.view(-1, self.fc_loc[0].in_features) # 展平特征图

        theta = self.fc_loc(xs) # 获取2x3变换矩阵
        theta = theta.view(-1, 2, 3) # 重塑为 (N, 2, 3)

        # 2. 生成采样网格
        # x.size() 是 (N, C, H, W)，确保输出的尺寸与输入相同
        grid = F.affine_grid(theta, x.size(), align_corners=True)

        # 3. 使用采样网格对输入特征图进行采样
        # 双线性插值使得整个模块可微分
        x_transformed = F.grid_sample(x, grid, align_corners=True)

        return x_transformed


class PAInpaint_RefUet:
    def __init__(self, sd_pipe, image_encoder_path, model_ckpt, depth_estimator, depth_guider, referencenet, device):
        # Takes model path as input
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.model_ckpt = model_ckpt
        self.referencenet = referencenet.to(self.device)
        self.depth_estimator = depth_estimator.to(self.device).eval()
        self.depth_guider = depth_guider.to(self.device, dtype=torch.float16)
        self.stn = SpatialTransformer(input_channels=4).to(self.device, dtype=torch.float16)
        self.pipe = sd_pipe.to(self.device)
        self.pipe.unet.set_attn_processor(AttnProcessor())

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()
        self.image_processor = VaeImageProcessor()

        # 关键修改：必须在加载权重之前修改网络结构
        self.reference_control_writer = ReferenceNetAttention(
            self.referencenet,
            do_classifier_free_guidance=True,
            mode='write', 
            fusion_blocks='midup', 
            batch_size=1,
            is_image=True
        )
        self.reference_control_reader = ReferenceNetAttention(
            self.pipe.unet, 
            do_classifier_free_guidance=True,
            mode='read', 
            fusion_blocks='midup', 
            batch_size=1,
            is_image=True
        )
        
        if self.model_ckpt:
            self.load_checkpoint()

    def init_proj(self):
        image_proj_model = LinearResampler(
            input_dim=1280,
            output_dim=self.pipe.unet.config.cross_attention_dim,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def load_checkpoint(self):
        state_dict = torch.load(self.model_ckpt, map_location="cpu")["module"]

        unet_dict = {}
        image_proj_dict = {}
        depth_guider_dict = {}
        stn_dict = {}

        for k in state_dict.keys():
            if k.startswith("unet."):
                unet_dict[k.replace("unet.", "")] = state_dict[k]           
            elif k.startswith("image_proj_model.") or k.startswith("proj."):
                # 兼容不同命名
                image_proj_dict[k.replace("image_proj_model.", "").replace("proj.", "")] = state_dict[k]
            elif k.startswith("depth_guider."):
                depth_guider_dict[k.replace("depth_guider.", "")] = state_dict[k]
            elif k.startswith("stn."):
                stn_dict[k.replace("stn.", "")] = state_dict[k]
            else:
                print(f"Unrecognized key: {k}")


        if unet_dict:
            self.pipe.unet.load_state_dict(unet_dict)
            self.pipe.unet.to(self.device)
            print('=== load unet ===')
        if image_proj_dict:
            self.image_proj_model.load_state_dict(image_proj_dict)
            self.image_proj_model.to(self.device)
            print('=== load image_proj ===')
        if depth_guider_dict:
            self.depth_guider.load_state_dict(depth_guider_dict)
            self.depth_guider.to(self.device)
            print('=== load depth_guider ===')
        if stn_dict:
            self.stn.load_state_dict(stn_dict)
            self.stn.to(self.device)
            print('=== load stn ===')

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values

        clip_image = clip_image.to(self.device, dtype=torch.float16)
        # print('&&&&&&&&&&&&&&')
        # print('clip_image',clip_image.shape) #torch.Size([1, 3, 224, 224])
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        # print('clip_image_embeds',clip_image_embeds.shape) #torch.Size([1, 257, 1280])
        # print('&&&&&&&&&&&&&&')
        image_prompt_embeds = self.image_proj_model(clip_image_embeds).to(dtype=torch.float16)
        
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


    def generate(
        self,
        pil_image=None,
        depth_image = None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        num_samples=4,
        seed=None,
        image = None,
        guidance_scale=7.5,
        num_inference_steps=30,
        return_loss=False,
        **kwargs,
    ):
        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        depth_image = depth_image.to(self.device, dtype=torch.float16)
        depth_map = self.depth_estimator(depth_image).unsqueeze(1)
        depth_feature = self.depth_guider(depth_map.to(self.device, dtype=torch.float16))

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            prompt_embeds=image_prompt_embeds , # image clip embedding 
            negative_prompt_embeds=uncond_image_prompt_embeds,  # uncond image clip embedding 
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            referencenet=self.referencenet,
            reference_control_reader=self.reference_control_reader,
            reference_control_writer=self.reference_control_writer,
            source_image=pil_image,
            image = image,
            clip_image_embed= torch.cat([uncond_image_prompt_embeds, image_prompt_embeds], dim=0), # for reference U-Net
            depth_feature = depth_feature,
            stn=self.stn, # Pass STN to pipeline
            return_loss=return_loss,
            **kwargs,
        ).images
        return images, depth_map
    



