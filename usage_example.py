#!/usr/bin/env python3
"""
SpatialTransformer使用示例

这个脚本展示了如何在实际的图像修复项目中集成和使用SpatialTransformer模块。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from visualize_spatial_transformer import SpatialTransformer

class ImageInpaintingWithSpatialTransformer(nn.Module):
    """
    集成SpatialTransformer的图像修复模型示例
    """
    def __init__(self, vae_encoder, vae_decoder, spatial_transformer=None):
        super().__init__()
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        
        # 如果没有提供SpatialTransformer，创建一个新的
        if spatial_transformer is None:
            latent_channels = 4  # 假设VAE的latent有4个通道
            self.spatial_transformer = SpatialTransformer(latent_channels)
        else:
            self.spatial_transformer = spatial_transformer
            
        self.use_spatial_transform = True
    
    def forward(self, masked_image, mask, reference_image=None):
        """
        前向传播过程
        
        Args:
            masked_image: 待修复的掩码图像 (B, 3, H, W)
            mask: 掩码 (B, 1, H, W)
            reference_image: 参考图像（可选，用于对齐）(B, 3, H, W)
        
        Returns:
            inpainted_image: 修复后的图像 (B, 3, H, W)
            transformation_info: 变换信息字典
        """
        # 1. 编码到latent空间
        latent = self.vae_encoder(masked_image)
        
        transformation_info = {}
        
        # 2. 如果启用空间变换且有参考图像，进行对齐
        if self.use_spatial_transform and reference_image is not None:
            # 编码参考图像
            ref_latent = self.vae_encoder(reference_image)
            
            # 应用空间变换对齐
            aligned_latent, theta = self.spatial_transformer(latent)
            
            transformation_info = {
                'transformation_matrix': theta,
                'spatial_alignment_applied': True,
                'original_latent': latent,
                'aligned_latent': aligned_latent
            }
            
            # 使用对齐后的latent进行后续处理
            processing_latent = aligned_latent
        else:
            processing_latent = latent
            transformation_info['spatial_alignment_applied'] = False
        
        # 3. 在latent空间进行修复处理（这里简化为直接解码）
        # 在实际应用中，这里会有更复杂的修复网络
        inpainted_latent = self.inpaint_in_latent_space(processing_latent, mask)
        
        # 4. 解码回图像空间
        inpainted_image = self.vae_decoder(inpainted_latent)
        
        return inpainted_image, transformation_info
    
    def inpaint_in_latent_space(self, latent, mask):
        """
        在latent空间进行修复（示例实现）
        
        在实际应用中，这里会是你的修复网络的核心部分
        """
        # 简化示例：直接返回latent（在实际中会有复杂的修复逻辑）
        return latent
    
    def set_spatial_transform_enabled(self, enabled):
        """启用或禁用空间变换"""
        self.use_spatial_transform = enabled

# 使用示例
def example_usage():
    """展示如何使用集成了SpatialTransformer的修复模型"""
    
    # 模拟模型组件（在实际应用中，这些会是预训练的模型）
    from visualize_spatial_transformer import MockVAEEncoder, MockVAEDecoder
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化组件
    vae_encoder = MockVAEEncoder(latent_channels=4).to(device)
    vae_decoder = MockVAEDecoder(latent_channels=4).to(device)
    spatial_transformer = SpatialTransformer(input_channels=4).to(device)
    
    # 创建完整的修复模型
    inpainting_model = ImageInpaintingWithSpatialTransformer(
        vae_encoder, vae_decoder, spatial_transformer
    ).to(device)
    
    # 模拟输入数据
    batch_size = 2
    masked_image = torch.randn(batch_size, 3, 256, 256).to(device)
    mask = torch.randint(0, 2, (batch_size, 1, 256, 256)).float().to(device)
    reference_image = torch.randn(batch_size, 3, 256, 256).to(device)
    
    # 设置为评估模式
    inpainting_model.eval()
    
    with torch.no_grad():
        # 进行修复
        inpainted_image, transform_info = inpainting_model(
            masked_image, mask, reference_image
        )
        
        print("修复完成！")
        print(f"输入图像尺寸: {masked_image.shape}")
        print(f"输出图像尺寸: {inpainted_image.shape}")
        print(f"是否应用空间对齐: {transform_info['spatial_alignment_applied']}")
        
        if transform_info['spatial_alignment_applied']:
            theta = transform_info['transformation_matrix']
            print(f"变换矩阵形状: {theta.shape}")
            print(f"第一个样本的变换矩阵:\n{theta[0].cpu().numpy()}")

# 训练示例
def example_training():
    """展示如何训练包含SpatialTransformer的模型"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型（这里使用简化的组件）
    from visualize_spatial_transformer import MockVAEEncoder, MockVAEDecoder
    
    vae_encoder = MockVAEEncoder(latent_channels=4).to(device)
    vae_decoder = MockVAEDecoder(latent_channels=4).to(device)
    
    model = ImageInpaintingWithSpatialTransformer(
        vae_encoder, vae_decoder
    ).to(device)
    
    # 优化器（通常只训练SpatialTransformer，VAE保持固定）
    optimizer = torch.optim.Adam(
        model.spatial_transformer.parameters(), 
        lr=0.001
    )
    
    # 训练循环示例
    model.train()
    for epoch in range(5):  # 简化的训练循环
        # 模拟训练数据
        masked_images = torch.randn(4, 3, 256, 256).to(device)
        masks = torch.randint(0, 2, (4, 1, 256, 256)).float().to(device)
        reference_images = torch.randn(4, 3, 256, 256).to(device)
        target_images = torch.randn(4, 3, 256, 256).to(device)
        
        # 前向传播
        inpainted, transform_info = model(masked_images, masks, reference_images)
        
        # 计算损失（示例）
        reconstruction_loss = F.mse_loss(inpainted, target_images)
        
        # 如果应用了空间变换，可以添加额外的正则化
        total_loss = reconstruction_loss
        if transform_info['spatial_alignment_applied']:
            # 例如：鼓励小的变换
            theta = transform_info['transformation_matrix']
            identity = torch.eye(2, 3).unsqueeze(0).expand_as(theta).to(device)
            regularization = F.mse_loss(theta, identity) * 0.01
            total_loss = total_loss + regularization
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss.item():.6f}")
    
    print("训练完成！")

# 实际集成建议
def integration_tips():
    """实际项目集成的建议和技巧"""
    
    tips = """
    ## SpatialTransformer集成建议
    
    ### 1. 何时使用SpatialTransformer
    - 处理存在明显几何失配的图像修复任务
    - 需要将参考图像与目标图像对齐时
    - VAE latent空间中存在空间不一致性时
    
    ### 2. 训练策略
    - 首先固定VAE参数，只训练SpatialTransformer
    - 使用合成的错位数据对进行预训练
    - 然后在真实数据上进行微调
    - 考虑使用感知损失而不是简单的MSE损失
    
    ### 3. 性能优化
    - 在推理时可以选择性地启用/禁用空间变换
    - 对于已经对齐的图像，跳过变换以节省计算
    - 可以预先计算变换矩阵并缓存结果
    
    ### 4. 调试技巧
    - 可视化变换前后的latent特征差异
    - 监控变换矩阵的统计信息
    - 使用梯度检查验证反向传播
    
    ### 5. 扩展可能性
    - 支持多尺度变换
    - 添加注意力机制
    - 结合其他几何变换（如透视变换）
    """
    
    print(tips)

if __name__ == "__main__":
    print("SpatialTransformer使用示例")
    print("=" * 50)
    
    print("\n1. 基础使用示例:")
    example_usage()
    
    print("\n2. 训练示例:")
    example_training()
    
    print("\n3. 集成建议:")
    integration_tips()