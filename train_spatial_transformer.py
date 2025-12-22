import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from visualize_spatial_transformer import SpatialTransformer, MockVAEEncoder, MockVAEDecoder, create_synthetic_test_data, denormalize_image
import os

class SpatialTransformerTrainer:
    def __init__(self, latent_channels=4, device='cpu'):
        self.device = device
        self.latent_channels = latent_channels
        
        # 初始化模型
        self.vae_encoder = MockVAEEncoder(latent_channels).to(device)
        self.vae_decoder = MockVAEDecoder(latent_channels).to(device)
        self.spatial_transformer = SpatialTransformer(latent_channels).to(device)
        
        # 固定VAE参数，只训练SpatialTransformer
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
        for param in self.vae_decoder.parameters():
            param.requires_grad = False
            
        self.vae_encoder.eval()
        self.vae_decoder.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.spatial_transformer.parameters(), lr=0.001)
        
        # 损失历史
        self.loss_history = []
        self.transformation_history = []

def create_misaligned_data(batch_size=8):
    """创建故意错位的数据对来训练空间变换器"""
    # 创建原始图像
    original_images = create_synthetic_test_data(batch_size)
    
    # 创建对应的"目标"图像（经过已知变换的版本）
    target_images = []
    true_transformations = []
    
    for i in range(batch_size):
        img = original_images[i:i+1]  # 保持batch维度
        
        # 随机生成变换参数
        scale_x = torch.normal(1.0, 0.1, (1,)).clamp(0.8, 1.2)
        scale_y = torch.normal(1.0, 0.1, (1,)).clamp(0.8, 1.2)
        trans_x = torch.normal(0.0, 0.1, (1,)).clamp(-0.2, 0.2)
        trans_y = torch.normal(0.0, 0.1, (1,)).clamp(-0.2, 0.2)
        shear_x = torch.normal(0.0, 0.05, (1,)).clamp(-0.1, 0.1)
        shear_y = torch.normal(0.0, 0.05, (1,)).clamp(-0.1, 0.1)
        
        # 构建变换矩阵
        theta = torch.tensor([[
            [scale_x.item(), shear_x.item(), trans_x.item()],
            [shear_y.item(), scale_y.item(), trans_y.item()]
        ]], dtype=torch.float32)
        
        # 应用变换
        grid = F.affine_grid(theta, img.size(), align_corners=True)
        transformed_img = F.grid_sample(img, grid, align_corners=True)
        
        target_images.append(transformed_img)
        true_transformations.append(theta)
    
    target_images = torch.cat(target_images, dim=0)
    true_transformations = torch.cat(true_transformations, dim=0)
    
    return original_images, target_images, true_transformations

def alignment_loss(reconstructed_original, reconstructed_transformed, target_images):
    """计算对齐损失 - 希望变换后的图像能够匹配目标图像"""
    # 主要损失：变换后的重建图像应该接近目标图像
    alignment_loss = F.mse_loss(reconstructed_transformed, target_images)
    
    # 辅助损失：保持原始图像的重建质量
    reconstruction_loss = F.mse_loss(reconstructed_original, target_images) * 0.1
    
    return alignment_loss + reconstruction_loss

def train_spatial_transformer(epochs=50, batch_size=8, save_dir='training_results'):
    """训练空间变换器来学习对齐变换"""
    
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # 初始化训练器
    trainer = SpatialTransformerTrainer(device=device)
    
    # 训练循环
    for epoch in range(epochs):
        # 生成训练数据
        original_images, target_images, true_transformations = create_misaligned_data(batch_size)
        original_images = original_images.to(device)
        target_images = target_images.to(device)
        true_transformations = true_transformations.to(device)
        
        # 前向传播
        with torch.no_grad():
            # 编码为latent
            latent_original = trainer.vae_encoder(original_images)
            latent_target = trainer.vae_encoder(target_images)
        
        # 应用空间变换
        trainer.spatial_transformer.train()
        transformed_latent, predicted_theta = trainer.spatial_transformer(latent_original)
        
        # 解码重建
        with torch.no_grad():
            reconstructed_original = trainer.vae_decoder(latent_original)
            reconstructed_transformed = trainer.vae_decoder(transformed_latent)
        
        # 计算损失
        # 方法1: 直接对齐损失
        loss_alignment = alignment_loss(reconstructed_original, reconstructed_transformed, target_images)
        
        # 方法2: 在latent空间的对齐损失（更直接）
        loss_latent = F.mse_loss(transformed_latent, latent_target) * 10.0
        
        # 方法3: 变换参数的监督损失（如果我们知道真实变换）
        loss_theta = F.mse_loss(predicted_theta, true_transformations) * 1.0
        
        # 总损失
        total_loss = loss_alignment + loss_latent + loss_theta
        
        # 反向传播
        trainer.optimizer.zero_grad()
        total_loss.backward()
        trainer.optimizer.step()
        
        # 记录
        trainer.loss_history.append(total_loss.item())
        trainer.transformation_history.append(predicted_theta.detach().cpu())
        
        # 打印进度
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}")
            print(f"  Total Loss: {total_loss.item():.6f}")
            print(f"  Alignment Loss: {loss_alignment.item():.6f}")
            print(f"  Latent Loss: {loss_latent.item():.6f}")
            print(f"  Theta Loss: {loss_theta.item():.6f}")
            
            # 打印一些预测的变换参数
            print(f"  Sample predicted theta:\n{predicted_theta[0].detach().cpu().numpy()}")
            print(f"  Sample true theta:\n{true_transformations[0].cpu().numpy()}")
            print()
    
    return trainer

def visualize_training_progress(trainer, save_dir='training_results'):
    """可视化训练进度"""
    
    # 1. 损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(trainer.loss_history)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f'{save_dir}/loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 变换参数的演变
    if len(trainer.transformation_history) > 0:
        transformations = torch.stack(trainer.transformation_history)  # (epochs, batch_size, 2, 3)
        
        # 分析第一个样本的变换参数演变
        first_sample_transforms = transformations[:, 0, :, :]  # (epochs, 2, 3)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        param_names = ['Scale X', 'Shear X', 'Trans X', 'Shear Y', 'Scale Y', 'Trans Y']
        
        for i in range(6):
            row = i // 3
            col = i % 3
            
            if i < 3:
                param_values = first_sample_transforms[:, 0, i].numpy()
            else:
                param_values = first_sample_transforms[:, 1, i-3].numpy()
            
            axes[row, col].plot(param_values)
            axes[row, col].set_title(f'{param_names[i]} Evolution')
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('Parameter Value')
            axes[row, col].grid(True)
        
        plt.suptitle('Transformation Parameters Evolution During Training', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/parameter_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()

def test_trained_model(trainer, save_dir='training_results'):
    """测试训练后的模型"""
    trainer.spatial_transformer.eval()
    
    # 生成测试数据
    test_original, test_target, test_true_theta = create_misaligned_data(4)
    test_original = test_original.to(trainer.device)
    test_target = test_target.to(trainer.device)
    
    with torch.no_grad():
        # 编码
        latent_original = trainer.vae_encoder(test_original)
        latent_target = trainer.vae_encoder(test_target)
        
        # 预测变换
        transformed_latent, predicted_theta = trainer.spatial_transformer(latent_original)
        
        # 解码
        reconstructed_original = trainer.vae_decoder(latent_original)
        reconstructed_transformed = trainer.vae_decoder(transformed_latent)
    
    # 可视化测试结果
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    for i in range(4):
        # 原始图像
        axes[i, 0].imshow(denormalize_image(test_original[i]).permute(1, 2, 0))
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # 目标图像
        axes[i, 1].imshow(denormalize_image(test_target[i]).permute(1, 2, 0))
        axes[i, 1].set_title('Target')
        axes[i, 1].axis('off')
        
        # 重建的原始图像
        axes[i, 2].imshow(denormalize_image(reconstructed_original[i]).permute(1, 2, 0))
        axes[i, 2].set_title('Reconstructed Original')
        axes[i, 2].axis('off')
        
        # 变换后的重建图像
        axes[i, 3].imshow(denormalize_image(reconstructed_transformed[i]).permute(1, 2, 0))
        axes[i, 3].set_title('Transformed Reconstruction')
        axes[i, 3].axis('off')
    
    plt.suptitle('Trained Model Test Results', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印变换参数比较
    print("Transformation Parameters Comparison:")
    for i in range(4):
        print(f"\nSample {i+1}:")
        print(f"True transformation:\n{test_true_theta[i].numpy()}")
        print(f"Predicted transformation:\n{predicted_theta[i].cpu().numpy()}")
        
        # 计算误差
        error = torch.abs(predicted_theta[i].cpu() - test_true_theta[i]).mean()
        print(f"Average absolute error: {error.item():.6f}")

def main():
    print("Starting SpatialTransformer training demonstration...")
    
    # 训练模型
    trainer = train_spatial_transformer(epochs=100, batch_size=8)
    
    # 可视化训练进度
    visualize_training_progress(trainer)
    
    # 测试训练后的模型
    test_trained_model(trainer)
    
    # 保存训练后的模型
    torch.save({
        'spatial_transformer_state': trainer.spatial_transformer.state_dict(),
        'loss_history': trainer.loss_history,
        'final_transformations': trainer.transformation_history[-1] if trainer.transformation_history else None
    }, 'training_results/trained_spatial_transformer.pth')
    
    print("Training complete! Results saved in 'training_results' directory.")

if __name__ == "__main__":
    main()