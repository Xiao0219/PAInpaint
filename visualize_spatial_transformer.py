import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

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

        return x_transformed, theta

class MockVAEEncoder(nn.Module):
    """模拟VAE编码器，将图像编码为latent特征"""
    def __init__(self, latent_channels=4):
        super(MockVAEEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # 假设输入图像为 (3, 256, 256)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # (64, 128, 128)
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (128, 64, 64)
            nn.ReLU(True),
            nn.Conv2d(128, latent_channels, kernel_size=1), # (latent_channels, 64, 64)
        )
    
    def forward(self, x):
        return self.encoder(x)

class MockVAEDecoder(nn.Module):
    """模拟VAE解码器，将latent特征解码为图像"""
    def __init__(self, latent_channels=4):
        super(MockVAEDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 128, kernel_size=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (64, 128, 128)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1), # (3, 256, 256)
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.decoder(x)

def load_and_preprocess_image(image_path, size=(256, 256)):
    """加载并预处理图像"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化到[-1, 1]
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 添加batch维度

def denormalize_image(tensor):
    """反归一化图像张量用于显示"""
    tensor = tensor.clone()
    tensor = tensor * 0.5 + 0.5  # 从[-1, 1]转换到[0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

def visualize_transformation_matrix(theta_batch, save_path=None):
    """可视化变换矩阵参数"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Affine Transformation Parameters', fontsize=16)
    
    batch_size = theta_batch.shape[0]
    
    # 提取变换参数
    theta_np = theta_batch.detach().cpu().numpy()
    
    # 参数标签
    param_names = ['Scale X (θ₁₁)', 'Shear X (θ₁₂)', 'Translation X (θ₁₃)',
                   'Shear Y (θ₂₁)', 'Scale Y (θ₂₂)', 'Translation Y (θ₂₃)']
    
    for i in range(6):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 获取第i个参数的所有样本值
        param_values = theta_np[:, row if i < 3 else row, col if i < 3 else col-3]
        
        ax.bar(range(batch_size), param_values)
        ax.set_title(param_names[i])
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Parameter Value')
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for j, v in enumerate(param_values):
            ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_synthetic_test_data(batch_size=4):
    """创建合成测试数据来模拟不同类型的变换需求"""
    # 创建一些有特定模式的合成图像
    images = []
    
    for i in range(batch_size):
        # 创建不同的测试模式
        img = torch.zeros(3, 256, 256)
        
        if i == 0:
            # 网格模式
            img[0, ::32, :] = 1.0  # 红色水平线
            img[1, :, ::32] = 1.0  # 绿色垂直线
        elif i == 1:
            # 圆形模式
            y, x = torch.meshgrid(torch.arange(256), torch.arange(256), indexing='ij')
            center_y, center_x = 128, 128
            radius = 80
            mask = ((x - center_x)**2 + (y - center_y)**2) < radius**2
            img[2, mask] = 1.0  # 蓝色圆形
        elif i == 2:
            # 对角线模式
            for j in range(256):
                if j < 256:
                    img[:, j, j] = 1.0  # 白色对角线
        else:
            # 矩形模式
            img[0, 64:192, 64:192] = 1.0  # 红色矩形
            img[1, 80:176, 80:176] = 1.0  # 绿色内矩形
        
        images.append(img)
    
    # 归一化到[-1, 1]
    images = torch.stack(images)
    images = images * 2.0 - 1.0
    
    return images

def visualize_results(original_images, latent_features, transformed_latent, 
                     reconstructed_original, reconstructed_transformed, 
                     theta_batch, save_dir='visualization_results'):
    """综合可视化结果"""
    
    os.makedirs(save_dir, exist_ok=True)
    batch_size = original_images.shape[0]
    
    # 1. 可视化完整流程
    fig, axes = plt.subplots(batch_size, 5, figsize=(20, 4*batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # 原始图像
        axes[i, 0].imshow(denormalize_image(original_images[i]).permute(1, 2, 0))
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Latent特征 (显示第一个通道)
        latent_vis = latent_features[i, 0].detach().cpu()
        axes[i, 1].imshow(latent_vis, cmap='viridis')
        axes[i, 1].set_title('Latent Features (Ch.0)')
        axes[i, 1].axis('off')
        
        # 变换后的Latent特征
        transformed_vis = transformed_latent[i, 0].detach().cpu()
        axes[i, 2].imshow(transformed_vis, cmap='viridis')
        axes[i, 2].set_title('Transformed Latent (Ch.0)')
        axes[i, 2].axis('off')
        
        # 重建的原始图像
        axes[i, 3].imshow(denormalize_image(reconstructed_original[i]).permute(1, 2, 0))
        axes[i, 3].set_title('Reconstructed Original')
        axes[i, 3].axis('off')
        
        # 重建的变换图像
        axes[i, 4].imshow(denormalize_image(reconstructed_transformed[i]).permute(1, 2, 0))
        axes[i, 4].set_title('Reconstructed Transformed')
        axes[i, 4].axis('off')
    
    plt.suptitle('Spatial Transformer Visualization Pipeline', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pipeline_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 可视化变换矩阵
    visualize_transformation_matrix(theta_batch, f'{save_dir}/transformation_matrix.png')
    
    # 3. 可视化latent特征的差异
    fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4*batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        for ch in range(min(4, latent_features.shape[1])):
            original_ch = latent_features[i, ch].detach().cpu()
            transformed_ch = transformed_latent[i, ch].detach().cpu()
            
            # 计算差异
            diff = torch.abs(transformed_ch - original_ch)
            
            axes[i, ch].imshow(diff, cmap='hot')
            axes[i, ch].set_title(f'Sample {i+1}, Channel {ch+1} Difference')
            axes[i, ch].axis('off')
    
    plt.suptitle('Latent Feature Transformation Differences', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/latent_differences.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 初始化模型
    latent_channels = 4
    vae_encoder = MockVAEEncoder(latent_channels).to(device)
    vae_decoder = MockVAEDecoder(latent_channels).to(device)
    spatial_transformer = SpatialTransformer(latent_channels).to(device)
    
    # 设置为评估模式
    vae_encoder.eval()
    vae_decoder.eval()
    spatial_transformer.eval()
    
    print("Models initialized successfully!")
    
    # 方法1: 使用真实图像（如果有的话）
    test_image_dir = "test_images"
    if os.path.exists(test_image_dir):
        image_files = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.png'))]
        if image_files:
            print(f"Found {len(image_files)} test images")
            # 选择前4张图像
            selected_images = []
            for img_file in image_files[:4]:
                img_path = os.path.join(test_image_dir, img_file)
                img_tensor = load_and_preprocess_image(img_path)
                selected_images.append(img_tensor)
            
            if selected_images:
                test_images = torch.cat(selected_images, dim=0).to(device)
                print(f"Loaded {test_images.shape[0]} real test images")
            else:
                test_images = create_synthetic_test_data(4).to(device)
                print("Using synthetic test data")
        else:
            test_images = create_synthetic_test_data(4).to(device)
            print("No valid images found, using synthetic test data")
    else:
        test_images = create_synthetic_test_data(4).to(device)
        print("Test image directory not found, using synthetic test data")
    
    print(f"Test images shape: {test_images.shape}")
    
    # 执行完整的测试流程
    with torch.no_grad():
        # 1. 编码为latent特征
        latent_features = vae_encoder(test_images)
        print(f"Latent features shape: {latent_features.shape}")
        
        # 2. 应用空间变换
        transformed_latent, theta_batch = spatial_transformer(latent_features)
        print(f"Transformed latent shape: {transformed_latent.shape}")
        print(f"Transformation matrices shape: {theta_batch.shape}")
        
        # 3. 解码重建图像
        reconstructed_original = vae_decoder(latent_features)
        reconstructed_transformed = vae_decoder(transformed_latent)
        
        print(f"Reconstructed images shape: {reconstructed_original.shape}")
        
        # 4. 打印变换矩阵
        print("\nTransformation matrices:")
        for i, theta in enumerate(theta_batch):
            print(f"Sample {i+1}:")
            print(f"  Matrix: \n{theta.cpu().numpy()}")
            
            # 解释变换类型
            theta_np = theta.cpu().numpy()
            scale_x, shear_x, trans_x = theta_np[0]
            shear_y, scale_y, trans_y = theta_np[1]
            
            print(f"  - Scale: ({scale_x:.3f}, {scale_y:.3f})")
            print(f"  - Translation: ({trans_x:.3f}, {trans_y:.3f})")
            print(f"  - Shear: ({shear_x:.3f}, {shear_y:.3f})")
        
        # 5. 计算变换效果统计
        latent_diff = torch.mean(torch.abs(transformed_latent - latent_features), dim=[1,2,3])
        print(f"\nAverage latent feature differences per sample: {latent_diff.cpu().numpy()}")
        
        # 6. 可视化结果
        visualize_results(
            test_images.cpu(), 
            latent_features.cpu(), 
            transformed_latent.cpu(),
            reconstructed_original.cpu(), 
            reconstructed_transformed.cpu(),
            theta_batch.cpu()
        )
    
    print("\nVisualization complete! Check the 'visualization_results' directory for saved plots.")
    
    # 7. 保存模型状态用于进一步分析
    torch.save({
        'spatial_transformer_state': spatial_transformer.state_dict(),
        'sample_theta': theta_batch.cpu(),
        'latent_features': latent_features.cpu(),
        'transformed_latent': transformed_latent.cpu()
    }, 'visualization_results/spatial_transformer_test.pth')
    
    print("Model states and test results saved!")

if __name__ == "__main__":
    main()