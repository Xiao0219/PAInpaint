import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from visualize_spatial_transformer import SpatialTransformer, create_synthetic_test_data
import os

def analyze_localization_network(model, input_tensor, save_dir='analysis_results'):
    """分析定位网络的激活模式"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        x = input_tensor
        
        # 逐层分析定位网络
        activations = []
        layer_names = []
        
        # 第一个卷积层
        x = model.localization[0](x)  # Conv2d
        activations.append(x.clone())
        layer_names.append('Conv1 (7x7)')
        
        x = model.localization[1](x)  # MaxPool2d
        activations.append(x.clone())
        layer_names.append('MaxPool1')
        
        x = model.localization[2](x)  # ReLU
        activations.append(x.clone())
        layer_names.append('ReLU1')
        
        # 第二个卷积层
        x = model.localization[3](x)  # Conv2d
        activations.append(x.clone())
        layer_names.append('Conv2 (5x5)')
        
        x = model.localization[4](x)  # MaxPool2d
        activations.append(x.clone())
        layer_names.append('MaxPool2')
        
        x = model.localization[5](x)  # ReLU
        activations.append(x.clone())
        layer_names.append('ReLU2')
        
        # 可视化激活
        batch_size = input_tensor.shape[0]
        n_layers = len(activations)
        
        fig, axes = plt.subplots(batch_size, n_layers, figsize=(20, 4*batch_size))
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(batch_size):
            for j, (activation, name) in enumerate(zip(activations, layer_names)):
                # 显示第一个通道的激活
                act_vis = activation[i, 0].cpu().numpy()
                
                im = axes[i, j].imshow(act_vis, cmap='viridis')
                axes[i, j].set_title(f'{name}\nShape: {activation.shape[1:]}')
                axes[i, j].axis('off')
                plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)
        
        plt.suptitle('Localization Network Activations', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/localization_activations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return activations

def analyze_transformation_parameters(model, input_tensor, save_dir='analysis_results'):
    """分析变换参数的分布和统计特性"""
    model.eval()
    
    # 生成更多样本来分析参数分布
    batch_sizes = [1, 4, 8, 16]
    all_thetas = []
    
    with torch.no_grad():
        for bs in batch_sizes:
            # 复制输入到指定的batch size
            if bs <= input_tensor.shape[0]:
                test_input = input_tensor[:bs]
            else:
                # 重复样本
                repeat_times = (bs + input_tensor.shape[0] - 1) // input_tensor.shape[0]
                test_input = input_tensor.repeat(repeat_times, 1, 1, 1)[:bs]
            
            _, theta = model(test_input)
            all_thetas.append(theta.cpu())
    
    # 合并所有theta
    all_thetas = torch.cat(all_thetas, dim=0)
    
    # 分析参数统计
    theta_np = all_thetas.numpy()
    n_samples = theta_np.shape[0]
    
    # 提取各个参数
    scale_x = theta_np[:, 0, 0]
    shear_x = theta_np[:, 0, 1] 
    trans_x = theta_np[:, 0, 2]
    shear_y = theta_np[:, 1, 0]
    scale_y = theta_np[:, 1, 1]
    trans_y = theta_np[:, 1, 2]
    
    params = [scale_x, shear_x, trans_x, shear_y, scale_y, trans_y]
    param_names = ['Scale X', 'Shear X', 'Trans X', 'Shear Y', 'Scale Y', 'Trans Y']
    
    # 可视化参数分布
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (param, name) in enumerate(zip(params, param_names)):
        axes[i].hist(param, bins=20, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{name}\nMean: {np.mean(param):.3f}, Std: {np.std(param):.3f}')
        axes[i].set_xlabel('Parameter Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
        
        # 添加统计信息
        axes[i].axvline(np.mean(param), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(param):.3f}')
        axes[i].legend()
    
    plt.suptitle(f'Transformation Parameter Distributions (N={n_samples})', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/parameter_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 计算变换类型统计
    identity_threshold = 0.1
    is_identity = (np.abs(scale_x - 1) < identity_threshold) & \
                  (np.abs(scale_y - 1) < identity_threshold) & \
                  (np.abs(shear_x) < identity_threshold) & \
                  (np.abs(shear_y) < identity_threshold) & \
                  (np.abs(trans_x) < identity_threshold) & \
                  (np.abs(trans_y) < identity_threshold)
    
    print(f"\nTransformation Analysis:")
    print(f"Total samples: {n_samples}")
    print(f"Identity-like transformations: {np.sum(is_identity)} ({np.mean(is_identity)*100:.1f}%)")
    print(f"Non-identity transformations: {n_samples - np.sum(is_identity)} ({(1-np.mean(is_identity))*100:.1f}%)")
    
    return all_thetas

def test_transformation_invariance(model, save_dir='analysis_results'):
    """测试变换的不变性和一致性"""
    model.eval()
    
    # 创建具有已知变换的测试数据
    base_image = create_synthetic_test_data(1)
    
    # 手动创建一些具有明显特征的图像
    test_cases = []
    
    # 1. 中心有明显特征的图像
    center_feature = torch.zeros(1, 4, 64, 64)
    center_feature[0, :, 28:36, 28:36] = 1.0  # 中心方块
    test_cases.append(("Center Feature", center_feature))
    
    # 2. 角落有特征的图像
    corner_feature = torch.zeros(1, 4, 64, 64)
    corner_feature[0, :, 0:8, 0:8] = 1.0  # 左上角
    corner_feature[0, :, 56:64, 56:64] = -1.0  # 右下角
    test_cases.append(("Corner Features", corner_feature))
    
    # 3. 边缘有特征的图像
    edge_feature = torch.zeros(1, 4, 64, 64)
    edge_feature[0, :, :, 0:4] = 1.0  # 左边缘
    edge_feature[0, :, :, 60:64] = -1.0  # 右边缘
    test_cases.append(("Edge Features", edge_feature))
    
    results = []
    
    with torch.no_grad():
        for name, test_input in test_cases:
            transformed, theta = model(test_input)
            
            # 计算变换的效果
            transformation_magnitude = torch.norm(transformed - test_input).item()
            
            results.append({
                'name': name,
                'input': test_input,
                'transformed': transformed,
                'theta': theta,
                'magnitude': transformation_magnitude
            })
            
            print(f"{name}:")
            print(f"  Transformation matrix:\n{theta[0].numpy()}")
            print(f"  Transformation magnitude: {transformation_magnitude:.4f}")
    
    # 可视化结果
    fig, axes = plt.subplots(len(test_cases), 3, figsize=(12, 4*len(test_cases)))
    if len(test_cases) == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        # 原始输入 (显示第一个通道)
        axes[i, 0].imshow(result['input'][0, 0].cpu(), cmap='RdBu_r')
        axes[i, 0].set_title(f"{result['name']}\nOriginal (Ch.0)")
        axes[i, 0].axis('off')
        
        # 变换后 (显示第一个通道)
        axes[i, 1].imshow(result['transformed'][0, 0].cpu(), cmap='RdBu_r')
        axes[i, 1].set_title(f"Transformed (Ch.0)")
        axes[i, 1].axis('off')
        
        # 差异
        diff = torch.abs(result['transformed'][0, 0] - result['input'][0, 0]).cpu()
        im = axes[i, 2].imshow(diff, cmap='hot')
        axes[i, 2].set_title(f"Difference\nMag: {result['magnitude']:.3f}")
        axes[i, 2].axis('off')
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
    
    plt.suptitle('Transformation Invariance Test', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/invariance_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def gradient_analysis(model, input_tensor, save_dir='analysis_results'):
    """分析梯度流和可训练性"""
    model.train()  # 启用训练模式以计算梯度
    
    # 创建一个简单的损失函数来测试梯度
    input_tensor.requires_grad_(True)
    
    # 前向传播
    transformed, theta = model(input_tensor)
    
    # 简单的损失：鼓励某种特定的变换
    target_theta = torch.tensor([[[1.0, 0.0, 0.1], [0.0, 1.0, 0.1]]], device=theta.device)
    target_theta = target_theta.expand_as(theta)
    
    loss = F.mse_loss(theta, target_theta)
    
    # 反向传播
    loss.backward()
    
    # 分析梯度
    print(f"\nGradient Analysis:")
    print(f"Loss value: {loss.item():.6f}")
    
    # 检查模型参数的梯度
    total_grad_norm = 0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            param_count += 1
            print(f"  {name}: grad_norm = {grad_norm:.6f}, shape = {param.shape}")
        else:
            print(f"  {name}: No gradient computed")
    
    avg_grad_norm = total_grad_norm / param_count if param_count > 0 else 0
    print(f"Average gradient norm: {avg_grad_norm:.6f}")
    
    # 清理梯度
    model.zero_grad()
    input_tensor.requires_grad_(False)
    
    return {
        'loss': loss.item(),
        'avg_grad_norm': avg_grad_norm,
        'total_params': param_count
    }

def main():
    print("Starting detailed SpatialTransformer analysis...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型和测试数据
    model = SpatialTransformer(input_channels=4).to(device)
    test_data = create_synthetic_test_data(4).to(device)
    
    # 模拟latent特征 (从3通道转换到4通道，降采样到64x64)
    latent_test = F.interpolate(test_data, size=(64, 64), mode='bilinear', align_corners=True)
    # 添加一个额外的通道来模拟4通道latent
    extra_channel = torch.mean(latent_test, dim=1, keepdim=True)  # 从RGB计算平均值作为第4通道
    latent_test = torch.cat([latent_test, extra_channel], dim=1)  # 变成4通道
    
    print(f"Test data shape: {latent_test.shape}")
    
    # 创建结果目录
    save_dir = 'analysis_results'
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 分析定位网络
    print("\n1. Analyzing localization network...")
    activations = analyze_localization_network(model, latent_test, save_dir)
    
    # 2. 分析变换参数
    print("\n2. Analyzing transformation parameters...")
    all_thetas = analyze_transformation_parameters(model, latent_test, save_dir)
    
    # 3. 测试变换不变性
    print("\n3. Testing transformation invariance...")
    invariance_results = test_transformation_invariance(model, save_dir)
    
    # 4. 梯度分析
    print("\n4. Analyzing gradients...")
    grad_results = gradient_analysis(model, latent_test.clone(), save_dir)
    
    # 5. 保存分析结果
    analysis_summary = {
        'model_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'gradient_analysis': grad_results,
        'sample_transformations': all_thetas[:10].numpy().tolist(),  # 保存前10个样本
    }
    
    torch.save(analysis_summary, f'{save_dir}/analysis_summary.pth')
    
    print(f"\nAnalysis complete!")
    print(f"Total parameters: {analysis_summary['model_params']:,}")
    print(f"Trainable parameters: {analysis_summary['trainable_params']:,}")
    print(f"Results saved in '{save_dir}' directory")

if __name__ == "__main__":
    main()