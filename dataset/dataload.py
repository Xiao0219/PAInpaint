from tqdm.auto import tqdm
from torch.utils.data import Dataset
import os
from glob import glob
import random
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import torch
import math
from transformers import CLIPImageProcessor
from torchvision.transforms import functional as F


# === import Depth Anything ===
import sys
sys.path.append("./depthanything")
from torchvision.transforms import Compose
from depthanything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

# =====================
# Data Augmentation Helper Functions
# =====================

def apply_geometric_transform(tar_image, ref_image, mask, transform_params):
    """
    Apply consistent geometric transformations to target image, reference image, and mask.

    Args:
        tar_image: PIL Image - target image
        ref_image: PIL Image - reference image
        mask: PIL Image - mask
        transform_params: dict - transformation parameters

    Returns:
        tuple: (transformed_tar_image, transformed_ref_image, transformed_mask)
    """
    # Apply horizontal flip
    if transform_params.get('hflip', False):
        tar_image = F.hflip(tar_image)
        ref_image = F.hflip(ref_image)
        mask = F.hflip(mask)

    # Apply vertical flip
    if transform_params.get('vflip', False):
        tar_image = F.vflip(tar_image)
        ref_image = F.vflip(ref_image)
        mask = F.vflip(mask)

    # Apply rotation
    if 'rotation' in transform_params:
        angle = transform_params['rotation']
        tar_image = F.rotate(tar_image, angle, interpolation=F.InterpolationMode.BILINEAR, fill=0)
        ref_image = F.rotate(ref_image, angle, interpolation=F.InterpolationMode.BILINEAR, fill=0)
        mask = F.rotate(mask, angle, interpolation=F.InterpolationMode.NEAREST, fill=0)

    # Apply perspective transform
    if 'perspective' in transform_params:
        distortion_scale = transform_params['perspective']
        width, height = tar_image.size

        # 生成一致的随机偏移
        random_state = random.getstate()

        # 生成透视变换的起始点和终点
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = []
        for i in range(4):
            x_offset = random.uniform(-distortion_scale * width, distortion_scale * width)
            y_offset = random.uniform(-distortion_scale * height, distortion_scale * height)
            endpoints.append([startpoints[i][0] + x_offset, startpoints[i][1] + y_offset])

        # 应用到目标图像
        random.setstate(random_state)
        tar_image = F.perspective(tar_image, startpoints, endpoints,
                                interpolation=F.InterpolationMode.BILINEAR, fill=0)

        # 应用到参考图像
        random.setstate(random_state)
        ref_image = F.perspective(ref_image, startpoints, endpoints,
                                interpolation=F.InterpolationMode.BILINEAR, fill=0)

        # 应用到掩膜
        random.setstate(random_state)
        mask = F.perspective(mask, startpoints, endpoints,
                           interpolation=F.InterpolationMode.NEAREST, fill=0)

    # Apply scaling (resize)
    if 'scale' in transform_params:
        scale_factor = transform_params['scale']
        original_size = tar_image.size
        new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))

        tar_image = F.resize(tar_image, new_size, interpolation=F.InterpolationMode.BILINEAR)
        ref_image = F.resize(ref_image, new_size, interpolation=F.InterpolationMode.BILINEAR)
        mask = F.resize(mask, new_size, interpolation=F.InterpolationMode.NEAREST)

    # Apply random crop
    if 'crop' in transform_params:
        crop_params = transform_params['crop']
        tar_image = F.crop(tar_image, *crop_params)
        ref_image = F.crop(ref_image, *crop_params)
        mask = F.crop(mask, *crop_params)

    return tar_image, ref_image, mask


def get_random_crop_params(img_size, crop_scale):
    """
    Get random crop parameters for consistent cropping across images and masks.

    Args:
        img_size: tuple - (width, height) of the image
        crop_scale: tuple - (min_scale, max_scale) for crop size

    Returns:
        tuple: (top, left, height, width) for cropping
    """
    width, height = img_size
    scale = random.uniform(crop_scale[0], crop_scale[1])

    crop_width = int(width * scale)
    crop_height = int(height * scale)

    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)

    return top, left, crop_height, crop_width


# =====================
# MVImageNetDataset
# 新增 start_ratio, end_ratio 用于灵活控制采样区段
# =====================

class MVImageNetDataset(Dataset):
    def __init__(self, root_dir, resolution=512, transform=None, tokenizer=None,
                 start_ratio: float = 0.0, end_ratio: float = 0.1,
                 class_start_ratio: float = 0.0, class_end_ratio: float = 1.0,
                 allowed_prefixes=None, augmentation_config=None):

        # 参数检查
        if not (0.0 <= start_ratio < end_ratio <= 1.0):
            raise ValueError(f"start_ratio ({start_ratio}) 必须小于 end_ratio ({end_ratio})，并且二者介于 0~1 之间")
        if not (0.0 <= class_start_ratio < class_end_ratio <= 1.0):
            raise ValueError(
                f"class_start_ratio ({class_start_ratio}) 必须小于 class_end_ratio ({class_end_ratio})，并且二者介于 0~1 之间")

        self.root_dir = root_dir
        self.resolution = resolution
        self.transform = transform
        self.tokenizer = tokenizer
        self.samples = []

        # 设置默认的数据增强配置
        self.aug_config = {
            'enabled': True,                # 是否启用数据增强
            'hflip_prob': 0.5,              # 水平翻转概率
            'vflip_prob': 0.0,              # 垂直翻转概率
            'rotate_prob': 0.3,             # 旋转概率
            'rotate_degrees': 10,           # 最大旋转角度
            'scale_prob': 0.3,              # 缩放概率
            'scale_range': (0.9, 1.1),      # 缩放范围
            'crop_prob': 0.0,               # 随机裁剪概率
            'crop_scale': (0.8, 1.0),       # 裁剪比例范围
            'color_jitter_prob': 0.3,       # 颜色抖动概率
            'brightness': 0.1,              # 亮度调整范围
            'contrast': 0.1,                # 对比度调整范围
            'saturation': 0.1,              # 饱和度调整范围
            'hue': 0.05,                    # 色调调整范围
            'gaussian_blur_prob': 0.1,      # 高斯模糊概率
            'gaussian_blur_sigma': (0.1, 2.0),  # 高斯模糊标准差范围
            'elastic_transform_prob': 0.0,  # 弹性变换概率（对修复任务可能过于激进）
            'perspective_prob': 0.1,        # 透视变换概率
            'perspective_distortion': 0.1,  # 透视变换强度
        }

        # 更新数据增强配置
        if augmentation_config is not None:
            self.aug_config.update(augmentation_config)

        # 基本变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.toTensor = transforms.ToTensor()

        # 颜色抖动变换（仅用于图像，不用于掩膜）
        self.color_jitter = transforms.ColorJitter(
            brightness=self.aug_config['brightness'],
            contrast=self.aug_config['contrast'],
            saturation=self.aug_config['saturation'],
            hue=self.aug_config['hue']
        )

        self.clip_image_processor = CLIPImageProcessor()
        print(f"Scanning dataset in: {root_dir}")

        # 允许的文件名前缀
        # 默认取 011~019；也可以在初始化时通过 allowed_prefixes 参数自定义
        if allowed_prefixes is None:
            self.allowed_prefixes = {f"{i:03d}" for i in range(11, 20)}
        else:
            # 支持传入 list / tuple / set，成员可以是 int 或 str
            try:
                self.allowed_prefixes = {f"{int(p):03d}" for p in allowed_prefixes}
            except Exception as e:
                raise ValueError(f"allowed_prefixes 参数必须是数字或数字字符串的可迭代对象，错误信息: {e}")
        
        # 记录采样区间，方便调试
        self.start_ratio = start_ratio  # 序列级采样区间
        self.end_ratio = end_ratio
        self.class_start_ratio = class_start_ratio  # 类别级采样区间
        self.class_end_ratio = class_end_ratio

        # ------------------ 类别采样 ------------------
        # 仅取 root_dir 下第一层目录作为"类别"
        all_class_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                          if os.path.isdir(os.path.join(root_dir, d))]
        all_class_dirs_sorted = sorted(all_class_dirs)
        if len(all_class_dirs_sorted) == 0:
            raise RuntimeError(f"No class directories found in {root_dir}")

        cls_start_idx = round(len(all_class_dirs_sorted) * self.class_start_ratio)
        cls_end_idx = max(cls_start_idx + 1, round(len(all_class_dirs_sorted) * self.class_end_ratio))
        selected_class_dirs = all_class_dirs_sorted[cls_start_idx:cls_end_idx]

        # 逐类别遍历
        for class_dir in tqdm(selected_class_dirs, desc="Loading dataset", leave=False):
            seq_dirs = [d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))]

            # 序列采样：根据 start_ratio~end_ratio 取指定百分比区段
            seq_dirs_sorted = sorted(seq_dirs)
            if len(seq_dirs_sorted) == 0:
                continue

            start_idx = round(len(seq_dirs_sorted) * self.start_ratio)
            end_idx = max(start_idx + 1, round(len(seq_dirs_sorted) * self.end_ratio))
            selected_seq_dirs = seq_dirs_sorted[start_idx:end_idx]
                
            for seq_dir in selected_seq_dirs:
                full_seq_dir = os.path.join(class_dir, seq_dir)
                # 直接列出目录内容
                try:
                    files = set(os.listdir(full_seq_dir))
                except:
                    continue
                    
                # 找出所有tar_image文件
                tar_images = [f for f in files if "tar_image" in f]
                
                for tar_file in tar_images:
                    num = tar_file.split("tar_image")[0]
                    # 文件采样：仅保留指定前缀
                    if num not in self.allowed_prefixes:
                        continue
                    ext = os.path.splitext(tar_file)[1]
                    
                    # 检查配套文件是否都存在
                    ref_file = f"{num}ref_image{ext}"
                    mask_file = f"{num}mask.png"
                    caption_file = f"{num}caption.txt"
                    
                    if ref_file in files and mask_file in files and caption_file in files:
                        self.samples.append((
                            os.path.join(full_seq_dir, tar_file),
                            os.path.join(full_seq_dir, ref_file),
                            os.path.join(full_seq_dir, mask_file),
                            os.path.join(full_seq_dir, caption_file)
                        ))
        
        print(f"Found {len(self.samples)} valid samples")
        if len(self.samples) == 0:
            print("Warning: No valid samples found! Please check the dataset structure.")
            # 打印一些目录结构信息以帮助调试
            print(f"Example class dirs: {glob(os.path.join(root_dir, '*'))[:5]}")
            if glob(os.path.join(root_dir, '*')):
                first_class = glob(os.path.join(root_dir, '*'))[0]
                print(f"Example sequence dirs in first class: {glob(os.path.join(first_class, '*'))[:5]}")
                if glob(os.path.join(first_class, '*')):
                    first_seq = glob(os.path.join(first_class, '*'))[0]
                    print(f"Files in first sequence: {os.listdir(first_seq)}")

    def __len__(self):
        return len(self.samples)

    def update_augmentation_config(self, new_config):
        """
        更新数据增强配置

        Args:
            new_config (dict): 新的配置参数

        Example:
            dataset.update_augmentation_config({
                'enabled': True,
                'hflip_prob': 0.7,
                'color_jitter_prob': 0.5,
                'brightness': 0.2
            })
        """
        self.aug_config.update(new_config)

        # 更新颜色抖动变换
        self.color_jitter = transforms.ColorJitter(
            brightness=self.aug_config['brightness'],
            contrast=self.aug_config['contrast'],
            saturation=self.aug_config['saturation'],
            hue=self.aug_config['hue']
        )

        print(f"Updated augmentation config: {self.aug_config}")

    def get_augmentation_config(self):
        """
        获取当前的数据增强配置

        Returns:
            dict: 当前的配置参数
        """
        return self.aug_config.copy()

    def disable_augmentation(self):
        """禁用数据增强"""
        self.aug_config['enabled'] = False
        print("Data augmentation disabled")

    def enable_augmentation(self):
        """启用数据增强"""
        self.aug_config['enabled'] = True
        print("Data augmentation enabled")

    def __getitem__(self, idx):
        tar_path, ref_path, mask_path, caption_path = self.samples[idx]

        # 读取图像
        tar_image = Image.open(tar_path).resize((self.resolution,self.resolution)).convert("RGB")
        ref_image = Image.open(ref_path).resize((self.resolution,self.resolution)).convert("RGB")
        mask = Image.open(mask_path).resize((self.resolution,self.resolution))


        if tar_image is None:
            print(f"Error loading target image: {tar_path}")
            # tar_image = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        if ref_image is None:
            print(f"Error loading reference image: {ref_path}")
            # ref_image = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        if mask is None:
            print(f"Error loading mask: {mask_path}")
            # mask = np.ones((self.resolution, self.resolution), dtype=np.uint8) * 255

        # ==================== 数据增强 ====================
        if self.aug_config['enabled']:
            # 准备几何变换参数
            transform_params = {}

            # 水平翻转
            if random.random() < self.aug_config['hflip_prob']:
                transform_params['hflip'] = True

            # 垂直翻转
            if random.random() < self.aug_config['vflip_prob']:
                transform_params['vflip'] = True

            # 旋转
            if random.random() < self.aug_config['rotate_prob']:
                angle = random.uniform(-self.aug_config['rotate_degrees'],
                                     self.aug_config['rotate_degrees'])
                transform_params['rotation'] = angle

            # 缩放
            if random.random() < self.aug_config['scale_prob']:
                scale = random.uniform(self.aug_config['scale_range'][0],
                                     self.aug_config['scale_range'][1])
                transform_params['scale'] = scale

            # 随机裁剪
            if random.random() < self.aug_config['crop_prob']:
                crop_params = get_random_crop_params(tar_image.size, self.aug_config['crop_scale'])
                transform_params['crop'] = crop_params

            # 透视变换
            if random.random() < self.aug_config['perspective_prob']:
                transform_params['perspective'] = self.aug_config['perspective_distortion']

            # 应用几何变换（同时应用到目标图像、参考图像和掩膜）
            if transform_params:
                tar_image, ref_image, mask = apply_geometric_transform(
                    tar_image, ref_image, mask, transform_params)

            # 应用颜色抖动（仅应用到图像，不应用到掩膜）
            if random.random() < self.aug_config['color_jitter_prob']:
                tar_image = self.color_jitter(tar_image)
                ref_image = self.color_jitter(ref_image)

            # 应用高斯模糊（仅应用到图像，不应用到掩膜）
            if random.random() < self.aug_config['gaussian_blur_prob']:
                sigma = random.uniform(self.aug_config['gaussian_blur_sigma'][0],
                                     self.aug_config['gaussian_blur_sigma'][1])
                tar_image = F.gaussian_blur(tar_image, kernel_size=3, sigma=sigma)
                ref_image = F.gaussian_blur(ref_image, kernel_size=3, sigma=sigma)

        # 确保图像尺寸正确（如果经过缩放或裁剪可能需要重新调整）
        if tar_image.size != (self.resolution, self.resolution):
            tar_image = tar_image.resize((self.resolution, self.resolution), Image.BILINEAR)
            ref_image = ref_image.resize((self.resolution, self.resolution), Image.BILINEAR)
            mask = mask.resize((self.resolution, self.resolution), Image.NEAREST)

        # create depth image
        depth_np = np.array(tar_image)
        depth_image = transform({'image': depth_np})['image']
        depth_image = torch.from_numpy(depth_image)

            
        # 读取文本
        try:
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        except Exception as e:
            print(f"Error loading caption {caption_path}: {e}")
            caption = ""
        

        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < 0.05:
            drop_image_embed = 1
        elif rand_num < 0.1:  # 0.55: #0.1:
            caption = ""
        elif rand_num < 0.15:  # 0.6: #0.15:
            caption = ""
            drop_image_embed = 1
        
        # 使用 tokenizer 处理文本
        if self.tokenizer is not None:
            caption = self.tokenizer(
                caption,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids[0]

        tar_images = self.transform(tar_image)
        ref_images = self.transform(ref_image)
        masks = self.toTensor(mask)
        mask = masks[:1]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        masked_images = tar_images * (1 - mask)

        clip_image = self.clip_image_processor(images=ref_image, return_tensors="pt").pixel_values
        
        return {
            "tar_image": tar_images,
            "ref_image": ref_images,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "caption": caption,
            "mask": masks,
            "masked_image": masked_images,
            'name': tar_path,
            "depth_image": depth_image,
        }


def collate_fn(samples):
    tar_images = torch.stack([example["tar_image"] for example in samples]).to(
        memory_format=torch.contiguous_format).float()
    ref_images = torch.stack([example["ref_image"] for example in samples]).to(
        memory_format=torch.contiguous_format).float()
    masked_images = torch.stack([example["masked_image"] for example in samples]).to(
        memory_format=torch.contiguous_format).float()
    masks = torch.stack([example["mask"] for example in samples]).to(
        memory_format=torch.contiguous_format).float()
    depth_images = torch.stack([example["depth_image"] for example in samples]).to(
        memory_format=torch.contiguous_format).float()
    clip_image = torch.cat([example["clip_image"] for example in samples], dim=0)
    drop_image_embed = [example["drop_image_embed"] for example in samples]

    # text = [example["text"] for example in data]
    # captions = torch.cat([example["caption"] for example in samples], dim=0)
    captions = torch.stack([example["caption"] for example in samples], dim=0)
    # null_input_ids = torch.cat([example["null_text_input_ids"] for example in data], dim=0)

    return {
        "tar_image": tar_images,
        "ref_image": ref_images,
        "mask": masks,
        "masked_image": masked_images,
        "clip_image": clip_image,
        "drop_image_embed": drop_image_embed,
        "caption":captions,
        "depth_image": depth_images,
    }