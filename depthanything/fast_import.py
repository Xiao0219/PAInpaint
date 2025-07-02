import sys
sys.path.append('/mnt/lustre/GPU3/home/zhoushengxiao/workspace/codes/MimicBrush/depthanything')
from depth_anything.dpt import DepthAnything
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose

model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}
encoder = 'vitb' # or 'vitb', 'vits'
# 避免在 import 时就占用 GPU0，模型先保持在 CPU，之后再由调用方显式移动到设备上
depth_anything_model = DepthAnything(model_configs[encoder]).eval()