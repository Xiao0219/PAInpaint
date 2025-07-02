import torch
import torch.nn.functional as F
from safetensors.numpy import save_file, load_file
from omegaconf import OmegaConf
from transformers import AutoConfig
import cv2
from PIL import Image
import numpy as np
import json
import os
import sys
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm


#
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, StableDiffusionInpaintPipeline, DDIMScheduler, AutoencoderKL
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler
#
from models.pipeline_PAInpaint import PAInpaintPipeline
from models.ReferenceNet import ReferenceNet
from models.depth_guider import DepthGuider
from PAInpaint.PAInpaint_RefUet import PAInpaint_RefUet
from dataset.data_utils import *
from accelerate.logging import get_logger

val_configs = OmegaConf.load('./configs/inference.yaml')

# === import Depth Anything ===
import sys
sys.path.append("./depthanything")
from torchvision.transforms import Compose
from depthanything.fast_import import depth_anything_model
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
depth_anything_model.load_state_dict(torch.load(val_configs.model_path.depth_model))
depth_anything_model = depth_anything_model.to(dtype=torch.float16)
logger = get_logger(__name__)

# === load the checkpoint ===
base_model_path = val_configs.model_path.pretrained_Ref_Unet_path
vae_model_path = val_configs.model_path.pretrained_vae_name_or_path
image_encoder_path = val_configs.model_path.image_encoder_path
ref_model_path = val_configs.model_path.pretrained_Target_Unet_path
PAInpaint_ckpt = val_configs.model_path.ckpt_path
device = "cuda"


noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet", in_channels=13, low_cpu_mem_usage=False, ignore_mismatched_sizes=True).to(dtype=torch.float16)

pipe = PAInpaintPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    unet=unet,
    feature_extractor=None,
    safety_checker=None,
)

depth_guider = DepthGuider()
referencenet = ReferenceNet.from_pretrained(ref_model_path, subfolder="unet").to(dtype=torch.float16)
PAInpaint_model = PAInpaint_RefUet(pipe, image_encoder_path, PAInpaint_ckpt, depth_anything_model, depth_guider, referencenet, device)


save_root = val_configs.data_path.output_dir
os.makedirs(save_root, exist_ok=True)

source_dir = val_configs.data_path.test_dir

reference_names = [i for i in os.listdir(source_dir) if 'ref_image.jpg' in i  ]

# ===================================================================
# Part 1: Inference and Saving Results
# ===================================================================
tqdm.write("\n" + "="*20 + " Part 1: Inference and Image Saving " + "="*20)
for i, reference_name in enumerate(tqdm(reference_names, desc="Generating Images")):
    
    # --- Check if result already exists ---
    result_save_path = os.path.join(save_root, reference_name.replace('ref_image.jpg', 'result.png'))
    if os.path.exists(result_save_path):
        tqdm.write(f"Skipping generation for {reference_name} as result.png already exists.")
        continue

    source_image_name = reference_name.replace('ref_image', 'tar_image')
    source_mask_name = reference_name.replace('ref_image.jpg', 'mask.png')

    source_image_path = os.path.join(source_dir,source_image_name)
    source_mask_path = os.path.join(source_dir,source_mask_name)
    reference_image_path = os.path.join(source_dir,reference_name)

    # All images fed to the model should be resized to 512x512
    # Load target image and resize
    target_image_raw = cv2.imread(source_image_path)
    target_image_rgb = cv2.cvtColor(target_image_raw, cv2.COLOR_BGR2RGB)
    target_image_512 = cv2.resize(target_image_rgb, (512, 512), interpolation=cv2.INTER_AREA)

    # Load reference image and resize
    ref_image_raw = cv2.imread(reference_image_path)
    ref_image_rgb = cv2.cvtColor(ref_image_raw, cv2.COLOR_BGR2RGB)
    ref_image_512 = cv2.resize(ref_image_rgb, (512, 512), interpolation=cv2.INTER_AREA)

    # Load mask and resize
    mask_raw = cv2.imread(source_mask_path, cv2.IMREAD_GRAYSCALE)
    mask_512 = cv2.resize(mask_raw, (512, 512), interpolation=cv2.INTER_AREA)
    mask_512_binary = (mask_512 > 128).astype(np.uint8) # Binary mask (0 or 1)

    # Prepare inputs for the model
    ref_image_pil = Image.fromarray(ref_image_512.astype(np.uint8))
    target_image_pil = Image.fromarray(target_image_512.astype(np.uint8))
    # For mask_image, it expects a PIL RGB image. Stack binary mask to 3 channels.
    mask_image_pil = Image.fromarray(np.stack([mask_512_binary*255, mask_512_binary*255, mask_512_binary*255], axis=-1).astype(np.uint8))

    # Masked input image for collage (target_image_512 with blacked out masked regions)
    masked_input_for_collage = target_image_512 * (1 - mask_512_binary[:, :, np.newaxis])

    # Prepare depth image input
    depth_image_input = target_image_512.copy()
    depth_image_input = transform({'image': depth_image_input})['image']
    depth_image_input = torch.from_numpy(depth_image_input).unsqueeze(0).to(dtype=torch.float16) / 255      
    
    # Generate prediction
    pred_list, depth_pred = PAInpaint_model.generate(pil_image=ref_image_pil, depth_image = depth_image_input, num_samples=1, num_inference_steps=50,
                            seed=1, image=target_image_pil, mask_image=mask_image_pil, strength=1.0, guidance_scale=5, progress_bar=False)

    # Model output is already 512x512 (RGB from PIL image)
    pred_pure_model_output = np.array(pred_list[0]).astype(np.uint8)

    # Blending the pure model output with the original target image (512x512) using a blurred mask
    mask_alpha_blending = np.stack([mask_512_binary*255, mask_512_binary*255, mask_512_binary*255], axis=-1).astype(np.uint8)
    for _ in range(10): 
        mask_alpha_blending = cv2.GaussianBlur(mask_alpha_blending, (3, 3), 0)
    mask_alpha_norm = mask_alpha_blending / 255

    final_pred_blended = pred_pure_model_output * mask_alpha_norm + target_image_512 * (1 - mask_alpha_norm)
    final_pred_blended = np.clip(final_pred_blended, 0, 255).astype(np.uint8)

    # Process depth prediction for visualization
    depth_pred = F.interpolate(depth_pred, size=(512,512), mode = 'bilinear', align_corners=True)[0][0]
    depth_pred = depth_pred.detach().cpu().numpy().astype(np.uint8)
    depth_pred_colored = cv2.applyColorMap(depth_pred, cv2.COLORMAP_INFERNO)[:,:,::-1] # Convert to RGB
    depth_pred_pil = Image.fromarray(depth_pred_colored)

    # Save final blended image
    cv2.imwrite(result_save_path, cv2.cvtColor(final_pred_blended, cv2.COLOR_RGB2BGR))

    # Save reference image separately
    ref_image_save_path = os.path.join(save_root, reference_name.replace('ref_image.jpg', 'reference.png'))
    cv2.imwrite(ref_image_save_path, cv2.cvtColor(ref_image_512, cv2.COLOR_RGB2BGR))

    # Save masked target image separately
    masked_target_save_path = os.path.join(save_root, reference_name.replace('ref_image.jpg', 'target_masked.png'))
    cv2.imwrite(masked_target_save_path, cv2.cvtColor(masked_input_for_collage, cv2.COLOR_RGB2BGR))

    # Save Ground Truth image separately
    gt_image_save_path = os.path.join(save_root, reference_name.replace('ref_image.jpg', 'target.png'))
    cv2.imwrite(gt_image_save_path, cv2.cvtColor(target_image_512, cv2.COLOR_RGB2BGR))

    # --- Add collage saving logic ---
    collage_save_path = os.path.join(save_root, reference_name.replace('ref_image.jpg', '_collage.png'))
    
    # 1. Reference Image (NumPy RGB) -> Convert to BGR for cv2.hconcat
    img1_bgr_coll = cv2.cvtColor(ref_image_512, cv2.COLOR_RGB2BGR)

    # 2. Masked Input Image (NumPy RGB) -> Convert to BGR
    img2_bgr_coll = cv2.cvtColor(masked_input_for_collage, cv2.COLOR_RGB2BGR)

    # 3. Pure Prediction (Model output, NumPy RGB) -> Convert to BGR
    img3_bgr_coll = cv2.cvtColor(pred_pure_model_output, cv2.COLOR_RGB2BGR)

    # 4. Repaired Image (Final Blended, NumPy RGB) -> Convert to BGR
    img4_bgr_coll = cv2.cvtColor(final_pred_blended, cv2.COLOR_RGB2BGR)

    # 5. Target Image (GT, NumPy RGB) -> Convert to BGR
    img5_bgr_coll = cv2.cvtColor(target_image_512, cv2.COLOR_RGB2BGR)

    # 6. Depth Prediction (PIL RGB) -> Convert to NumPy RGB -> Convert to BGR
    img6_bgr_coll = cv2.cvtColor(np.array(depth_pred_pil), cv2.COLOR_RGB2BGR)

    # Concatenate images horizontally and save
    collage_image = cv2.hconcat([img1_bgr_coll, img2_bgr_coll, img3_bgr_coll, img4_bgr_coll, img5_bgr_coll, img6_bgr_coll])
    cv2.imwrite(collage_save_path, collage_image)


# ===================================================================
# Part 2: Performance Evaluation
# ===================================================================
tqdm.write("\n" + "="*20 + " Part 2: Performance Evaluation " + "="*20)

index = 0
# 初始化各项指标列表
psnr_scores = []
ssim_scores = []
lpips_scores = []
clip_i_scores = []  # 可能为空，需在使用前检查
dino_scores = []    # 可能为空，需在使用前检查

# 额外的指标日志文件
metrics_log_path = os.path.join(save_root, "metrics.txt")

for i, reference_name in enumerate(tqdm(reference_names, desc="Calculating Metrics")):
    
    result_save_path = os.path.join(save_root, reference_name.replace('ref_image.jpg', 'result.png'))
    gt_image_save_path = os.path.join(save_root, reference_name.replace('ref_image.jpg', 'target.png'))
    
    if not os.path.exists(result_save_path) or not os.path.exists(gt_image_save_path):
        tqdm.write(f"Skipping metrics for {reference_name} as result or target file is missing.")
        continue

    # --- Metrics Calculation ---
    # Load images for metrics. cv2 reads in BGR, convert to RGB for consistency.
    pred_for_metrics_bgr = cv2.imread(result_save_path)
    pred_for_metrics = cv2.cvtColor(pred_for_metrics_bgr, cv2.COLOR_BGR2RGB)

    gt_for_metrics_bgr = cv2.imread(gt_image_save_path)
    gt_for_metrics = cv2.cvtColor(gt_for_metrics_bgr, cv2.COLOR_BGR2RGB)

    # PSNR is calculated between the 512x512 ground truth and the 512x512 blended prediction
    current_psnr = peak_signal_noise_ratio(gt_for_metrics, pred_for_metrics, data_range=255)

    # 计算其他指标
    current_metrics = compute_metrics_for_pair(gt_for_metrics, pred_for_metrics, device=device)

    # 记录各项指标到对应列表
    psnr_scores.append(current_psnr)
    ssim_scores.append(current_metrics["SSIM"])
    lpips_scores.append(current_metrics["LPIPS"])
    if "CLIP_I" in current_metrics:
        clip_i_scores.append(current_metrics["CLIP_I"])
    if "DINO" in current_metrics:
        dino_scores.append(current_metrics["DINO"])

    # 打印当前样本的全部指标
    msg = (
        f"Processed {reference_name}: "
        f"PSNR={current_psnr:.4f}, "
        f"SSIM={current_metrics['SSIM']:.4f}, "
        f"LPIPS={current_metrics['LPIPS']:.4f}"
    )
    if "CLIP_I" in current_metrics:
        msg += f", CLIP_I={current_metrics['CLIP_I']:.2f}"
    if "DINO" in current_metrics:
        msg += f", DINO={current_metrics['DINO']:.2f}"
    tqdm.write(msg)

    # 将当前指标写入综合日志文件
    with open(metrics_log_path, 'a') as f:
        f.write(msg + "\n")

# --- Final Average Metrics ---
print("\n" + "="*50)
if psnr_scores:
    average_psnr = np.mean(psnr_scores)
    avg_msg = f"All images processed. Average PSNR: {average_psnr:.4f}"

    # 计算其他指标平均值
    if ssim_scores:
        avg_ssim = np.mean(ssim_scores)
        avg_msg += f", Average SSIM: {avg_ssim:.4f}"
    if lpips_scores:
        avg_lpips = np.mean(lpips_scores)
        avg_msg += f", Average LPIPS: {avg_lpips:.4f}"
    if clip_i_scores:
        avg_clip = np.mean(clip_i_scores)
        avg_msg += f", Average CLIP_I: {avg_clip:.2f}"
    if dino_scores:
        avg_dino = np.mean(dino_scores)
        avg_msg += f", Average DINO: {avg_dino:.2f}"

    print(avg_msg)

    # 将平均指标写入综合日志文件
    with open(metrics_log_path, 'a') as f:
        f.write("\n" + avg_msg + "\n")
print("="*50)