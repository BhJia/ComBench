import argparse
import glob
import numpy as np
import os
import random
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import transformers
import json
from tqdm import tqdm

########################################################################################################################################################################
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanSquaredError

# 1. MetricsCalculator
class MetricsCalculator:
    def __init__(self, device, clip_model_path) -> None:
        self.device = device

        # CLIP similarity
        self.clip_metric_calculator = CLIPScore(model_name_or_path=clip_model_path).to(device)

        # background preservation
        self.psnr_metric_calculator = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        self.mse_metric_calculator = MeanSquaredError().to(device)
        self.ssim_metric_calculator = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    ####################################################################################
    # 1. CLIP similarity
    def calculate_clip_similarity(self, img, txt, mask=None):
        img = np.array(img)

        if mask is not None:
            mask = np.array(mask)
            img = np.uint8(img * mask)

        img_tensor = torch.tensor(img).permute(2, 0, 1).to(self.device)
        score = self.clip_metric_calculator(img_tensor, txt)
        score = score.cpu().item()
        return score

    # 2. PSNR
    def calculate_psnr(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        score = self.psnr_metric_calculator(img_pred_tensor, img_gt_tensor)
        score = score.cpu().item()
        return score

    # 3. LPIPS
    def calculate_lpips(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        score = self.lpips_metric_calculator(img_pred_tensor * 2 - 1, img_gt_tensor * 2 - 1)
        score = score.cpu().item()
        return score

    # 4. MSE (kept for calculation but removed from output)
    def calculate_mse(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).to(self.device)
        score = self.mse_metric_calculator(img_pred_tensor.contiguous(), img_gt_tensor.contiguous())
        score = score.cpu().item()
        return score

    # 5. SSIM
    def calculate_ssim(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        score = self.ssim_metric_calculator(img_pred_tensor, img_gt_tensor)
        score = score.cpu().item()
        return score

def calculate_single_metrics(original_img_path, edited_img_path, mask_img_path, clip_text, clip_model_path, metrics_size=512):
    # 初始化指标计算器
    metrics_calculator = MetricsCalculator("cuda", clip_model_path)
    clip_metric_calculator = CLIPScore(model_name_or_path=clip_model_path).to("cuda")
    
    # # 读取图像和掩码
    original_image = Image.open(original_img_path).convert("RGB")
    edited_image = Image.open(edited_img_path).convert("RGB")
    mask_image = Image.open(mask_img_path).convert("L")

    # ·图像预处理
    original_image = original_image.resize((metrics_size, metrics_size), Image.Resampling.BICUBIC)
    edited_image = edited_image.resize((metrics_size, metrics_size), Image.Resampling.BICUBIC)
    mask_image = mask_image.resize((metrics_size, metrics_size), Image.Resampling.NEAREST)
    # import pdb; pdb/.set_trace()
    # 处理掩码
    mask_array = np.array(mask_image)
    y, x = np.where(mask_array)
    if len(y) == 0 or len(x) == 0:  # 处理空掩码情况
        top = bottom = left = right = 0
    else:
        top, bottom = np.min(y), np.max(y)
        left, right = np.min(x), np.max(x)

    # 计算CLIP分数（裁剪编辑区域）
    if right > left and bottom > top:
        cropped_edited = edited_image.crop((left, top, right, bottom))
        cropped_tensor = torch.tensor(np.array(cropped_edited)).permute(2, 0, 1).to("cuda")
        clip_score = clip_metric_calculator(cropped_tensor, clip_text).cpu().item()
    else:
        clip_score = 0.0  # 无效掩码时的默认值

    # 准备背景保留计算用的掩码
    mask_np = np.array(mask_image, dtype=np.float32) / 255.0
    mask_np = 1 - mask_np  # 反转掩码（关注未编辑区域）
    mask_3ch = mask_np[:, :, np.newaxis].repeat(3, axis=2)

    # 转换图像为numpy数组
    original_np = np.array(original_image, dtype=np.float32) / 255.0
    edited_np = np.array(edited_image, dtype=np.float32) / 255.0

    # 计算所有指标 (MSE still calculated but not included in results)
    results = {
        "CLIP": clip_score,
        "PSNR": metrics_calculator.calculate_psnr(original_np*255, edited_np*255, mask_3ch, mask_3ch),
        "LPIPS": metrics_calculator.calculate_lpips(original_np*255, edited_np*255, mask_3ch, mask_3ch),
        "SSIM": metrics_calculator.calculate_ssim(original_np*255, edited_np*255, mask_3ch, mask_3ch)
    }

    return results

def main():
    # Add command line arguments
    parser = argparse.ArgumentParser(description='Calculate image editing metrics')
    parser.add_argument('--edited_dir', type=str, required=True, help='Directory containing edited images')
    parser.add_argument('--output_json', type=str, default="./implicit_reasoning_result.json", help='Path to output JSON file')
    parser.add_argument('--clip_model_path', type=str, default="/data/jbh/clip-vit-large-patch14", 
                        help='Path to CLIP model (default: /data/jbh/clip-vit-large-patch14)')
    args = parser.parse_args()
    
    # Use user-specified edited directory and clip model path
    edited_dir = args.edited_dir
    clip_model_path = args.clip_model_path
    
    # Keep the gt_dir as it was
    gt_dir = "./tasks/implicit_reasoning"
    
    # Input JSON file
    implicit_info = f"{gt_dir}/metadata.json"

    # Directory setup for remove operations
    ori_dir = f"{gt_dir}/input_image"
    GT_dir = f"{gt_dir}/edited_image"
    mask_dir = f"{gt_dir}/mask"

    # Initialize metric dictionaries for all operations
    metrics = {
        'implicit': {'CLIP': [], 'PSNR': [], 'LPIPS': [], 'SSIM': []},
    }
    
    # 用于保存每个图像的指标
    image_metrics = []

    # 读取JSON文件而不是JSONL文件
    try:
        with open(implicit_info, 'r') as f:
            # 将整个文件作为一个JSON对象读取，而不是逐行解析
            json_data = json.load(f)
            
        # 遍历JSON数组中的每个对象
        for item in tqdm(json_data, desc="Processing images"):
            try:
                # 使用image_path字段作为图像名称
                image_name = item.get("image_path")
                if not image_name:
                    print(f"Missing 'image_path' in data item, skipping")
                    continue
                
                caption = item.get("caption")
                if not caption:
                    print(f"Missing 'caption' in data for {image_name}, skipping")
                    continue
                
                # 确保文件存在
                if not os.path.exists(f"{edited_dir}/{image_name}"):
                    print(f"Warning: File {edited_dir}/{image_name} not found. Skipping.")
                    continue
                
                if not os.path.exists(f"{ori_dir}/{image_name}"):
                    print(f"Warning: Original file {ori_dir}/{image_name} not found. Skipping.")
                    continue
                
                if not os.path.exists(f"{mask_dir}/{image_name}"):
                    print(f"Warning: Mask file {mask_dir}/{image_name} not found. Skipping.")
                    continue
                
                # 计算指标
                metrics_result = calculate_single_metrics(
                    original_img_path=f"{ori_dir}/{image_name}",
                    edited_img_path=f"{edited_dir}/{image_name}",
                    mask_img_path=f"{mask_dir}/{image_name}",
                    clip_text=caption,
                    clip_model_path=clip_model_path
                )
                
                # 添加到总体指标列表
                for metric in metrics_result:
                    metrics['implicit'][metric].append(metrics_result[metric])
                
                # 保存每个图像的指标
                image_metrics.append({
                    "image_name": image_name,
                    "caption": caption,
                    "metrics": metrics_result
                })
                
            except Exception as e:
                print(f"Error processing item: {e}")
                continue
    
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {implicit_info}: {e}")
        return
    except Exception as e:
        print(f"Error reading JSON file {implicit_info}: {e}")
        return
    
    # Calculate average metrics
    operation_avg = {}
    for operation in metrics:
        operation_avg[operation] = {
            'CLIP': np.mean(metrics[operation]['CLIP']),
            'PSNR': np.mean(metrics[operation]['PSNR']),
            'LPIPS': np.mean(metrics[operation]['LPIPS']),
            'SSIM': np.mean(metrics[operation]['SSIM'])
        }
    
    # Output to JSON file
    output_data = {
        'per_image_metrics': image_metrics,
        'average_metrics': operation_avg
    }
    
    # Save JSON file to user-specified path
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"All metrics saved to {args.output_json}")


if __name__ == "__main__":
    main()