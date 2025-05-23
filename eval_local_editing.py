import pdb
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

    # 4. MSE
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

def calculate_single_metrics(original_img_path, edited_img_path, mask_img_path, clip_text, clip_model_path,metrics_size=512):
    # 初始化指标计算器
    metrics_calculator = MetricsCalculator("cuda", clip_model_path)
    clip_metric_calculator = CLIPScore(model_name_or_path=clip_model_path).to("cuda")

    # 读取图像和掩码
    original_image = Image.open(original_img_path).convert("RGB")
    edited_image = Image.open(edited_img_path).convert("RGB")
    mask_image = Image.open(mask_img_path).convert("L")

    # 图像预处理
    original_image = original_image.resize((metrics_size, metrics_size), resample=Image.Resampling.BICUBIC)
    edited_image = edited_image.resize((metrics_size, metrics_size), resample=Image.Resampling.BICUBIC)
    mask_image = mask_image.resize((metrics_size, metrics_size), resample=Image.Resampling.NEAREST)

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

    # 计算所有指标
    results = {
        "CLIP": clip_score,
        "PSNR": metrics_calculator.calculate_psnr(original_np*255, edited_np*255, mask_3ch, mask_3ch),
        "LPIPS": metrics_calculator.calculate_lpips(original_np*255, edited_np*255, mask_3ch, mask_3ch),
        "MSE": metrics_calculator.calculate_mse(original_np*255, edited_np*255, mask_3ch, mask_3ch),
        "SSIM": metrics_calculator.calculate_ssim(original_np*255, edited_np*255, mask_3ch, mask_3ch)
    }

    return results


def main():
    # 定义路径
    parser = argparse.ArgumentParser(description='Calculate image editing metrics')
    parser.add_argument('--edited_dir', type=str, required=True, help='Directory containing edited images')
    parser.add_argument('--output_json', type=str, default="./local_editing_result.json", help='Path to output JSON file')
    parser.add_argument('--clip_model_path', type=str, default="/data/jbh/clip-vit-large-patch14", 
                        help='Path to CLIP model (default: /data/jbh/clip-vit-large-patch14)')
    args = parser.parse_args()
    
    # Use user-specified edited directory and clip model path
    edited_dir = args.edited_dir
    clip_model_path = args.clip_model_path
    
    # Keep the gt_dir as it was
    gt_dir = "./tasks"

    # 定义各任务路径配置
    operation_configs = {
        'add': {
            'ori_dir': f"{gt_dir}/add/input_image",
            'edited_dir': f"{edited_dir}/add",
            'mask_dir': f"{gt_dir}/add/mask",
            'info_file': f"{gt_dir}/add/metadata.json",
            'gt_dir': f"{gt_dir}/add/edited_image"
        },
        'remove': {
            'ori_dir': f"{gt_dir}/remove/input_image",
            'edited_dir': f"{edited_dir}/remove",
            'mask_dir': f"{gt_dir}/remove/mask",
            'info_file': f"{gt_dir}/remove/metadata.json",
            'gt_dir': f"{gt_dir}/remove/edited_image"
        },
        'replace': {
            'ori_dir': f"{gt_dir}/replace/input_image",
            'edited_dir': f"{edited_dir}/replace",
            'mask_dir': f"{gt_dir}/replace/mask",  # 修正原拼写错误
            'info_file': f"{gt_dir}/replace/metadata.json",
            'gt_dir': f"{gt_dir}/replace/edited_image"
        }
    }

    # 初始化指标存储
    metrics = {
        op: {'CLIP': [], 'PSNR': [], 'LPIPS': [], 'MSE': [], 'SSIM': []}
        for op in ['add', 'remove', 'replace']
    }

    image_metrics = []

    # 遍历所有操作处理数据
    for operation in ['add', 'remove','replace']:
        config = operation_configs[operation]
        with open(config['info_file'], 'r') as f:
            data = json.load(f)
            for item in tqdm(data, desc=f"Processing {operation}"):
                image_name = item.get('image_path')
                caption = item.get('caption')
                
                # 计算单样本指标
                metrics_implicit = calculate_single_metrics(
                    original_img_path=f"{config['ori_dir']}/{image_name}",
                    edited_img_path=f"{config['edited_dir']}/{image_name}",
                    mask_img_path=f"{config['mask_dir']}/{image_name}",
                    clip_text=caption,
                    clip_model_path=clip_model_path,
                )
                # 存储指标
                for metric in metrics_implicit:
                    metrics[operation][metric].append(metrics_implicit[metric])

                image_metrics.append({
                    'image_name': image_name,
                    'caption': caption,
                    'metrics': metrics_implicit
                })

    # 计算各任务平均指标
    operation_avg = {
        op: {
            metric: np.mean(metrics[op][metric]) 
            for metric in ['CLIP', 'PSNR', 'LPIPS', 'MSE', 'SSIM']
        }
        for op in ['add', 'remove','replace']
    }

    # Output to JSON file
    output_data = {
        'per_image_metrics': image_metrics,
        'average_metrics': operation_avg
    }
    
    # Save JSON file to user-specified path
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=4)

    # 计算总平均指标
    # total_avg = {
    #     metric: np.mean([operation_avg[op][metric] for op in operation_avg])
    #     for metric in ['CLIP', 'PSNR', 'LPIPS', 'MSE', 'SSIM']
    # }

    # 保存结果到文件
    output_file = f"./local_editing_result.txt"
    with open(output_file, 'w') as f:
        # 写入各任务指标
        for op in ['add', 'remove','replace']:
            f.write(f"Average Metrics for {op}:\n")
            f.write("="*40 + "\n")
            f.write(f"CLIP Score: {operation_avg[op]['CLIP']:.4f}\n")
            f.write(f"PSNR: {operation_avg[op]['PSNR']:.4f}\n")
            f.write(f"LPIPS: {operation_avg[op]['LPIPS']:.4f}\n")
            f.write(f"MSE: {operation_avg[op]['MSE']:.4f}\n")
            f.write(f"SSIM: {operation_avg[op]['SSIM']:.4f}\n")
            f.write("="*40 + "\n\n")
        
        # 写入总平均指标
        # f.write("Overall Average Metrics:\n")
        # f.write("="*40 + "\n")
        # f.write(f"CLIP Score: {total_avg['CLIP']:.4f}\n")
        # f.write(f"PSNR: {total_avg['PSNR']:.4f}\n")
        # f.write(f"LPIPS: {total_avg['LPIPS']:.4f}\n")
        # f.write(f"MSE: {total_avg['MSE']:.4f}\n")
        # f.write(f"SSIM: {total_avg['SSIM']:.4f}\n")
        # f.write("="*40 + "\n")

    print(f"All metrics summary saved to {output_file}")


if __name__ == "__main__":
    main()