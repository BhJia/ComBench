import pdb
import argparse
import glob
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import transformers
import json
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

########################################################################################################################################################################
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanSquaredError


def calculate_single_metrics(gt_img_path, edited_img_path, mask_img_path, clip_model_path,metrics_size=512):
    clip_model = CLIPModel.from_pretrained(clip_model_path).to("cuda")
    clip_processor = CLIPProcessor.from_pretrained(clip_model_path)

    # 读取图像和掩码
    gt_image = Image.open(gt_img_path).convert("RGB").resize((512, 512))
    edited_image = Image.open(edited_img_path).convert("RGB").resize((512, 512))
    mask_image = Image.open(mask_img_path).convert("L").resize((512, 512))

    # 图像预处理
    gt_image = gt_image.resize((metrics_size, metrics_size), resample=Image.Resampling.BICUBIC)
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

    if right > left and bottom > top:
        # 裁剪编辑区域
        cropped_edited = edited_image.crop((left, top, right, bottom))
        cropped_gt = gt_image.crop((left, top, right, bottom))

        # 提取 CLIP 特征
        with torch.no_grad():
            inputs_edited = clip_processor(images=cropped_edited, return_tensors="pt").to("cuda")
            inputs_gt = clip_processor(images=cropped_gt, return_tensors="pt").to("cuda")
            features_edited = clip_model.get_image_features(**inputs_edited)
            features_gt = clip_model.get_image_features(**inputs_gt)

        # 计算余弦相似度
        clip_score = F.cosine_similarity(features_edited, features_gt).cpu().item()
    else:
        clip_score = 0.0  # 无效掩码时的默认值

    # 计算所有指标
    results = {
        "CLIP": clip_score,
    }

    return results

def main():
    # 定义路径
    parser = argparse.ArgumentParser(description='Calculate image editing metrics')
    parser.add_argument('--edited_dir', type=str, required=True, help='Directory containing edited images')
    parser.add_argument('--output_json', type=str, default="./local_clip_result.json", help='Path to output JSON file')
    parser.add_argument('--clip_model_path', type=str, default="/data/jbh/clip-vit-large-patch14", 
                        help='Path to CLIP model (default: /data/jbh/clip-vit-large-patch14)')
    args = parser.parse_args()
    
    edited_dir = args.edited_dir
    clip_model_path = args.clip_model_path

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
        op: {'CLIP': []}
        for op in ['add', 'remove','replace']
    }

    image_metrics = []

    # 遍历所有操作处理数据
    for operation in ['add', 'remove','replace']:
        config = operation_configs[operation]
        with open(config['info_file'], 'r') as f:
            data = json.load(f)
            for item in tqdm(data, desc=f"Processing {operation}"):
                image_name = item.get('image_path')
                
                # 计算单样本指标
                metrics_implicit = calculate_single_metrics(
                    gt_img_path=f"{config['gt_dir']}/{image_name}",
                    edited_img_path=f"{config['edited_dir']}/{image_name}",
                    mask_img_path=f"{config['mask_dir']}/{image_name}",
                    clip_model_path=clip_model_path,
                )
                # 存储指标
                for metric in metrics_implicit:
                    metrics[operation][metric].append(metrics_implicit[metric])

                image_metrics.append({
                    'image_name': image_name,
                    'metrics': metrics_implicit
                })

    # 计算各任务平均指标
    operation_avg = {
        op: {
            metric: np.mean(metrics[op][metric]) 
            for metric in ['CLIP']
        }
        for op in ['add', 'remove','replace']
    }

    # 计算总平均指标
    total_avg = {
        metric: np.mean([operation_avg[op][metric] for op in operation_avg])
        for metric in ['CLIP']
    }

    output_data = {
        'per_image_metrics': image_metrics,
        'average_metrics': operation_avg,
    }

    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=4)

    # 保存结果到文件
    output_file = f"./local_clip_result.txt"
    with open(output_file, 'w') as f:
        # 写入各任务指标
        for op in ['add', 'remove','replace']:
            f.write(f"Average Metrics for {op}:\n")
            f.write("="*40 + "\n")
            f.write(f"CLIP Score: {operation_avg[op]['CLIP']:.4f}\n")
            f.write("="*40 + "\n\n")
        
        # 写入总平均指标
        f.write("Overall Average Metrics:\n")
        f.write("="*40 + "\n")
        f.write(f"CLIP Score: {total_avg['CLIP']:.4f}\n")
        f.write("="*40 + "\n")

    print(f"All metrics summary saved to {output_file}")


if __name__ == "__main__":
    main()