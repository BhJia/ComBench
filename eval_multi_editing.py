import pdb
import argparse
import glob
import numpy as np
import os
import random
import torch
from PIL import Image
import torch.nn.functional as F
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

class MetricsCalculator:
    def __init__(self, device, clip_model_path) -> None:
        self.device = device

        # CLIP similarity
        self.clip_metric_calculator = CLIPScore(model_name_or_path=clip_model_path).to(device)

        # background preservation
        self.psnr_metric_calculator = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        self.ssim_metric_calculator = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    ####################################################################################
    # 1. CLIP similarity
    def calculate_clip_similarity(self, img, txt, mask=None):
        import pdb; pdb.set_trace()
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

    # 4. SSIM
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

def calculate_single_metrics(ori_img_path, edited_img_path, mask_img_path, clip_model_path, clip_text, metrics_size=512):
    metrics_calculator = MetricsCalculator("cuda", clip_model_path)
    clip_metric_calculator = CLIPScore(model_name_or_path=clip_model_path).to("cuda")
    
    # # 读取图像和掩码
    original_image = Image.open(ori_img_path).convert("RGB")
    edited_image = Image.open(edited_img_path).convert("RGB")
    mask_image = Image.open(mask_img_path).convert("L")

    # ·图像预处理
    original_image = original_image.resize((metrics_size, metrics_size), Image.Resampling.BICUBIC)
    edited_image = edited_image.resize((metrics_size, metrics_size), Image.Resampling.BICUBIC)
    mask_image = mask_image.resize((metrics_size, metrics_size), Image.Resampling.NEAREST)
    # import pdb; pdb.set_trace()
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
    # 添加命令行参数
    parser = argparse.ArgumentParser(description='Calculate image editing metrics')
    parser.add_argument('--edited_dir', type=str, required=True, help='Directory containing edited images')
    parser.add_argument('--output_json', type=str, default="./multi_editing_scores.json", help='Path to output JSON file')
    parser.add_argument('--clip_model_path', type=str, default="/data/jbh/clip-vit-large-patch14", 
                        help='Path to CLIP model (default: /data/jbh/clip-vit-large-patch14)')
    args = parser.parse_args()
    
    edited_dir = args.edited_dir
    clip_model_path = args.clip_model_path
    
    gt_dir = "./tasks"
    
    # Input JSON files
    multi_turn_remove_info = f"{gt_dir}/multi_turn_remove/converted_metadata.json"
    multi_turn_add_info = f"{gt_dir}/multi_turn_add/converted_metadata.json"
    multi_object_remove_info = f"{gt_dir}/multi_object_remove/metadata.json"
    multi_object_add_info = f"{gt_dir}/multi_object_add/metadata.json"

    # Directory setup for remove operations
    turn1_remove_edited_dir = f"{edited_dir}/turn1_remove"
    turn1_remove_ori_dir = f"{gt_dir}/multi_turn_remove/input_image/turn1_remove"
    turn1_remove_mask_dir = f"{gt_dir}/multi_turn_remove/mask/turn1_remove"
    turn1_remove_gt_dir = f"{gt_dir}/multi_turn_remove/edited_image"

    turn2_remove_edited_dir = f"{edited_dir}/turn2_remove"
    turn2_remove_gt_dir = f"{gt_dir}/multi_turn_remove/edited_image"
    turn2_remove_mask_dir = f"{gt_dir}/multi_turn_remove/mask/turn2_remove"
    turn2_remove_ori_dir = f"{gt_dir}/multi_turn_remove/input_image/turn2_remove"

    # Directory setup for add operations
    turn1_add_edited_dir = f"{edited_dir}/turn1_add"
    turn1_add_gt_dir = f"{gt_dir}/multi_turn_add/edited_image"
    turn1_add_mask_dir = f"{gt_dir}/multi_turn_add/mask/turn1_add"
    turn1_add_ori_dir = f"{gt_dir}/multi_turn_add/input_image/turn1_add"

    turn2_add_edited_dir = f"{edited_dir}/turn2_add"
    turn2_add_gt_dir = f"{gt_dir}/multi_turn_add/edited_image"
    turn2_add_mask_dir = f"{gt_dir}/multi_turn_add/mask/turn2_add"
    turn2_add_ori_dir = f"{gt_dir}/multi_turn_add/input_image/turn2_add"

    # multi-object remove & add
    multi_obj_rm_edited_dir = f"{edited_dir}/multi_object_remove"
    multi_obj_rm_gt_dir = f"{gt_dir}/multi_object_remove/edited_image"
    multi_obj_rm_ori_dir = f"{gt_dir}/multi_object_remove/input_image"

    multi_obj_add_edited_dir = f"{edited_dir}/multi_object_add"
    multi_obj_add_gt_dir = f"{gt_dir}/multi_object_add/edited_image"
    multi_obj_add_ori_dir = f"{gt_dir}/multi_object_add/input_image"

    # Initialize metric dictionaries for all operations
    metrics = {
        'multi_object_remove': {'CLIP': [], 'PSNR': [], 'LPIPS': [], 'SSIM': []},
        'multi_object_add': {'CLIP': [], 'PSNR': [], 'LPIPS': [], 'SSIM': []},
        'turn1_remove': {'CLIP': [], 'PSNR': [], 'LPIPS': [], 'SSIM': []},
        'turn2_remove': {'CLIP': [], 'PSNR': [], 'LPIPS': [], 'SSIM': []},
        'turn1_add': {'CLIP': [], 'PSNR': [], 'LPIPS': [], 'SSIM': []},
        'turn2_add': {'CLIP': [], 'PSNR': [], 'LPIPS': [], 'SSIM': []}
    }
    
    # 用于保存详细的每个图像指标
    image_metrics = {
        'multi_object_remove': [],
        'multi_object_add': [],
        'turn1_remove': [],
        'turn2_remove': [],
        'turn1_add': [],
        'turn2_add': []
    }

    # 处理multi-turn remove operations
    try:
        with open(multi_turn_remove_info, 'r') as f:
            remove_data = json.load(f)
            
        for item in tqdm(remove_data, desc="Processing remove operations"):
            try:
                image_path = item.get("image_path")
                if not image_path:
                    print(f"Missing 'image_path' in remove data, skipping")
                    continue
                
                # 确保文件存在
                if not os.path.exists(f"{turn1_remove_edited_dir}/{image_path}"):
                    print(f"Warning: File {turn1_remove_edited_dir}/{image_path} not found. Skipping turn1 remove.")
                else:
                    # Process turn1 remove
                    turn1_metrics = calculate_single_metrics(
                        ori_img_path=f"{turn1_remove_ori_dir}/{image_path}",
                        edited_img_path=f"{turn1_remove_edited_dir}/{image_path}",
                        mask_img_path=f"{turn1_remove_mask_dir}/{image_path}",
                        clip_text=item.get("turn1_caption"),
                        clip_model_path=clip_model_path
                    )
                    for metric in metrics['turn1_remove']:
                        metrics['turn1_remove'][metric].append(turn1_metrics[metric])
                    
                    # 保存详细指标
                    image_metrics['turn1_remove'].append({
                        "image_path": image_path,
                        "metrics": turn1_metrics
                    })
                
                if not os.path.exists(f"{turn2_remove_edited_dir}/{image_path}"):
                    print(f"Warning: File {turn2_remove_edited_dir}/{image_path} not found. Skipping turn2 remove.")
                else:
                    # Process turn2 remove
                    turn2_metrics = calculate_single_metrics(
                        ori_img_path=f"{turn2_remove_ori_dir}/{image_path}",
                        edited_img_path=f"{turn2_remove_edited_dir}/{image_path}",
                        mask_img_path=f"{turn2_remove_mask_dir}/{image_path}",
                        clip_text=item.get("turn2_caption"),
                        clip_model_path=clip_model_path
                    )
                    for metric in metrics['turn2_remove']:
                        metrics['turn2_remove'][metric].append(turn2_metrics[metric])
                    
                    # 保存详细指标
                    image_metrics['turn2_remove'].append({
                        "image_path": image_path,
                        "metrics": turn2_metrics
                    })
            except Exception as e:
                print(f"Error processing remove item {image_path if 'image_path' in locals() else 'unknown'}: {e}")
                continue
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {multi_turn_remove_info}: {e}")
    except Exception as e:
        print(f"Error reading JSON file {multi_turn_remove_info}: {e}")

    # 处理multi-turn add operations
    try:
        with open(multi_turn_add_info, 'r') as f:
            add_data = json.load(f)
            
        for item in tqdm(add_data, desc="Processing add operations"):
            try:
                image_path = item.get("image_path")
                if not image_path:
                    print(f"Missing 'image_path' in add data, skipping")
                    continue
                
                # 确保文件存在
                if not os.path.exists(f"{turn1_add_edited_dir}/{image_path}"):
                    print(f"Warning: File {turn1_add_edited_dir}/{image_path} not found. Skipping turn1 add.")
                else:
                    # Process turn1 add
                    turn1_metrics = calculate_single_metrics(
                        ori_img_path=f"{turn1_add_ori_dir}/{image_path}",
                        edited_img_path=f"{turn1_add_edited_dir}/{image_path}",
                        mask_img_path=f"{turn1_add_mask_dir}/{image_path}",
                        clip_text=item.get("turn1_caption"),
                        clip_model_path=clip_model_path
                    )
                    for metric in metrics['turn1_add']:
                        metrics['turn1_add'][metric].append(turn1_metrics[metric])
                    
                    # 保存详细指标
                    image_metrics['turn1_add'].append({
                        "image_path": image_path,
                        "metrics": turn1_metrics
                    })
                
                if not os.path.exists(f"{turn2_add_edited_dir}/{image_path}"):
                    print(f"Warning: File {turn2_add_edited_dir}/{image_path} not found. Skipping turn2 add.")
                else:
                    # Process turn2 add
                    turn2_metrics = calculate_single_metrics(
                        ori_img_path=f"{turn2_add_ori_dir}/{image_path}",
                        edited_img_path=f"{turn2_add_edited_dir}/{image_path}",
                        mask_img_path=f"{turn2_add_mask_dir}/{image_path}",
                        clip_text=item.get("turn2_caption"),
                        clip_model_path=clip_model_path
                    )
                    for metric in metrics['turn2_add']:
                        metrics['turn2_add'][metric].append(turn2_metrics[metric])
                    
                    # 保存详细指标
                    image_metrics['turn2_add'].append({
                        "image_path": image_path,
                        "metrics": turn2_metrics
                    })
            except Exception as e:
                print(f"Error processing add item {image_path if 'image_path' in locals() else 'unknown'}: {e}")
                continue
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {multi_turn_add_info}: {e}")
    except Exception as e:
        print(f"Error reading JSON file {multi_turn_add_info}: {e}")
    
    # 处理multi-object remove
    try:
        with open(multi_object_remove_info, 'r') as f:
            multi_obj_remove_data = json.load(f)
            
        for item in tqdm(multi_obj_remove_data, desc="Processing multi-object remove"):
            try:
                image_path = item.get("image_path")
                if not image_path:
                    print(f"Missing 'image_path' in multi-object remove data, skipping")
                    continue
                
                base_name = image_path.split('.')[0]  # 提取无扩展的文件名
                
                # 确保文件存在
                if not os.path.exists(f"{multi_obj_rm_edited_dir}/{image_path}"):
                    print(f"Warning: File {multi_obj_rm_edited_dir}/{image_path} not found. Skipping multi-object remove.")
                    continue
                
                # 处理两个mask的指标并取平均
                mask1_path = f"{turn1_remove_mask_dir}/{base_name}_t1.png"
                mask2_path = f"{turn2_remove_mask_dir}/{base_name}_t1.png"

                caption1, caption2 = item.get('caption').split('|')
                # import pdb; pdb.set_trace()
                if not os.path.exists(mask1_path) or not os.path.exists(mask2_path):
                    print(f"Warning: Mask files for {image_path} not found. Skipping multi-object remove.")
                    continue
                
                mask1_metrics = calculate_single_metrics(
                    ori_img_path=f"{multi_obj_rm_ori_dir}/{image_path}",
                    edited_img_path=f"{multi_obj_rm_edited_dir}/{image_path}",
                    mask_img_path=mask1_path,
                    clip_text=caption1,
                    clip_model_path=clip_model_path
                )
                mask2_metrics = calculate_single_metrics(
                    ori_img_path=f"{multi_obj_rm_ori_dir}/{image_path}",
                    edited_img_path=f"{multi_obj_rm_edited_dir}/{image_path}",
                    mask_img_path=mask2_path,
                    clip_text=caption2,
                    clip_model_path=clip_model_path
                )
                
                # 计算两个mask指标的平均值
                avg_metrics = {
                    metric: (mask1_metrics[metric] + mask2_metrics[metric]) / 2 
                    for metric in metrics['multi_object_remove']
                }
                
                # 将平均值添加到metrics中
                for metric in metrics['multi_object_remove']:
                    metrics['multi_object_remove'][metric].append(avg_metrics[metric])
                
                # 保存详细指标
                image_metrics['multi_object_remove'].append({
                    "image_path": image_path,
                    "metrics": avg_metrics,
                    "mask1_metrics": mask1_metrics,
                    "mask2_metrics": mask2_metrics
                })
            except Exception as e:
                print(f"Error processing multi-object remove item {image_path if 'image_path' in locals() else 'unknown'}: {e}")
                continue
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {multi_object_remove_info}: {e}")
    except Exception as e:
        print(f"Error reading JSON file {multi_object_remove_info}: {e}")

    # 处理multi-object add
    try:
        with open(multi_object_add_info, 'r') as f:
            multi_obj_add_data = json.load(f)

            
        for item in tqdm(multi_obj_add_data, desc="Processing multi-object add"):       
            try:
                image_path = item.get("image_path")

                if not image_path:
                    print(f"Missing 'image_path' in multi-object add data, skipping")
                    continue
                
                base_name = image_path.split('.')[0]
                
                # 确保文件存在
                if not os.path.exists(f"{multi_obj_add_edited_dir}/{image_path}"):
                    print(f"Warning: File {multi_obj_add_edited_dir}/{image_path} not found. Skipping multi-object add.")
                    continue
                
                # 处理两个mask的指标并取平均
                mask1_path = f"{turn1_add_mask_dir}/{base_name}_t1.png"
                mask2_path = f"{turn2_add_mask_dir}/{base_name}_t1.png"

                caption1, caption2 = item.get('caption').split('|')
                
                # import pdb; pdb.set_trace()
                
                if not os.path.exists(mask1_path) or not os.path.exists(mask2_path):
                    print(f"Warning: Mask files for {image_path} not found. Skipping multi-object add.")
                    continue
                
                mask1_metrics = calculate_single_metrics(
                    ori_img_path=f"{multi_obj_add_ori_dir}/{image_path}",
                    edited_img_path=f"{multi_obj_add_edited_dir}/{image_path}",
                    mask_img_path=mask1_path,
                    clip_text=caption1,
                    clip_model_path=clip_model_path
                )
                mask2_metrics = calculate_single_metrics(
                    ori_img_path=f"{multi_obj_add_ori_dir}/{image_path}",
                    edited_img_path=f"{multi_obj_add_edited_dir}/{image_path}",
                    mask_img_path=mask2_path,
                    clip_text=caption2,
                    clip_model_path=clip_model_path
                )
                
                # 计算两个mask指标的平均值
                avg_metrics = {
                    metric: (mask1_metrics[metric] + mask2_metrics[metric]) / 2 
                    for metric in metrics['multi_object_add']
                }
                
                # 将平均值添加到metrics中
                for metric in metrics['multi_object_add']:
                    metrics['multi_object_add'][metric].append(avg_metrics[metric])
                
                # 保存详细指标
                image_metrics['multi_object_add'].append({
                    "image_path": image_path,
                    "metrics": avg_metrics,
                    "mask1_metrics": mask1_metrics,
                    "mask2_metrics": mask2_metrics
                })
            except Exception as e:
                print(f"Error processing multi-object add item {image_path if 'image_path' in locals() else 'unknown'}: {e}")
                continue
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {multi_object_add_info}: {e}")
    except Exception as e:
        print(f"Error reading JSON file {multi_object_add_info}: {e}")

    # Calculate and save average metrics
    # 1. First calculate individual operation averages
    operation_avg = {}
    for operation in metrics:
        operation_avg[operation] = {
            metric: np.mean(values) if values else 0 
            for metric, values in metrics[operation].items()
        }
    
    # 2. Calculate combined averages
    # All remove operations (turn1_remove + turn2_remove)
    remove_avg = {
        metric: np.mean([
            operation_avg['turn1_remove'][metric],
            operation_avg['turn2_remove'][metric]
        ])
        for metric in metrics['turn1_remove']
    }

    # All add operations (turn1_add + turn2_add)
    add_avg = {
        metric: np.mean([
            operation_avg['turn1_add'][metric],
            operation_avg['turn2_add'][metric]
        ])
        for metric in metrics['turn1_add']
    }

    # Overall average (all operations)
    overall_avg = {
        metric: np.mean([
            operation_avg['turn1_remove'][metric],
            operation_avg['turn2_remove'][metric],
            operation_avg['turn1_add'][metric],
            operation_avg['turn2_add'][metric]
        ])
        for metric in metrics['turn1_remove']
    }

    multi_object_avg = {
        metric: np.mean([
            operation_avg['multi_object_remove'][metric],
            operation_avg['multi_object_add'][metric]
        ])
        for metric in metrics['multi_object_remove']
    }

    # 保存到JSON文件 - 现在包含所有指标
    output_data = {
        'per_image_metrics': image_metrics,
        'individual_operations': operation_avg,  # 重命名以更清晰
        'combined_metrics': {
            'remove_operations': remove_avg,
            'add_operations': add_avg,
            'overall_multi_turn': overall_avg,
            'multi_object': multi_object_avg
        },
        'detailed_summary': {
            'individual_operations': {
                operation: {
                    'CLIP_Score': operation_avg[operation]['CLIP'],
                    'PSNR': operation_avg[operation]['PSNR'],
                    'LPIPS': operation_avg[operation]['LPIPS'],
                    'SSIM': operation_avg[operation]['SSIM']
                }
                for operation in operation_avg
            },
            'combined_remove': {
                'CLIP_Score': remove_avg['CLIP'],
                'PSNR': remove_avg['PSNR'],
                'LPIPS': remove_avg['LPIPS'],
                'SSIM': remove_avg['SSIM']
            },
            'combined_add': {
                'CLIP_Score': add_avg['CLIP'],
                'PSNR': add_avg['PSNR'],
                'LPIPS': add_avg['LPIPS'],
                'SSIM': add_avg['SSIM']
            },
            'overall': {
                'CLIP_Score': overall_avg['CLIP'],
                'PSNR': overall_avg['PSNR'],
                'LPIPS': overall_avg['LPIPS'],
                'SSIM': overall_avg['SSIM']
            },
            'multi_object_detailed': {
                'remove': {
                    'CLIP_Score': operation_avg['multi_object_remove']['CLIP'],
                    'PSNR': operation_avg['multi_object_remove']['PSNR'],
                    'LPIPS': operation_avg['multi_object_remove']['LPIPS'],
                    'SSIM': operation_avg['multi_object_remove']['SSIM']
                },
                'add': {
                    'CLIP_Score': operation_avg['multi_object_add']['CLIP'],
                    'PSNR': operation_avg['multi_object_add']['PSNR'],
                    'LPIPS': operation_avg['multi_object_add']['LPIPS'],
                    'SSIM': operation_avg['multi_object_add']['SSIM']
                },
                'combined': {
                    'CLIP_Score': multi_object_avg['CLIP'],
                    'PSNR': multi_object_avg['PSNR'],
                    'LPIPS': multi_object_avg['LPIPS'],
                    'SSIM': multi_object_avg['SSIM']
                }
            }
        }
    }

    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    # 同时保存文本结果文件，保持向后兼容
    output_text_file = f"{edited_dir}/multi_editing_scores.txt"
    with open(output_text_file, 'w') as f:
        # Write individual operation metrics
        for operation in metrics:
            f.write(f"Average Metrics for {operation}:\n")
            f.write("="*40 + "\n")
            f.write(f"CLIP Score: {operation_avg[operation]['CLIP']:.4f}\n")
            f.write(f"PSNR: {operation_avg[operation]['PSNR']:.4f}\n")
            f.write(f"LPIPS: {operation_avg[operation]['LPIPS']:.4f}\n")
            f.write(f"SSIM: {operation_avg[operation]['SSIM']:.4f}\n")
            f.write("="*40 + "\n\n")
        
        # Write combined remove metrics
        f.write("Combined Metrics for ALL REMOVE operations:\n")
        f.write("="*40 + "\n")
        f.write(f"CLIP Score: {remove_avg['CLIP']:.4f}\n")
        f.write(f"PSNR: {remove_avg['PSNR']:.4f}\n")
        f.write(f"LPIPS: {remove_avg['LPIPS']:.4f}\n")
        f.write(f"SSIM: {remove_avg['SSIM']:.4f}\n")
        f.write("="*40 + "\n\n")
        
        # Write combined add metrics
        f.write("Combined Metrics for ALL ADD operations:\n")
        f.write("="*40 + "\n")
        f.write(f"CLIP Score: {add_avg['CLIP']:.4f}\n")
        f.write(f"PSNR: {add_avg['PSNR']:.4f}\n")
        f.write(f"LPIPS: {add_avg['LPIPS']:.4f}\n")
        f.write(f"SSIM: {add_avg['SSIM']:.4f}\n")
        f.write("="*40 + "\n\n")
        
        # Write overall metrics
        f.write("OVERALL Metrics for Multi-turn operations:\n")
        f.write("="*40 + "\n")
        f.write(f"CLIP Score: {overall_avg['CLIP']:.4f}\n")
        f.write(f"PSNR: {overall_avg['PSNR']:.4f}\n")
        f.write(f"LPIPS: {overall_avg['LPIPS']:.4f}\n")
        f.write(f"SSIM: {overall_avg['SSIM']:.4f}\n")
        f.write("="*40 + "\n")

        # 新增：写入multi_object的指标
        f.write("Metrics for Multi-object Operations:\n")
        f.write("="*40 + "\n")
        f.write(f"CLIP Score (Remove): {operation_avg['multi_object_remove']['CLIP']:.4f}\n")
        f.write(f"CLIP Score (Add): {operation_avg['multi_object_add']['CLIP']:.4f}\n")
        f.write(f"Combined CLIP Score: {multi_object_avg['CLIP']:.4f}\n\n")
        
        f.write(f"PSNR (Remove): {operation_avg['multi_object_remove']['PSNR']:.4f}\n")
        f.write(f"PSNR (Add): {operation_avg['multi_object_add']['PSNR']:.4f}\n")
        f.write(f"Combined PSNR: {multi_object_avg['PSNR']:.4f}\n\n")
        
        f.write(f"LPIPS (Remove): {operation_avg['multi_object_remove']['LPIPS']:.4f}\n")
        f.write(f"LPIPS (Add): {operation_avg['multi_object_add']['LPIPS']:.4f}\n")
        f.write(f"Combined LPIPS: {multi_object_avg['LPIPS']:.4f}\n\n")
                
        f.write(f"SSIM (Remove): {operation_avg['multi_object_remove']['SSIM']:.4f}\n")
        f.write(f"SSIM (Add): {operation_avg['multi_object_add']['SSIM']:.4f}\n")
        f.write(f"Combined SSIM: {multi_object_avg['SSIM']:.4f}\n")
        f.write("="*40 + "\n\n")
    
    print(f"JSON metrics saved to {args.output_json}")
    print(f"Text metrics summary saved to {output_text_file}")

if __name__ == "__main__":
    main()