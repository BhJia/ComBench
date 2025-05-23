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

def calculate_single_metrics(gt_img_path, edited_img_path, mask_img_path, clip_model_path, metrics_size=512):
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
    # import pdb; pdb.set_trace()
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
    # 添加命令行参数
    parser = argparse.ArgumentParser(description='Calculate image editing metrics')
    parser.add_argument('--edited_dir', type=str, required=True, help='Directory containing edited images')
    parser.add_argument('--output_json', type=str, default="./multi_clip_scores.json", help='Path to output JSON file')
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
        'multi_object_remove': {'CLIP': []},
        'multi_object_add': {'CLIP': []},
        'turn1_remove': {'CLIP': []},
        'turn2_remove': {'CLIP': []},
        'turn1_add': {'CLIP': []},
        'turn2_add': {'CLIP': []}
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
                        gt_img_path=f"{turn1_remove_gt_dir}/{image_path}",
                        edited_img_path=f"{turn1_remove_edited_dir}/{image_path}",
                        mask_img_path=f"{turn1_remove_mask_dir}/{image_path}",
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
                        gt_img_path=f"{turn2_remove_gt_dir}/{image_path}",
                        edited_img_path=f"{turn2_remove_edited_dir}/{image_path}",
                        mask_img_path=f"{turn2_remove_mask_dir}/{image_path}",
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
                        gt_img_path=f"{turn1_add_gt_dir}/{image_path}",
                        edited_img_path=f"{turn1_add_edited_dir}/{image_path}",
                        mask_img_path=f"{turn1_add_mask_dir}/{image_path}",
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
                        gt_img_path=f"{turn2_add_gt_dir}/{image_path}",
                        edited_img_path=f"{turn2_add_edited_dir}/{image_path}",
                        mask_img_path=f"{turn2_add_mask_dir}/{image_path}",
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

                # import pdb; pdb.set_trace()
                
                if not os.path.exists(mask1_path) or not os.path.exists(mask2_path):
                    print(f"Warning: Mask files for {image_path} not found. Skipping multi-object remove.")
                    continue
                
                mask1_metrics = calculate_single_metrics(
                    gt_img_path=f"{turn1_remove_gt_dir}/{base_name}_t1.png",
                    edited_img_path=f"{multi_obj_rm_edited_dir}/{image_path}",
                    mask_img_path=mask1_path,
                    clip_model_path=clip_model_path
                )
                mask2_metrics = calculate_single_metrics(
                    gt_img_path=f"{turn2_remove_gt_dir}/{base_name}_t1.png",
                    edited_img_path=f"{multi_obj_rm_edited_dir}/{image_path}",
                    mask_img_path=mask2_path,
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
                
                if not os.path.exists(mask1_path) or not os.path.exists(mask2_path):
                    print(f"Warning: Mask files for {image_path} not found. Skipping multi-object add.")
                    continue
                
                mask1_metrics = calculate_single_metrics(
                    gt_img_path=f"{turn1_add_gt_dir}/{base_name}_t1.png",
                    edited_img_path=f"{multi_obj_add_edited_dir}/{image_path}",
                    mask_img_path=mask1_path,
                    clip_model_path=clip_model_path
                )
                mask2_metrics = calculate_single_metrics(
                    gt_img_path=f"{turn2_add_gt_dir}/{base_name}_t1.png",
                    edited_img_path=f"{multi_obj_add_edited_dir}/{image_path}",
                    mask_img_path=mask2_path,
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
    
    # 保存到JSON文件
    output_data = {
        'per_image_metrics': image_metrics,
        'operation_avg': operation_avg,
        'remove_avg': remove_avg,
        'add_avg': add_avg,
        'overall_avg': overall_avg,
        'multi_object_avg': multi_object_avg
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    # 同时保存文本结果文件，保持向后兼容
    output_text_file = f"{edited_dir}/multi_clip_scores.txt"
    with open(output_text_file, 'w') as f:
        # Write individual operation metrics
        for operation in metrics:
            f.write(f"Average Metrics for {operation}:\n")
            f.write("="*40 + "\n")
            f.write(f"CLIP Score: {operation_avg[operation]['CLIP']:.4f}\n")
            f.write("="*40 + "\n\n")
        
        # Write combined remove metrics
        f.write("Combined Metrics for ALL REMOVE operations:\n")
        f.write("="*40 + "\n")
        f.write(f"CLIP Score: {remove_avg['CLIP']:.4f}\n")
        f.write("="*40 + "\n\n")
        
        # Write combined add metrics
        f.write("Combined Metrics for ALL ADD operations:\n")
        f.write("="*40 + "\n")
        f.write(f"CLIP Score: {add_avg['CLIP']:.4f}\n")
        f.write("="*40 + "\n\n")
        
        # Write overall metrics
        f.write("OVERALL Metrics for Multi-turn operations:\n")
        f.write("="*40 + "\n")
        f.write(f"CLIP Score: {overall_avg['CLIP']:.4f}\n")
        f.write("="*40 + "\n")

        # 新增：写入multi_object的指标
        f.write("Metrics for Multi-object Operations:\n")
        f.write("="*40 + "\n")
        f.write(f"CLIP Score (Remove): {operation_avg['multi_object_remove']['CLIP']:.4f}\n")
        f.write(f"CLIP Score (Add): {operation_avg['multi_object_add']['CLIP']:.4f}\n")
        f.write(f"Combined CLIP Score: {multi_object_avg['CLIP']:.4f}\n\n")
        f.write("="*40 + "\n\n")
    
    print(f"JSON metrics saved to {args.output_json}")
    print(f"Text metrics summary saved to {output_text_file}")

if __name__ == "__main__":
    main()