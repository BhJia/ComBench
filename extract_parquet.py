import os
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64
import argparse
import json

def ensure_directory(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def numpy_array_to_image(arr):
    """将NumPy数组转换为PIL图像"""
    # 确保是uint8类型
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
            
    # 根据维度创建图像
    if len(arr.shape) == 2:  # 灰度图像
        return Image.fromarray(arr, mode='L')
    elif len(arr.shape) == 3 and arr.shape[2] == 3:  # RGB图像
        return Image.fromarray(arr, mode='RGB')
    elif len(arr.shape) == 3 and arr.shape[2] == 4:  # RGBA图像
        return Image.fromarray(arr, mode='RGBA')
    elif len(arr.shape) == 3 and arr.shape[2] == 1:  # 单通道图像
        return Image.fromarray(arr.squeeze(), mode='L')
    else:
        raise ValueError(f"无法处理形状为 {arr.shape} 的数组")

def save_image(img_data, save_path):
    """保存图像数据到指定路径"""
    directory = os.path.dirname(save_path)
    ensure_directory(directory)
    
    # 如果是PIL Image对象，直接保存
    if isinstance(img_data, Image.Image):
        img_data.save(save_path)
        print(f"成功保存PIL图像: {save_path}")
        return True
        
    # 如果是NumPy数组，转换为PIL Image再保存
    elif isinstance(img_data, np.ndarray):
        try:
            img = numpy_array_to_image(img_data)
            img.save(save_path)
            print(f"成功保存NumPy图像: {save_path}")
            return True
        except Exception as e:
            print(f"保存NumPy图像时出错 {save_path}: {e}")
            return False
            
    # 如果是字节数据，转换为PIL Image再保存
    elif isinstance(img_data, bytes):
        try:
            img = Image.open(io.BytesIO(img_data))
            img.save(save_path)
            print(f"成功保存字节图像: {save_path}")
            return True
        except Exception as e:
            print(f"保存字节图像时出错 {save_path}: {e}")
            return False
            
    # 如果是字典格式，尝试提取图像数据
    elif isinstance(img_data, dict):
        try:
            # 打印字典键以便调试
            # print(f"字典键: {list(img_data.keys())}")
            
            # 尝试直接从字典中提取PIL.Image
            if 'image' in img_data and isinstance(img_data['image'], Image.Image):
                img_data['image'].save(save_path)
                # print(f"成功保存字典中的PIL图像: {save_path}")
                return True
                
            # 尝试从字典中提取NumPy数组
            elif 'array' in img_data and isinstance(img_data['array'], np.ndarray):
                img = numpy_array_to_image(img_data['array'])
                img.save(save_path)
                # print(f"成功保存字典中的NumPy数组: {save_path}")
                return True
                
            # 尝试从字典中的常见键提取数据
            elif 'bytes' in img_data and img_data['bytes'] is not None:
                if isinstance(img_data['bytes'], bytes):
                    img = Image.open(io.BytesIO(img_data['bytes']))
                    img.save(save_path)
                    # print(f"成功保存字典中的字节数据: {save_path}")
                    return True
                    
            elif 'data' in img_data and img_data['data'] is not None:
                if isinstance(img_data['data'], bytes):
                    img = Image.open(io.BytesIO(img_data['data']))
                    img.save(save_path)
                    # print(f"成功保存字典中的data字段: {save_path}")
                    return True
                elif isinstance(img_data['data'], np.ndarray):
                    img = numpy_array_to_image(img_data['data'])
                    img.save(save_path)
                    # print(f"成功保存字典中的data数组: {save_path}")
                    return True
                    
            elif 'encoded' in img_data and img_data['encoded'] is not None:
                # 可能是Base64编码的数据
                image_bytes = base64.b64decode(img_data['encoded'])
                img = Image.open(io.BytesIO(image_bytes))
                img.save(save_path)
                # print(f"成功保存Base64编码的图像: {save_path}")
                return True
                
            # 如果字典中有PIL支持的模式和大小，可能是图像数据
            elif 'mode' in img_data and 'size' in img_data:
                try:
                    # 尝试重建PIL图像
                    if 'data' in img_data:
                        mode = img_data['mode']
                        size = img_data['size']
                        data = img_data['data']
                        img = Image.frombytes(mode, size, data)
                        img.save(save_path)
                        print(f"成功从模式和大小重建PIL图像: {save_path}")
                        return True
                except Exception as e:
                    print(f"尝试从模式和大小重建图像失败: {e}")
            
            print(f"无法从字典中提取有效的图像数据")
            return False
            
        except Exception as e:
            print(f"从字典提取并保存图像时出错 {save_path}: {e}")
            return False
    else:
        print(f"不支持的图像数据类型: {type(img_data)}")
        return False

def process_parquet_file(file_path, output_base_dir):
    """处理单个parquet文件并提取图像和元数据"""
    # print(f"处理文件: {file_path}")
    
    try:
        # 读取parquet文件
        df = pd.read_parquet(file_path)
        
        # 打印列名，帮助调试
        print(f"列名: {df.columns.tolist()}")
        
        # 添加计数器来跟踪成功保存的图像
        saved_images_count = 0
        
        # 创建一个字典来存储每个任务的元数据
        task_metadata = {}
        
        # 遍历每一行
        for index, row in df.iterrows():
            task = row.get('task', 'unknown')
            image_path = row.get('image_path', f"image_{index}.png")
            instruction = row.get('instruction', '')
            caption = row.get('caption', '')
            
            # 创建任务的元数据字典（如果不存在）
            if task not in task_metadata:
                task_metadata[task] = []
            
            # 为这个条目添加元数据
            entry_metadata = {
                'image_path': image_path,
                'instruction': instruction,
                'caption': caption
            }
            task_metadata[task].append(entry_metadata)
            
            # 打印第一行的数据类型，帮助调试
            if index == 0:
                for col in df.columns:
                    if col in row and row[col] is not None:
                        print(f"列 '{col}' 的数据类型: {type(row[col])}")
                        # 如果是字典，打印它的键
                        if isinstance(row[col], dict):
                            print(f"列 '{col}' 字典的键: {list(row[col].keys())}")
            
            # 保存原始图像
            if 'input_image' in row and row['input_image'] is not None:
                input_save_path = os.path.join(output_base_dir, task, 'input_image', image_path)
                if save_image(row['input_image'], input_save_path):
                    saved_images_count += 1
            
            # 保存编辑后的图像
            if 'edited_image' in row and row['edited_image'] is not None:
                edited_save_path = os.path.join(output_base_dir, task, 'edited_image', image_path)
                if save_image(row['edited_image'], edited_save_path):
                    saved_images_count += 1
            
            # 保存mask（如果存在）
            if 'mask' in row and row['mask'] is not None:
                mask_save_path = os.path.join(output_base_dir, task, 'mask', image_path)
                if save_image(row['mask'], mask_save_path):
                    saved_images_count += 1
        
        # 保存每个任务的元数据到JSON文件
        for task, entries in task_metadata.items():
            task_dir = os.path.join(output_base_dir, task)
            ensure_directory(task_dir)

            metadata_file = os.path.join(task_dir, 'metadata.json')

            # 如果已有metadata，先读出来合并
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    try:
                        existing_entries = json.load(f)
                        if isinstance(existing_entries, list):
                            entries = existing_entries + entries
                    except Exception as e:
                        print(f"读取现有元数据失败，跳过合并: {e}")

            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
            print(f"已保存 {task} 的元数据到 {metadata_file}，共 {len(entries)} 条记录")
        
        print(f"成功保存了 {saved_images_count} 张图像")
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='从Parquet文件提取图像数据')
    parser.add_argument('--input_dir', type=str, required=True, help='包含parquet文件的输入目录')
    parser.add_argument('--output_dir', type=str, required=True, help='保存图像的输出目录')
    parser.add_argument('--debug', action='store_true', help='启用调试模式，处理单个文件并打印详细信息')
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    debug_mode = args.debug
    
    # 确保输出目录存在
    ensure_directory(output_dir)
    
    # 处理目录中的所有parquet文件
    processed_files = 0
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.parquet'):
            file_path = os.path.join(input_dir, filename)
            
            if debug_mode:
                # 在调试模式下，读取文件并打印第一行的数据结构
                try:
                    df = pd.read_parquet(file_path)
                    print(f"文件 {filename} 的列名: {df.columns.tolist()}")
                    if not df.empty:
                        row = df.iloc[0]
                        for col in df.columns:
                            if col in ['input_image', 'edited_image', 'mask'] and row[col] is not None:
                                print(f"列 '{col}' 的数据类型: {type(row[col])}")
                                # 如果是字典，尝试打印它的键
                                if isinstance(row[col], dict):
                                    print(f"列 '{col}' 字典的键: {list(row[col].keys())}")
                                # 尝试探测一些常见的键值
                                if isinstance(row[col], dict) and 'mode' in row[col]:
                                    print(f"图像模式: {row[col].get('mode')}")
                                if isinstance(row[col], dict) and 'size' in row[col]:
                                    print(f"图像大小: {row[col].get('size')}")
                except Exception as e:
                    print(f"调试模式读取文件 {file_path} 时出错: {e}")
                
                # 在调试模式下只处理第一个文件
                process_parquet_file(file_path, output_dir)
                break
            else:
                process_parquet_file(file_path, output_dir)
                processed_files += 1
    
    if not debug_mode:
        print(f"所有图像已提取并保存到 {output_dir}，共处理了 {processed_files} 个parquet文件")
    else:
        print("调试模式结束，仅处理了第一个parquet文件")

if __name__ == "__main__":
    main()