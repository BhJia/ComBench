import json
import re
from collections import defaultdict

def convert_json_format(input_file, output_file):
    """
    将JSON从格式A转换为格式B
    格式A: 按turn分开的记录
    格式B: 按image_name分组的记录
    """
    
    # 读取输入JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 用于存储按image_name分组的数据
    grouped_data = defaultdict(dict)
    
    # 处理每条记录
    for item in data:
        image_path = item['image_path']
        instruction = item['instruction']
        caption = item['caption']
        
        # 从image_path中提取turn信息和image_name
        # 例如: "turn1_add/02b72b8d_00002_t1.png" -> turn1, "02b72b8d_00002_t1.png"
        path_parts = image_path.split('/')
        folder_name = path_parts[0]  # 例如: "turn1_add"
        image_name = path_parts[1]   # 例如: "02b72b8d_00002_t1.png"
        
        # 提取turn号码
        turn_match = re.search(r'turn(\d+)', folder_name)
        if turn_match:
            turn_num = turn_match.group(1)
            
            # 初始化image_name对应的字典
            if 'image_name' not in grouped_data[image_name]:
                grouped_data[image_name]['image_path'] = image_name
            
            # 添加instruction和caption
            grouped_data[image_name][f'turn{turn_num}_add_ins'] = instruction
            grouped_data[image_name][f'turn{turn_num}_caption'] = caption
    
    # 将结果转换为列表
    result = list(grouped_data.values())
    
    # 写入输出文件，标准JSON数组格式
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！输出文件: {output_file}")
    print(f"共处理了 {len(result)} 个图像组")
    
    return result

def convert_json_format_from_string(json_string):
    """
    直接从JSON字符串转换格式（用于测试）
    """
    data = json.loads(json_string)
    grouped_data = defaultdict(dict)
    
    for item in data:
        image_path = item['image_path']
        instruction = item['instruction']
        caption = item['caption']
        
        path_parts = image_path.split('/')
        folder_name = path_parts[0]
        image_name = path_parts[1]
        
        turn_match = re.search(r'turn(\d+)', folder_name)
        if turn_match:
            turn_num = turn_match.group(1)
            
            if 'image_name' not in grouped_data[image_name]:
                grouped_data[image_name]['image_name'] = image_name
            
            grouped_data[image_name][f'turn{turn_num}_add_ins'] = instruction
            grouped_data[image_name][f'turn{turn_num}_caption'] = caption
    
    return list(grouped_data.values())

# 使用示例
if __name__ == "__main__":
    # 测试用的示例数据
    test_data = '''[
        {
            "image_path": "turn1_add/02b72b8d_00002_t1.png",
            "instruction": "add a white fish above the biggest rock on the left between the  waterweeds",
            "caption": "a yellow fish"
        },
        {
            "image_path": "turn2_add/02b72b8d_00002_t1.png",
            "instruction": "add a yellow fish in front of the rock under the white fish",
            "caption": "a white fish"
        },
        {
            "image_path": "turn1_add/02b72b8d_00002_t2.png",
            "instruction": "add a yellow fish in front of the biggest rock on the left",
            "caption": "a white fish"
        },
        {
            "image_path": "turn2_add/02b72b8d_00002_t2.png",
            "instruction": "add a white fish above the rock to the upper left of the yellow fish",
            "caption": "a yellow fish"
        }
    ]'''
    
    # # 测试转换功能
    # result = convert_json_format_from_string(test_data)
    
    # print("转换结果:")
    # for item in result:
    #     print(json.dumps(item, ensure_ascii=False))

    convert_json_format('/data/jbh/CompBench_dataset/tasks/multi_turn_remove/metadata.json', '/data/jbh/CompBench_dataset/tasks/multi_turn_remove/converted_metadata.json')