import sys
sys.path.append('..')

import json
import torch
from transformers import AutoTokenizer
from src.data.dataset import LCSTSDataset

# 读取数据集文件
data_path = '../data/lcsts/lcsts_data.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"数据集大小: {len(data)}条")
print("\n示例数据:")
for i in range(3):
    print(f"\n例{i+1}:")
    print(f"标题: {data[i]['title']}")
    print(f"内容: {data[i]['content']}")

# 初始化tokenizer
model_name = "IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese-V1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 创建数据集实例
dataset = LCSTSDataset(
    data_path=data_path,
    tokenizer=tokenizer,
    max_source_length=512,
    max_target_length=128
)

# 测试数据集的第一个样本
sample = dataset[0]
print("\n数据集样本格式:")
for key, value in sample.items():
    print(f"{key}: shape {value.shape}")

# 解码测试
print("\n解码测试:")
decoded_source = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
decoded_target = tokenizer.decode(sample['labels'], skip_special_tokens=True)
print(f"源文本: {decoded_source}")
print(f"目标摘要: {decoded_target}")
