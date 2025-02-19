import nbformat as nbf

nb = nbf.v4.new_notebook()

# Markdown cell - Introduction
intro_cell = nbf.v4.new_markdown_cell("""# Randeng-Pegasus 模型微调测试

本notebook用于测试LCSTS数据集的加载和Randeng-Pegasus模型的微调准备工作。""")

# Code cell - Imports
imports_cell = nbf.v4.new_code_cell("""import sys
sys.path.append('..')

import json
import torch
from transformers import AutoTokenizer
from src.data.dataset import LCSTSDataset""")

# Markdown cell - Dataset Loading
dataset_md = nbf.v4.new_markdown_cell("""## 1. 加载数据集

首先测试LCSTS数据集的加载""")

# Code cell - Load Dataset
load_data = nbf.v4.new_code_cell("""# 读取数据集文件
data_path = '../data/lcsts/lcsts_data.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"数据集大小: {len(data)}条")
print("\\n示例数据:")
for i in range(3):
    print(f"\\n例{i+1}:")
    print(f"标题: {data[i]['title']}")
    print(f"内容: {data[i]['content']}")""")

# Markdown cell - Dataset Class Test
dataset_class_md = nbf.v4.new_markdown_cell("""## 2. 测试数据集类

测试我们的LCSTSDataset类是否正常工作""")

# Code cell - Test Dataset Class
test_dataset = nbf.v4.new_code_cell("""# 初始化tokenizer
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
print("数据集样本格式:")
for key, value in sample.items():
    print(f"{key}: shape {value.shape}")

# 解码测试
print("\\n解码测试:")
decoded_source = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
decoded_target = tokenizer.decode(sample['labels'], skip_special_tokens=True)
print(f"源文本: {decoded_source}")
print(f"目标摘要: {decoded_target}")""")

# Add cells to notebook
nb.cells = [intro_cell, imports_cell, dataset_md, load_data, dataset_class_md, test_dataset]

# Write the notebook
with open('randeng_pegasus_finetuning.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
