import nbformat as nbf

nb = nbf.v4.new_notebook()

# 创建单元格
cells = [
    nbf.v4.new_markdown_cell("""# Randeng-Pegasus 模型微调测试

本notebook用于测试LCSTS数据集的加载和Randeng-Pegasus模型的微调准备工作。

## 1. 环境准备"""),
    
    nbf.v4.new_code_cell("""import sys
sys.path.append('..')

import json
import torch
from transformers import AutoTokenizer, PegasusForConditionalGeneration
from src.data.dataset import LCSTSDataset"""),
    
    nbf.v4.new_markdown_cell("""## 2. 数据集加载与验证

首先加载LCSTS数据集并查看样本数据"""),
    
    nbf.v4.new_code_cell("""# 加载数据集文件
data_path = 'data/lcsts/lcsts_data.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'数据集大小: {len(data)}条')
print('\\n示例数据:')
for i in range(3):
    print(f'\\n例{i+1}:')
    print(f'标题: {data[i]["title"]}')
    print(f'内容: {data[i]["content"]}')"""),
    
    nbf.v4.new_markdown_cell("""## 3. 模型与分词器初始化

使用本地的Randeng-Pegasus模型和分词器"""),
    
    nbf.v4.new_code_cell("""# 初始化分词器和模型
# 使用BERT中文分词器进行测试
model_name = "bert-base-chinese"
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
print('数据集样本格式:')
for key, value in sample.items():
    print(f'{key}: shape {value.shape}')

# 解码测试
print('\\n解码测试:')
decoded_source = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
decoded_target = tokenizer.decode(sample['labels'], skip_special_tokens=True)
print(f'源文本: {decoded_source}')
print(f'目标摘要: {decoded_target}')""")
]

nb.cells = cells

# 写入notebook
with open('test_randeng_pegasus.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
