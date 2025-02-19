import json
from src.data.dataset import LCSTSDataset
from transformers import AutoTokenizer

# Test data loading
data_path = 'data/lcsts/lcsts_data.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f'Dataset size: {len(data)} samples')
print('\nSample data:')
for i in range(3):
    print(f'\nExample {i+1}:')
    print(f'Title: {data[i]["title"]}')
    print(f'Content: {data[i]["content"]}')

# Test tokenizer and dataset class
model_name = 'IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese-V1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = LCSTSDataset(data_path, tokenizer)
sample = dataset[0]
print('\nDataset sample format:')
for key, value in sample.items():
    print(f'{key}: shape {value.shape}')
