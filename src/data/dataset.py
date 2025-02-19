from torch.utils.data import Dataset
import json
import torch
from typing import Dict, List

class LCSTSDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_source_length: int = 512, max_target_length: int = 128):
        """
        Initialize LCSTS dataset
        Args:
            data_path: Path to the LCSTS json file
            tokenizer: HuggingFace tokenizer
            max_source_length: Maximum length of source text
            max_target_length: Maximum length of target summary
        """
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Tokenize source text
        source_encoding = self.tokenizer(
            item['content'],
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target summary
        target_encoding = self.tokenizer(
            item['title'],
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
            'decoder_attention_mask': target_encoding['attention_mask'].squeeze()
        }
