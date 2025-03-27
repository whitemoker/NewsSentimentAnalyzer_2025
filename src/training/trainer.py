import torch
from torch.utils.data import DataLoader
from transformers import PegasusForConditionalGeneration, AutoTokenizer, BertConfig
from tqdm import tqdm
import logging
from typing import Dict, List, Union, Optional, Any
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class SummaryTrainer:
    def __init__(
        self,
        model_name: str = "IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese-V1",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        model_type: str = "pegasus",  # 'pegasus' 或 'bertsum'
        use_sentiment: bool = False,
        sentiment_loss_weight: float = 1.0
    ):
        self.device = device
        self.model_type = model_type
        self.use_sentiment = use_sentiment
        self.sentiment_loss_weight = sentiment_loss_weight
        
        if model_type == "pegasus":
            if use_sentiment:
                from models.pegasus.sentiment_pegasus import SentimentAwarePegasus
                self.model = SentimentAwarePegasus(model_name).to(device)
            else:
                self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif model_type == "bertsum":
            from models.bertsum.modeling import BertSumExtractor
            config = BertConfig.from_pretrained("bert-base-chinese")
            self.model = BertSumExtractor(config).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.max_grad_norm = max_grad_norm
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        total_sentiment_loss = 0
        total_summary_loss = 0
        sentiment_preds = []
        sentiment_labels = []
        
        for batch in tqdm(dataloader, desc="Training"):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            if self.model_type == "pegasus":
                if self.use_sentiment and 'content_sentiment' in batch:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'],
                        decoder_attention_mask=batch['decoder_attention_mask'],
                        sentiment_labels=batch['content_sentiment']
                    )
                    loss = outputs["loss"]
                    sentiment_logits = outputs["sentiment_logits"]
                    
                    pred_sentiment = torch.argmax(sentiment_logits, dim=1).cpu().numpy()
                    true_sentiment = batch['content_sentiment'].cpu().numpy()
                    sentiment_preds.extend(pred_sentiment)
                    sentiment_labels.extend(true_sentiment)
                    
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'],
                        decoder_attention_mask=batch['decoder_attention_mask']
                    )
                    loss = outputs.loss
            
            elif self.model_type == "bertsum":
                if self.use_sentiment and 'content_sentiment' in batch:
                    loss, summary_logits, sentiment_logits = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'],
                        sentiment_labels=batch['content_sentiment']
                    )
                    
                    pred_sentiment = torch.argmax(sentiment_logits, dim=1).cpu().numpy()
                    true_sentiment = batch['content_sentiment'].cpu().numpy()
                    sentiment_preds.extend(pred_sentiment)
                    sentiment_labels.extend(true_sentiment)
                    
                else:
                    summary_logits, _ = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss_fct = torch.nn.BCEWithLogitsLoss()
                    loss = loss_fct(summary_logits.view(-1), batch['labels'].float().view(-1))
            
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        results = {"loss": total_loss / len(dataloader)}
        if sentiment_preds and sentiment_labels:
            sentiment_accuracy = accuracy_score(sentiment_labels, sentiment_preds)
            sentiment_f1 = f1_score(sentiment_labels, sentiment_preds, average='weighted')
            results.update({
                "sentiment_accuracy": sentiment_accuracy,
                "sentiment_f1": sentiment_f1
            })
            
        return results
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        sentiment_preds = []
        sentiment_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if self.model_type == "pegasus":
                    if self.use_sentiment and 'content_sentiment' in batch:
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels'],
                            decoder_attention_mask=batch['decoder_attention_mask'],
                            sentiment_labels=batch['content_sentiment']
                        )
                        loss = outputs["loss"]
                        sentiment_logits = outputs["sentiment_logits"]
                        
                        pred_sentiment = torch.argmax(sentiment_logits, dim=1).cpu().numpy()
                        true_sentiment = batch['content_sentiment'].cpu().numpy()
                        sentiment_preds.extend(pred_sentiment)
                        sentiment_labels.extend(true_sentiment)
                    else:
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels'],
                            decoder_attention_mask=batch['decoder_attention_mask']
                        )
                        loss = outputs.loss
                
                elif self.model_type == "bertsum":
                    if self.use_sentiment and 'content_sentiment' in batch:
                        loss, summary_logits, sentiment_logits = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels'],
                            sentiment_labels=batch['content_sentiment']
                        )
                        
                        pred_sentiment = torch.argmax(sentiment_logits, dim=1).cpu().numpy()
                        true_sentiment = batch['content_sentiment'].cpu().numpy()
                        sentiment_preds.extend(pred_sentiment)
                        sentiment_labels.extend(true_sentiment)
                    else:
                        summary_logits, _ = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels']
                        )
                        loss_fct = torch.nn.BCEWithLogitsLoss()
                        loss = loss_fct(summary_logits.view(-1), batch['labels'].float().view(-1))
                
                total_loss += loss.item()
        
        results = {"eval_loss": total_loss / len(dataloader)}
        
        if sentiment_preds and sentiment_labels:
            sentiment_accuracy = accuracy_score(sentiment_labels, sentiment_preds)
            sentiment_f1 = f1_score(sentiment_labels, sentiment_preds, average='weighted')
            results.update({
                "sentiment_accuracy": sentiment_accuracy,
                "sentiment_f1": sentiment_f1
            })
            
        return results
    
    def save_checkpoint(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.logger.info(f"Model saved to {path}")
    
    def generate_summary(self, text: str, max_length: int = 128, return_sentiment: bool = False) -> Union[str, Dict[str, Any]]:
        """
        生成摘要，可选择返回情感分析结果
        
        Args:
            text: 输入文本
            max_length: 最大摘要长度
            return_sentiment: 是否返回情感分析结果
            
        Returns:
            如果return_sentiment为False，返回摘要字符串
            如果return_sentiment为True，返回包含摘要和情感分析的字典
        """
        self.model.eval()
        inputs = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            if self.model_type == "pegasus":
                if hasattr(self.model, 'generate'):
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=max_length,
                        num_beams=4,
                        length_penalty=2.0,
                        early_stopping=True
                    )
                else:
                    outputs = self.model.pegasus.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=max_length,
                        num_beams=4,
                        length_penalty=2.0,
                        early_stopping=True
                    )
                
                summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            elif self.model_type == "bertsum":
                logits, _ = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                
                sentences = text.split('。')
                sentences = [s + '。' for s in sentences if s.strip()]
                
                scores = torch.sigmoid(logits).cpu().numpy()
                top_indices = scores.argsort()[0][-3:]  # 选择得分最高的3个句子
                
                selected_indices = sorted(top_indices)
                summary = ''.join([sentences[i] for i in selected_indices if i < len(sentences)])
        
        if return_sentiment and self.use_sentiment:
            from src.sentiment.analyzer import SentimentAnalyzer
            analyzer = SentimentAnalyzer()
            sentiment_result = analyzer.compute_consistency(text, summary)
            return {
                "summary": summary,
                "sentiment_consistency": sentiment_result
            }
        
        return summary
