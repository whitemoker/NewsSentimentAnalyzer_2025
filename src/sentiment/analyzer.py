import torch
import snownlp
from snownlp import SnowNLP
from typing import Dict, List, Union

class SentimentAnalyzer:
    """中文情感分析器，使用SnowNLP作为权威情感分类器"""
    
    def __init__(self):
        """初始化情感分析器"""
        pass
        
    def analyze(self, text: str) -> Dict[str, Union[float, str]]:
        """
        分析文本情感
        
        Args:
            text: 待分析文本
            
        Returns:
            Dict包含情感分数，0-1之间，越接近1越积极
        """
        sentiment = SnowNLP(text)
        score = sentiment.sentiments
        
        sentiment_label = "neutral"
        if score > 0.6:
            sentiment_label = "positive"
        elif score < 0.4:
            sentiment_label = "negative"
            
        return {
            "score": score,
            "label": sentiment_label
        }
        
    def compute_consistency(self, original_text: str, summary: str) -> Dict[str, Union[float, Dict[str, Union[float, str]]]]:
        """
        计算原文与摘要的情感一致性
        
        Args:
            original_text: 原文
            summary: 摘要
            
        Returns:
            Dict包含情感一致性分数
        """
        orig_sentiment = self.analyze(original_text)
        summ_sentiment = self.analyze(summary)
        
        label_match = 1.0 if orig_sentiment["label"] == summ_sentiment["label"] else 0.0
        
        orig_score = float(orig_sentiment["score"])
        summ_score = float(summ_sentiment["score"])
        score_diff = abs(orig_score - summ_score)
        score_consistency = 1.0 - score_diff  # 分数越接近，一致性越高
        
        return {
            "label_consistency": label_match,
            "score_consistency": score_consistency,
            "original_sentiment": orig_sentiment,
            "summary_sentiment": summ_sentiment
        }
