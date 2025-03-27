import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

class BertSumExtractor(BertPreTrainedModel):
    """BertSum抽取式摘要模型，参考论文：https://arxiv.org/abs/1903.10318"""
    
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.sentiment_classifier = nn.Linear(config.hidden_size, 3)  # 3分类：积极、中性、消极
        
        self.init_weights()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        sentiment_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        
        cls_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        
        logits = self.classifier(cls_output)  # [batch_size, 1]
        
        sentiment_logits = self.sentiment_classifier(cls_output)  # [batch_size, 3]
        
        loss = None
        sentiment_loss = None
        
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float().view(-1))
            
        if sentiment_labels is not None:
            sentiment_loss_fct = nn.CrossEntropyLoss()
            sentiment_loss = sentiment_loss_fct(sentiment_logits.view(-1, 3), sentiment_labels.view(-1))
        
        if loss is not None and sentiment_loss is not None:
            total_loss = loss + sentiment_loss
            return total_loss, logits, sentiment_logits
        
        return logits, sentiment_logits
