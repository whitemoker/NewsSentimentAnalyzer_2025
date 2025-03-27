# 中文新闻自动摘要系统（情感一致性）

本项目实现了一个基于预训练模型的中文新闻自动摘要系统，该系统在保证文本摘要性能的前提下，保证情感一致性。

## 项目特点

- **抽取式摘要**：基于BertSum实现，使用bert-base-chinese预训练模型
- **生成式摘要**：基于Pegasus实现，文本内容和情感特征双通道联合训练
- **多任务学习**：同时优化摘要质量和情感一致性
- **评估指标**：使用Rouge评价指标和情感一致性评估
- **可视化界面**：提供模型性能对比的可视化界面

## 项目结构

```
NewsSentimentAnalyzer_2025/
├── data/                      # 数据目录
│   ├── lcsts/                 # LCSTS数据集
│   └── nlpcc/                 # NLPCC2017 Task3数据集
├── models/                    # 模型定义
│   ├── bertsum/               # 抽取式摘要模型
│   └── pegasus/               # 生成式摘要模型
├── src/                       # 源代码
│   ├── data/                  # 数据处理
│   ├── evaluation/            # 评估模块
│   ├── sentiment/             # 情感分析
│   ├── training/              # 训练模块
│   └── visualization/         # 可视化模块
├── notebooks/                 # Jupyter笔记本
└── docs/                      # 文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据集

本项目使用以下数据集：

1. **LCSTS**：大规模中文短文本摘要数据集
2. **NLPCC2017 Task3**：中文新闻摘要数据集

## 模型实现

### 抽取式摘要模型

基于BertSum实现的抽取式摘要模型，使用bert-base-chinese作为预训练模型，通过多任务学习同时优化摘要质量和情感一致性。

```python
class BertSumExtractor(BertPreTrainedModel):
    """基于BERT的抽取式摘要模型，带情感分类器"""
    
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.sentiment_classifier = nn.Linear(config.hidden_size, 3)  # 3分类：积极、中性、消极
        self.init_weights()
```

### 生成式摘要模型

基于Pegasus实现的生成式摘要模型，采用文本内容和情感特征双通道联合训练，通过多任务学习保证情感一致性。

```python
class SentimentAwarePegasus(nn.Module):
    """情感感知Pegasus模型，双通道联合训练"""
    
    def __init__(self, model_name: str = "IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese-V1"):
        super().__init__()
        self.pegasus = PegasusForConditionalGeneration.from_pretrained(model_name)
        self.sentiment_classifier = nn.Linear(self.pegasus.config.d_model, 3)
        self.sentiment_attention = nn.Linear(self.pegasus.config.d_model, 1)
```

### 情感分析器

使用SnowNLP作为中文情感分析工具，评估原文和摘要之间的情感一致性。

```python
class SentimentAnalyzer:
    """中文情感分析器，使用SnowNLP"""
    
    def analyze(self, text: str) -> Dict[str, Union[float, str]]:
        """分析文本情感，返回情感分数和标签"""
        s = SnowNLP(text)
        score = s.sentiments
        
        if score > 0.6:
            label = "positive"
        elif score < 0.4:
            label = "negative"
        else:
            label = "neutral"
            
        return {"score": score, "label": label}
```

### 多任务学习

通过SummaryTrainer实现多任务学习，同时优化摘要质量和情感一致性。

```python
class SummaryTrainer:
    """摘要训练器，支持多任务学习"""
    
    def __init__(self, model_name, learning_rate=2e-5, weight_decay=0.01, 
                 model_type="pegasus", use_sentiment=False, sentiment_loss_weight=1.0):
        self.model_type = model_type
        self.use_sentiment = use_sentiment
        self.sentiment_loss_weight = sentiment_loss_weight
        
        # 初始化模型
        if model_type == "pegasus":
            if use_sentiment:
                self.model = SentimentAwarePegasus(model_name)
            else:
                self.model = PegasusForConditionalGeneration.from_pretrained(model_name)
        elif model_type == "bertsum":
            if use_sentiment:
                self.model = BertSumExtractor.from_pretrained(model_name)
            else:
                # 普通BertSum模型
                pass
```

## 训练

### 抽取式摘要模型训练

```bash
python src/training/train.py \
    --model_type bertsum \
    --model_name bert-base-chinese \
    --lcsts_path data/lcsts/lcsts_data.json \
    --output_dir checkpoints \
    --use_sentiment \
    --num_epochs 5
```

### 生成式摘要模型训练

```bash
python src/training/train.py \
    --model_type pegasus \
    --model_name IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese-V1 \
    --lcsts_path data/lcsts/lcsts_data.json \
    --output_dir checkpoints \
    --use_sentiment \
    --num_epochs 3
```

## 评估

```bash
python src/evaluation/evaluate.py \
    --test_data data/lcsts/lcsts_test.json \
    --models "checkpoints/bertsum_sentiment/final_model,bertsum,true" "checkpoints/pegasus_sentiment/final_model,pegasus,true" \
    --output_dir evaluation_results
```

## 可视化

评估结果会自动生成可视化图表，包括Rouge指标对比、情感一致性对比和性能指标热力图。

```python
class SummaryEvaluator:
    """摘要评估器，计算Rouge指标和情感一致性"""
    
    def evaluate_summaries(self, original_texts: List[str], summaries: List[str], model_name: str) -> Dict[str, Any]:
        """评估摘要质量和情感一致性"""
        # 计算Rouge指标
        rouge_scores = []
        for orig, summ in tqdm(zip(original_texts, summaries), total=len(original_texts), desc="计算Rouge指标"):
            score = self.rouge.get_scores(summ, orig)[0]
            rouge_scores.append(score)
        
        # 计算情感一致性
        sentiment_scores = []
        for orig, summ in tqdm(zip(original_texts, summaries), total=len(original_texts), desc="计算情感一致性"):
            sentiment_result = self.sentiment_analyzer.compute_consistency(orig, summ)
            sentiment_scores.append(sentiment_result)
```

## 情感一致性评估

本项目使用SnowNLP作为中文情感分析工具，评估原文和摘要之间的情感一致性。评估指标包括：

- **标签一致性**：原文和摘要的情感标签（积极、中性、消极）一致的比例
- **分数一致性**：原文和摘要的情感分数相似度

```python
def compute_consistency(self, original_text: str, summary: str) -> Dict[str, Union[float, Dict]]:
    """计算原文和摘要之间的情感一致性"""
    original_sentiment = self.analyze(original_text)
    summary_sentiment = self.analyze(summary)
    
    # 标签一致性（0或1）
    label_consistency = 1.0 if original_sentiment["label"] == summary_sentiment["label"] else 0.0
    
    # 分数一致性（0-1之间的值）
    score_diff = abs(original_sentiment["score"] - summary_sentiment["score"])
    score_consistency = 1.0 - min(score_diff, 1.0)
    
    return {
        "label_consistency": label_consistency,
        "score_consistency": score_consistency,
        "original_sentiment": original_sentiment,
        "summary_sentiment": summary_sentiment
    }
```

## 参考文献

1. Liu, Y., & Lapata, M. (2019). Text summarization with pretrained encoders. arXiv preprint arXiv:1908.08345.
2. Zhang, J., Zhao, Y., Saleh, M., & Liu, P. (2020). Pegasus: Pre-training with extracted gap-sentences for abstractive summarization. In International Conference on Machine Learning (pp. 11328-11339). PMLR.
3. LCSTS数据集: Hu, B., Chen, Q., & Zhu, F. (2015). LCSTS: A large scale Chinese short text summarization dataset. arXiv preprint arXiv:1506.05865.
4. SnowNLP: https://github.com/isnowfy/snownlp
