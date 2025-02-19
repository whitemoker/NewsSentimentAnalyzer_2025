# Randeng-Pegasus 与 T5 模型对比研究

## 1. 模型架构对比

### 1.1 基础架构
- Randeng-Pegasus
  - 基于PEGASUS架构 [[论文链接]](https://arxiv.org/abs/1912.08777)
  - 专门针对摘要任务设计
  - 使用Gap Sentences Generation (GSG) 预训练目标

- T5
  - 统一的文本到文本框架 [[论文链接]](https://arxiv.org/abs/1910.10683)
  - 通用NLP任务设计
  - 使用Span Corruption预训练目标

### 1.2 主要区别

1. 预训练目标：
   - Randeng-Pegasus：使用GSG策略，通过预测重要句子来学习文档结构
   - T5：使用Span Corruption，随机掩码连续文本片段

2. 模型规模：
   - Randeng-Pegasus-523M：约5.23亿参数
   - T5-base：约2.2亿参数

3. 中文优化：
   - Randeng-Pegasus：专门针对中文场景优化
     - 使用中文分词
     - 在多个中文数据集上微调
   - T5：需要额外的中文适配

## 2. 训练方法对比

### 2.1 预训练策略
- Randeng-Pegasus：
  - 使用新闻文章作为预训练数据
  - 选择重要句子作为生成目标
  - 参考：[IDEA-CCNL/Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)

- T5：
  - 使用通用文本数据
  - 随机掩码策略
  - 多任务训练方式

### 2.2 微调方法
- Randeng-Pegasus：
  - 在LCSTS等中文数据集上微调
  - 针对新闻摘要任务优化
  - ROUGE-L分数达到43.51

- T5：
  - 需要额外的中文数据集微调
  - 通用任务适配能力强

## 3. 性能对比

### 3.1 LCSTS数据集上的表现
- Randeng-Pegasus：
  - ROUGE-1：46.94
  - ROUGE-2：33.92
  - ROUGE-L：43.51

- T5（中文微调后）：
  - 性能数据待补充
  - 需要专门的中文适配

### 3.2 推理效率
- Randeng-Pegasus：
  - 针对摘要任务优化
  - 生成速度较快
  - 支持批处理

- T5：
  - 通用任务处理
  - 需要额外的任务适配层

## 4. 选型建议

基于以上对比，建议选择Randeng-Pegasus模型，原因如下：
1. 专门针对中文摘要任务优化
2. 在LCSTS等中文数据集上表现优秀
3. 预训练目标更适合摘要任务
4. 提供完整的中文支持

## 5. 参考资料

1. [PEGASUS论文](https://arxiv.org/abs/1912.08777)
2. [T5论文](https://arxiv.org/abs/1910.10683)
3. [Randeng-Pegasus介绍](https://zhuanlan.zhihu.com/p/528753707)
4. [Fengshenbang-LM项目](https://github.com/IDEA-CCNL/Fengshenbang-LM)
5. [LCSTS数据集论文](https://aclanthology.org/D15-1229/)
