import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from transformers import set_seed
import argparse
import os
import json
from datetime import datetime
import logging

from src.data.dataset import LCSTSDataset
from src.training.trainer import SummaryTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_nlpcc_data(file_path):
    """加载NLPCC2017 Task3数据集"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append({
                    "content": item.get("document", ""),
                    "title": item.get("summary", "")
                })
            except json.JSONDecodeError:
                logger.warning(f"无法解析NLPCC数据行: {line}")
    return data

def main(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    trainer = SummaryTrainer(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        model_type=args.model_type,
        use_sentiment=args.use_sentiment,
        sentiment_loss_weight=args.sentiment_loss_weight
    )
    
    datasets = []
    
    if args.lcsts_path:
        logger.info(f"加载LCSTS数据集: {args.lcsts_path}")
        if args.use_sentiment:
            from src.data.sentiment_dataset import SentimentAwareLCSTSDataset
            lcsts_dataset = SentimentAwareLCSTSDataset(
                data_path=args.lcsts_path,
                tokenizer=trainer.tokenizer,
                max_source_length=args.max_source_length,
                max_target_length=args.max_target_length,
                cache_dir=args.cache_dir
            )
        else:
            lcsts_dataset = LCSTSDataset(
                data_path=args.lcsts_path,
                tokenizer=trainer.tokenizer,
                max_source_length=args.max_source_length,
                max_target_length=args.max_target_length
            )
        datasets.append(lcsts_dataset)
    
    if args.nlpcc_path:
        logger.info(f"加载NLPCC数据集: {args.nlpcc_path}")
        nlpcc_data = load_nlpcc_data(args.nlpcc_path)
        
        nlpcc_json_path = os.path.join(args.cache_dir, "nlpcc_data.json") if args.cache_dir else "nlpcc_data.json"
        with open(nlpcc_json_path, 'w', encoding='utf-8') as f:
            json.dump(nlpcc_data, f, ensure_ascii=False, indent=2)
        
        if args.use_sentiment:
            from src.data.sentiment_dataset import SentimentAwareLCSTSDataset
            nlpcc_dataset = SentimentAwareLCSTSDataset(
                data_path=nlpcc_json_path,
                tokenizer=trainer.tokenizer,
                max_source_length=args.max_source_length,
                max_target_length=args.max_target_length,
                cache_dir=args.cache_dir
            )
        else:
            nlpcc_dataset = LCSTSDataset(
                data_path=nlpcc_json_path,
                tokenizer=trainer.tokenizer,
                max_source_length=args.max_source_length,
                max_target_length=args.max_target_length
            )
        datasets.append(nlpcc_dataset)
    
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)
        logger.info(f"合并数据集，总样本数: {len(dataset)}")
    elif len(datasets) == 1:
        dataset = datasets[0]
        logger.info(f"使用单一数据集，样本数: {len(dataset)}")
    else:
        raise ValueError("至少需要指定一个数据集路径 (--lcsts_path 或 --nlpcc_path)")

    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_short = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    run_name = f"{model_name_short}_{args.model_type}_{'sentiment' if args.use_sentiment else 'base'}_{timestamp}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    config = vars(args)
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    # Training loop
    best_loss = float('inf')
    best_sentiment_accuracy = 0.0
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_dataloader)
        train_loss = train_metrics["loss"]
        logger.info(f"Training loss: {train_loss:.4f}")
        
        if "sentiment_accuracy" in train_metrics:
            logger.info(f"Training sentiment accuracy: {train_metrics['sentiment_accuracy']:.4f}")
            logger.info(f"Training sentiment F1: {train_metrics['sentiment_f1']:.4f}")
        
        # Evaluate
        eval_metrics = trainer.evaluate(val_dataloader)
        eval_loss = eval_metrics['eval_loss']
        logger.info(f"Validation loss: {eval_loss:.4f}")
        
        if "sentiment_accuracy" in eval_metrics:
            sentiment_accuracy = eval_metrics["sentiment_accuracy"]
            logger.info(f"Validation sentiment accuracy: {sentiment_accuracy:.4f}")
            logger.info(f"Validation sentiment F1: {eval_metrics['sentiment_f1']:.4f}")
        else:
            sentiment_accuracy = 0.0
        
        metrics_path = os.path.join(output_dir, f"metrics_epoch_{epoch+1}.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump({
                "epoch": epoch + 1,
                "train": train_metrics,
                "eval": eval_metrics
            }, f, ensure_ascii=False, indent=2)
        
        if eval_loss < best_loss:
            best_loss = eval_loss
            checkpoint_dir = os.path.join(
                output_dir,
                f"best_loss_checkpoint_epoch_{epoch+1}"
            )
            trainer.save_checkpoint(checkpoint_dir)
            logger.info(f"保存最佳损失模型到 {checkpoint_dir}")
        
        if args.use_sentiment and sentiment_accuracy > best_sentiment_accuracy:
            best_sentiment_accuracy = sentiment_accuracy
            sentiment_checkpoint_dir = os.path.join(
                output_dir,
                f"best_sentiment_checkpoint_epoch_{epoch+1}"
            )
            trainer.save_checkpoint(sentiment_checkpoint_dir)
            logger.info(f"保存最佳情感模型到 {sentiment_checkpoint_dir}")
        
        if args.save_every_epoch:
            epoch_checkpoint_dir = os.path.join(
                output_dir,
                f"checkpoint_epoch_{epoch+1}"
            )
            trainer.save_checkpoint(epoch_checkpoint_dir)
            logger.info(f"保存Epoch {epoch+1}检查点到 {epoch_checkpoint_dir}")
    
    logger.info(f"训练完成！最佳验证损失: {best_loss:.4f}")
    if args.use_sentiment:
        logger.info(f"最佳情感准确率: {best_sentiment_accuracy:.4f}")
    
    final_checkpoint_dir = os.path.join(output_dir, "final_model")
    trainer.save_checkpoint(final_checkpoint_dir)
    logger.info(f"保存最终模型到 {final_checkpoint_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="中文新闻摘要训练脚本，支持情感一致性")
    
    parser.add_argument("--model_name", type=str, default="IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese-V1",
                        help="预训练模型名称或路径")
    parser.add_argument("--model_type", type=str, choices=["pegasus", "bertsum"], default="pegasus",
                        help="模型类型: pegasus(生成式) 或 bertsum(抽取式)")
    
    parser.add_argument("--lcsts_path", type=str, default=None,
                        help="LCSTS数据集路径")
    parser.add_argument("--nlpcc_path", type=str, default=None,
                        help="NLPCC2017 Task3数据集路径")
    parser.add_argument("--cache_dir", type=str, default="cache",
                        help="缓存目录，用于存储情感标签等")
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录，用于保存模型和结果")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--max_source_length", type=int, default=512,
                        help="源文本最大长度")
    parser.add_argument("--max_target_length", type=int, default=128,
                        help="目标摘要最大长度")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载器工作进程数")
    
    parser.add_argument("--use_sentiment", action="store_true",
                        help="是否使用情感一致性训练")
    parser.add_argument("--sentiment_loss_weight", type=float, default=1.0,
                        help="情感损失权重")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--save_every_epoch", action="store_true",
                        help="是否每个epoch都保存检查点")
    
    args = parser.parse_args()
    main(args)
