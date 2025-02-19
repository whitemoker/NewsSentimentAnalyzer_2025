import torch
from torch.utils.data import DataLoader, random_split
from transformers import set_seed
import argparse
import os
from datetime import datetime

from src.data.dataset import LCSTSDataset
from src.training.trainer import SummaryTrainer

def main(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Initialize trainer
    trainer = SummaryTrainer(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Load dataset
    dataset = LCSTSDataset(
        data_path=args.data_path,
        tokenizer=trainer.tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length
    )
    
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
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_dataloader)
        print(f"Training loss: {train_loss:.4f}")
        
        # Evaluate
        eval_metrics = trainer.evaluate(val_dataloader)
        eval_loss = eval_metrics['eval_loss']
        print(f"Validation loss: {eval_loss:.4f}")
        
        # Save checkpoint if best model
        if eval_loss < best_loss:
            best_loss = eval_loss
            checkpoint_dir = os.path.join(
                args.output_dir,
                f"checkpoint-epoch-{epoch+1}-loss-{eval_loss:.4f}"
            )
            trainer.save_checkpoint(checkpoint_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese-V1")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)
