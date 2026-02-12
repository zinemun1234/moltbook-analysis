"""
Training script for Moltbook models
Includes training loop, evaluation, and model saving
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from tqdm import tqdm
import argparse
from datetime import datetime

from data_loader import MoltbookDataLoader
from models import get_model, get_loss_function

class MoltbookTrainer:
    """Trainer class for Moltbook models"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device='cuda', save_dir='checkpoints'):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
    def train_epoch(self, optimizer, criterion, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'content_logits'):  # Multi-task model
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['content_logits']  # Use content logits for training
            else:
                logits = self.model(input_ids, attention_mask)
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy_score(all_labels, all_preds):.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def evaluate(self, data_loader, criterion):
        """Evaluate model on validation or test set"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'content_logits'):  # Multi-task model
                    outputs = self.model(input_ids, attention_mask)
                    logits = outputs['content_logits']
                else:
                    logits = self.model(input_ids, attention_mask)
                
                # Calculate loss
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_labels)
        
        # Detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels
    
    def train(self, num_epochs, learning_rate=2e-5, weight_decay=0.01, 
              warmup_steps=0, loss_type='crossentropy'):
        """Main training loop"""
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        total_steps = len(self.train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        
        # Setup loss function
        if hasattr(self.model, 'num_classes'):
            criterion = get_loss_function(loss_type, num_classes=self.model.num_classes)
        else:
            criterion = get_loss_function(loss_type, num_classes=9)  # Default to content classes
        
        best_val_acc = 0
        best_model_path = None
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion, scheduler)
            
            # Evaluate
            val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = self.evaluate(
                self.val_loader, criterion
            )
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(scheduler.get_last_lr()[0])
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(self.save_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'history': self.history
                }, best_model_path)
                print(f"New best model saved with validation accuracy: {val_acc:.4f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'history': self.history
                }, checkpoint_path)
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
        return best_model_path
    
    def test_evaluation(self, model_path=None, criterion=None):
        """Final evaluation on test set"""
        
        if model_path and os.path.exists(model_path):
            # Load best model
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        
        if criterion is None:
            if hasattr(self.model, 'num_classes'):
                criterion = get_loss_function('crossentropy', num_classes=self.model.num_classes)
            else:
                criterion = get_loss_function('crossentropy', num_classes=9)
        
        # Evaluate on test set
        test_loss, test_acc, test_precision, test_recall, test_f1, preds, labels = self.evaluate(
            self.test_loader, criterion
        )
        
        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(labels, preds))
        
        # Save results
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'classification_report': classification_report(labels, preds, output_dict=True)
        }
        
        results_path = os.path.join(self.save_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(labels, preds)
        
        return results
    
    def plot_confusion_matrix(self, labels, preds):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plot_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {plot_path}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Learning rate plot
        axes[1, 0].plot(self.history['learning_rates'])
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        
        # Remove empty subplot
        axes[1, 1].remove()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved to {plot_path}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Moltbook models')
    parser.add_argument('--task', type=str, default='content', choices=['content', 'toxicity', 'multi'],
                        help='Task to train: content, toxicity, or multi')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='Pre-trained model name')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save models')
    parser.add_argument('--loss_type', type=str, default='crossentropy',
                        choices=['crossentropy', 'focal', 'label_smoothing'],
                        help='Loss function type')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading and preprocessing data...")
    data_loader = MoltbookDataLoader(model_name=args.model_name, max_length=args.max_length)
    
    # Load dataset
    dataset = data_loader.load_dataset('posts')
    df = data_loader.preprocess_data(dataset)
    
    # Prepare data for the specified task
    if args.task == 'multi':
        # For multi-task, use content classification data
        data_splits = data_loader.prepare_data_for_classification(df, task='content')
    else:
        data_splits = data_loader.prepare_data_for_classification(df, task=args.task)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(
        data_splits, batch_size=args.batch_size
    )
    
    # Create model
    print(f"Creating {args.task} model...")
    model = get_model(args.task, args.model_name)
    model.to(device)
    
    # Create trainer
    trainer = MoltbookTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        save_dir=args.save_dir
    )
    
    # Train model
    best_model_path = trainer.train(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        loss_type=args.loss_type
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Final evaluation
    results = trainer.test_evaluation(best_model_path)
    
    print(f"\nTraining completed successfully!")
    print(f"Best model saved at: {best_model_path}")
    print(f"Results saved in: {args.save_dir}")

if __name__ == "__main__":
    main()
