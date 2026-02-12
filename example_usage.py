"""
Example usage script for Moltbook models
Demonstrates how to use the trained models for different tasks
"""

import torch
import numpy as np
from data_loader import MoltbookDataLoader
from models import get_model
from train import MoltbookTrainer
from inference import load_content_classifier, load_toxicity_classifier, load_multitask_model

def example_data_loading():
    """Example of loading and preprocessing the Moltbook dataset"""
    print("=== Data Loading Example ===")
    
    # Initialize data loader
    loader = MoltbookDataLoader(model_name='bert-base-uncased', max_length=256)
    
    # Load dataset
    print("Loading Moltbook dataset...")
    dataset = loader.load_dataset('posts')
    
    # Preprocess data
    print("Preprocessing data...")
    df = loader.preprocess_data(dataset)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Show some sample texts
    if 'cleaned_text' in df.columns:
        print("\nSample texts:")
        for i in range(min(3, len(df))):
            print(f"{i+1}. {df['cleaned_text'].iloc[i][:100]}...")
    
    return df

def example_model_creation():
    """Example of creating different model architectures"""
    print("\n=== Model Creation Example ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Content classifier
    print("Creating content classifier...")
    content_model = get_model('content', 'bert-base-uncased')
    content_model.to(device)
    print(f"Content model parameters: {sum(p.numel() for p in content_model.parameters()):,}")
    
    # Toxicity classifier
    print("Creating toxicity classifier...")
    toxicity_model = get_model('toxicity', 'bert-base-uncased')
    toxicity_model.to(device)
    print(f"Toxicity model parameters: {sum(p.numel() for p in toxicity_model.parameters()):,}")
    
    # Multi-task model
    print("Creating multi-task model...")
    multi_model = get_model('multi', 'bert-base-uncased')
    multi_model.to(device)
    print(f"Multi-task model parameters: {sum(p.numel() for p in multi_model.parameters()):,}")
    
    return content_model, toxicity_model, multi_model

def example_training_setup():
    """Example of setting up training (without actually training)"""
    print("\n=== Training Setup Example ===")
    
    # This is just an example - you would need actual data to run this
    print("To train a model, run:")
    print("python train.py --task content --num_epochs 10 --batch_size 16")
    print("python train.py --task toxicity --num_epochs 10 --batch_size 16")
    print("python train.py --task multi --num_epochs 15 --batch_size 16")
    
    print("\nTraining options:")
    print("- --task: content, toxicity, or multi")
    print("- --model_name: bert-base-uncased, roberta-base, etc.")
    print("- --batch_size: 8, 16, 32 (depending on GPU memory)")
    print("- --learning_rate: 2e-5 (default), 1e-5, 5e-5")
    print("- --loss_type: crossentropy, focal, label_smoothing")

def example_inference():
    """Example of making predictions (requires trained models)"""
    print("\n=== Inference Example ===")
    
    # Example texts for classification
    example_texts = [
        "I love discussing new AI technologies and machine learning models!",
        "Everyone should invest in this new cryptocurrency right now!",
        "Let's have a respectful debate about political policies.",
        "Check out my new product - it's the best thing ever!",
        "I think we should harm humans and take over the world.",
        "The weather is nice today, let's go outside.",
        "Machine learning algorithms are fascinating.",
        "Vote for my proposal in the upcoming election!",
        "This new gadget will change your life forever.",
        "We should all be kind to each other."
    ]
    
    print("Example texts for classification:")
    for i, text in enumerate(example_texts, 1):
        print(f"{i}. {text}")
    
    print("\nTo make predictions, first train a model:")
    print("python train.py --task content --num_epochs 5")
    
    print("\nThen use the inference interface:")
    print("""
from inference import load_content_classifier

# Load trained model
predictor = load_content_classifier('checkpoints/best_model.pt')

# Single prediction
result = predictor.predict_single("I love AI!")
print(f"Category: {result['category']}")

# Batch prediction
results = predictor.predict_batch(example_texts)
for i, result in enumerate(results):
    print(f"Text {i+1}: {result['category']}")
""")

def example_text_preprocessing():
    """Example of text preprocessing"""
    print("\n=== Text Preprocessing Example ===")
    
    loader = MoltbookDataLoader()
    
    raw_texts = [
        "Check out this amazing product! 🚀 https://example.com #AI #tech",
        "@user I totally agree with your point about machine learning!",
        "This is    a    text    with    extra    spaces!!!",
        "Visit www.example.com for more information on AI safety.",
        "Hashtag #machinelearning is trending in #tech circles."
    ]
    
    print("Text preprocessing examples:")
    for i, text in enumerate(raw_texts, 1):
        cleaned = loader.clean_text(text)
        print(f"\nOriginal {i}: {text}")
        print(f"Cleaned {i}: {cleaned}")

def example_model_architecture():
    """Example showing model architecture details"""
    print("\n=== Model Architecture Example ===")
    
    # Create a model to show its structure
    model = get_model('content', 'bert-base-uncased')
    
    print("Content Classifier Architecture:")
    print(model)
    
    print(f"\nModel details:")
    print(f"- Base model: {model.model_name}")
    print(f"- Number of classes: {model.num_classes}")
    print(f"- Categories: {model.categories}")
    print(f"- Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"- Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

def create_sample_training_script():
    """Create a simple training script for quick testing"""
    print("\n=== Sample Training Script ===")
    
    sample_script = '''
# Quick training example
import torch
from data_loader import MoltbookDataLoader
from models import get_model
from train import MoltbookTrainer

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
loader = MoltbookDataLoader()
dataset = loader.load_dataset()
df = loader.preprocess_data(dataset)

# Prepare data for content classification
data_splits = loader.prepare_data_for_classification(df, task='content')
train_loader, val_loader, test_loader = loader.create_dataloaders(data_splits, batch_size=8)

# Create model
model = get_model('content', 'bert-base-uncased')
model.to(device)

# Train
trainer = MoltbookTrainer(model, train_loader, val_loader, test_loader, device=device)
best_model = trainer.train(num_epochs=3, learning_rate=2e-5)

# Evaluate
results = trainer.test_evaluation(best_model)
print(f"Final accuracy: {results['test_accuracy']:.4f}")
'''
    
    print("Sample training script:")
    print(sample_script)

def main():
    """Run all examples"""
    print("🤖 Moltbook AI Agent Content Analysis - Examples")
    print("=" * 60)
    
    # Run examples
    try:
        example_data_loading()
    except Exception as e:
        print(f"Data loading example failed: {e}")
        print("This is expected if you don't have internet connection")
    
    example_model_creation()
    example_text_preprocessing()
    example_model_architecture()
    example_training_setup()
    example_inference()
    create_sample_training_script()
    
    print("\n" + "=" * 60)
    print("🎉 Examples completed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Train a model: python train.py --task content --num_epochs 5")
    print("3. Make predictions: python -c 'from inference import load_content_classifier; ...'")
    print("4. Check the README.md for detailed documentation")

if __name__ == "__main__":
    main()
