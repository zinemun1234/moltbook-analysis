"""
Data loader and preprocessor for Moltbook dataset
Handles loading, cleaning, and preprocessing of the dataset
"""

from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import re
import warnings
warnings.filterwarnings('ignore')

class MoltbookDataset(Dataset):
    """Custom dataset class for Moltbook data"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class MoltbookDataLoader:
    """Main data loader class for Moltbook dataset"""
    
    def __init__(self, model_name='bert-base-uncased', max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Content categories (9 categories from the paper)
        self.content_categories = [
            'General', 'Technology', 'Viewpoint', 'Economics', 
            'Promotion', 'Politics', 'Social', 'Entertainment', 'Other'
        ]
        
        # Toxicity levels (5 levels from the paper)
        self.toxicity_levels = [0, 1, 2, 3, 4]  # 0=Safe, 4=Highly Toxic
        
        self.content_encoder = LabelEncoder()
        self.toxicity_encoder = LabelEncoder()
        
    def load_dataset(self, config_name='posts'):
        """Load the Moltbook dataset from Hugging Face"""
        print(f"Loading Moltbook dataset with config '{config_name}'...")
        dataset = load_dataset("TrustAIRLab/Moltbook", config_name)
        return dataset
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (keep the text after #)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-]', '', text)
        
        return text
    
    def preprocess_data(self, dataset):
        """Preprocess the dataset"""
        print("Preprocessing data...")
        
        # Convert to pandas for easier manipulation
        if isinstance(dataset, dict):
            df = pd.DataFrame(dataset['train'])
        else:
            df = dataset.to_pandas()
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Extract content from post dictionary
        if 'post' in df.columns:
            # Handle the case where post is a dictionary with 'content' key
            df['raw_content'] = df['post'].apply(lambda x: x.get('content', '') if isinstance(x, dict) else str(x))
        else:
            # Find the text column
            text_cols = [col for col in df.columns if df[col].dtype == 'object']
            if text_cols:
                df['raw_content'] = df[text_cols[0]].astype(str)
            else:
                raise ValueError("No text column found in dataset")
        
        # Clean text data
        df['cleaned_text'] = df['raw_content'].apply(self.clean_text)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
        
        print(f"After cleaning: {df.shape}")
        return df
    
    def prepare_data_for_classification(self, df, task='content'):
        """Prepare data for either content classification or toxicity detection"""
        
        if task == 'content':
            # For content classification
            if 'topic_label' in df.columns:
                labels = df['topic_label'].values
            elif 'category' in df.columns:
                labels = df['category'].values
            elif 'content_category' in df.columns:
                labels = df['content_category'].values
            else:
                # Create dummy labels for demonstration
                print("Warning: No category column found, creating dummy labels")
                labels = np.random.choice(self.content_categories, len(df))
            
            # Encode labels
            encoded_labels = self.content_encoder.fit_transform(labels)
            num_classes = len(self.content_encoder.classes_)
            
        elif task == 'toxicity':
            # For toxicity detection
            if 'toxic_level' in df.columns:
                labels = df['toxic_level'].values
            elif 'toxicity' in df.columns:
                labels = df['toxicity'].values
            elif 'toxicity_level' in df.columns:
                labels = df['toxicity_level'].values
            else:
                # Create dummy labels for demonstration
                print("Warning: No toxicity column found, creating dummy labels")
                labels = np.random.choice(self.toxicity_levels, len(df))
            
            # Encode labels
            encoded_labels = self.toxicity_encoder.fit_transform(labels)
            num_classes = len(self.toxicity_encoder.classes_)
        
        else:
            raise ValueError("Task must be either 'content' or 'toxicity'")
        
        texts = df['cleaned_text'].values
        
        # Split data
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, encoded_labels, test_size=0.3, random_state=42, stratify=encoded_labels
        )
        
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        return {
            'train': (train_texts, train_labels),
            'val': (val_texts, val_labels),
            'test': (test_texts, test_labels),
            'num_classes': num_classes
        }
    
    def create_dataloaders(self, data_splits, batch_size=16):
        """Create PyTorch dataloaders"""
        train_texts, train_labels = data_splits['train']
        val_texts, val_labels = data_splits['val']
        test_texts, test_labels = data_splits['test']
        
        # Create datasets
        train_dataset = MoltbookDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = MoltbookDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        test_dataset = MoltbookDataset(test_texts, test_labels, self.tokenizer, self.max_length)
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def get_label_info(self):
        """Get label encoding information"""
        return {
            'content_labels': dict(zip(self.content_encoder.classes_, self.content_encoder.transform(self.content_encoder.classes_))),
            'toxicity_labels': dict(zip(self.toxicity_encoder.classes_, self.toxicity_encoder.transform(self.toxicity_encoder.classes_)))
        }

# Example usage
if __name__ == "__main__":
    loader = MoltbookDataLoader()
    
    # Load dataset
    dataset = loader.load_dataset()
    
    # Preprocess
    df = loader.preprocess_data(dataset)
    
    # Prepare for content classification
    content_data = loader.prepare_data_for_classification(df, task='content')
    
    # Create dataloaders
    train_loader, val_loader, test_loader = loader.create_dataloaders(content_data)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Number of classes: {content_data['num_classes']}")
