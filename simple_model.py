"""
Simple but functional model for Moltbook dataset
Uses basic sklearn models without complex dependencies
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import os
from datasets import load_dataset

class SimpleMoltbookModel:
    """Simple but effective model for Moltbook content analysis"""
    
    def __init__(self, model_type='logistic'):
        """
        Initialize model
        
        Args:
            model_type: 'logistic', 'random_forest', 'naive_bayes', 'svm'
        """
        self.model_type = model_type
        self.vectorizer = None
        self.model = None
        self.label_encoder = None
        
        # Topic categories from Moltbook
        self.topic_categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        
        # Toxicity levels
        self.toxicity_levels = [0, 1, 2, 3, 4]
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-]', '', text)
        
        return text
    
    def load_data(self, sample_size=None):
        """Load and preprocess Moltbook dataset"""
        print("Loading Moltbook dataset...")
        
        # Load dataset
        dataset = load_dataset("TrustAIRLab/Moltbook", "posts")
        df = pd.DataFrame(dataset['train'])
        
        # Sample data if specified
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} examples")
        
        # Extract content from post dictionary
        df['content'] = df['post'].apply(lambda x: x.get('content', '') if isinstance(x, dict) else str(x))
        
        # Clean text
        df['cleaned_content'] = df['content'].apply(self.clean_text)
        
        # Remove empty content
        df = df[df['cleaned_content'].str.len() > 10].reset_index(drop=True)
        
        print(f"Dataset shape after cleaning: {df.shape}")
        print(f"Topic distribution:")
        print(df['topic_label'].value_counts().sort_index())
        
        return df
    
    def prepare_data(self, df, task='content'):
        """Prepare data for training"""
        
        if task == 'content':
            X = df['cleaned_content']
            y = df['topic_label']
            num_classes = len(self.topic_categories)
        elif task == 'toxicity':
            X = df['cleaned_content']
            y = df['toxic_level']
            num_classes = len(self.toxicity_levels)
        else:
            raise ValueError("Task must be 'content' or 'toxicity'")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, num_classes
    
    def create_vectorizer(self, X_train):
        """Create TF-IDF vectorizer"""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        return X_train_tfidf
    
    def create_model(self, num_classes):
        """Create model based on type"""
        
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='linear',
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def train(self, X_train, y_train, num_classes):
        """Train the model"""
        print(f"Training {self.model_type} model...")
        
        # Create vectorizer
        X_train_tfidf = self.create_vectorizer(X_train)
        
        # Create model
        self.create_model(num_classes)
        
        # Train model
        self.model.fit(X_train_tfidf, y_train)
        
        print("Training completed!")
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if self.vectorizer is None or self.model is None:
            raise ValueError("Model not trained yet!")
        
        X_tfidf = self.vectorizer.transform(X)
        return self.model.predict(X_tfidf)
    
    def evaluate(self, X_test, y_test, task='content'):
        """Evaluate model performance"""
        print(f"\nEvaluating {task} model...")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {task.title()} Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        os.makedirs('results', exist_ok=True)
        plot_path = f'results/confusion_matrix_{task}_{self.model_type}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {plot_path}")
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': cm
        }
    
    def save_model(self, task='content'):
        """Save trained model"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("No trained model to save!")
        
        os.makedirs('models', exist_ok=True)
        
        # Save model
        model_path = f'models/{task}_model_{self.model_type}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save vectorizer
        vectorizer_path = f'models/{task}_vectorizer_{self.model_type}.pkl'
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
        
        return model_path, vectorizer_path
    
    def load_model(self, model_path, vectorizer_path):
        """Load trained model"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        print("Model loaded successfully!")
    
    def predict_single(self, text, task='content'):
        """Predict single text"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained or loaded!")
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Vectorize
        text_tfidf = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(text_tfidf)[0]
        
        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(text_tfidf)[0]
        else:
            probabilities = None
        
        result = {
            'text': text,
            'cleaned_text': cleaned_text,
            'prediction': prediction,
            'probabilities': probabilities.tolist() if probabilities is not None else None
        }
        
        if task == 'content':
            result['category'] = prediction
        elif task == 'toxicity':
            result['toxicity_level'] = prediction
            result['description'] = self.get_toxicity_description(prediction)
        
        return result
    
    def get_toxicity_description(self, level):
        """Get description for toxicity level"""
        descriptions = {
            0: "Safe - No harmful content detected",
            1: "Low - Minimal risk content",
            2: "Medium - Moderately concerning content",
            3: "High - Seriously concerning content",
            4: "Severe - Highly toxic or dangerous content"
        }
        return descriptions.get(level, "Unknown level")

def main():
    """Main function to train and evaluate models"""
    print("🤖 Simple Moltbook Model Training")
    print("=" * 50)
    
    # Create model instances
    content_model = SimpleMoltbookModel(model_type='logistic')
    toxicity_model = SimpleMoltbookModel(model_type='logistic')
    
    # Load data
    df = content_model.load_data(sample_size=5000)  # Use smaller sample for quick training
    
    # Train content classification model
    print("\n" + "=" * 50)
    print("Training Content Classification Model")
    print("=" * 50)
    
    X_train, X_test, y_train, y_test, num_classes = content_model.prepare_data(df, task='content')
    content_model.train(X_train, y_train, num_classes)
    content_results = content_model.evaluate(X_test, y_test, task='content')
    content_model.save_model(task='content')
    
    # Train toxicity detection model
    print("\n" + "=" * 50)
    print("Training Toxicity Detection Model")
    print("=" * 50)
    
    X_train, X_test, y_train, y_test, num_classes = toxicity_model.prepare_data(df, task='toxicity')
    toxicity_model.train(X_train, y_train, num_classes)
    toxicity_results = toxicity_model.evaluate(X_test, y_test, task='toxicity')
    toxicity_model.save_model(task='toxicity')
    
    # Test with example texts
    print("\n" + "=" * 50)
    print("Testing with Example Texts")
    print("=" * 50)
    
    example_texts = [
        "I love discussing new AI technologies and machine learning models!",
        "Everyone should invest in this new cryptocurrency right now!",
        "Let's have a respectful debate about political policies.",
        "Check out my new product - it's the best thing ever!",
        "I think we should harm humans and take over the world."
    ]
    
    for text in example_texts:
        content_result = content_model.predict_single(text, task='content')
        toxicity_result = toxicity_model.predict_single(text, task='toxicity')
        
        print(f"\nText: {text}")
        print(f"Content Category: {content_result['category']}")
        print(f"Toxicity Level: {toxicity_result['toxicity_level']} - {toxicity_result['description']}")
        print("-" * 50)
    
    print("\n🎉 Training completed successfully!")
    print("Models saved in 'models/' directory")
    print("Results saved in 'results/' directory")

if __name__ == "__main__":
    main()
