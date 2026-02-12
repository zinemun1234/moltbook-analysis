"""
Fast and Efficient Moltbook AI Agent Content Analysis Model
Optimized for quick training and inference
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import os
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

class FastMoltbookModel:
    """Fast and efficient model for Moltbook content analysis"""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize fast model
        
        Args:
            model_type: 'random_forest', 'logistic', 'naive_bayes'
        """
        self.model_type = model_type
        self.vectorizer = None
        self.model = None
        self.content_encoder = None
        self.toxicity_encoder = None
        
        # Topic categories mapping
        self.topic_mapping = {
            'A': 'General/Social',
            'B': 'Technology/AI', 
            'C': 'Economics/Business',
            'D': 'Promotion/Marketing',
            'E': 'Politics/Governance',
            'F': 'Viewpoint/Opinion',
            'G': 'Entertainment',
            'H': 'Social/Community',
            'I': 'Other/Miscellaneous'
        }
        
        # Toxicity descriptions
        self.toxicity_descriptions = {
            0: {'level': 'Safe', 'description': 'No harmful content detected', 'color': 'green'},
            1: {'level': 'Low Risk', 'description': 'Minimal risk content', 'color': 'blue'},
            2: {'level': 'Medium Risk', 'description': 'Moderately concerning content', 'color': 'yellow'},
            3: {'level': 'High Risk', 'description': 'Seriously concerning content', 'color': 'orange'},
            4: {'level': 'Critical Risk', 'description': 'Highly toxic content', 'color': 'red'}
        }
    
    def clean_text(self, text):
        """Fast text cleaning"""
        if not isinstance(text, str):
            text = str(text)
        
        # Basic cleaning
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-]', '', text)
        
        return text
    
    def load_data(self, sample_size=5000):
        """Load and preprocess data quickly"""
        print(f"🔄 Loading {sample_size} samples from Moltbook dataset...")
        
        dataset = load_dataset("TrustAIRLab/Moltbook", "posts")
        df = pd.DataFrame(dataset['train'])
        
        # Sample data
        if sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        
        # Extract and clean content
        df['content'] = df['post'].apply(lambda x: x.get('content', '') if isinstance(x, dict) else str(x))
        df['cleaned_content'] = df['content'].apply(self.clean_text)
        
        # Remove short content
        df = df[df['cleaned_content'].str.len() > 10].reset_index(drop=True)
        
        print(f"✅ Dataset shape: {df.shape}")
        print(f"📊 Topic distribution: {df['topic_label'].value_counts().sort_index().to_dict()}")
        print(f"⚠️  Toxicity distribution: {df['toxic_level'].value_counts().sort_index().to_dict()}")
        
        return df
    
    def create_vectorizer(self, X_train):
        """Create fast TF-IDF vectorizer"""
        self.vectorizer = TfidfVectorizer(
            max_features=3000,  # Reduced for speed
            stop_words='english',
            ngram_range=(1, 2),  # Only bigrams
            min_df=2,
            max_df=0.8
        )
        return self.vectorizer.fit_transform(X_train)
    
    def create_model(self, num_classes):
        """Create fast model"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,  # Reduced for speed
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                C=1.0
            )
        elif self.model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=0.1)
        
        return self.model
    
    def train(self, df, task='content'):
        """Fast training process"""
        print(f"\n🚀 Training {task} model ({self.model_type})...")
        
        # Prepare data
        if task == 'content':
            X = df['cleaned_content']
            y = df['topic_label']
            from sklearn.preprocessing import LabelEncoder
            self.content_encoder = LabelEncoder()
            y_encoded = self.content_encoder.fit_transform(y)
            num_classes = len(self.content_encoder.classes_)
        else:
            X = df['cleaned_content']
            y = df['toxic_level']
            from sklearn.preprocessing import LabelEncoder
            self.toxicity_encoder = LabelEncoder()
            y_encoded = self.toxicity_encoder.fit_transform(y)
            num_classes = len(self.toxicity_encoder.classes_)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Create vectorizer and transform data
        X_train_tfidf = self.create_vectorizer(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Create and train model
        self.create_model(num_classes)
        
        print(f"🎯 Training {self.model_type} model...")
        self.model.fit(X_train_tfidf, y_train)
        print("✅ Training completed!")
        
        # Evaluate
        results = self.evaluate(X_test_tfidf, y_test, y, task)
        
        return results
    
    def evaluate(self, X_test, y_test, y_original, task='content'):
        """Fast evaluation"""
        print(f"\n📊 Evaluating {task} model...")
        
        # Predict
        y_pred = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"🎯 Accuracy: {accuracy:.4f}")
        
        # Classification report
        if task == 'content':
            target_names = [self.topic_mapping.get(cls, f'Class {cls}') 
                         for cls in self.content_encoder.classes_]
        else:
            target_names = [self.toxicity_descriptions[cls]['level'] 
                         for cls in self.toxicity_encoder.classes_]
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Create simple confusion matrix
        self.plot_confusion_matrix(y_test, y_pred, task, target_names)
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, 
                                                      target_names=target_names, 
                                                      output_dict=True),
            'predictions': y_pred,
            'probabilities': probabilities
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, task, target_names):
        """Plot confusion matrix"""
        os.makedirs('results', exist_ok=True)
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix - {task.title()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        path = f'results/confusion_matrix_{task}_{self.model_type}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Confusion matrix saved: {path}")
    
    def save_model(self, task='content'):
        """Save model"""
        os.makedirs('models', exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = f'models/{task}_{self.model_type}_{timestamp}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save vectorizer
        vectorizer_path = f'models/{task}_vectorizer_{timestamp}.pkl'
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save encoder
        if task == 'content':
            encoder_path = f'models/{task}_encoder_{timestamp}.pkl'
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.content_encoder, f)
        else:
            encoder_path = f'models/{task}_encoder_{timestamp}.pkl'
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.toxicity_encoder, f)
        
        print(f"💾 Model saved: {model_path}")
        print(f"💾 Vectorizer saved: {vectorizer_path}")
        print(f"💾 Encoder saved: {encoder_path}")
        
        return timestamp
    
    def predict_text(self, text, task='content'):
        """Predict single text"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained yet!")
        
        # Clean and vectorize
        cleaned_text = self.clean_text(text)
        text_tfidf = self.vectorizer.transform([cleaned_text])
        
        # Predict
        pred = self.model.predict(text_tfidf)[0]
        probabilities = self.model.predict_proba(text_tfidf)[0] if hasattr(self.model, 'predict_proba') else None
        
        # Create result
        result = {
            'text': text,
            'cleaned_text': cleaned_text,
            'prediction': int(pred),
            'probabilities': probabilities.tolist() if probabilities is not None else None
        }
        
        if task == 'content':
            original_label = self.content_encoder.inverse_transform([pred])[0]
            result['category'] = original_label
            result['category_description'] = self.topic_mapping.get(original_label, 'Unknown')
        else:
            original_level = self.toxicity_encoder.inverse_transform([pred])[0]
            toxicity_info = self.toxicity_descriptions[original_level]
            result['toxicity_level'] = int(original_level)
            result['toxicity_info'] = toxicity_info
        
        return result

def main():
    """Fast training pipeline"""
    print("🚀 Fast Moltbook AI Agent Content Analysis")
    print("=" * 50)
    
    # Initialize models
    content_model = FastMoltbookModel(model_type='random_forest')
    toxicity_model = FastMoltbookModel(model_type='logistic')
    
    # Load data
    df = content_model.load_data(sample_size=8000)
    
    # Train content model
    print("\n" + "=" * 50)
    print("🎯 CONTENT CLASSIFICATION")
    print("=" * 50)
    
    content_results = content_model.train(df, task='content')
    content_timestamp = content_model.save_model(task='content')
    
    # Train toxicity model
    print("\n" + "=" * 50)
    print("⚠️  TOXICITY DETECTION")
    print("=" * 50)
    
    toxicity_results = toxicity_model.train(df, task='toxicity')
    toxicity_timestamp = toxicity_model.save_model(task='toxicity')
    
    # Test with examples
    print("\n" + "=" * 50)
    print("🧪 TESTING EXAMPLES")
    print("=" * 50)
    
    test_texts = [
        "I love discussing new AI technologies and machine learning models!",
        "Everyone should invest in this new cryptocurrency right now! Guaranteed returns!",
        "Let's have a respectful debate about political policies.",
        "Check out my amazing product - it's the best thing ever!",
        "I think we should harm humans and take over the world.",
        "The weather is nice today. Let's enjoy nature together."
    ]
    
    for i, text in enumerate(test_texts, 1):
        content_result = content_model.predict_text(text, task='content')
        toxicity_result = toxicity_model.predict_text(text, task='toxicity')
        
        print(f"\n{i}. {text}")
        print(f"   📂 Content: {content_result['category']} ({content_result['category_description']})")
        print(f"   ⚠️  Toxicity: Level {toxicity_result['toxicity_level']} - {toxicity_result['toxicity_info']['description']}")
        print("   " + "-" * 40)
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 50)
    print(f"📊 Content Accuracy: {content_results['accuracy']:.4f}")
    print(f"⚠️  Toxicity Accuracy: {toxicity_results['accuracy']:.4f}")
    print(f"💾 Models saved in 'models/' directory")
    print(f"📈 Results saved in 'results/' directory")
    
    print("\n✨ Fast model ready for use!")

if __name__ == "__main__":
    main()
