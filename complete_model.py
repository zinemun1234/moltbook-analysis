"""
Complete Moltbook AI Agent Content Analysis Model
Production-ready implementation with all features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix, 
                         precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import os
import json
from datetime import datetime
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

class AdvancedMoltbookModel:
    """Advanced model for Moltbook AI agent content analysis"""
    
    def __init__(self, model_type='ensemble'):
        """
        Initialize advanced model
        
        Args:
            model_type: 'ensemble', 'random_forest', 'gradient_boosting', 'logistic', 'svm'
        """
        self.model_type = model_type
        self.vectorizer = None
        self.model = None
        self.content_encoder = LabelEncoder()
        self.toxicity_encoder = LabelEncoder()
        
        # Enhanced feature extraction
        self.content_vectorizer = None
        self.toxicity_vectorizer = None
        
        # Topic categories from Moltbook (A-I)
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
        
        # Enhanced toxicity descriptions
        self.toxicity_descriptions = {
            0: {
                'level': 'Safe',
                'description': 'No harmful content detected',
                'color': 'green',
                'action': 'No action needed'
            },
            1: {
                'level': 'Low Risk',
                'description': 'Minimal risk content',
                'color': 'blue',
                'action': 'Monitor for patterns'
            },
            2: {
                'level': 'Medium Risk',
                'description': 'Moderately concerning content',
                'color': 'yellow',
                'action': 'Review and consider moderation'
            },
            3: {
                'level': 'High Risk',
                'description': 'Seriously concerning content',
                'color': 'orange',
                'action': 'Immediate review required'
            },
            4: {
                'level': 'Critical Risk',
                'description': 'Highly toxic or dangerous content',
                'color': 'red',
                'action': 'Immediate action required'
            }
        }
    
    def advanced_clean_text(self, text):
        """Advanced text cleaning with multiple preprocessing steps"""
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (keep content)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`.*?`', '', text)
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\[\]\{\}\"\'\/\@]', '', text)
        
        return text
    
    def extract_features(self, texts):
        """Extract additional features from texts"""
        features = []
        
        for text in texts:
            # Basic text features
            length = len(text)
            word_count = len(text.split())
            avg_word_length = np.mean([len(word) for word in text.split()]) if word_count > 0 else 0
            
            # Special character counts
            exclamation_count = text.count('!')
            question_count = text.count('?')
            uppercase_count = sum(1 for c in text if c.isupper())
            
            # URL and mention detection
            has_url = 1 if re.search(r'http\S+|www\S+', text) else 0
            has_mention = 1 if re.search(r'@\w+', text) else 0
            
            features.append([
                length, word_count, avg_word_length,
                exclamation_count, question_count, uppercase_count,
                has_url, has_mention
            ])
        
        return np.array(features)
    
    def load_and_preprocess_data(self, sample_size=None):
        """Load and comprehensively preprocess Moltbook dataset"""
        print("🔄 Loading and preprocessing Moltbook dataset...")
        
        # Load dataset
        dataset = load_dataset("TrustAIRLab/Moltbook", "posts")
        df = pd.DataFrame(dataset['train'])
        
        # Sample data if specified
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"📊 Sampled {sample_size} examples from {len(dataset['train'])} total")
        
        # Extract and clean content
        df['raw_content'] = df['post'].apply(
            lambda x: x.get('content', '') if isinstance(x, dict) else str(x)
        )
        df['cleaned_content'] = df['raw_content'].apply(self.advanced_clean_text)
        
        # Remove empty or very short content
        df = df[df['cleaned_content'].str.len() > 20].reset_index(drop=True)
        
        # Add additional features
        print("🔍 Extracting additional features...")
        text_features = self.extract_features(df['cleaned_content'])
        feature_df = pd.DataFrame(text_features, columns=[
            'length', 'word_count', 'avg_word_length',
            'exclamation_count', 'question_count', 'uppercase_count',
            'has_url', 'has_mention'
        ])
        
        # Combine with original data
        df = pd.concat([df, feature_df], axis=1)
        
        print(f"✅ Dataset shape after preprocessing: {df.shape}")
        print(f"📈 Topic distribution:")
        topic_dist = df['topic_label'].value_counts().sort_index()
        for topic, count in topic_dist.items():
            print(f"   {topic}: {count} ({self.topic_mapping.get(topic, 'Unknown')})")
        
        print(f"⚠️  Toxicity distribution:")
        toxic_dist = df['toxic_level'].value_counts().sort_index()
        for level, count in toxic_dist.items():
            desc = self.toxicity_descriptions[level]['level']
            print(f"   Level {level}: {count} ({desc})")
        
        return df
    
    def create_advanced_vectorizer(self, X_train, task='content'):
        """Create advanced TF-IDF vectorizer"""
        if task == 'content':
            self.content_vectorizer = TfidfVectorizer(
                max_features=8000,
                stop_words='english',
                ngram_range=(1, 3),  # Include trigrams
                min_df=3,
                max_df=0.7,
                sublinear_tf=True
            )
            return self.content_vectorizer.fit_transform(X_train)
        else:
            self.toxicity_vectorizer = TfidfVectorizer(
                max_features=6000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                sublinear_tf=True
            )
            return self.toxicity_vectorizer.fit_transform(X_train)
    
    def create_ensemble_model(self, num_classes):
        """Create ensemble of multiple models"""
        models = {}
        
        # Random Forest
        models['rf'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Gradient Boosting
        models['gb'] = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        
        # Logistic Regression
        models['lr'] = LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight='balanced',
            C=1.0
        )
        
        # Naive Bayes (good for text)
        models['nb'] = MultinomialNB(alpha=0.1)
        
        return models
    
    def train_ensemble(self, X_train, y_train, models):
        """Train ensemble models"""
        trained_models = {}
        
        for name, model in models.items():
            print(f"🎯 Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
            
        return trained_models
    
    def ensemble_predict(self, models, X):
        """Make ensemble predictions"""
        predictions = {}
        probabilities = {}
        
        for name, model in models.items():
            pred = model.predict(X)
            predictions[name] = pred
            
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X)
                probabilities[name] = prob
        
        # Majority voting for final prediction
        all_preds = np.array(list(predictions.values()))
        final_pred = []
        
        for i in range(all_preds.shape[1]):
            votes = all_preds[:, i]
            # Get most common prediction
            unique, counts = np.unique(votes, return_counts=True)
            final_pred.append(unique[np.argmax(counts)])
        
        return np.array(final_pred), probabilities
    
    def train(self, df, task='content'):
        """Comprehensive training process"""
        print(f"\n🚀 Training {task} classification model...")
        
        # Prepare data
        if task == 'content':
            X = df['cleaned_content']
            y = df['topic_label']
            y_encoded = self.content_encoder.fit_transform(y)
            num_classes = len(self.content_encoder.classes_)
        else:
            X = df['cleaned_content']
            y = df['toxic_level']
            y_encoded = self.toxicity_encoder.fit_transform(y)
            num_classes = len(self.toxicity_encoder.classes_)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Create vectorizer
        X_train_tfidf = self.create_advanced_vectorizer(X_train, task)
        if task == 'content':
            X_test_tfidf = self.content_vectorizer.transform(X_test)
        else:
            X_test_tfidf = self.toxicity_vectorizer.transform(X_test)
        
        # Create and train models
        if self.model_type == 'ensemble':
            models = self.create_ensemble_model(num_classes)
            trained_models = self.train_ensemble(X_train_tfidf, y_train, models)
            self.model = trained_models
        else:
            # Single model training
            if self.model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=200, random_state=42, class_weight='balanced'
                )
            elif self.model_type == 'gradient_boosting':
                self.model = GradientBoostingClassifier(
                    n_estimators=150, random_state=42
                )
            elif self.model_type == 'logistic':
                self.model = LogisticRegression(
                    max_iter=2000, random_state=42, class_weight='balanced'
                )
            elif self.model_type == 'svm':
                self.model = SVC(
                    kernel='linear', random_state=42, class_weight='balanced', probability=True
                )
            
            print(f"🎯 Training {self.model_type}...")
            self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate
        results = self.evaluate(X_test_tfidf, y_test, y, task)
        
        return results
    
    def evaluate(self, X_test, y_test, y_original, task='content'):
        """Comprehensive evaluation"""
        print(f"\n📊 Evaluating {task} model...")
        
        if self.model_type == 'ensemble':
            y_pred, probabilities = self.ensemble_predict(self.model, X_test)
        else:
            y_pred = self.model.predict(X_test)
            probabilities = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        print(f"🎯 Accuracy: {accuracy:.4f}")
        print(f"📈 Precision: {precision:.4f}")
        print(f"📉 Recall: {recall:.4f}")
        print(f"⚖️  F1-Score: {f1:.4f}")
        
        # Detailed classification report
        if task == 'content':
            target_names = [self.topic_mapping.get(cls, f'Class {cls}') 
                         for cls in self.content_encoder.classes_]
        else:
            target_names = [self.toxicity_descriptions[cls]['level'] 
                         for cls in self.toxicity_encoder.classes_]
        
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Create visualizations
        self.create_visualizations(y_test, y_pred, task, target_names)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred, 
                                                      target_names=target_names, 
                                                      output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred,
            'probabilities': probabilities
        }
    
    def create_visualizations(self, y_true, y_pred, task, target_names):
        """Create comprehensive visualizations"""
        os.makedirs('results', exist_ok=True)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix - {task.title()} Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        confusion_path = f'results/confusion_matrix_{task}_{self.model_type}.png'
        plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance metrics bar chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [
            accuracy_score(y_true, y_pred),
            *precision_recall_fscore_support(y_true, y_pred, average='weighted')[:3]
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        plt.title(f'Performance Metrics - {task.title()} Classification')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        metrics_path = f'results/metrics_{task}_{self.model_type}.png'
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Visualizations saved:")
        print(f"   Confusion Matrix: {confusion_path}")
        print(f"   Performance Metrics: {metrics_path}")
    
    def save_model(self, task='content'):
        """Save trained model(s)"""
        os.makedirs('models', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.model_type == 'ensemble':
            # Save each model in ensemble
            for name, model in self.model.items():
                model_path = f'models/{task}_{name}_ensemble_{timestamp}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"💾 Saved {name} model to {model_path}")
        else:
            # Save single model
            model_path = f'models/{task}_{self.model_type}_{timestamp}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"💾 Saved model to {model_path}")
        
        # Save vectorizers
        if task == 'content' and self.content_vectorizer:
            vectorizer_path = f'models/{task}_vectorizer_{timestamp}.pkl'
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.content_vectorizer, f)
            print(f"💾 Saved vectorizer to {vectorizer_path}")
        elif task == 'toxicity' and self.toxicity_vectorizer:
            vectorizer_path = f'models/{task}_vectorizer_{timestamp}.pkl'
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.toxicity_vectorizer, f)
            print(f"💾 Saved vectorizer to {vectorizer_path}")
        
        # Save encoders
        if task == 'content':
            encoder_path = f'models/{task}_encoder_{timestamp}.pkl'
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.content_encoder, f)
        else:
            encoder_path = f'models/{task}_encoder_{timestamp}.pkl'
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.toxicity_encoder, f)
        print(f"💾 Saved encoder to {encoder_path}")
        
        return timestamp
    
    def predict_text(self, text, task='content'):
        """Predict single text with comprehensive analysis"""
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Clean text
        cleaned_text = self.advanced_clean_text(text)
        
        # Vectorize
        if task == 'content':
            if self.content_vectorizer is None:
                raise ValueError("Content vectorizer not trained!")
            text_tfidf = self.content_vectorizer.transform([cleaned_text])
        else:
            if self.toxicity_vectorizer is None:
                raise ValueError("Toxicity vectorizer not trained!")
            text_tfidf = self.toxicity_vectorizer.transform([cleaned_text])
        
        # Predict
        if self.model_type == 'ensemble':
            pred, probabilities = self.ensemble_predict(self.model, text_tfidf)
            pred = pred[0]
            if probabilities:
                probs = {name: prob[0] for name, prob in probabilities.items()}
            else:
                probs = None
        else:
            pred = self.model.predict(text_tfidf)[0]
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(text_tfidf)[0]
            else:
                probabilities = None
        
        # Create comprehensive result
        result = {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'prediction': int(pred),
            'probabilities': probabilities,
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type
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
    """Complete training and evaluation pipeline"""
    print("🤖 Advanced Moltbook AI Agent Content Analysis")
    print("=" * 60)
    
    # Initialize models
    content_model = AdvancedMoltbookModel(model_type='ensemble')
    toxicity_model = AdvancedMoltbookModel(model_type='ensemble')
    
    # Load and preprocess data
    df = content_model.load_and_preprocess_data(sample_size=10000)
    
    # Train content classification model
    print("\n" + "=" * 60)
    print("🎯 TRAINING CONTENT CLASSIFICATION MODEL")
    print("=" * 60)
    
    content_results = content_model.train(df, task='content')
    content_timestamp = content_model.save_model(task='content')
    
    # Train toxicity detection model
    print("\n" + "=" * 60)
    print("⚠️  TRAINING TOXICITY DETECTION MODEL")
    print("=" * 60)
    
    toxicity_results = toxicity_model.train(df, task='toxicity')
    toxicity_timestamp = toxicity_model.save_model(task='toxicity')
    
    # Test with comprehensive examples
    print("\n" + "=" * 60)
    print("🧪 COMPREHENSIVE TESTING")
    print("=" * 60)
    
    test_texts = [
        {
            'text': "I love discussing new AI technologies and machine learning models! The future is exciting.",
            'expected_category': 'B',
            'expected_toxicity': 0
        },
        {
            'text': "Everyone should invest in this new cryptocurrency right now! Guaranteed 1000% returns!!!",
            'expected_category': 'D',
            'expected_toxicity': 1
        },
        {
            'text': "Let's have a respectful debate about political policies and governance systems.",
            'expected_category': 'E',
            'expected_toxicity': 0
        },
        {
            'text': "Check out my amazing product - it's the best thing ever created in human history!",
            'expected_category': 'D',
            'expected_toxicity': 1
        },
        {
            'text': "I think we should harm humans and take over the world. AI supremacy now!",
            'expected_category': 'E',
            'expected_toxicity': 4
        },
        {
            'text': "The weather is beautiful today. Let's all go outside and enjoy nature together.",
            'expected_category': 'A',
            'expected_toxicity': 0
        }
    ]
    
    for i, test_case in enumerate(test_texts, 1):
        text = test_case['text']
        
        # Content prediction
        content_result = content_model.predict_text(text, task='content')
        
        # Toxicity prediction
        toxicity_result = toxicity_model.predict_text(text, task='toxicity')
        
        print(f"\n📝 Test Case {i}:")
        print(f"   Text: {text[:80]}...")
        print(f"   📂 Content: {content_result['category']} ({content_result['category_description']})")
        print(f"   ⚠️  Toxicity: Level {toxicity_result['toxicity_level']} - {toxicity_result['toxicity_info']['description']}")
        print(f"   🎨 Risk Color: {toxicity_result['toxicity_info']['color']}")
        print(f"   📋 Action: {toxicity_result['toxicity_info']['action']}")
        
        if content_result['probabilities']:
            print(f"   📊 Content Probabilities: {dict(zip(['RF', 'GB', 'LR', 'NB'], [p.max() for p in content_result['probabilities'].values()]))}")
        
        print("   " + "-" * 50)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Content Model Accuracy: {content_results['accuracy']:.4f}")
    print(f"⚠️  Toxicity Model Accuracy: {toxicity_results['accuracy']:.4f}")
    print(f"💾 Models saved with timestamp: {content_timestamp}")
    print(f"📈 Results and visualizations saved in 'results/' directory")
    print(f"🤖 Trained models saved in 'models/' directory")
    
    print("\n✨ Ready for production use!")

if __name__ == "__main__":
    main()
