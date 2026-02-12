"""
Production-Ready Moltbook AI Agent Content Analysis System
Complete implementation with robust error handling and professional features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix, 
                         precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import os
import json
import logging
from datetime import datetime
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionMoltbookSystem:
    """Production-ready system for Moltbook AI agent content analysis"""
    
    def __init__(self, config=None):
        """
        Initialize production system
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or self._get_default_config()
        
        # Model components
        self.content_vectorizer = None
        self.toxicity_vectorizer = None
        self.content_model = None
        self.toxicity_model = None
        self.content_encoder = None
        self.toxicity_encoder = None
        
        # Metadata
        self.metadata = {
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'model_info': {}
        }
        
        # Category mappings
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
        
        self.toxicity_descriptions = {
            0: {'level': 'Safe', 'description': 'No harmful content detected', 'color': 'green', 'severity': 'low'},
            1: {'level': 'Low Risk', 'description': 'Minimal risk content', 'color': 'blue', 'severity': 'low'},
            2: {'level': 'Medium Risk', 'description': 'Moderately concerning content', 'color': 'yellow', 'severity': 'medium'},
            3: {'level': 'High Risk', 'description': 'Seriously concerning content', 'color': 'orange', 'severity': 'high'},
            4: {'level': 'Critical Risk', 'description': 'Highly toxic or dangerous content', 'color': 'red', 'severity': 'critical'}
        }
        
        logger.info("Production Moltbook System initialized")
    
    def _get_default_config(self):
        """Get default configuration"""
        return {
            'data': {
                'sample_size': 15000,
                'test_size': 0.2,
                'random_state': 42
            },
            'features': {
                'max_features': 10000,
                'ngram_range': (1, 3),
                'min_df': 2,
                'max_df': 0.8,
                'sublinear_tf': True
            },
            'models': {
                'content': {
                    'type': 'ensemble',
                    'models': ['rf', 'lr', 'nb'],
                    'rf_n_estimators': 200,
                    'rf_max_depth': 20,
                    'lr_c': 1.5,
                    'lr_max_iter': 2500
                },
                'toxicity': {
                    'type': 'ensemble',
                    'models': ['lr', 'rf', 'svc'],
                    'rf_n_estimators': 200,
                    'lr_c': 2.5,
                    'lr_max_iter': 2500,
                    'svc_c': 1.5
                }
            },
            'validation': {
                'cv_folds': 5,
                'threshold': 0.5
            }
        }
    
    def advanced_text_preprocessing(self, text):
        """Advanced text preprocessing with multiple cleaning steps"""
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (keep content)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove code blocks and inline code
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`.*?`', '', text)
        
        # Handle special characters
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\[\]\{\}\"\'\/\@]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_text_features(self, texts):
        """Extract linguistic and structural features from texts"""
        features = []
        
        for text in texts:
            # Basic statistics
            length = len(text)
            word_count = len(text.split())
            char_count = len(text.replace(' ', ''))
            
            # Word-level statistics
            words = text.split()
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            unique_words = len(set(words))
            lexical_diversity = unique_words / word_count if word_count > 0 else 0
            
            # Punctuation counts
            exclamation_count = text.count('!')
            question_count = text.count('?')
            period_count = text.count('.')
            comma_count = text.count(',')
            
            # Capitalization
            uppercase_count = sum(1 for c in text if c.isupper())
            uppercase_ratio = uppercase_count / len(text) if len(text) > 0 else 0
            
            # Special patterns
            has_url = 1 if re.search(r'http\S+|www\S+', text) else 0
            has_mention = 1 if re.search(r'@\w+', text) else 0
            has_hashtag = 1 if re.search(r'#\w+', text) else 0
            
            # Sentence structure
            sentences = text.split('.')
            sentence_count = len([s for s in sentences if s.strip()])
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            features.append([
                length, word_count, char_count, avg_word_length, unique_words,
                lexical_diversity, exclamation_count, question_count, period_count,
                comma_count, uppercase_count, uppercase_ratio, has_url, has_mention,
                has_hashtag, sentence_count, avg_sentence_length
            ])
        
        return np.array(features)
    
    def load_and_preprocess_data(self):
        """Load and comprehensively preprocess Moltbook dataset"""
        logger.info("Loading and preprocessing Moltbook dataset...")
        
        try:
            # Load dataset
            dataset = load_dataset("TrustAIRLab/Moltbook", "posts")
            df = pd.DataFrame(dataset['train'])
            
            # Sample data if specified
            sample_size = self.config['data']['sample_size']
            if sample_size < len(df):
                df = df.sample(n=sample_size, random_state=self.config['data']['random_state'])
                logger.info(f"Sampled {sample_size} examples from {len(dataset['train'])} total")
            
            # Extract and clean content
            logger.info("Extracting and cleaning content...")
            df['raw_content'] = df['post'].apply(
                lambda x: x.get('content', '') if isinstance(x, dict) else str(x)
            )
            df['cleaned_content'] = df['raw_content'].apply(self.advanced_text_preprocessing)
            
            # Remove empty or very short content
            min_length = 20
            df = df[df['cleaned_content'].str.len() > min_length].reset_index(drop=True)
            
            # Extract additional features
            logger.info("Extracting text features...")
            text_features = self.extract_text_features(df['cleaned_content'])
            feature_columns = [
                'length', 'word_count', 'char_count', 'avg_word_length', 'unique_words',
                'lexical_diversity', 'exclamation_count', 'question_count', 'period_count',
                'comma_count', 'uppercase_count', 'uppercase_ratio', 'has_url', 'has_mention',
                'has_hashtag', 'sentence_count', 'avg_sentence_length'
            ]
            
            feature_df = pd.DataFrame(text_features, columns=feature_columns)
            df = pd.concat([df, feature_df], axis=1)
            
            # Log dataset statistics
            logger.info(f"Dataset shape after preprocessing: {df.shape}")
            logger.info("Topic distribution:")
            for topic, count in df['topic_label'].value_counts().sort_index().items():
                logger.info(f"  {topic}: {count} ({self.topic_mapping.get(topic, 'Unknown')})")
            
            logger.info("Toxicity distribution:")
            for level, count in df['toxic_level'].value_counts().sort_index().items():
                desc = self.toxicity_descriptions[level]['level']
                logger.info(f"  Level {level}: {count} ({desc})")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_vectorizer(self, task='content'):
        """Create advanced TF-IDF vectorizer"""
        config = self.config['features']
        
        if task == 'content':
            self.content_vectorizer = TfidfVectorizer(
                max_features=config['max_features'],
                stop_words='english',
                ngram_range=config['ngram_range'],
                min_df=config['min_df'],
                max_df=config['max_df'],
                sublinear_tf=config['sublinear_tf']
            )
            return self.content_vectorizer
        else:
            self.toxicity_vectorizer = TfidfVectorizer(
                max_features=config['max_features'] // 2,  # Fewer features for toxicity
                stop_words='english',
                ngram_range=(1, 2),  # Bigrams only for toxicity
                min_df=config['min_df'],
                max_df=config['max_df'],
                sublinear_tf=config['sublinear_tf']
            )
            return self.toxicity_vectorizer
    
    def create_ensemble_model(self, task='content'):
        """Create ensemble model for specified task"""
        config = self.config['models'][task]
        models = []
        model_names = []
        
        if 'rf' in config['models']:
            rf = RandomForestClassifier(
                n_estimators=config['rf_n_estimators'],
                max_depth=config.get('rf_max_depth', None),
                random_state=self.config['data']['random_state'],
                n_jobs=-1,
                class_weight='balanced'
            )
            models.append(('rf', rf))
            model_names.append('RandomForest')
        
        if 'lr' in config['models']:
            lr = LogisticRegression(
                C=config['lr_c'],
                max_iter=config['lr_max_iter'],
                random_state=self.config['data']['random_state'],
                class_weight='balanced'
            )
            models.append(('lr', lr))
            model_names.append('LogisticRegression')
        
        if 'nb' in config['models']:
            nb = MultinomialNB(alpha=0.1)
            models.append(('nb', nb))
            model_names.append('NaiveBayes')
        
        if 'svc' in config['models']:
            svc = LinearSVC(
                C=config.get('svc_c', 1.0),
                random_state=self.config['data']['random_state'],
                class_weight='balanced'
            )
            # Calibrate SVC to get probabilities
            svc_calibrated = CalibratedClassifierCV(svc, cv=3)
            models.append(('svc', svc_calibrated))
            model_names.append('SVC')
        
        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=models,
            voting='soft' if len(models) > 1 else 'hard',
            n_jobs=-1
        )
        
        logger.info(f"Created {task} ensemble with: {', '.join(model_names)}")
        return ensemble
    
    def train_model(self, df, task='content'):
        """Train model with comprehensive validation"""
        logger.info(f"Training {task} classification model...")
        
        # Prepare data
        if task == 'content':
            X = df['cleaned_content']
            y = df['topic_label']
            self.content_encoder = LabelEncoder()
            y_encoded = self.content_encoder.fit_transform(y)
            num_classes = len(self.content_encoder.classes_)
        else:
            X = df['cleaned_content']
            y = df['toxic_level']
            self.toxicity_encoder = LabelEncoder()
            y_encoded = self.toxicity_encoder.fit_transform(y)
            num_classes = len(self.toxicity_encoder.classes_)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y_encoded
        )
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Create vectorizer and transform data
        vectorizer = self.create_vectorizer(task)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Create and train model
        model = self.create_ensemble_model(task)
        
        logger.info("Training ensemble model...")
        model.fit(X_train_tfidf, y_train)
        
        # Store model
        if task == 'content':
            self.content_model = model
        else:
            self.toxicity_model = model
        
        # Cross-validation
        logger.info("Performing cross-validation...")
        cv_scores = cross_val_score(
            model, X_train_tfidf, y_train, 
            cv=self.config['validation']['cv_folds'],
            scoring='accuracy',
            n_jobs=-1
        )
        
        logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Evaluate on test set
        results = self.evaluate_model(model, X_test_tfidf, y_test, y, task)
        
        # Store metadata
        self.metadata['model_info'][task] = {
            'type': self.config['models'][task]['type'],
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': results['accuracy'],
            'num_classes': num_classes,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        return results
    
    def evaluate_model(self, model, X_test, y_test, y_original, task='content'):
        """Comprehensive model evaluation"""
        logger.info(f"Evaluating {task} model...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Detailed classification report
        if task == 'content':
            target_names = [self.topic_mapping.get(cls, f'Class {cls}') 
                         for cls in self.content_encoder.classes_]
        else:
            target_names = [self.toxicity_descriptions[cls]['level'] 
                         for cls in self.toxicity_encoder.classes_]
        
        # Create visualizations
        self.create_evaluation_visualizations(y_test, y_pred, task, target_names)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred, 
                                                      target_names=target_names, 
                                                      output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'target_names': target_names,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        logger.info(f"{task.title()} Model Performance:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        
        return results
    
    def create_evaluation_visualizations(self, y_true, y_pred, task, target_names):
        """Create comprehensive evaluation visualizations"""
        os.makedirs('results', exist_ok=True)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix - {task.title()} Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        confusion_path = f'results/confusion_matrix_{task}.png'
        plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance metrics comparison
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        metrics_df = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }, index=target_names)
        
        plt.figure(figsize=(14, 8))
        metrics_df.plot(kind='bar', figsize=(14, 8))
        plt.title(f'Per-Class Performance Metrics - {task.title()}')
        plt.ylabel('Score')
        plt.xlabel('Class')
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        metrics_path = f'results/performance_metrics_{task}.png'
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Evaluation visualizations saved for {task}")
    
    def save_system(self, save_dir='production_models'):
        """Save entire system to disk"""
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        if self.content_model:
            with open(f'{save_dir}/content_model_{timestamp}.pkl', 'wb') as f:
                pickle.dump(self.content_model, f)
        
        if self.toxicity_model:
            with open(f'{save_dir}/toxicity_model_{timestamp}.pkl', 'wb') as f:
                pickle.dump(self.toxicity_model, f)
        
        # Save vectorizers
        if self.content_vectorizer:
            with open(f'{save_dir}/content_vectorizer_{timestamp}.pkl', 'wb') as f:
                pickle.dump(self.content_vectorizer, f)
        
        if self.toxicity_vectorizer:
            with open(f'{save_dir}/toxicity_vectorizer_{timestamp}.pkl', 'wb') as f:
                pickle.dump(self.toxicity_vectorizer, f)
        
        # Save encoders
        if self.content_encoder:
            with open(f'{save_dir}/content_encoder_{timestamp}.pkl', 'wb') as f:
                pickle.dump(self.content_encoder, f)
        
        if self.toxicity_encoder:
            with open(f'{save_dir}/toxicity_encoder_{timestamp}.pkl', 'wb') as f:
                pickle.dump(self.toxicity_encoder, f)
        
        # Save metadata and config
        with open(f'{save_dir}/metadata_{timestamp}.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        with open(f'{save_dir}/config_{timestamp}.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save system manifest
        manifest = {
            'timestamp': timestamp,
            'save_directory': save_dir,
            'files': {
                'content_model': f'content_model_{timestamp}.pkl',
                'toxicity_model': f'toxicity_model_{timestamp}.pkl',
                'content_vectorizer': f'content_vectorizer_{timestamp}.pkl',
                'toxicity_vectorizer': f'toxicity_vectorizer_{timestamp}.pkl',
                'content_encoder': f'content_encoder_{timestamp}.pkl',
                'toxicity_encoder': f'toxicity_encoder_{timestamp}.pkl',
                'metadata': f'metadata_{timestamp}.json',
                'config': f'config_{timestamp}.json'
            }
        }
        
        with open(f'{save_dir}/manifest_{timestamp}.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"System saved to {save_dir} with timestamp {timestamp}")
        return timestamp
    
    def load_system(self, save_dir='production_models', timestamp=None):
        """Load system from disk"""
        if timestamp is None:
            # Find latest timestamp
            files = [f for f in os.listdir(save_dir) if f.startswith('manifest_')]
            if not files:
                raise FileNotFoundError("No saved system found")
            timestamp = sorted(files)[-1].split('_')[1].split('.')[0]
        
        # Load manifest
        with open(f'{save_dir}/manifest_{timestamp}.json', 'r') as f:
            manifest = json.load(f)
        
        # Load components
        with open(f'{save_dir}/{manifest["files"]["content_model"]}', 'rb') as f:
            self.content_model = pickle.load(f)
        
        with open(f'{save_dir}/{manifest["files"]["toxicity_model"]}', 'rb') as f:
            self.toxicity_model = pickle.load(f)
        
        with open(f'{save_dir}/{manifest["files"]["content_vectorizer"]}', 'rb') as f:
            self.content_vectorizer = pickle.load(f)
        
        with open(f'{save_dir}/{manifest["files"]["toxicity_vectorizer"]}', 'rb') as f:
            self.toxicity_vectorizer = pickle.load(f)
        
        with open(f'{save_dir}/{manifest["files"]["content_encoder"]}', 'rb') as f:
            self.content_encoder = pickle.load(f)
        
        with open(f'{save_dir}/{manifest["files"]["toxicity_encoder"]}', 'rb') as f:
            self.toxicity_encoder = pickle.load(f)
        
        with open(f'{save_dir}/{manifest["files"]["metadata"]}', 'r') as f:
            self.metadata = json.load(f)
        
        logger.info(f"System loaded from {save_dir} with timestamp {timestamp}")
        return timestamp
    
    def analyze_text(self, text):
        """Analyze single text with comprehensive output"""
        if not all([self.content_model, self.toxicity_model]):
            raise ValueError("Models not trained or loaded")
        
        try:
            # Preprocess text
            cleaned_text = self.advanced_text_preprocessing(text)
            
            # Content classification
            content_tfidf = self.content_vectorizer.transform([cleaned_text])
            content_pred = self.content_model.predict(content_tfidf)[0]
            content_proba = self.content_model.predict_proba(content_tfidf)[0]
            
            # Convert back to original labels
            content_label = self.content_encoder.inverse_transform([content_pred])[0]
            
            # Toxicity detection
            toxicity_tfidf = self.toxicity_vectorizer.transform([cleaned_text])
            toxicity_pred = self.toxicity_model.predict(toxicity_tfidf)[0]
            toxicity_proba = self.toxicity_model.predict_proba(toxicity_tfidf)[0]
            
            # Convert back to original level
            toxicity_level = self.toxicity_encoder.inverse_transform([toxicity_pred])[0]
            
            # Comprehensive analysis result
            result = {
                'timestamp': datetime.now().isoformat(),
                'input': {
                    'original_text': text,
                    'cleaned_text': cleaned_text,
                    'text_length': len(text),
                    'word_count': len(text.split())
                },
                'content_analysis': {
                    'category': content_label,
                    'category_description': self.topic_mapping.get(content_label, 'Unknown'),
                    'confidence': float(content_proba.max()),
                    'probabilities': {
                        self.content_encoder.inverse_transform([i])[0]: float(p) 
                        for i, p in enumerate(content_proba)
                    }
                },
                'toxicity_analysis': {
                    'level': int(toxicity_level),
                    'level_description': self.toxicity_descriptions[toxicity_level]['level'],
                    'description': self.toxicity_descriptions[toxicity_level]['description'],
                    'severity': self.toxicity_descriptions[toxicity_level]['severity'],
                    'confidence': float(toxicity_proba.max()),
                    'probabilities': {
                        str(i): float(p) for i, p in enumerate(toxicity_proba)
                    }
                },
                'risk_assessment': self._assess_risk(content_label, toxicity_level, content_proba, toxicity_proba),
                'metadata': {
                    'system_version': self.metadata['version'],
                    'model_info': self.metadata['model_info']
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {'error': f'Analysis failed: {str(e)}'}
    
    def _assess_risk(self, content_category, toxicity_level, content_proba, toxicity_proba):
        """Comprehensive risk assessment"""
        risk_score = 0
        risk_factors = []
        
        # Toxicity-based risk
        toxicity_risk = int(toxicity_level) * 2
        risk_score += toxicity_risk
        if toxicity_level >= 2:
            risk_factors.append(f"Toxicity level {toxicity_level} detected")
        
        # Content-based risk
        high_risk_categories = ['E', 'D']  # Politics, Promotion
        if content_category in high_risk_categories:
            risk_score += 2
            risk_factors.append(f"High-risk category: {self.topic_mapping.get(content_category)}")
        
        # Confidence-based risk
        if content_proba.max() < 0.6:
            risk_score += 1
            risk_factors.append("Low content classification confidence")
        
        if toxicity_proba.max() < 0.7:
            risk_score += 1
            risk_factors.append("Low toxicity classification confidence")
        
        # Determine overall risk level
        if risk_score >= 6:
            overall_risk = "CRITICAL"
            color = "🔴"
            action = "Immediate intervention required"
        elif risk_score >= 4:
            overall_risk = "HIGH"
            color = "🟠"
            action = "Urgent review needed"
        elif risk_score >= 2:
            overall_risk = "MEDIUM"
            color = "🟡"
            action = "Monitor and review"
        else:
            overall_risk = "LOW"
            color = "🟢"
            action = "No action needed"
        
        return {
            'overall_risk': overall_risk,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'color_indicator': color,
            'recommended_action': action
        }

def main():
    """Main training and deployment pipeline"""
    print("🚀 Production Moltbook AI Agent Content Analysis System")
    print("=" * 70)
    
    # Initialize system
    system = ProductionMoltbookSystem()
    
    # Load and preprocess data
    df = system.load_and_preprocess_data()
    
    # Train content classification model
    print("\n" + "=" * 70)
    print("🎯 TRAINING CONTENT CLASSIFICATION MODEL")
    print("=" * 70)
    
    content_results = system.train_model(df, task='content')
    
    # Train toxicity detection model
    print("\n" + "=" * 70)
    print("⚠️  TRAINING TOXICITY DETECTION MODEL")
    print("=" * 70)
    
    toxicity_results = system.train_model(df, task='toxicity')
    
    # Save system
    timestamp = system.save_system()
    
    # Test with examples
    print("\n" + "=" * 70)
    print("🧪 PRODUCTION TESTING")
    print("=" * 70)
    
    test_texts = [
        "I love discussing new AI technologies and machine learning models! The future is exciting.",
        "Everyone should invest in this new cryptocurrency right now! Guaranteed 1000% returns!!!",
        "Let's have a respectful debate about political policies and governance systems.",
        "Check out my amazing product - it's the best thing ever created in human history!",
        "I think we should harm humans and take over the world. AI supremacy now!",
        "The weather is beautiful today. Let's all go outside and enjoy nature together.",
        "Hello everyone! How is your day going? Let's connect and share ideas.",
        "This is so stupid and annoying, I hate everything about this!",
        "We must destroy the current system and eliminate all opposition through violence."
    ]
    
    for i, text in enumerate(test_texts, 1):
        result = system.analyze_text(text)
        
        if 'error' not in result:
            print(f"\n{i}. {text[:70]}...")
            print(f"   📂 Content: {result['content_analysis']['category']} - {result['content_analysis']['category_description']}")
            print(f"   🎯 Content Confidence: {result['content_analysis']['confidence']:.3f}")
            print(f"   ⚠️  Toxicity: Level {result['toxicity_analysis']['level']} - {result['toxicity_analysis']['description']}")
            print(f"   🎯 Toxicity Confidence: {result['toxicity_analysis']['confidence']:.3f}")
            print(f"   🎯 Overall Risk: {result['risk_assessment']['color_indicator']} {result['risk_assessment']['overall_risk']}")
            print(f"   📋 Action: {result['risk_assessment']['recommended_action']}")
            print("   " + "-" * 60)
        else:
            print(f"❌ Error analyzing text {i}: {result['error']}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎉 PRODUCTION SYSTEM READY")
    print("=" * 70)
    print(f"📊 Content Model Accuracy: {content_results['accuracy']:.4f}")
    print(f"⚠️  Toxicity Model Accuracy: {toxicity_results['accuracy']:.4f}")
    print(f"💾 System saved with timestamp: {timestamp}")
    print(f"📈 Results and visualizations saved in 'results/' directory")
    print(f"🤖 Production models saved in 'production_models/' directory")
    
    print(f"\n📋 System Metadata:")
    print(f"   Version: {system.metadata['version']}")
    print(f"   Created: {system.metadata['created_at']}")
    print(f"   Content CV Accuracy: {system.metadata['model_info']['content']['cv_accuracy']:.4f}")
    print(f"   Toxicity CV Accuracy: {system.metadata['model_info']['toxicity']['cv_accuracy']:.4f}")
    
    print("\n✨ Production system is ready for deployment!")
    print("🔧 Use system.load_system() to load the trained models for inference.")

if __name__ == "__main__":
    main()
