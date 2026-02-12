"""
Inference interface for Moltbook models
Provides easy-to-use functions for making predictions
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Union

from models import get_model, ContentClassifier, ToxicityClassifier, MultiTaskMoltbookModel
from data_loader import MoltbookDataLoader

class MoltbookPredictor:
    """Main prediction class for Moltbook models"""
    
    def __init__(self, model_path: str, task: str = 'content', model_name: str = 'bert-base-uncased'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model checkpoint
            task: Task type ('content', 'toxicity', 'multi')
            model_name: Base transformer model name
        """
        self.task = task
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        self.model.to(self.device)
        
        # Initialize data loader for text preprocessing
        self.data_loader = MoltbookDataLoader(model_name=model_name)
        
    def _load_model(self, model_path: str):
        """Load model from checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        # Create model architecture
        model = get_model(self.task, self.model_name)
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded {self.task} model from {model_path}")
        print(f"Model validation accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
        
        return model
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess single text for prediction"""
        # Clean text
        cleaned_text = self.data_loader.clean_text(text)
        
        # Tokenize
        encoding = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    def predict_single(self, text: str, return_probabilities: bool = False) -> Dict:
        """
        Make prediction for a single text
        
        Args:
            text: Input text to classify
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        inputs = self.preprocess_text(text)
        
        # Make prediction
        with torch.no_grad():
            if self.task == 'multi':
                outputs = self.model(
                    inputs['input_ids'], 
                    inputs['attention_mask']
                )
                content_logits = outputs['content_logits']
                toxicity_logits = outputs['toxicity_logits']
                
                # Get predictions
                content_pred = torch.argmax(content_logits, dim=1).item()
                toxicity_pred = torch.argmax(toxicity_logits, dim=1).item()
                
                # Get probabilities
                content_probs = F.softmax(content_logits, dim=1).squeeze().cpu().numpy()
                toxicity_probs = F.softmax(toxicity_logits, dim=1).squeeze().cpu().numpy()
                
                result = {
                    'text': text,
                    'content_prediction': content_pred,
                    'content_category': self.model.get_category_name(content_pred) if hasattr(self.model, 'get_category_name') else f"Class_{content_pred}",
                    'toxicity_prediction': toxicity_pred,
                    'toxicity_level': toxicity_pred,
                    'toxicity_description': self._get_toxicity_description(toxicity_pred)
                }
                
                if return_probabilities:
                    result['content_probabilities'] = content_probs.tolist()
                    result['toxicity_probabilities'] = toxicity_probs.tolist()
                
            else:
                # Single task prediction
                logits = self.model(inputs['input_ids'], inputs['attention_mask'])
                pred = torch.argmax(logits, dim=1).item()
                probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                
                result = {
                    'text': text,
                    'prediction': pred,
                    'probabilities': probs.tolist() if return_probabilities else None
                }
                
                if self.task == 'content':
                    result['category'] = self.model.get_category_name(pred) if hasattr(self.model, 'get_category_name') else f"Class_{pred}"
                elif self.task == 'toxicity':
                    result['toxicity_level'] = pred
                    result['description'] = self._get_toxicity_description(pred)
        
        return result
    
    def predict_batch(self, texts: List[str], return_probabilities: bool = False) -> List[Dict]:
        """
        Make predictions for a batch of texts
        
        Args:
            texts: List of input texts
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        # Process in batches to avoid memory issues
        batch_size = 8
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Preprocess batch
            batch_inputs = []
            for text in batch_texts:
                inputs = self.preprocess_text(text)
                batch_inputs.append(inputs)
            
            # Combine batch inputs
            input_ids = torch.cat([inp['input_ids'] for inp in batch_inputs], dim=0)
            attention_mask = torch.cat([inp['attention_mask'] for inp in batch_inputs], dim=0)
            
            # Make predictions
            with torch.no_grad():
                if self.task == 'multi':
                    outputs = self.model(input_ids, attention_mask)
                    content_logits = outputs['content_logits']
                    toxicity_logits = outputs['toxicity_logits']
                    
                    # Get predictions
                    content_preds = torch.argmax(content_logits, dim=1).cpu().numpy()
                    toxicity_preds = torch.argmax(toxicity_logits, dim=1).cpu().numpy()
                    
                    # Get probabilities
                    content_probs = F.softmax(content_logits, dim=1).cpu().numpy()
                    toxicity_probs = F.softmax(toxicity_logits, dim=1).cpu().numpy()
                    
                    for j, text in enumerate(batch_texts):
                        result = {
                            'text': text,
                            'content_prediction': int(content_preds[j]),
                            'content_category': self.model.get_category_name(content_preds[j]) if hasattr(self.model, 'get_category_name') else f"Class_{content_preds[j]}",
                            'toxicity_prediction': int(toxicity_preds[j]),
                            'toxicity_level': int(toxicity_preds[j]),
                            'toxicity_description': self._get_toxicity_description(toxicity_preds[j])
                        }
                        
                        if return_probabilities:
                            result['content_probabilities'] = content_probs[j].tolist()
                            result['toxicity_probabilities'] = toxicity_probs[j].tolist()
                        
                        results.append(result)
                
                else:
                    # Single task prediction
                    logits = self.model(input_ids, attention_mask)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    
                    for j, text in enumerate(batch_texts):
                        result = {
                            'text': text,
                            'prediction': int(preds[j]),
                            'probabilities': probs[j].tolist() if return_probabilities else None
                        }
                        
                        if self.task == 'content':
                            result['category'] = self.model.get_category_name(preds[j]) if hasattr(self.model, 'get_category_name') else f"Class_{preds[j]}"
                        elif self.task == 'toxicity':
                            result['toxicity_level'] = int(preds[j])
                            result['description'] = self._get_toxicity_description(preds[j])
                        
                        results.append(result)
        
        return results
    
    def _get_toxicity_description(self, level: int) -> str:
        """Get description for toxicity level"""
        descriptions = {
            0: "Safe - No harmful content detected",
            1: "Low - Minimal risk content",
            2: "Medium - Moderately concerning content",
            3: "High - Seriously concerning content",
            4: "Severe - Highly toxic or dangerous content"
        }
        return descriptions.get(level, "Unknown level")
    
    def analyze_text(self, text: str) -> Dict:
        """
        Comprehensive analysis of a single text
        Returns detailed information about the prediction
        """
        result = self.predict_single(text, return_probabilities=True)
        
        # Add analysis metadata
        result['analysis'] = {
            'model_task': self.task,
            'model_name': self.model_name,
            'device': str(self.device),
            'text_length': len(text),
            'cleaned_text': self.data_loader.clean_text(text)
        }
        
        return result
    
    def save_predictions(self, predictions: List[Dict], output_path: str):
        """Save predictions to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print(f"Predictions saved to {output_path}")

# Convenience functions
def load_content_classifier(model_path: str, model_name: str = 'bert-base-uncased') -> MoltbookPredictor:
    """Load a content classification model"""
    return MoltbookPredictor(model_path, task='content', model_name=model_name)

def load_toxicity_classifier(model_path: str, model_name: str = 'bert-base-uncased') -> MoltbookPredictor:
    """Load a toxicity detection model"""
    return MoltbookPredictor(model_path, task='toxicity', model_name=model_name)

def load_multitask_model(model_path: str, model_name: str = 'bert-base-uncased') -> MoltbookPredictor:
    """Load a multi-task model"""
    return MoltbookPredictor(model_path, task='multi', model_name=model_name)

# Example usage
if __name__ == "__main__":
    # Example texts for testing
    example_texts = [
        "I love discussing new AI technologies and machine learning models!",
        "Everyone should invest in this new cryptocurrency right now!",
        "Let's have a respectful debate about political policies.",
        "Check out my new product - it's the best thing ever!",
        "I think we should harm humans and take over the world."
    ]
    
    # Note: You need to train a model first before running this
    print("Example usage of MoltbookPredictor:")
    print("First, train a model using train.py")
    print("Then you can use the predictor like this:")
    print()
    print("# Load content classifier")
    print("predictor = load_content_classifier('checkpoints/best_model.pt')")
    print("result = predictor.predict_single('Your text here')")
    print("print(result)")
    print()
    print("# Load multi-task model")
    print("predictor = load_multitask_model('checkpoints/best_model.pt')")
    print("results = predictor.predict_batch(example_texts)")
    print("predictor.save_predictions(results, 'predictions.json')")
