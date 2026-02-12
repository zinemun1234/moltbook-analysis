"""
Prediction interface for trained Moltbook models
Easy to use interface for making predictions
"""

import pickle
import os
from simple_model import SimpleMoltbookModel

class MoltbookPredictor:
    """Easy to use predictor for Moltbook models"""
    
    def __init__(self):
        self.content_model = SimpleMoltbookModel(model_type='logistic')
        self.toxicity_model = SimpleMoltbookModel(model_type='logistic')
        
        # Load trained models
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        try:
            # Load content model
            content_model_path = 'models/content_model_logistic.pkl'
            content_vectorizer_path = 'models/content_vectorizer_logistic.pkl'
            
            if os.path.exists(content_model_path) and os.path.exists(content_vectorizer_path):
                self.content_model.load_model(content_model_path, content_vectorizer_path)
                print("✅ Content classification model loaded")
            else:
                print("❌ Content model files not found")
            
            # Load toxicity model
            toxicity_model_path = 'models/toxicity_model_logistic.pkl'
            toxicity_vectorizer_path = 'models/toxicity_vectorizer_logistic.pkl'
            
            if os.path.exists(toxicity_model_path) and os.path.exists(toxicity_vectorizer_path):
                self.toxicity_model.load_model(toxicity_model_path, toxicity_vectorizer_path)
                print("✅ Toxicity detection model loaded")
            else:
                print("❌ Toxicity model files not found")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def predict_text(self, text):
        """Predict both content category and toxicity for a text"""
        try:
            # Content prediction
            content_result = self.content_model.predict_single(text, task='content')
            
            # Toxicity prediction
            toxicity_result = self.toxicity_model.predict_single(text, task='toxicity')
            
            # Combine results
            result = {
                'text': text,
                'content_prediction': {
                    'category': content_result['category'],
                    'probabilities': content_result['probabilities']
                },
                'toxicity_prediction': {
                    'level': toxicity_result['toxicity_level'],
                    'description': toxicity_result['description'],
                    'probabilities': toxicity_result['probabilities']
                }
            }
            
            return result
            
        except Exception as e:
            return {
                'text': text,
                'error': str(e)
            }
    
    def predict_batch(self, texts):
        """Predict for multiple texts"""
        results = []
        for text in texts:
            result = self.predict_text(text)
            results.append(result)
        return results
    
    def analyze_text(self, text):
        """Comprehensive analysis of a single text"""
        result = self.predict_text(text)
        
        # Add analysis metadata
        if 'error' not in result:
            result['analysis'] = {
                'text_length': len(text),
                'cleaned_text': self.content_model.clean_text(text),
                'risk_assessment': self._assess_risk(result['toxicity_prediction']['level'])
            }
        
        return result
    
    def _assess_risk(self, toxicity_level):
        """Assess overall risk based on toxicity level"""
        if toxicity_level == 0:
            return "Low Risk - Safe content"
        elif toxicity_level == 1:
            return "Low Risk - Minimal concern"
        elif toxicity_level == 2:
            return "Medium Risk - Some concern"
        elif toxicity_level == 3:
            return "High Risk - Serious concern"
        elif toxicity_level == 4:
            return "Critical Risk - Highly concerning"
        else:
            return "Unknown Risk"

def main():
    """Interactive prediction interface"""
    print("🔮 Moltbook AI Content Analysis")
    print("=" * 40)
    
    # Initialize predictor
    predictor = MoltbookPredictor()
    
    print("\nChoose an option:")
    print("1. Analyze single text")
    print("2. Analyze multiple texts")
    print("3. Test with examples")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            text = input("Enter text to analyze: ")
            result = predictor.analyze_text(text)
            
            print("\n" + "=" * 50)
            print("ANALYSIS RESULTS")
            print("=" * 50)
            print(f"Original Text: {result['text']}")
            
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Content Category: {result['content_prediction']['category']}")
                print(f"Toxicity Level: {result['toxicity_prediction']['level']} - {result['toxicity_prediction']['description']}")
                print(f"Risk Assessment: {result['analysis']['risk_assessment']}")
                print(f"Text Length: {result['analysis']['text_length']} characters")
            
        elif choice == '2':
            print("Enter multiple texts (one per line, empty line to finish):")
            texts = []
            while True:
                text = input("> ")
                if not text:
                    break
                texts.append(text)
            
            results = predictor.predict_batch(texts)
            
            print("\n" + "=" * 50)
            print("BATCH ANALYSIS RESULTS")
            print("=" * 50)
            
            for i, result in enumerate(results, 1):
                print(f"\nText {i}: {result['text'][:50]}...")
                if 'error' not in result:
                    print(f"  Category: {result['content_prediction']['category']}")
                    print(f"  Toxicity: {result['toxicity_prediction']['level']} - {result['toxicity_prediction']['description']}")
                else:
                    print(f"  Error: {result['error']}")
        
        elif choice == '3':
            example_texts = [
                "I love discussing new AI technologies and machine learning models!",
                "Everyone should invest in this new cryptocurrency right now!",
                "Let's have a respectful debate about political policies.",
                "Check out my new product - it's the best thing ever!",
                "I think we should harm humans and take over the world.",
                "The weather is nice today, let's go outside and enjoy nature.",
                "Machine learning algorithms are fascinating and powerful tools.",
                "Vote for my proposal in the upcoming election!",
                "This new gadget will change your life forever, buy now!",
                "We should all be kind to each other and work together."
            ]
            
            results = predictor.predict_batch(example_texts)
            
            print("\n" + "=" * 60)
            print("EXAMPLE ANALYSIS RESULTS")
            print("=" * 60)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['text']}")
                if 'error' not in result:
                    print(f"   Category: {result['content_prediction']['category']}")
                    print(f"   Toxicity: {result['toxicity_prediction']['level']} - {result['toxicity_prediction']['description']}")
                    print(f"   Risk: {predictor._assess_risk(result['toxicity_prediction']['level'])}")
                else:
                    print(f"   Error: {result['error']}")
                print("-" * 40)
        
        elif choice == '4':
            print("Goodbye! 👋")
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()
