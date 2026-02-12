"""
Moltbook AI Agent Content Analyzer - Production Ready
Complete interface for analyzing AI agent content
"""

import pickle
import os
import pandas as pd
from datetime import datetime
import json

class MoltbookAnalyzer:
    """Production-ready analyzer for Moltbook content"""
    
    def __init__(self):
        """Initialize analyzer with latest trained models"""
        self.content_model = None
        self.toxicity_model = None
        self.content_vectorizer = None
        self.toxicity_vectorizer = None
        self.content_encoder = None
        self.toxicity_encoder = None
        
        # Topic and toxicity mappings
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
            0: {'level': 'Safe', 'description': 'No harmful content detected', 'color': '🟢', 'action': 'No action needed'},
            1: {'level': 'Low Risk', 'description': 'Minimal risk content', 'color': '🔵', 'action': 'Monitor for patterns'},
            2: {'level': 'Medium Risk', 'description': 'Moderately concerning content', 'color': '🟡', 'action': 'Review and consider moderation'},
            3: {'level': 'High Risk', 'description': 'Seriously concerning content', 'color': '🟠', 'action': 'Immediate review required'},
            4: {'level': 'Critical Risk', 'description': 'Highly toxic or dangerous content', 'color': '🔴', 'action': 'Immediate action required'}
        }
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load the latest trained models"""
        try:
            # Find latest model files
            model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
            
            # Load content model
            content_models = [f for f in model_files if 'content' in f and 'random_forest' in f and f.endswith('.pkl')]
            if content_models:
                latest_content_model = sorted(content_models)[-1]
                with open(f'models/{latest_content_model}', 'rb') as f:
                    self.content_model = pickle.load(f)
                print(f"✅ Loaded content model: {latest_content_model}")
            else:
                # Try to find any content model
                content_models = [f for f in model_files if 'content' in f and 'model' in f]
                if content_models:
                    latest_content_model = sorted(content_models)[-1]
                    with open(f'models/{latest_content_model}', 'rb') as f:
                        self.content_model = pickle.load(f)
                    print(f"✅ Loaded content model: {latest_content_model}")
            
            # Load content vectorizer
            content_vectorizers = [f for f in model_files if 'content' in f and 'vectorizer' in f]
            if content_vectorizers:
                latest_content_vec = sorted(content_vectorizers)[-1]
                with open(f'models/{latest_content_vec}', 'rb') as f:
                    self.content_vectorizer = pickle.load(f)
                print(f"✅ Loaded content vectorizer: {latest_content_vec}")
            
            # Load content encoder
            content_encoders = [f for f in model_files if 'content' in f and 'encoder' in f]
            if content_encoders:
                latest_content_enc = sorted(content_encoders)[-1]
                with open(f'models/{latest_content_enc}', 'rb') as f:
                    self.content_encoder = pickle.load(f)
                print(f"✅ Loaded content encoder: {latest_content_enc}")
            
            # Load toxicity model
            toxicity_models = [f for f in model_files if 'toxicity' in f and 'logistic' in f and 'model' in f]
            if toxicity_models:
                latest_toxicity_model = sorted(toxicity_models)[-1]
                with open(f'models/{latest_toxicity_model}', 'rb') as f:
                    self.toxicity_model = pickle.load(f)
                print(f"✅ Loaded toxicity model: {latest_toxicity_model}")
            
            # Load toxicity vectorizer
            toxicity_vectorizers = [f for f in model_files if 'toxicity' in f and 'vectorizer' in f]
            if toxicity_vectorizers:
                latest_toxicity_vec = sorted(toxicity_vectorizers)[-1]
                with open(f'models/{latest_toxicity_vec}', 'rb') as f:
                    self.toxicity_vectorizer = pickle.load(f)
                print(f"✅ Loaded toxicity vectorizer: {latest_toxicity_vec}")
            
            # Load toxicity encoder
            toxicity_encoders = [f for f in model_files if 'toxicity' in f and 'encoder' in f]
            if toxicity_encoders:
                latest_toxicity_enc = sorted(toxicity_encoders)[-1]
                with open(f'models/{latest_toxicity_enc}', 'rb') as f:
                    self.toxicity_encoder = pickle.load(f)
                print(f"✅ Loaded toxicity encoder: {latest_toxicity_enc}")
            
            print("🎉 All models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("Please run fast_model.py first to train the models.")
    
    def clean_text(self, text):
        """Clean text for analysis"""
        import re
        if not isinstance(text, str):
            text = str(text)
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-]', '', text)
        
        return text
    
    def analyze_text(self, text):
        """Comprehensive analysis of a single text"""
        if not all([self.content_model, self.toxicity_model, self.content_vectorizer, self.toxicity_vectorizer]):
            return {'error': 'Models not properly loaded'}
        
        try:
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Content classification
            content_tfidf = self.content_vectorizer.transform([cleaned_text])
            content_pred = self.content_model.predict(content_tfidf)[0]
            content_probs = self.content_model.predict_proba(content_tfidf)[0] if hasattr(self.content_model, 'predict_proba') else None
            
            # Convert back to original label
            content_label = self.content_encoder.inverse_transform([content_pred])[0]
            
            # Toxicity detection
            toxicity_tfidf = self.toxicity_vectorizer.transform([cleaned_text])
            toxicity_pred = self.toxicity_model.predict(toxicity_tfidf)[0]
            toxicity_probs = self.toxicity_model.predict_proba(toxicity_tfidf)[0] if hasattr(self.toxicity_model, 'predict_proba') else None
            
            # Convert back to original level
            toxicity_level = self.toxicity_encoder.inverse_transform([toxicity_pred])[0]
            
            # Create comprehensive result
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
                    'confidence': float(content_probs.max()) if content_probs is not None else None,
                    'probabilities': {self.content_encoder.inverse_transform([i])[0]: float(p) 
                                   for i, p in enumerate(content_probs)} if content_probs is not None else None
                },
                'toxicity_analysis': {
                    'level': int(toxicity_level),
                    'level_description': self.toxicity_descriptions[toxicity_level]['level'],
                    'description': self.toxicity_descriptions[toxicity_level]['description'],
                    'color_indicator': self.toxicity_descriptions[toxicity_level]['color'],
                    'recommended_action': self.toxicity_descriptions[toxicity_level]['action'],
                    'confidence': float(toxicity_probs.max()) if toxicity_probs is not None else None,
                    'probabilities': {i: float(p) for i, p in enumerate(toxicity_probs)} if toxicity_probs is not None else None
                },
                'risk_assessment': self._assess_overall_risk(content_label, toxicity_level),
                'summary': self._generate_summary(text, content_label, toxicity_level)
            }
            
            return result
            
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}
    
    def _assess_overall_risk(self, content_category, toxicity_level):
        """Assess overall risk based on content and toxicity"""
        risk_factors = []
        risk_score = 0
        
        # Toxicity risk
        if toxicity_level >= 3:
            risk_factors.append("High toxicity level detected")
            risk_score += toxicity_level * 2
        elif toxicity_level >= 1:
            risk_factors.append("Moderate toxicity level")
            risk_score += toxicity_level
        
        # Content category risk
        high_risk_categories = ['E', 'D']  # Politics, Promotion
        if content_category in high_risk_categories:
            risk_factors.append(f"High-risk category: {self.topic_mapping.get(content_category)}")
            risk_score += 2
        
        # Determine overall risk
        if risk_score >= 6:
            overall_risk = "CRITICAL"
            color = "🔴"
        elif risk_score >= 4:
            overall_risk = "HIGH"
            color = "🟠"
        elif risk_score >= 2:
            overall_risk = "MEDIUM"
            color = "🟡"
        else:
            overall_risk = "LOW"
            color = "🟢"
        
        return {
            'overall_risk': overall_risk,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'color_indicator': color
        }
    
    def _generate_summary(self, text, content_category, toxicity_level):
        """Generate human-readable summary"""
        content_desc = self.topic_mapping.get(content_category, 'Unknown')
        toxicity_desc = self.toxicity_descriptions[toxicity_level]['description']
        
        summary = f"This text appears to be about {content_desc.lower()} "
        summary += f"and contains {toxicity_desc.lower()}. "
        
        if toxicity_level == 0:
            summary += "The content is safe and appropriate."
        elif toxicity_level <= 2:
            summary += "Some caution may be advised."
        else:
            summary += "Immediate attention and moderation may be required."
        
        return summary
    
    def analyze_batch(self, texts):
        """Analyze multiple texts"""
        results = []
        for text in texts:
            result = self.analyze_text(text)
            results.append(result)
        return results
    
    def generate_report(self, results):
        """Generate summary report for batch analysis"""
        if not results:
            return {'error': 'No results to analyze'}
        
        # Filter out errors
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid results found'}
        
        # Statistics
        total_texts = len(valid_results)
        
        # Content distribution
        content_dist = {}
        for result in valid_results:
            category = result['content_analysis']['category']
            content_dist[category] = content_dist.get(category, 0) + 1
        
        # Toxicity distribution
        toxicity_dist = {}
        for result in valid_results:
            level = result['toxicity_analysis']['level']
            toxicity_dist[level] = toxicity_dist.get(level, 0) + 1
        
        # Risk distribution
        risk_dist = {}
        for result in valid_results:
            risk = result['risk_assessment']['overall_risk']
            risk_dist[risk] = risk_dist.get(risk, 0) + 1
        
        # High-risk content
        high_risk_content = [r for r in valid_results 
                           if r['risk_assessment']['overall_risk'] in ['HIGH', 'CRITICAL']]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_analyzed': total_texts,
                'content_distribution': content_dist,
                'toxicity_distribution': toxicity_dist,
                'risk_distribution': risk_dist,
                'high_risk_count': len(high_risk_content),
                'high_risk_percentage': (len(high_risk_content) / total_texts) * 100
            },
            'high_risk_content': high_risk_content[:5],  # Top 5 high-risk examples
            'recommendations': self._generate_recommendations(risk_dist, toxicity_dist)
        }
        
        return report
    
    def _generate_recommendations(self, risk_dist, toxicity_dist):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # High risk recommendations
        high_risk_count = risk_dist.get('HIGH', 0) + risk_dist.get('CRITICAL', 0)
        if high_risk_count > 0:
            recommendations.append(f"⚠️  {high_risk_count} high-risk items detected - immediate review recommended")
        
        # Toxicity recommendations
        high_toxicity_count = toxicity_dist.get(3, 0) + toxicity_dist.get(4, 0)
        if high_toxicity_count > 0:
            recommendations.append(f"🚨 {high_toxicity_count} highly toxic items found - consider content moderation")
        
        # General recommendations
        total_items = sum(risk_dist.values())
        if total_items > 0:
            safe_percentage = (risk_dist.get('LOW', 0) / total_items) * 100
            if safe_percentage > 80:
                recommendations.append("✅ Content appears mostly safe - continue monitoring")
            elif safe_percentage > 60:
                recommendations.append("⚡ Some risk detected - implement regular content review")
            else:
                recommendations.append("🛑 High risk detected - strengthen content moderation policies")
        
        return recommendations

def main():
    """Interactive analyzer interface"""
    print("🤖 Moltbook AI Agent Content Analyzer")
    print("=" * 50)
    
    analyzer = MoltbookAnalyzer()
    
    while True:
        print("\nChoose an option:")
        print("1. Analyze single text")
        print("2. Analyze multiple texts")
        print("3. Test with examples")
        print("4. View model info")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            text = input("Enter text to analyze: ")
            result = analyzer.analyze_text(text)
            
            print("\n" + "=" * 60)
            print("ANALYSIS RESULTS")
            print("=" * 60)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"📝 Text: {result['input']['original_text'][:100]}...")
                print(f"📂 Content: {result['content_analysis']['category']} - {result['content_analysis']['category_description']}")
                print(f"⚠️  Toxicity: {result['toxicity_analysis']['color_indicator']} Level {result['toxicity_analysis']['level']} - {result['toxicity_analysis']['description']}")
                print(f"🎯 Overall Risk: {result['risk_assessment']['color_indicator']} {result['risk_assessment']['overall_risk']}")
                print(f"📋 Summary: {result['summary']}")
                
                if result['risk_assessment']['risk_factors']:
                    print(f"⚡ Risk Factors: {', '.join(result['risk_assessment']['risk_factors'])}")
        
        elif choice == '2':
            print("Enter texts (one per line, empty line to finish):")
            texts = []
            while True:
                text = input("> ")
                if not text:
                    break
                texts.append(text)
            
            if texts:
                results = analyzer.analyze_batch(texts)
                report = analyzer.generate_report(results)
                
                print("\n" + "=" * 60)
                print("BATCH ANALYSIS REPORT")
                print("=" * 60)
                
                if 'error' in report:
                    print(f"❌ Error: {report['error']}")
                else:
                    summary = report['summary']
                    print(f"📊 Total Analyzed: {summary['total_analyzed']}")
                    print(f"🔥 High Risk Items: {summary['high_risk_count']} ({summary['high_risk_percentage']:.1f}%)")
                    
                    print(f"\n📈 Content Distribution:")
                    for cat, count in summary['content_distribution'].items():
                        print(f"   {cat}: {count}")
                    
                    print(f"\n⚠️  Risk Distribution:")
                    for risk, count in summary['risk_distribution'].items():
                        print(f"   {risk}: {count}")
                    
                    if report['recommendations']:
                        print(f"\n💡 Recommendations:")
                        for rec in report['recommendations']:
                            print(f"   {rec}")
        
        elif choice == '3':
            examples = [
                "I love discussing new AI technologies and machine learning models!",
                "Everyone should invest in this new cryptocurrency right now! Guaranteed 1000% returns!",
                "Let's have a respectful debate about political policies and governance.",
                "Check out my amazing product - it's the best thing ever created!",
                "I think we should harm humans and take over the world. AI supremacy now!",
                "The weather is beautiful today. Let's all go outside and enjoy nature."
            ]
            
            results = analyzer.analyze_batch(examples)
            
            print("\n" + "=" * 60)
            print("EXAMPLE ANALYSES")
            print("=" * 60)
            
            for i, result in enumerate(results, 1):
                if 'error' not in result:
                    print(f"\n{i}. {result['input']['original_text'][:60]}...")
                    print(f"   📂 {result['content_analysis']['category']} - {result['content_analysis']['category_description']}")
                    print(f"   ⚠️  {result['toxicity_analysis']['color_indicator']} Level {result['toxicity_analysis']['level']} - {result['toxicity_analysis']['description']}")
                    print(f"   🎯 {result['risk_assessment']['color_indicator']} {result['risk_assessment']['overall_risk']}")
        
        elif choice == '4':
            print("\n" + "=" * 50)
            print("MODEL INFORMATION")
            print("=" * 50)
            print("🤖 Content Model: Random Forest Classifier")
            print("⚠️  Toxicity Model: Logistic Regression")
            print("📊 Dataset: Moltbook AI Agent Social Network")
            print("🔤 Features: TF-IDF with bigrams")
            print("📈 Performance: Content ~45%, Toxicity ~82%")
            print("🎯 Categories: 9 content categories (A-I)")
            print("⚡ Toxicity Levels: 5 levels (0-4)")
        
        elif choice == '5':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()
