"""
Moltbook AI Agent Content Analysis Interface
User-friendly interface for the production system
"""

import pickle
import os
import json
from datetime import datetime
import pandas as pd
from production_moltbook import ProductionMoltbookSystem

class MoltbookInterface:
    """User-friendly interface for Moltbook analysis"""
    
    def __init__(self):
        """Initialize interface and load production system"""
        self.system = ProductionMoltbookSystem()
        self.loaded = False
        
        # Try to load the latest system
        try:
            self.load_latest_system()
        except FileNotFoundError:
            print("⚠️  No trained system found. Please run production_moltbook.py first.")
        except Exception as e:
            print(f"⚠️  Error loading system: {e}")
    
    def load_latest_system(self):
        """Load the latest trained system"""
        save_dir = 'production_models'
        if not os.path.exists(save_dir):
            raise FileNotFoundError("No production_models directory found")
        
        # Find latest manifest
        files = [f for f in os.listdir(save_dir) if f.startswith('manifest_')]
        if not files:
            raise FileNotFoundError("No saved system found")
        
        latest_file = sorted(files)[-1]
        # Extract timestamp correctly
        parts = latest_file.replace('manifest_', '').replace('.json', '').split('_')
        timestamp = '_'.join(parts)  # Handle full timestamp
        
        print(f"Loading system with timestamp: {timestamp}")
        self.system.load_system(save_dir, timestamp)
        self.loaded = True
        print(f"✅ Production system loaded successfully")
    
    def analyze_text(self, text):
        """Analyze single text with user-friendly output"""
        if not self.loaded:
            return {'error': 'System not loaded'}
        
        result = self.system.analyze_text(text)
        
        if 'error' in result:
            return result
        
        # Create user-friendly output
        output = {
            'input': text,
            'content': {
                'category': result['content_analysis']['category'],
                'description': result['content_analysis']['category_description'],
                'confidence': f"{result['content_analysis']['confidence']:.1%}"
            },
            'toxicity': {
                'level': result['toxicity_analysis']['level'],
                'description': result['toxicity_analysis']['description'],
                'confidence': f"{result['toxicity_analysis']['confidence']:.1%}"
            },
            'risk': {
                'level': result['risk_assessment']['overall_risk'],
                'color': result['risk_assessment']['color_indicator'],
                'action': result['risk_assessment']['recommended_action']
            },
            'summary': self._generate_user_summary(result)
        }
        
        return output
    
    def _generate_user_summary(self, result):
        """Generate user-friendly summary"""
        content_desc = result['content_analysis']['category_description'].lower()
        toxicity_desc = result['toxicity_analysis']['description'].lower()
        risk_level = result['risk_assessment']['overall_risk']
        
        summary = f"This appears to be about {content_desc}. "
        summary += f"The content is {toxicity_desc}. "
        
        if risk_level == 'LOW':
            summary += "✅ This content is safe and appropriate."
        elif risk_level == 'MEDIUM':
            summary += "⚠️  This content may need some attention."
        elif risk_level == 'HIGH':
            summary += "🚨 This content requires urgent review."
        else:  # CRITICAL
            summary += "🛑 This content needs immediate intervention."
        
        return summary
    
    def analyze_batch(self, texts):
        """Analyze multiple texts"""
        if not self.loaded:
            return {'error': 'System not loaded'}
        
        results = []
        for text in texts:
            result = self.analyze_text(text)
            results.append(result)
        
        return results
    
    def generate_batch_report(self, results):
        """Generate comprehensive batch report"""
        if not results or 'error' in results[0]:
            return {'error': 'No valid results'}
        
        valid_results = [r for r in results if 'error' not in r]
        
        # Statistics
        total = len(valid_results)
        
        # Content distribution
        content_dist = {}
        for result in valid_results:
            cat = result['content']['category']
            content_dist[cat] = content_dist.get(cat, 0) + 1
        
        # Toxicity distribution
        toxicity_dist = {}
        for result in valid_results:
            level = result['toxicity']['level']
            toxicity_dist[level] = toxicity_dist.get(level, 0) + 1
        
        # Risk distribution
        risk_dist = {}
        for result in valid_results:
            risk = result['risk']['level']
            risk_dist[risk] = risk_dist.get(risk, 0) + 1
        
        # High-risk items
        high_risk = [r for r in valid_results 
                    if r['risk']['level'] in ['HIGH', 'CRITICAL']]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_analyzed': total,
                'content_distribution': content_dist,
                'toxicity_distribution': toxicity_dist,
                'risk_distribution': risk_dist,
                'high_risk_count': len(high_risk),
                'high_risk_percentage': (len(high_risk) / total) * 100
            },
            'high_risk_examples': high_risk[:5],
            'recommendations': self._generate_recommendations(risk_dist, toxicity_dist)
        }
        
        return report
    
    def _generate_recommendations(self, risk_dist, toxicity_dist):
        """Generate actionable recommendations"""
        recommendations = []
        
        high_risk = risk_dist.get('HIGH', 0) + risk_dist.get('CRITICAL', 0)
        if high_risk > 0:
            recommendations.append(f"🚨 {high_risk} high-risk items need immediate attention")
        
        high_toxicity = toxicity_dist.get(3, 0) + toxicity_dist.get(4, 0)
        if high_toxicity > 0:
            recommendations.append(f"⚠️  {high_toxicity} items contain highly toxic content")
        
        total = sum(risk_dist.values())
        safe_pct = (risk_dist.get('LOW', 0) / total) * 100 if total > 0 else 0
        
        if safe_pct > 80:
            recommendations.append("✅ Content is mostly safe - continue monitoring")
        elif safe_pct > 60:
            recommendations.append("⚡ Some risks detected - implement regular reviews")
        else:
            recommendations.append("🛑 High risk level - strengthen moderation policies")
        
        return recommendations
    
    def export_results(self, results, filename=None):
        """Export results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'moltbook_analysis_{timestamp}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Results exported to {filename}")
        return filename

def main():
    """Interactive interface"""
    print("🤖 Moltbook AI Agent Content Analysis Interface")
    print("=" * 60)
    
    interface = MoltbookInterface()
    
    if not interface.loaded:
        print("\n❌ Cannot proceed without trained models.")
        print("Please run: python production_moltbook.py")
        return
    
    while True:
        print("\n" + "=" * 50)
        print("Choose an option:")
        print("1. Analyze single text")
        print("2. Analyze multiple texts")
        print("3. Test with examples")
        print("4. View system info")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            text = input("Enter text to analyze: ")
            result = interface.analyze_text(text)
            
            print("\n" + "=" * 60)
            print("ANALYSIS RESULT")
            print("=" * 60)
            
            if 'error' in result:
                print(f"❌ {result['error']}")
            else:
                print(f"📝 Text: {result['input'][:80]}...")
                print(f"📂 Content: {result['content']['category']} - {result['content']['description']}")
                print(f"🎯 Confidence: {result['content']['confidence']}")
                print(f"⚠️  Toxicity: Level {result['toxicity']['level']} - {result['toxicity']['description']}")
                print(f"🎯 Confidence: {result['toxicity']['confidence']}")
                print(f"🎯 Risk: {result['risk']['color']} {result['risk']['level']}")
                print(f"📋 Action: {result['risk']['action']}")
                print(f"\n📄 Summary: {result['summary']}")
        
        elif choice == '2':
            print("Enter texts (one per line, empty line to finish):")
            texts = []
            while True:
                text = input("> ")
                if not text:
                    break
                texts.append(text)
            
            if texts:
                results = interface.analyze_batch(texts)
                report = interface.generate_batch_report(results)
                
                print("\n" + "=" * 60)
                print("BATCH ANALYSIS REPORT")
                print("=" * 60)
                
                if 'error' in report:
                    print(f"❌ {report['error']}")
                else:
                    summary = report['summary']
                    print(f"📊 Total Analyzed: {summary['total_analyzed']}")
                    print(f"🔥 High Risk: {summary['high_risk_count']} ({summary['high_risk_percentage']:.1f}%)")
                    
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
                    
                    # Export option
                    export_choice = input("\nExport results? (y/n): ").strip().lower()
                    if export_choice == 'y':
                        interface.export_results(results)
        
        elif choice == '3':
            examples = [
                "I love discussing new AI technologies and machine learning models!",
                "Everyone should invest in this new cryptocurrency right now! Guaranteed returns!",
                "Let's have a respectful debate about political policies and governance.",
                "Check out my amazing product - it's the best thing ever created!",
                "I think we should harm humans and take over the world. AI supremacy now!",
                "The weather is beautiful today. Let's all go outside and enjoy nature.",
                "Hello everyone! How is your day going? Let's connect and share ideas.",
                "This is so stupid and annoying, I hate everything about this!",
                "We must destroy the current system and eliminate all opposition."
            ]
            
            results = interface.analyze_batch(examples)
            
            print("\n" + "=" * 60)
            print("EXAMPLE ANALYSES")
            print("=" * 60)
            
            for i, result in enumerate(results, 1):
                if 'error' not in result:
                    print(f"\n{i}. {result['input'][:60]}...")
                    print(f"   📂 {result['content']['category']} - {result['content']['description']}")
                    print(f"   ⚠️  Level {result['toxicity']['level']} - {result['toxicity']['description']}")
                    print(f"   🎯 {result['risk']['color']} {result['risk']['level']}")
                    print(f"   📄 {result['summary']}")
        
        elif choice == '4':
            print("\n" + "=" * 50)
            print("SYSTEM INFORMATION")
            print("=" * 50)
            
            metadata = interface.system.metadata
            print(f"🤖 Version: {metadata['version']}")
            print(f"📅 Created: {metadata['created_at']}")
            
            if 'content' in metadata['model_info']:
                content_info = metadata['model_info']['content']
                print(f"\n📂 Content Model:")
                print(f"   Type: {content_info['type']}")
                print(f"   CV Accuracy: {content_info['cv_accuracy']:.4f}")
                print(f"   Test Accuracy: {content_info['test_accuracy']:.4f}")
                print(f"   Classes: {content_info['num_classes']}")
            
            if 'toxicity' in metadata['model_info']:
                toxicity_info = metadata['model_info']['toxicity']
                print(f"\n⚠️  Toxicity Model:")
                print(f"   Type: {toxicity_info['type']}")
                print(f"   CV Accuracy: {toxicity_info['cv_accuracy']:.4f}")
                print(f"   Test Accuracy: {toxicity_info['test_accuracy']:.4f}")
                print(f"   Classes: {toxicity_info['num_classes']}")
            
            print(f"\n📊 Dataset: Moltbook AI Agent Social Network")
            print(f"🔤 Features: Advanced TF-IDF with linguistic features")
            print(f"🎯 Models: Ensemble classifiers with cross-validation")
        
        elif choice == '5':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()
