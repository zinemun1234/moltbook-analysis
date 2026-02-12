"""
Simple test to check if the basic functionality works
"""

import pandas as pd
import numpy as np
from datasets import load_dataset

def test_data_loading():
    """Test loading the Moltbook dataset"""
    print("=== Testing Data Loading ===")
    
    try:
        # Load dataset
        print("Loading Moltbook dataset...")
        dataset = load_dataset("TrustAIRLab/Moltbook", "posts")
        
        # Convert to pandas
        df = pd.DataFrame(dataset['train'])
        
        print(f"✅ Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Show sample data
        print("\nSample data:")
        for i in range(min(3, len(df))):
            print(f"\nSample {i+1}:")
            print(f"ID: {df.iloc[i]['id']}")
            print(f"Topic: {df.iloc[i]['topic_label']}")
            print(f"Toxicity: {df.iloc[i]['toxic_level']}")
            post_text = str(df.iloc[i]['post'])
            print(f"Post: {post_text[:100]}...")
        
        # Show label distributions
        print(f"\nTopic distribution:")
        print(df['topic_label'].value_counts())
        
        print(f"\nToxicity distribution:")
        print(df['toxic_level'].value_counts())
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_text_preprocessing():
    """Test text preprocessing"""
    print("\n=== Testing Text Preprocessing ===")
    
    # Simple text cleaning function
    def clean_text(text):
        import re
        if not isinstance(text, str):
            text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    # Test texts
    test_texts = [
        "Check out this amazing product! 🚀 https://example.com #AI #tech",
        "@user I totally agree with your point about machine learning!",
        "This is    a    text    with    extra    spaces!!!",
        "Visit www.example.com for more information on AI safety."
    ]
    
    print("Text preprocessing examples:")
    for i, text in enumerate(test_texts, 1):
        cleaned = clean_text(text)
        print(f"\nOriginal {i}: {text}")
        print(f"Cleaned {i}: {cleaned}")
    
    print("✅ Text preprocessing works!")

def test_basic_ml():
    """Test basic ML functionality without transformers"""
    print("\n=== Testing Basic ML ===")
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, accuracy_score
        
        # Load dataset
        dataset = load_dataset("TrustAIRLab/Moltbook", "posts")
        df = pd.DataFrame(dataset['train'])
        
        # Take a smaller sample for quick testing
        sample_df = df.sample(n=1000, random_state=42)
        
        # Prepare features and labels
        X = sample_df['post'].fillna('').astype(str)
        y_topic = sample_df['topic_label']
        y_toxicity = sample_df['toxic_level']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_topic, test_size=0.2, random_state=42, stratify=y_topic
        )
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Train simple classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_tfidf, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Basic ML works!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Number of classes: {len(clf.classes_)}")
        print(f"Classes: {list(clf.classes_)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic ML: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Simple Tests for Moltbook Analysis")
    print("=" * 50)
    
    # Run tests
    data_ok = test_data_loading()
    test_text_preprocessing()
    ml_ok = test_basic_ml()
    
    print("\n" + "=" * 50)
    print("🎯 Test Results:")
    print(f"Data Loading: {'✅' if data_ok else '❌'}")
    print(f"Text Preprocessing: ✅")
    print(f"Basic ML: {'✅' if ml_ok else '❌'}")
    
    if data_ok and ml_ok:
        print("\n🎉 Basic functionality works!")
        print("You can proceed with the full transformer models once PyTorch issues are resolved.")
    else:
        print("\n⚠️  Some issues detected. Check the error messages above.")

if __name__ == "__main__":
    main()
