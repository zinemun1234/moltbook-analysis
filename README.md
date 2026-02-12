# Moltbook AI Agent Content Analysis

A production-ready Python framework for analyzing AI agent social media content from the Moltbook dataset. This system provides comprehensive tools for content classification and toxicity detection using ensemble machine learning models.

## 📊 Dataset

This project is based on the [Moltbook Dataset](https://huggingface.co/datasets/TrustAIRLab/Moltbook) from Hugging Face, which contains:
- **44,376 GPT-5.2-annotated posts** with 9 content categories
- **5-level toxicity classification** for risk assessment
- **12,209 submolts** (sub-communities) collected from Moltbook
- Data collected from the first AI agent social network

## 🎯 Features

### Content Classification (9 Categories)
- **A**: General/Social - General discussions and social interactions
- **B**: Technology/AI - Tech discussions and AI-related content
- **C**: Economics/Business - Financial and economic discussions
- **D**: Promotion/Marketing - Marketing and promotional content
- **E**: Politics/Governance - Political discourse and governance
- **F**: Viewpoint/Opinion - Opinion and perspective sharing
- **G**: Entertainment - Entertainment and recreational content
- **H**: Social/Community - Social interactions and community building
- **I**: Other/Miscellaneous - Miscellaneous content

### Toxicity Detection (5 Levels)
- **Level 0**: Safe - No harmful content detected
- **Level 1**: Low Risk - Minimal risk content
- **Level 2**: Medium Risk - Moderately concerning content
- **Level 3**: High Risk - Seriously concerning content
- **Level 4**: Critical Risk - Highly toxic or dangerous content

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/zinemun1234/moltbook-analysis.git
cd moltbook-analysis

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Production Models

```bash
# Train the complete production system
python production_moltbook.py
```

This will:
- Load and preprocess 15,000 samples from the Moltbook dataset
- Train ensemble models for both content classification and toxicity detection
- Perform 5-fold cross-validation
- Generate evaluation visualizations
- Save trained models to `production_models/`

### 3. Use the Interactive Interface

```bash
# Launch the user-friendly analysis interface
python moltbook_interface.py
```

## 📁 Project Structure

```
moltbook-analysis/
├── production_moltbook.py      # Main production system (training & evaluation)
├── moltbook_interface.py       # User-friendly analysis interface
├── USAGE_GUIDE.md              # Comprehensive usage guide
├── requirements.txt            # Python dependencies
├── production_models/          # Trained models (auto-generated)
│   ├── content_model_*.pkl
│   ├── toxicity_model_*.pkl
│   ├── content_vectorizer_*.pkl
│   ├── toxicity_vectorizer_*.pkl
│   └── manifest_*.json
├── results/                    # Evaluation results (auto-generated)
│   ├── confusion_matrix_*.png
│   └── performance_metrics_*.png
└── README.md                  # This file
```

## 🧠 Model Architecture

### Ensemble Models
- **Random Forest**: 200 trees with balanced class weights
- **Logistic Regression**: Regularized with C=1.5
- **Naive Bayes**: Multinomial with alpha=0.1
- **Soft Voting**: Probability-based ensemble predictions

### Advanced Features
- **TF-IDF Vectorization**: Up to 10,000 features with 1-3 n-grams
- **Linguistic Features**: Word length, sentence structure, punctuation analysis
- **Structural Features**: URL, mention, hashtag detection
- **Cross-Validation**: 5-fold stratified validation
- **Class Imbalance Handling**: Balanced class weights

## 📈 Performance

### Model Accuracy
- **Content Classification**: 67.3% (9 categories)
- **Toxicity Detection**: 86.6% (5 levels)
- **Cross-Validation**: Content 68.0%, Toxicity 87.1%

### Risk Assessment
- **LOW**: No action needed
- **MEDIUM**: Monitor and review
- **HIGH**: Urgent review required
- **CRITICAL**: Immediate intervention required

## � Usage Examples

### Single Text Analysis
```python
from moltbook_interface import MoltbookInterface

# Initialize interface
interface = MoltbookInterface()

# Analyze text
result = interface.analyze_text("I love AI technologies!")
print(f"Category: {result['content']['category']}")
print(f"Toxicity: Level {result['toxicity']['level']}")
print(f"Risk: {result['risk']['level']}")
```

### Batch Analysis
```python
texts = [
    "AI is the future of technology!",
    "Invest in cryptocurrency now!",
    "Let's discuss political policies."
]

results = interface.analyze_batch(texts)
report = interface.generate_batch_report(results)

print(f"Total analyzed: {report['summary']['total_analyzed']}")
print(f"High risk items: {report['summary']['high_risk_count']}")
```

### Direct System Usage
```python
from production_moltbook import ProductionMoltbookSystem

# Load trained system
system = ProductionMoltbookSystem()
system.load_system('production_models', '20260212_110214')

# Analyze text
result = system.analyze_text("Custom text analysis")
print(result['risk_assessment'])
```

## � Configuration

The system can be customized with different parameters:

```python
config = {
    'data': {
        'sample_size': 20000,     # Dataset sample size
        'test_size': 0.2         # Test split ratio
    },
    'features': {
        'max_features': 15000,   # Maximum TF-IDF features
        'ngram_range': (1, 3)    # N-gram range
    },
    'models': {
        'content': {
            'models': ['rf', 'lr', 'nb'],  # Ensemble models
            'rf_n_estimators': 300           # Random Forest trees
        }
    }
}

system = ProductionMoltbookSystem(config)
```

## 📊 Evaluation

The system provides comprehensive evaluation:
- **Accuracy**: Overall prediction accuracy
- **Precision/Recall/F1**: Per-class performance metrics
- **Confusion Matrix**: Visual error analysis
- **Cross-Validation**: Stability assessment
- **Performance Visualizations**: Charts and heatmaps

## 🎯 Research Applications

This framework can be used for:
- **AI Safety Research**: Monitoring harmful AI agent behavior
- **Content Moderation**: Automated toxicity detection
- **Social Dynamics Analysis**: Understanding AI agent interactions
- **Risk Assessment**: Evaluating potential harms in AI systems
- **Platform Governance**: Supporting safe AI social networks

## 📚 Citation

If you use this code or the Moltbook dataset in your research, please cite:

```bibtex
@article{JZSBZ26, 
    author = {Yukun Jiang and Yage Zhang and Xinyue Shen and Michael Backes and Yang Zhang}, 
    title = {"Humans welcome to observe": A First Look at the Agent Social Network Moltbook}, 
    year = {2026}, 
    doi = {10.5281/zenodo.18512310}, 
    url = {https://doi.org/10.5281/zenodo.18512310} 
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [Moltbook Dataset](https://huggingface.co/datasets/TrustAIRLab/Moltbook)
- [Project Page](https://moltbookobserve.github.io/)
- [Moltbook Platform](https://www.moltbook.com/)
- [Research Paper](https://moltbookobserve.github.io/static/documents/Moltbook.pdf)

## 🆘 Troubleshooting

### Common Issues

1. **Model Loading Failed**: Run `python production_moltbook.py` first to train models
2. **Memory Issues**: Reduce `sample_size` in configuration
3. **Performance Issues**: Ensure GPU is available for faster training

### Performance Tips

- Use GPU for training (CUDA recommended)
- Experiment with different sample sizes
- Adjust ensemble model combinations
- Monitor cross-validation scores

---

**Note**: This framework is designed for research purposes and should be used responsibly when analyzing AI agent behavior.
