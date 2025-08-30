# NLP Question Answering: A Comparative Study

A comprehensive Natural Language Processing pipeline comparing traditional machine learning approaches with modern transformer architectures for question classification and answering tasks.

## 🎯 Project Overview

This project implements and evaluates multiple NLP approaches for question answering systems using the Stanford Question Answering Dataset (SQuAD). It provides a systematic comparison between traditional ML methods and state-of-the-art transformer models.

## ✨ Key Features

- **Comprehensive NLP Pipeline**: End-to-end implementation from data preprocessing to model evaluation
- **Multi-Method Comparison**: Traditional ML (TF-IDF, Word2Vec, FastText) vs Transformers (DistilBERT)
- **Question Classification**: Automatic categorization into semantic types (what, who, when, where, how, why)
- **Question Answering**: Extract precise answers from textual contexts
- **Advanced Visualizations**: t-SNE plots, performance heatmaps, confusion matrices
- **OOV Handling**: Comprehensive out-of-vocabulary word analysis using FastText
- **Production Ready**: 98% accuracy achieved with DistilBERT implementation


## 📊 Results Summary

| Model | Feature Type | Accuracy | Performance Category |
|-------|-------------|----------|---------------------|
| DistilBERT QA | Transformer | **98.00%** | State-of-the-art |
| Keyword Baseline | Rule-based | 95.87% | Baseline |
| Random Forest | TF-IDF | 76.40% | Traditional ML |
| Random Forest | Word2Vec | 71.07% | Traditional ML |
| SVM Linear | TF-IDF | 62.12% | Traditional ML |

### Key Findings
- **DistilBERT** achieved exceptional 98% accuracy for question answering
- **TF-IDF** consistently outperformed word embeddings for traditional ML (66.35% avg vs 53% avg)
- **FastText** successfully handled all out-of-vocabulary words through subword information
- **Class imbalance** in question types (67% "what" questions) reflects real-world distributions


## 🔧 Technical Implementation

### Data Processing
- **Dataset**: Stanford Question Answering Dataset (SQuAD)
- **Sample Size**: 20,000 examples for optimal performance
- **Preprocessing**: Tokenization, stemming, stopword removal, contraction expansion

### Feature Extraction
- **TF-IDF**: 3,000 features with n-gram range (1-3)
- **Word2Vec**: 150-dimensional vectors, skip-gram architecture
- **FastText**: 150-dimensional vectors with character n-grams (3-6)

### Models Implemented
- **Baselines**: Keyword overlap, length-based classification
- **Traditional ML**: Logistic Regression, SVM, Random Forest
- **Transformers**: DistilBERT for Question Answering

### Evaluation Metrics
- **Classification**: Accuracy, F1-score (weighted & macro)
- **Question Answering**: Token-level F1-score, confidence scoring
- **Visualizations**: Performance comparisons, embedding analysis, confusion matrices

## 📈 Visualizations

The project includes comprehensive visualizations:
- Model performance comparison charts
- Feature-model performance heatmaps
- t-SNE embeddings visualization
- Question type distribution analysis
- QA confidence analysis

## 🎓 Educational Value

This project demonstrates:
- **Complete NLP Pipeline**: From data loading to model deployment
- **Comparative Analysis**: Traditional vs modern approaches
- **Best Practices**: Proper evaluation, visualization, documentation
- **Real-world Considerations**: OOV handling, class imbalance, computational constraints

