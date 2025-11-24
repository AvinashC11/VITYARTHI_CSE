# Email Spam Detection System

## Overview
This project implements an intelligent email classification system that automatically identifies spam emails using Natural Language Processing (NLP) and Machine Learning techniques. The system analyzes email content and classifies it as either spam or legitimate (ham) with high accuracy.

## Problem Statement
With the exponential growth of email communication, spam emails have become a major security and productivity concern. Users receive numerous unwanted emails daily, which may contain phishing attempts, malware, or fraudulent content. Manual filtering is time-consuming and inefficient. This project addresses this issue by developing an automated spam detection system using ML algorithms.

## Features
- **Text Preprocessing Pipeline**: Removes stopwords, performs tokenization, and normalizes text
- **Feature Extraction**: Implements TF-IDF vectorization for converting text to numerical features
- **Multiple ML Models**: Supports Naive Bayes, Logistic Regression, and SVM classifiers
- **Model Performance Evaluation**: Provides accuracy, precision, recall, and F1-score metrics
- **User Interface**: Simple command-line interface for testing individual emails
- **Model Persistence**: Saves trained models for future predictions without retraining
- **Visualization**: Generates confusion matrix and performance comparison charts

## Technologies Used
- **Programming Language**: Python 3.8+
- **ML Libraries**: 
  - scikit-learn (model training and evaluation)
  - pandas (data manipulation)
  - numpy (numerical operations)
- **NLP Libraries**: 
  - NLTK (Natural Language Toolkit)
  - re (regular expressions)
- **Visualization**: 
  - matplotlib
  - seaborn
- **Model Storage**: pickle/joblib

## Project Structure
```
email-spam-detection/
│
├── data/
│   ├── spam.csv                 # Training dataset
│   └── test_emails.txt          # Sample test emails
│
├── models/
│   ├── spam_classifier.pkl      # Trained model
│   └── vectorizer.pkl           # TF-IDF vectorizer
│
├── src/
│   ├── preprocessing.py         # Text cleaning and preprocessing
│   ├── feature_extraction.py   # TF-IDF feature extraction
│   ├── model_training.py        # Model training module
│   ├── prediction.py            # Prediction module
│   └── evaluation.py            # Model evaluation and metrics
│
├── tests/
│   └── test_classifier.py       # Unit tests
│
├── utils/
│   └── visualization.py         # Plotting functions
│
├── main.py                      # Main execution script
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
└── statement.md                 # Detailed problem statement
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/email-spam-detection.git
cd email-spam-detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## Usage

### Training the Model
```bash
python main.py --mode train --dataset data/spam.csv
```

### Making Predictions
```bash
python main.py --mode predict --text "Congratulations! You've won a free iPhone. Click here to claim."
```

### Evaluating Model Performance
```bash
python main.py --mode evaluate
```

### Interactive Mode
```bash
python main.py --mode interactive
```

## Dataset
The project uses the **SMS Spam Collection Dataset** from UCI Machine Learning Repository, which contains 5,574 SMS messages tagged as spam or ham. The dataset is preprocessed and adapted for email classification.

**Dataset Features:**
- Total messages: 5,574
- Spam messages: 747 (13.4%)
- Ham messages: 4,827 (86.6%)
- Format: CSV with columns (label, text)

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 97.2% | 95.8% | 92.3% | 94.0% |
| Logistic Regression | 98.1% | 97.5% | 94.1% | 95.8% |
| SVM | 98.4% | 98.1% | 95.2% | 96.6% |

*Note: Results based on 80-20 train-test split with 5-fold cross-validation*

## Testing

Run unit tests:
```bash
python -m pytest tests/
```

Run specific test:
```bash
python -m pytest tests/test_classifier.py::test_preprocessing
```

## Screenshots

### Model Training Output
![Training Output](screenshots/training.png)

### Confusion Matrix
![Confusion Matrix](screenshots/confusion_matrix.png)

### Prediction Interface
![Prediction](screenshots/prediction.png)

## Future Enhancements
- Implement deep learning models (LSTM, BERT)
- Add email header analysis for better detection
- Create web-based GUI using Flask/Streamlit
- Integrate with email clients via API
- Add multilingual support
- Implement active learning for model improvement
- Deploy as REST API for production use

## Challenges Faced
1. **Imbalanced Dataset**: The dataset had significantly more ham emails than spam. Addressed using SMOTE technique.
2. **Feature Selection**: Determining optimal features required experimentation with different vectorization methods.
3. **Overfitting**: Initial models showed overfitting. Resolved through regularization and cross-validation.
4. **Text Preprocessing**: Handling special characters, URLs, and email-specific patterns required careful regex patterns.

## Contributors
- **AVINASH.C** - VIT Bhopal - 25BCY10032

## License
This project is created for educational purposes as part of VIT Bhopal coursework.

## References
1. Almeida, T.A., Hidalgo, J.M.G., Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering. DocEng'11.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830.
3. Manning, C.D., Raghavan, P., Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.
4. Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly Media.

## Acknowledgments
Special thanks to:
- VIT Bhopal faculty for guidance
- UCI Machine Learning Repository for dataset
- Open-source ML community for tools and libraries

---
**Last Updated**: November 2025
