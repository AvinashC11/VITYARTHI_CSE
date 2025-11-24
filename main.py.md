"""
Main execution script for Email Spam Detection System
Handles CLI interface and workflow coordination.
"""

import argparse
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import TextPreprocessor
from feature_extraction import FeatureExtractor
from model_training import SpamClassifier, train_and_compare_models
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(filepath='data/spam.csv'):
    """
    Load the spam dataset.
    
    Args:
        filepath (str): Path to dataset
        
    Returns:
        tuple: (texts, labels)
    """
    try:
        df = pd.read_csv(filepath, encoding='latin-1')
        # Assume columns are named 'label' and 'text'
        # Adjust column names based on your dataset
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df[['v1', 'v2']]
            df.columns = ['label', 'text']
        
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        print(f"Dataset loaded: {len(texts)} emails")
        print(f"Spam: {labels.count('spam')}, Ham: {labels.count('ham')}")
        
        return texts, labels
    except FileNotFoundError:
        print(f"Error: Dataset not found at {filepath}")
        print("Please download the dataset and place it in the data/ folder")
        sys.exit(1)


def train_model(dataset_path, model_type='logistic_regression'):
    """
    Train a spam classification model.
    
    Args:
        dataset_path (str): Path to training dataset
        model_type (str): Type of model to train
    """
    print("="*70)
    print("EMAIL SPAM DETECTION - MODEL TRAINING")
    print("="*70)
    
    # Load data
    texts, labels = load_dataset(dataset_path)
    
    # Preprocess
    print("\nPreprocessing emails...")
    preprocessor = TextPreprocessor(use_stemming=True)
    processed_texts = preprocessor.batch_preprocess(texts)
    
    # Extract features
    print("Extracting features...")
    extractor = FeatureExtractor(max_features=3000, ngram_range=(1, 2))
    features = extractor.fit_transform(processed_texts)
    
    # Save vectorizer
    extractor.save_vectorizer()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    classifier = SpamClassifier(model_type=model_type)
    classifier.train(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = classifier.evaluate(X_test, y_test)
    
    print("\n" + "="*70)
    print("MODEL PERFORMANCE")
    print("="*70)
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {metrics['f1_score']*100:.2f}%")
    
    # Save model
    classifier.save_model()
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)


def predict_email(text):
    """
    Predict if a single email is spam.
    
    Args:
        text (str): Email text
    """
    # Load components
    preprocessor = TextPreprocessor(use_stemming=True)
    extractor = FeatureExtractor()
    extractor.load_vectorizer()
    
    classifier = SpamClassifier()
    classifier.load_model()
    
    # Process and predict
    processed = preprocessor.preprocess(text)
    features = extractor.transform([processed])
    prediction = classifier.predict(features)[0]
    probabilities = classifier.predict_proba(features)[0]
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"\nEmail Text (truncated):\n{text[:200]}...")
    print("\n" + "-"*70)
    print(f"Classification: {prediction.upper()}")
    print(f"Confidence: {max(probabilities)*100:.2f}%")
    print(f"Spam probability: {probabilities[1]*100:.2f}%")
    print(f"Ham probability: {probabilities[0]*100:.2f}%")
    print("="*70)


def interactive_mode():
    """
    Interactive mode for testing emails.
    """
    print("="*70)
    print("EMAIL SPAM DETECTION - INTERACTIVE MODE")
    print("="*70)
    print("\nEnter email text to check if it's spam.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Load components
    try:
        preprocessor = TextPreprocessor(use_stemming=True)
        extractor = FeatureExtractor()
        extractor.load_vectorizer()
        
        classifier = SpamClassifier()
        classifier.load_model()
    except FileNotFoundError:
        print("Error: Model files not found. Please train the model first.")
        print("Run: python main.py --mode train --dataset data/spam.csv")
        return
    
    while True:
        print("\n" + "-"*70)
        user_input = input("\nEnter email text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '']:
            print("\nExiting interactive mode. Goodbye!")
            break
        
        # Process and predict
        processed = preprocessor.preprocess(user_input)
        features = extractor.transform([processed])
        prediction = classifier.predict(features)[0]
        probabilities = classifier.predict_proba(features)[0]
        
        # Display result
        print(f"\n{'🚨 SPAM' if prediction == 'spam' else '✅ LEGITIMATE'}")
        print(f"Confidence: {max(probabilities)*100:.2f}%")


def main():
    """
    Main entry point for the application.
    """
    parser = argparse.ArgumentParser(
        description='Email Spam Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict', 'interactive', 'evaluate'],
        default='interactive',
        help='Operation mode'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/spam.csv',
        help='Path to training dataset'
    )
    
    parser.add_argument(
        '--text',
        type=str,
        default='',
        help='Email text for prediction'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['naive_bayes', 'logistic_regression', 'svm'],
        default='logistic_regression',
        help='Model type for training'
    )
    
    args = parser.parse_args()
    
    # Execute based on mode
    if args.mode == 'train':
        train_model(args.dataset, args.model)
    elif args.mode == 'predict':
        if not args.text:
            print("Error: --text argument required for prediction mode")
            sys.exit(1)
        predict_email(args.text)
    elif args.mode == 'interactive':
        interactive_mode()
    elif args.mode == 'evaluate':
        print("Evaluation mode - comparing all models")
        texts, labels = load_dataset(args.dataset)
        preprocessor = TextPreprocessor(use_stemming=True)
        processed = preprocessor.batch_preprocess(texts)
        extractor = FeatureExtractor()
        features = extractor.fit_transform(processed)
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        results = train_and_compare_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
