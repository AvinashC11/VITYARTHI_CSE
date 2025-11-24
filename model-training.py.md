"""
Model Training Module for Email Spam Detection
Trains and saves multiple ML classifiers.
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import pandas as pd


class SpamClassifier:
    """
    Trains and manages spam classification models.
    """
    
    def __init__(self, model_type='naive_bayes'):
        """
        Initialize classifier with specified model type.
        
        Args:
            model_type (str): Type of model ('naive_bayes', 'logistic_regression', 'svm')
        """
        self.model_type = model_type
        self.model = self._initialize_model()
        self.is_trained = False
    
    def _initialize_model(self):
        """
        Initialize the ML model based on type.
        
        Returns:
            sklearn model: Initialized model
        """
        if self.model_type == 'naive_bayes':
            return MultinomialNB(alpha=1.0)
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == 'svm':
            return SVC(kernel='linear', probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train, verbose=True):
        """
        Train the model on training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            verbose (bool): Print training progress
        """
        if verbose:
            print(f"Training {self.model_type} model...")
            print(f"Training samples: {X_train.shape[0]}")
            print(f"Features: {X_train.shape[1]}")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        if verbose:
            print("Training completed!")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            array: Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            array: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Performance metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, pos_label='spam'),
            'recall': recall_score(y_test, predictions, pos_label='spam'),
            'f1_score': f1_score(y_test, predictions, pos_label='spam')
        }
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv (int): Number of folds
            
        Returns:
            dict: Cross-validation scores
        """
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores
        }
    
    def get_confusion_matrix(self, X_test, y_test):
        """
        Generate confusion matrix.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            array: Confusion matrix
        """
        predictions = self.predict(X_test)
        return confusion_matrix(y_test, predictions, labels=['ham', 'spam'])
    
    def save_model(self, filepath='models/spam_classifier.pkl'):
        """
        Save trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Create directory if doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/spam_classifier.pkl'):
        """
        Load trained model from disk.
        
        Args:
            filepath (str): Path to saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")


def train_and_compare_models(X_train, X_test, y_train, y_test):
    """
    Train and compare multiple models.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        
    Returns:
        dict: Comparison results
    """
    models = ['naive_bayes', 'logistic_regression', 'svm']
    results = {}
    
    for model_type in models:
        print(f"\n{'='*60}")
        print(f"Training {model_type}...")
        print('='*60)
        
        classifier = SpamClassifier(model_type=model_type)
        classifier.train(X_train, y_train)
        
        # Evaluate
        metrics = classifier.evaluate(X_test, y_test)
        results[model_type] = metrics
        
        print(f"\nPerformance Metrics:")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    return results


# Example usage
if __name__ == "__main__":
    # This would normally use actual data
    print("Model Training Module")
    print("This module should be imported and used with actual data")
    print("\nSupported models:")
    print("- naive_bayes")
    print("- logistic_regression")
    print("- svm")
