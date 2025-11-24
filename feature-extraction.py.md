"""
Feature Extraction Module for Email Spam Detection
Implements TF-IDF vectorization for text feature extraction.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os


class FeatureExtractor:
    """
    Extracts numerical features from preprocessed text using TF-IDF.
    """
    
    def __init__(self, max_features=3000, ngram_range=(1, 2), min_df=2):
        """
        Initialize the feature extractor.
        
        Args:
            max_features (int): Maximum number of features to extract
            ngram_range (tuple): Range of n-grams to consider
            min_df (int): Minimum document frequency for terms
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            lowercase=True,
            strip_accents='unicode'
        )
        
        self.is_fitted = False
    
    def fit_transform(self, texts):
        """
        Fit the vectorizer and transform texts to features.
        
        Args:
            texts (list): List of preprocessed text strings
            
        Returns:
            sparse matrix: TF-IDF feature matrix
        """
        features = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return features
    
    def transform(self, texts):
        """
        Transform texts to features using fitted vectorizer.
        
        Args:
            texts (list): List of preprocessed text strings
            
        Returns:
            sparse matrix: TF-IDF feature matrix
            
        Raises:
            ValueError: If vectorizer is not fitted yet
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform. Use fit_transform first.")
        
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """
        Get the feature names (vocabulary terms).
        
        Returns:
            list: Feature names
        """
        if not self.is_fitted:
            return []
        return self.vectorizer.get_feature_names_out()
    
    def get_vocabulary_size(self):
        """
        Get the size of the vocabulary.
        
        Returns:
            int: Vocabulary size
        """
        if not self.is_fitted:
            return 0
        return len(self.vectorizer.vocabulary_)
    
    def save_vectorizer(self, filepath='models/vectorizer.pkl'):
        """
        Save the fitted vectorizer to disk.
        
        Args:
            filepath (str): Path to save the vectorizer
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted vectorizer")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(self.vectorizer, filepath)
        print(f"Vectorizer saved to {filepath}")
    
    def load_vectorizer(self, filepath='models/vectorizer.pkl'):
        """
        Load a fitted vectorizer from disk.
        
        Args:
            filepath (str): Path to the saved vectorizer
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vectorizer file not found: {filepath}")
        
        self.vectorizer = joblib.load(filepath)
        self.is_fitted = True
        print(f"Vectorizer loaded from {filepath}")
    
    def get_top_features(self, n=20):
        """
        Get the top n features by average TF-IDF score.
        
        Args:
            n (int): Number of top features to return
            
        Returns:
            list: Top feature names
        """
        if not self.is_fitted:
            return []
        
        feature_names = self.get_feature_names()
        return feature_names[:min(n, len(feature_names))]


# Example usage
if __name__ == "__main__":
    # Sample preprocessed texts
    sample_texts = [
        "click win prize free money",
        "meeting tomorrow discuss project deadline",
        "congratulations selected winner claim prize",
        "please review attached document feedback",
        "urgent action required account verification"
    ]
    
    # Initialize feature extractor
    extractor = FeatureExtractor(max_features=50, ngram_range=(1, 2))
    
    # Fit and transform
    features = extractor.fit_transform(sample_texts)
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Vocabulary size: {extractor.get_vocabulary_size()}")
    print(f"\nTop 10 features:")
    for feature in extractor.get_top_features(10):
        print(f"  - {feature}")
    
    # Transform new text
    new_text = ["win free prize now"]
    new_features = extractor.transform(new_text)
    print(f"\nNew text feature shape: {new_features.shape}")
