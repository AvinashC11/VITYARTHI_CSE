"""
Text Preprocessing Module for Email Spam Detection
This module handles text cleaning and preprocessing operations.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK resources (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class TextPreprocessor:
    """
    Handles all text preprocessing operations for email spam detection.
    """
    
    def __init__(self, use_stemming=True):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            use_stemming (bool): Whether to apply stemming to tokens
        """
        self.use_stemming = use_stemming
        self.stemmer = PorterStemmer() if use_stemming else None
        self.stop_words = set(stopwords.words('english'))
        
    def remove_urls(self, text):
        """
        Remove URLs from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with URLs removed
        """
        # Pattern matches http, https, and www URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, '', text)
        text = re.sub(r'www\.[a-zA-Z0-9]+\.[a-z]+', '', text)
        return text
    
    def remove_email_addresses(self, text):
        """
        Remove email addresses from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with email addresses removed
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    def remove_special_chars(self, text):
        """
        Remove special characters and punctuation.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with special characters removed
        """
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def remove_numbers(self, text):
        """
        Remove standalone numbers from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with numbers removed
        """
        return re.sub(r'\b\d+\b', '', text)
    
    def tokenize(self, text):
        """
        Tokenize text into words.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tokens
        """
        return word_tokenize(text.lower())
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from token list.
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Filtered token list
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def apply_stemming(self, tokens):
        """
        Apply stemming to tokens.
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Stemmed tokens
        """
        if self.stemmer:
            return [self.stemmer.stem(token) for token in tokens]
        return tokens
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline.
        
        Args:
            text (str): Raw email text
            
        Returns:
            str: Preprocessed text ready for feature extraction
        """
        # Step 1: Remove URLs and emails
        text = self.remove_urls(text)
        text = self.remove_email_addresses(text)
        
        # Step 2: Remove special characters
        text = self.remove_special_chars(text)
        
        # Step 3: Remove numbers
        text = self.remove_numbers(text)
        
        # Step 4: Tokenize
        tokens = self.tokenize(text)
        
        # Step 5: Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Step 6: Apply stemming if enabled
        if self.use_stemming:
            tokens = self.apply_stemming(tokens)
        
        # Step 7: Join tokens back to string
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    
    def batch_preprocess(self, texts):
        """
        Preprocess multiple texts.
        
        Args:
            texts (list): List of raw email texts
            
        Returns:
            list: List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


# Example usage
if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor(use_stemming=True)
    
    sample_email = """
    Congratulations! You've WON a FREE iPhone 15 Pro!!! 
    Click here: http://fakewebsite.com/claim?id=12345
    Contact us at winner@scam.com or call 1-800-123-4567
    Act NOW before this offer expires!!!
    """
    
    print("Original Email:")
    print(sample_email)
    print("\n" + "="*60 + "\n")
    
    processed = preprocessor.preprocess(sample_email)
    print("Preprocessed Email:")
    print(processed)
    print("\n" + "="*60 + "\n")
    
    # Show intermediate steps
    print("Step-by-step processing:")
    text = sample_email
    print(f"1. After URL removal: {preprocessor.remove_urls(text)[:100]}...")
    print(f"2. After email removal: {preprocessor.remove_email_addresses(preprocessor.remove_urls(text))[:100]}...")
    print(f"3. Tokenized: {preprocessor.tokenize(text)[:15]}...")
