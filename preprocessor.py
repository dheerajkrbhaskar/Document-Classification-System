import re
from config import STOPWORDS

class TextPreprocessor:
    @staticmethod
    def preprocess_text(text):
        """
        Preprocess the input text by:
        1. Converting to lowercase
        2. Removing punctuation
        3. Removing stopwords
        4. Tokenizing
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            list: List of preprocessed tokens
        """
        # Convert text to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split text into tokens (words)
        tokens = text.split()
        
        # Remove common stopwords
        tokens = [token for token in tokens if token not in STOPWORDS]
        
        return tokens
    
    @classmethod
    def preprocess_documents(cls, documents):
        """
        Preprocess a list of documents
        
        Args:
            documents (list): List of text documents
            
        Returns:
            list: List of preprocessed document tokens
        """
        # Process each document and return the preprocessed tokens
        processed_docs = []
        for doc in documents:
            processed_docs.append(cls.preprocess_text(doc))
        return processed_docs
    
    @staticmethod
    def build_vocabulary(documents):
        """
        Build vocabulary from preprocessed documents
        
        Args:
            documents (list): List of preprocessed document tokens
            
        Returns:
            set: Set of unique words in the vocabulary
        """
        # Create a set to store unique words
        vocabulary = set()
        for doc in documents:
            vocabulary.update(doc)
        
        return vocabulary