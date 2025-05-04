import numpy as np
from collections import Counter
from config import SMOOTHING_ALPHA

class NaiveBayes:
    def __init__(self, alpha=1.0):
       
        self.alpha = alpha
        self.class_weights = None
        self.feature_weights = None
        self.class_likelihoods = None
        self.classes = None
        self.class_to_index = None

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier
        
        Args:
            X (list): List of feature vectors
            y (list): List of class labels
        """
        #convert input data to numpy arrays
        X = np.array(X)
        y = np.array(y)

        #Get unique classes & map them to indices
        self.classes = np.unique(y)
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}
        n_classes = len(self.classes)
        n_features = X.shape[1]

        # Initialize feature weights, calculate class weights/prior probabilities
        self.feature_weights = np.ones(n_features)

        class_counts = Counter(y)
        total_samples = len(y)

        self.class_weights = np.zeros(n_classes)

        for cls, idx in self.class_to_index.items():
            self.class_weights[idx] = (class_counts[cls] + self.alpha) / (total_samples + n_classes * self.alpha)

        #calculate each feature likelihoods of class
        self.class_likelihoods = np.zeros((n_classes, n_features))
        for cls, idx in self.class_to_index.items():
            class_samples = X[y == cls]
            feature_counts = np.sum(class_samples, axis=0)
            total_words = np.sum(feature_counts)
            self.class_likelihoods[idx] = (feature_counts + self.alpha) / (total_words + n_features * self.alpha)

    def predict_proba(self, X):
        """
        Predict class probabilities for input data
        
        Args:
            X (list): List of feature vectors
            
        Returns:
            numpy.ndarray: Class probabilities
        """
        # Ensure the model is trained before making predictions
        if self.class_weights is None or self.class_likelihoods is None:
            raise ValueError("Model not fitted. Call fit first.")

        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes)

        # Calculate probabilities for each class
        probabilities = np.zeros((n_samples, n_classes))
        for cls, idx in self.class_to_index.items():
            log_likelihood = np.sum(X * np.log(self.class_likelihoods[idx]) * self.feature_weights, axis=1)
            probabilities[:, idx] = log_likelihood + np.log(self.class_weights[idx])

        # Convert log probabilities to actual probabilities using softmax
        probabilities = np.exp(probabilities - np.max(probabilities, axis=1, keepdims=True))
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)

        return probabilities

    def predict(self, X):
        """
        Predict class labels for input data
        
        Args:
            X (list): List of feature vectors
            
        Returns:
            list: Predicted class labels
        """
        # Get probabilities for each class
        probabilities = self.predict_proba(X)

        # Get top N predictions for each sample
        top_n = 3
        top_n_indices = np.argsort(probabilities, axis=1)[:, -top_n:][:, ::-1]  # Indices of top N probabilities
        top_n_labels = [[self.classes[idx] for idx in indices] for indices in top_n_indices]
        top_n_probs = [probabilities[i, indices] for i, indices in enumerate(top_n_indices)]

        # Return the top predictions
        return top_n_labels, top_n_probs
