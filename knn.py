import numpy as np
from numpy.linalg import norm
from collections import Counter

# def euclidean_distance(x1, x2):
#     return np.sqrt(np.sum((x1 - x2) ** 2))
def cosine_distance(x1, x2):
    """
    Returns cosine distance between vectors x1 and x2.
    Cosine distance = 1 - cosine similarity
    """
    dot_product = np.dot(x1, x2)
    norm_x1 = norm(x1)
    norm_x2 = norm(x2)
    
    # Avoid division by zero
    if norm_x1 == 0 or norm_x2 == 0:
        return 1.0
    
    cosine_sim = dot_product / (norm_x1 * norm_x2)
    return 1.0 - cosine_sim  # Distance, not similarity

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test, top_n=3):
        """
        Predict class labels for input data
        
        Args:
            X_test (list): List of feature vectors
            top_n (int): Number of top predictions to return
            
        Returns:
            list: Predicted class labels
        """
        X_test = np.array(X_test)
        predictions = [self._predict(x, top_n) for x in X_test]
        return predictions

    def _predict(self, x, top_n):
        """
        Predict the top N categories for a single sample
        
        Args:
            x (array): Feature vector
            top_n (int): Number of top predictions to return
            
        Returns:
            list: Top N predicted class labels
        """
        distances = [cosine_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(top_n)
        return [label for label, _ in most_common]
