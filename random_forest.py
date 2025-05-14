from sklearn.ensemble import RandomForestClassifier

import numpy as np

class RandomForestClassifierModel:
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the Random Forest model with default parameters.
        :param n_estimators: Number of trees in the forest.
        :param random_state: Random seed for reproducibility.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X_train, y_train):
        """
        Train the Random Forest model.
        :param X_train: Training feature set.
        :param y_train: Training labels.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test, top_n=2):
        """
        Predict the top N categories using the trained Random Forest model.
        :param X_test: Test feature set.
        :param top_n: Number of top predictions to return.
        :return: List of top N predicted labels for each sample.
        """
        probabilities = self.model.predict_proba(X_test)
        top_n_indices = np.argsort(probabilities, axis=1)[:, -top_n:][:, ::-1]
        return [[self.model.classes_[idx] for idx in indices[:top_n]] for indices in top_n_indices]
