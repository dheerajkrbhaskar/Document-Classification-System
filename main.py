import os
from config import DATA_DIR, TEST_DATA_DIR, MODEL_DIR, CATEGORIES
from preprocessor import TextPreprocessor
from feature_extractor import FeatureExtractor
from naive_bayes import NaiveBayes as NaiveBayesClassifier
from knn import KNNClassifier
from random_forest import RandomForestClassifierModel
from utils import load_dataset, evaluate_model, save_model, load_model
import numpy as np

def train_model(train_data_dir, model_save_dir):
    """
    Train the document classification models (Naive Bayes, KNN, and Random Forest)
    
    Args:
        train_data_dir (str): Path to training data directory
        model_save_dir (str): Path to save trained models
    """
    print("\n=== Training Models ===")
    print(f"Training data directory: {train_data_dir}")
    print(f"Model save directory: {model_save_dir}")
    print(f"Categories: {', '.join(CATEGORIES)}")
    
    documents, labels, class_names = load_dataset(train_data_dir)
    preprocessor = TextPreprocessor()
    processed_docs = preprocessor.preprocess_documents(documents)
    
    feature_extractor = FeatureExtractor()
    feature_extractor.build_vocabulary(processed_docs)
    X = feature_extractor.extract_features_batch(processed_docs)
    
    nb_model_dir = os.path.join(model_save_dir, "naive_bayes")
    knn_model_dir = os.path.join(model_save_dir, "knn")
    rf_model_dir = os.path.join(model_save_dir, "random_forest")
    os.makedirs(nb_model_dir, exist_ok=True)
    os.makedirs(knn_model_dir, exist_ok=True)
    os.makedirs(rf_model_dir, exist_ok=True)

    # Train Naive Bayes
    nb_model = NaiveBayesClassifier()
    nb_model.fit(X, labels)
    save_model(nb_model, preprocessor, feature_extractor, nb_model_dir)

    # Train KNN
    knn_model = KNNClassifier(k=3)
    knn_model.fit(X, labels)
    save_model(knn_model, preprocessor, feature_extractor, knn_model_dir)

    # Train Random Forest
    rf_model = RandomForestClassifierModel()
    rf_model.fit(X, labels)
    save_model(rf_model, preprocessor, feature_extractor, rf_model_dir)

def display_confusion_matrix(cm, class_names):
    """
    Display the confusion matrix in a compact format.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        class_names (list): List of class names
    """
    print("\nConfusion Matrix:")
    print(" " * 15 + "Predicted Labels")
    header = " " * 14 + " ".join([f"{cls[:8]:>10}" for cls in class_names])  # Truncate class names for compactness
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{class_names[i][:8]:<14} |" + " ".join([f"{val:>10}" for val in row])
        print(row_str)

def evaluate_test_data(test_data_dir, model_load_dir):
    """
    Evaluate Naive Bayes, KNN, and Random Forest models on test data
    
    Args:
        test_data_dir (str): Path to test data directory
        model_load_dir (str): Path to load trained models
    """
    print("\n=== Evaluating Models ===")
    print(f"Test data directory: {test_data_dir}")
    print(f"Model load directory: {model_load_dir}")
    
    documents, labels, class_names = load_dataset(test_data_dir)
    nb_model_dir = os.path.join(model_load_dir, "naive_bayes")
    knn_model_dir = os.path.join(model_load_dir, "knn")
    rf_model_dir = os.path.join(model_load_dir, "random_forest")

    nb_model, preprocessor, feature_extractor = load_model(nb_model_dir)
    knn_model, _, _ = load_model(knn_model_dir)
    rf_model, _, _ = load_model(rf_model_dir)
    
    processed_docs = preprocessor.preprocess_documents(documents)
    X_test = feature_extractor.extract_features_batch(processed_docs)
    X_test = np.array(X_test)  # Ensure X_test is a NumPy array
    y_test = labels  # Ensure y_test matches the number of documents

    # Debugging: Print shapes of X_test and y_test
    print(f"X_test shape: {X_test.shape}, y_test length: {len(y_test)}")

    # Evaluate Naive Bayes
    print("\n=== Naive Bayes Results ===")
    y_pred_nb = nb_model.predict(X_test, top_n=1)  # Use only the top category for evaluation
    nb_results = evaluate_model(y_test, y_pred_nb, class_names)
    print(f"Overall Accuracy: {nb_results['accuracy']:.4f}")
    print(nb_results['report'])

    # Evaluate KNN
    print("\n=== KNN Results ===")
    y_pred_knn = knn_model.predict(X_test)  # Assuming KNN already returns top 1
    knn_results = evaluate_model(y_test, y_pred_knn, class_names)
    print(f"Overall Accuracy: {knn_results['accuracy']:.4f}")
    print(knn_results['report'])

    # Evaluate Random Forest
    print("\n=== Random Forest Results ===")
    y_pred_rf = rf_model.predict(X_test, top_n=3)  # Get top 3 predictions
    rf_results = evaluate_model(y_test, [pred[0] for pred in y_pred_rf], class_names)  # Use top prediction for evaluation
    print(f"Overall Accuracy: {rf_results['accuracy']:.4f}")
    print(rf_results['report'])

def classify_document(document_path, model_load_dir):
    """
    Classify a single document using Naive Bayes, KNN, and Random Forest models
    
    Args:
        document_path (str): Path to the document to classify
        model_load_dir (str): Path to load the trained models
    """
    print("\n=== Classifying Document ===")
    print(f"Document path: {document_path}")
    print(f"Model load directory: {model_load_dir}")

    # Load Naive Bayes model
    nb_model_dir = os.path.join(model_load_dir, "naive_bayes")
    nb_model, preprocessor, feature_extractor = load_model(nb_model_dir)

    # Load KNN model
    knn_model_dir = os.path.join(model_load_dir, "knn")
    knn_model, _, _ = load_model(knn_model_dir)

    # Load Random Forest model
    rf_model_dir = os.path.join(model_load_dir, "random_forest")
    rf_model, _, _ = load_model(rf_model_dir)

    # Read and preprocess the document
    with open(document_path, 'r', encoding='utf-8') as file:
        document = file.read()
    processed_doc = preprocessor.preprocess_documents([document])

    # Extract features
    X = feature_extractor.extract_features_batch(processed_doc)
    X = np.array(X)  # Ensure X is a NumPy array

    # Predict using Naive Bayes
    nb_predictions = nb_model.predict(X, top_n=3)[0]  # Get top 3 categories
    print(f"Naive Bayes - Top 3 Predicted Categories: {', '.join(nb_predictions)}")

    # Predict using KNN
    knn_predictions = knn_model.predict(X, top_n=3)[0]  # Get top 3 categories
    print(f"KNN - Top 3 Predicted Categories: {', '.join(knn_predictions)}")

    # Predict using Random Forest
    rf_predictions = rf_model.predict(X, top_n=3)[0]  # Get top 3 categories
    print(f"Random Forest - Top 3 Predicted Categories: {', '.join(rf_predictions)}")
    print("-------------------------------------------------------------------")

def main():
    while True:
        print("\n============================= Document Classification System ===============================")
        print("Welcome to the Document Classification System!")
        print("This program allows you to train a document classification model, evaluate its performance, and classify new documents.")
        print("Please choose an option:")   
        print("1. Train")
        print("2. Evaluate")
        print("3. Classify")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            #training
            train_model(DATA_DIR, MODEL_DIR)

        elif choice == "2":
            #Evaluating...
            evaluate_test_data(TEST_DATA_DIR, MODEL_DIR)

        elif choice == "3":
            #Classifying...
            classify_document('document.txt', MODEL_DIR)

        elif choice == "4":
            print("Exiting program. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 4.")


if __name__ == '__main__':
    main()