import argparse
import os
from config import DATA_DIR, MODEL_DIR, CATEGORIES
from preprocessor import TextPreprocessor
from feature_extractor import FeatureExtractor
from naive_bayes import NaiveBayes as NaiveBayesClassifier
from utils import load_dataset, evaluate_model, save_model, load_model

def train_model(train_data_dir, model_save_dir):
    """
    Train the document classification model
    
    Args:
        train_data_dir (str): Path to training data directory
        model_save_dir (str): Path to save trained model
    """
    # This function trains the model using the provided training data
    # and saves the trained model to the specified directory.
    print("\n=== Training Model ===")
    print(f"Training data directory: {train_data_dir}")
    print(f"Model save directory: {model_save_dir}")
    print(f"Categories: {', '.join(CATEGORIES)}")
    
    documents, labels, class_names = load_dataset(train_data_dir)
    preprocessor = TextPreprocessor()
    processed_docs = preprocessor.preprocess_documents(documents)
    
    feature_extractor = FeatureExtractor()
    feature_extractor.build_vocabulary(processed_docs)
    X = feature_extractor.extract_features_batch(processed_docs)
    
    model = NaiveBayesClassifier()
    model.fit(X, labels)
    
    save_model(model, preprocessor, feature_extractor, model_save_dir)

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
    Evaluate model on test data
    
    Args:
        test_data_dir (str): Path to test data directory
        model_load_dir (str): Path to load trained model
    """
    # This function evaluates the model's performance on test data.
    # It calculates accuracy and provides a detailed report.
    print("\n=== Evaluating Model ===")
    print(f"Test data directory: {test_data_dir}")
    print(f"Model load directory: {model_load_dir}")
    
    documents, labels, class_names = load_dataset(test_data_dir)
    model, preprocessor, feature_extractor = load_model(model_load_dir)
    
    processed_docs = preprocessor.preprocess_documents(documents)
    X = feature_extractor.extract_features_batch(processed_docs)
    
    top_n_labels, _ = model.predict(X)
    predictions = [labels[0] for labels in top_n_labels]
    
    results = evaluate_model(labels, predictions, class_names)
    
    print("\n=== Evaluation Results ===")
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    for category, acc in results['category_accuracy'].items():
        print(f"{category}: {acc:.4f}")
    print(results['report'])
    

def classify_document(document_path, model_load_dir):
    """
    Classify a single document
    
    Args:
        document_path (str): Path to document to classify
        model_load_dir (str): Path to load trained model
    """
    # This function classifies a single document and predicts its category.
    print("\n=== Classification ===")
    print(f"Change {document_path} for other classifications")
    print(f"Model stored at {model_load_dir}")
    
    with open(document_path, 'r', encoding='utf-8') as f:
        document = f.read()
    
    model, preprocessor, feature_extractor = load_model(model_load_dir)
    
    processed_doc = preprocessor.preprocess_text(document)
    X = feature_extractor.extract_features(processed_doc)
    
    prediction = model.predict([X])[0]
    probabilities = model.predict_proba([X])[0]
    
    print("\n Classification Result")
    print("-------------------------------------------------------------------")
    print(f"Top 3 Predicted Categories: {', '.join(prediction[0])}")
    print("-------------------------------------------------------------------")

# def main():
#     # This is the main function that handles different commands like train, evaluate, and classify.
#     parser = argparse.ArgumentParser(description='Document Classification System')
#     subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
#     train_parser = subparsers.add_parser('train', help='Train the model')
#     train_parser.add_argument('--train_data_dir', type=str, default=DATA_DIR,
#                             help='Path to training data directory')
#     train_parser.add_argument('--model_save_dir', type=str, default=MODEL_DIR,
#                             help='Path to save trained model')
    
#     eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
#     eval_parser.add_argument('--test_data_dir', type=str, default=DATA_DIR,
#                            help='Path to test data directory')
#     eval_parser.add_argument('--model_load_dir', type=str, default=MODEL_DIR,
#                            help='Path to load trained model')
    
#     classify_parser = subparsers.add_parser('classify', help='Classify a document')
#     classify_parser.add_argument('--document_path', type=str, default='document.txt',
#                                help='Path to document to classify')
#     classify_parser.add_argument('--model_load_dir', type=str, default=MODEL_DIR,
#                                help='Path to load trained model')
    
#     args = parser.parse_args()
    
#     if args.command == 'train':
#         train_model(args.train_data_dir, args.model_save_dir)
#     elif args.command == 'evaluate':
#         evaluate_test_data(args.test_data_dir, args.model_load_dir)
#     elif args.command == 'classify':
#         classify_document(args.document_path, args.model_load_dir)
#     else:
#         parser.print_help()

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
            evaluate_test_data(DATA_DIR, MODEL_DIR)

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