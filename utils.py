import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config import CATEGORIES
import joblib
from naive_bayes import NaiveBayes

def load_dataset(data_dir):
    """
    Load dataset from directory structure where each subdirectory represents a class
    
    Args:
        data_dir (str): Path to dataset directory
        
    Returns:
        tuple: (documents, labels, class_names)
    """
    documents = []
    labels = []
    class_names = []
    category_counts = {}  # Track number of documents per category
    
    # Validate categories
    for category in os.listdir(data_dir):
        if category not in CATEGORIES:
            print(f"Warning: Found unexpected category '{category}' in data directory")
            continue
            
        class_dir = os.path.join(data_dir, category)
        if os.path.isdir(class_dir):
            class_names.append(category)
            txt_files = [f for f in os.listdir(class_dir) if f.endswith('.txt')]
            
            if not txt_files:
                print(f"Warning: No .txt files found in category '{category}'")
                continue
                
            category_counts[category] = 0
            for filename in txt_files:
                file_path = os.path.join(class_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:  # Only add non-empty documents
                            documents.append(content)
                            labels.append(category)
                            category_counts[category] += 1
                except Exception as e:
                    print(f"Error reading file {file_path}: {str(e)}")
    
    if not documents:
        raise ValueError("No valid documents found in the data directory")
    
    # Print document distribution
    print("\nDocument Distribution:")
    print("-" * 30)
    for category, count in category_counts.items():
        print(f"{category}: {count} documents")
    print("-" * 30)
    print(f"Total documents: {len(documents)}")
    print(f"Categories: {len(set(labels))}")
    
    return documents, labels, class_names

# def evaluate_model(y_true, y_pred, class_names):
#     """
#     Evaluate model performance
    
#     Args:
#         y_true (list): True labels
#         y_pred (list): Predicted labels
#         class_names (list): List of class names
        
#     Returns:
#         dict: Evaluation metrics
#     """
#     accuracy = accuracy_score(y_true, y_pred)
#     report = classification_report(y_true, y_pred, target_names=class_names)
    
#     # Calculate confusion matrix
#     cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
#     # Calculate per-category accuracy and confusion
#     category_accuracy = {}
#     category_confusion = {}
    
#     for category in class_names:
#         # Calculate accuracy for this category
#         category_indices = [i for i, label in enumerate(y_true) if label == category]
#         if category_indices:  # Only calculate if there are documents in this category
#             category_true = [y_true[i] for i in category_indices]
#             category_pred = [y_pred[i] for i in category_indices]
#             category_accuracy[category] = accuracy_score(category_true, category_pred)
            
#             # Calculate confusion for this category
#             category_confusion[category] = {
#                 'true_positives': sum(1 for t, p in zip(category_true, category_pred) if t == p),
#                 'false_positives': sum(1 for t, p in zip(y_true, y_pred) if t != category and p == category),
#                 'false_negatives': sum(1 for t, p in zip(category_true, category_pred) if t != p)
#             }
    
#     # Print detailed evaluation metrics
#     print("\nDetailed Evaluation Metrics:")
#     print("-" * 50)
#     for category in class_names:
#         if category in category_accuracy:
#             conf = category_confusion[category]
#             print(f"\nCategory: {category}")
#             print(f"Accuracy: {category_accuracy[category]:.4f}")
#             print(f"True Positives: {conf['true_positives']}")
#             print(f"False Positives: {conf['false_positives']}")
#             print(f"False Negatives: {conf['false_negatives']}")
    
#     print("\nConfusion Matrix:")
#     print("-" * 50)
#     print("Rows: True labels, Columns: Predicted labels")
#     print("\n".join([f"{row}" for row in cm]))
    
#     return {
#         'accuracy': accuracy,
#         'report': report,
#         'category_accuracy': category_accuracy,
#         'confusion_matrix': cm
#     }

def evaluate_model(y_true, y_pred, class_names):
    """
    Evaluate model performance
    
    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels
        class_names (list): List of class names
        
    Returns:
        dict: Evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    # Calculate per-category accuracy and confusion
    category_accuracy = {}
    category_confusion = {}
    
    for category in class_names:
        # Calculate accuracy for this category
        category_indices = [i for i, label in enumerate(y_true) if label == category]
        if category_indices:  # Only calculate if there are documents in this category
            category_true = [y_true[i] for i in category_indices]
            category_pred = [y_pred[i] for i in category_indices]
            category_accuracy[category] = accuracy_score(category_true, category_pred)
            
            # Calculate confusion for this category
            category_confusion[category] = {
                'true_positives': sum(1 for t, p in zip(category_true, category_pred) if t == p),
                'false_positives': sum(1 for t, p in zip(y_true, y_pred) if t != category and p == category),
                'false_negatives': sum(1 for t, p in zip(category_true, category_pred) if t != p)
            }
    
    # Print detailed evaluation metrics
    print("\nDetailed Evaluation Metrics:")
    print("-" * 50)
    for category in class_names:
        if category in category_accuracy:
            conf = category_confusion[category]
            print(f"\nCategory: {category}")
            print(f"Accuracy: {category_accuracy[category]:.4f}")
            print(f"True Positives: {conf['true_positives']}")
            print(f"False Positives: {conf['false_positives']}")
            print(f"False Negatives: {conf['false_negatives']}")
    
    # Print confusion matrix in a formatted table
    print("\nConfusion Matrix:")
    print(" " * 15 + "Predicted Labels")
    header = " " * 14 + " ".join([f"{cls:>12}" for cls in class_names])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:<14} |" + " ".join([f"{val:>12}" for val in row])
        print(row_str)
    
    return {
        'accuracy': accuracy,
        'report': report,
        'category_accuracy': category_accuracy,
        'confusion_matrix': cm
    }

def save_model(model, preprocessor, feature_extractor, save_dir):
    """
    Save model components to disk
    
    Args:
        model: Trained model
        preprocessor: Text preprocessor
        feature_extractor: Feature extractor
        save_dir: Directory to save components
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model components
    if isinstance(model, NaiveBayes):
        # Save Naive Bayes model components
        np.save(os.path.join(save_dir, 'class_weights.npy'), model.class_weights)
        np.save(os.path.join(save_dir, 'feature_weights.npy'), model.feature_weights)
        np.save(os.path.join(save_dir, 'class_likelihoods.npy'), model.class_likelihoods)
        np.save(os.path.join(save_dir, 'classes.npy'), model.classes)
        np.save(os.path.join(save_dir, 'class_to_index.npy'), model.class_to_index)
        np.save(os.path.join(save_dir, 'alpha.npy'), np.array([model.alpha]))
    else:
        # Save other model types
        joblib.dump(model, os.path.join(save_dir, 'model.joblib'))
    
    # Save preprocessor
    joblib.dump(preprocessor, os.path.join(save_dir, 'preprocessor.joblib'))
    
    # Save feature extractor
    joblib.dump(feature_extractor, os.path.join(save_dir, 'feature_extractor.joblib'))
    
    print(f"Model components saved to {save_dir}")

def load_model(model_dir):
    """
    Load model components from disk
    
    Args:
        model_dir: Directory containing saved components
        
    Returns:
        tuple: (model, preprocessor, feature_extractor)
    """
    # Load model
    if os.path.exists(os.path.join(model_dir, 'class_weights.npy')):
        # Load Naive Bayes model
        model = NaiveBayes()
        model.class_weights = np.load(os.path.join(model_dir, 'class_weights.npy'))
        model.feature_weights = np.load(os.path.join(model_dir, 'feature_weights.npy'))
        model.class_likelihoods = np.load(os.path.join(model_dir, 'class_likelihoods.npy'))
        model.classes = np.load(os.path.join(model_dir, 'classes.npy'))
        model.class_to_index = np.load(os.path.join(model_dir, 'class_to_index.npy'), allow_pickle=True).item()
        model.alpha = np.load(os.path.join(model_dir, 'alpha.npy'))[0]
    else:
        # Load other model types
        model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    
    # Load preprocessor
    preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.joblib'))
    
    # Load feature extractor
    feature_extractor = joblib.load(os.path.join(model_dir, 'feature_extractor.joblib'))
    
    return model, preprocessor, feature_extractor