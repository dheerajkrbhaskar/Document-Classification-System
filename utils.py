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
    # Debugging: Print lengths of y_true and y_pred
    print(f"Evaluating model: y_true length = {len(y_true)}, y_pred length = {len(y_pred)}")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Inconsistent lengths: y_true ({len(y_true)}) and y_pred ({len(y_pred)})")

    # Ensure y_pred contains only the top category
    y_pred = [label[0] if isinstance(label, list) else label for label in y_pred]

    # Map short labels to full names
    label_mapping = {
        'b': 'business', 'e': 'entertainment', 'f': 'food', 'g': 'graphics',
        'h': 'historical', 'm': 'medical', 'p': 'politics', 's': 'space',
        'sport': 'sport', 't': 'technology'
    }
    y_true = [label_mapping.get(label, label) for label in y_true]
    y_pred = [label_mapping.get(label, label) for label in y_pred]

    # Debugging: Ensure all categories are evaluated
    print(f"Unique labels in y_true: {set(y_true)}")
    print(f"Unique labels in y_pred: {set(y_pred)}")

    # Debugging: Print unique labels
    unique_labels = sorted(set(y_true) | set(y_pred))
    print(f"Unique labels: {unique_labels}")

    # Generate classification report with explicit labels
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print("\nClassification Report:")
    print(report)
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate per-category accuracy
    category_accuracy = {}
    for label in class_names:
        indices = [i for i, true_label in enumerate(y_true) if true_label == label]
        if indices:
            true_subset = [y_true[i] for i in indices]
            pred_subset = [y_pred[i] for i in indices]
            category_accuracy[label] = accuracy_score(true_subset, pred_subset)

    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    print("\nConfusion Matrix:")
    print(cm)

    return {
        'accuracy': accuracy,
        'report': report,
        'category_accuracy': category_accuracy,
        'confusion_matrix': cm
    }

def save_model(model, preprocessor, feature_extractor, save_dir, model_name="model"):
    """
    Save model components to disk
    
    Args:
        model: Trained model
        preprocessor: Text preprocessor
        feature_extractor: Feature extractor
        save_dir: Directory to save components
        model_name: Name of the model file
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
        joblib.dump(model, os.path.join(save_dir, f"{model_name}.joblib"))
    
    # Save preprocessor
    joblib.dump(preprocessor, os.path.join(save_dir, 'preprocessor.joblib'))
    
    # Save feature extractor
    joblib.dump(feature_extractor, os.path.join(save_dir, 'feature_extractor.joblib'))
    
    print(f"Model components saved to {save_dir}")

def load_model(model_dir, model_name="model"):
    """
    Load model components from disk
    
    Args:
        model_dir: Directory containing saved components
        model_name: Name of the model file
        
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
        model = joblib.load(os.path.join(model_dir, f"{model_name}.joblib"))
    
    # Load preprocessor
    preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.joblib'))
    
    # Load feature extractor
    feature_extractor = joblib.load(os.path.join(model_dir, 'feature_extractor.joblib'))
    
    return model, preprocessor, feature_extractor