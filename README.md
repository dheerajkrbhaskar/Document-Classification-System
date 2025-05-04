# Document Classification System

A Naive Bayes-based document classification system implemented from scratch in Python.

## Project Overview

This project implements a document classification system using a Naive Bayes classifier. The system can categorize text documents into predefined categories based on their content. The implementation is done without relying on external machine learning libraries for the core classification logic.

## Features

- Text preprocessing (tokenization, stopword removal, punctuation removal)
- Bag of Words (BoW) feature extraction
- Naive Bayes classification with Laplace smoothing
- Support for multiple document categories
- Model evaluation and performance analysis

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset in the following format:
   - Each document should be in a separate text file
   - Documents should be organized in folders by category

2. Run the classifier:
```bash
python main.py --train_data_path /path/to/training/data --test_data_path /path/to/test/data
```

## Project Structure

- `preprocessor.py`: Text preprocessing module
- `feature_extractor.py`: Feature extraction module
- `naive_bayes.py`: Naive Bayes classifier implementation
- `main.py`: Main script to run the classifier
- `utils.py`: Utility functions
- `config.py`: Configuration settings

## Authors

- Dheeraj Kumar Bhaskar (102303860)
- Lakshay Sharma (102303856)
- Saumil Makkar (102303862)

## Supervisor

Mr. Amardeep Singh 