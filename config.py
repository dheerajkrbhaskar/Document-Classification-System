import os
from pathlib import Path

# project paths
BASE_DIR = Path(__file__).parent.absolute()  # base directory of  project
DATA_DIR = os.path.join(BASE_DIR, "data")  #directory for  storing data
MODEL_DIR = os.path.join(BASE_DIR, "models")  # directory for saving model


#categories
CATEGORIES = [
    'business',
    'entertainment',
    'food',
    'graphics',
    'historical',
    'medical',
    'politics',
    'space',
    'sport',
    'technology'
]

#stopwords to remove
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
    'will', 'with', 'this', 'that', 'these', 'those', 'they', 'their', 'them',
    'there', 'then', 'than', 'what', 'when', 'where', 'which', 'who', 'whom',
    'why', 'how', 'if', 'else', 'because', 'while', 'until', 'since', 'about',
    'above', 'below', 'between', 'into', 'through', 'during', 'before', 'after',
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
}


MIN_WORD_FREQUENCY = 2  #minimum frequency for a word to  consider
MAX_FEATURES = 2000  # maximum features to extract

# Model settings
SMOOTHING_ALPHA = 1.0  # smoothing in NaiveBayes

# create dir if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)  # check data directory exists
os.makedirs(MODEL_DIR, exist_ok=True)  # check model directory exists