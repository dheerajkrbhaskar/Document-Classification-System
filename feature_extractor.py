from collections import Counter
from config import MIN_WORD_FREQUENCY, MAX_FEATURES

class FeatureExtractor:
    def __init__(self):
        # create and initialise vocabulary and word to index mapping
        self.vocabulary = None
        self.word_to_index = None

    def build_vocabulary(self, documents):
        """
        Build vocabulary from documents and create word to index mapping
        
        Args:
            documents (list): List of preprocessed document tokens
        """
        #get word frequencies across for all docs
        word_counts = Counter()
        for doc in documents:

            word_counts.update(doc)


        #Filter words by minimum frequency & limit vocabulary size
        vocabulary = {word for word, count in word_counts.items() 
                     if count >= MIN_WORD_FREQUENCY}
        if len(vocabulary) > MAX_FEATURES:
            vocabulary = set(sorted(vocabulary, key=lambda x: word_counts[x], 
                                 reverse=True)[:MAX_FEATURES])

        #save vocabulary &create word-to-index mapping
        self.vocabulary = vocabulary
        self.word_to_index = {word: int(idx) for idx, word in enumerate(sorted(vocabulary))}

    def extract_features(self, document):
        """
        Convert document to feature vector using Bag of Words
        
        Args:
            document (list): Preprocessed document tokens
            
        Returns:
            list: Feature vector
        """
        # Ensure vocabulary is built before extracting features
        if not self.vocabulary:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")

        # Initialize feature vector and count word occurrences
        features = [0] * len(self.vocabulary)
        word_counts = Counter(document)
        for word, count in word_counts.items():
            if word in self.word_to_index:
                idx = int(self.word_to_index[word])
                features[idx] = count

        return features

    def extract_features_batch(self, documents):
        """
        Convert multiple documents to feature vectors
        
        Args:
            documents (list): List of preprocessed document tokens
            
        Returns:
            list: List of feature vectors
        """
        # Process each document and extract features
        feature_vectors = []
        for doc in documents:
            feature_vectors.append(self.extract_features(doc))
        return feature_vectors