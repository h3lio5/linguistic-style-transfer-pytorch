import torch
from torch.utils.data import Dataset
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords


class TextDataset(Dataset):
    """
    """

    def __init__(self, batch_size=32, bow_size=):
        super(self, TextDataset).__init__()

        self.bow_size = bow_size

    def _get_sentiment_words(self):
        """
        Returns all the sentiment words (positive and negative) 
        which are excluded from the main vocab to form the BoW vocab
        """
        with open(file='../data/lexicon/positive-words.txt',
                  mode='r', encoding='ISO-8859-1') as pos_sentiment_words_file,\
            open(file='../data/lexicon/negative-words.txt',
                 mode='r', encoding='ISO-8859-1') as neg_sentiment_words_file:
            pos_words = pos_sentiment_words_file.readlines()
            neg_words = neg_sentiment_words_file.readlines()
            words = pos_words + neg_words
        words = set(word.strip() for word in words)

        return words

    def get_stopwords():
        """
        Returns all the stopwords which excluded from the 
        main vocab to form the BoW vocab
        """
        nltk_stopwords = set(stopwords.words('english'))
        sklearn_stopwords = stop_words.ENGLISH_STOP_WORDS

        all_stopwords = set()
        all_stopwords |= spacy_stopwords
        all_stopwords |= nltk_stopwords
        all_stopwords |= sklearn_stopwords

        return all_stopwords

    def _get_bow_representations(self, text_sequences):
        """
        Returns BOW representation of every sequence of the batch
        """
        bow_representation = list()
        # Iterate over each sequence in the batch
        for text_sequence in text_sequences:
            sequence_bow_representation = np.zeros(
                shape=self.bow_size, dtype=np.float32)
            # Iterate over each word in the sequence
            for index in text_sequence:
                if index in bow_filtered_vocab_indices:
                    bow_index = bow_filtered_vocab_indices[index]
                    sequence_bow_representation[bow_index] += 1
            sequence_bow_representation = np.max(
                [np.sum(sequence_bow_representation), 1])
            bow_representation.append(sequence_bow_representation)

        return np.asarray(bow_representation)
