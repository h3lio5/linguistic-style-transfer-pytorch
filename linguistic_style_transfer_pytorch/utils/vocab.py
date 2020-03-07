from linguistic_style_transfer_pytorch.config import GeneralConfig
import json
import numpy as np
import collections
import logging
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
from gensim.models import KeyedVectors

# Part of Code taken from https://github.com/vineetjohn/linguistic-style-transfer/tree/master/linguistic_style_transfer_model/utils


class Vocab:
    """
    Holds all the necessary methods to create vocabulary
    """

    def __init__(self, config):

        self.config = config
        self.vocab_size = config.vocab_size
        self.train_file_path = config.train_text_file_path
        self.vocab_save_path = config.vocab_save_path
        self.predefined_word_index = config.predefined_word_index
        self.filter_sentiment_words = config.filter_sentiment_words
        self.filter_stopwords = config.filter_stopwords

    def create_vocab(self):
        """
        Creates word2index and index2word dictionaries
        """
        index2word = dict()
        words = collections.Counter()
        word2index = self.predefined_word_index

        with open(self.train_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if len(line) == 0:
                    continue
                words.update(line.split())
        # Store only 9200 most common words
        words = words.most_common(self.vocab_size)
        logging.debug("collected {} most common words".format(self.vocab_size))
        # create embedding matrix
        emb_matrix = np.zeros(
            (self.config.vocab_size+len(self.config.predefined_word_index), self.config.embedding_size))
        # randomly initialize the special tokens
        emb_matrix[:len(self.config.predefined_word_index), :] = np.random.rand(
            len(self.config.predefined_word_index), self.config.embedding_size)
        # load the pretrained word embeddings
        w2v_model = KeyedVectors.load_word2vec_format(
            self.config.word_embedding_text_file_path).wv

        # fill the dictionary with special tokens first
        idx = 0
        for word, index in self.config.predefined_word_index.items():
            word2index[word] = index
            index2word[index] = word
            idx += 1
        # Create word2index, index2word by iterating over
        # the most common words
        for token in words:
            if token[0] in w2v_model:
                word2index[token[0]] = idx
                index2word[idx] = token[0]
                emb_matrix[idx, :] = w2v_model[token[0]]
                idx = idx + 1
        print("Created embeddings")
        print("Created word2index dictionary")
        print("Created index2word dictionary")
        # Saving the vocab file
        with open(self.vocab_save_path + '/word2index.json', 'w') as json_file:
            json.dump(word2index, json_file)
        print("Saved word2index.json at {}".format(
            self.vocab_save_path+'/word2index.json'))
        with open(self.vocab_save_path + '/index2word.json', 'w') as json_file:
            json.dump(index2word, json_file)
        print("Saved index2word.json at {}".format(
            self.vocab_save_path+'/index2word.json'))
        # Save word embedding file
        np.save(self.config.word_embedding_path, emb_matrix)
        # create bow vocab
        self._populate_word_blacklist(word2index)

    def _populate_word_blacklist(self, word_index):
        """
        Creates a dict of vocab indeces of non-stopwords and non-sentiment words
        """
        blacklisted_words = set()
        bow_filtered_vocab_indices = dict()
        # The '|' operator on sets in python acts as a union operator
        blacklisted_words |= set(self.predefined_word_index.values())
        if self.filter_sentiment_words:
            blacklisted_words |= self._get_sentiment_words()
        if self.filter_stopwords:
            blacklisted_words |= self._get_stopwords()

        allowed_vocab = word_index.keys() - blacklisted_words
        i = 0

        for word in allowed_vocab:
            vocab_index = word_index[word]
            bow_filtered_vocab_indices[vocab_index] = i
            i += 1

        self.config.bow_size = len(allowed_vocab)
        print("Created word index blacklist for BoW")
        print("BoW size: {}".format(self.config.bow_size))
        # saving bow vocab
        with open(self.vocab_save_path + '/bow.json', 'w') as json_file:
            json.dump(bow_filtered_vocab_indices, json_file)
        print("Saved bow.json at {}".format(
            self.vocab_save_path+'/bow.json'))

    def _get_sentiment_words(self):
        """
        Returns all the sentiment words (positive and negative)
        which are excluded from the main vocab to form the BoW vocab
        """
        with open(file=config.pos_sentiment_file_path,
                  mode='r', encoding='ISO-8859-1') as pos_sentiment_words_file,\
            open(file=config.neg_sentiment_file_path,
                 mode='r', encoding='ISO-8859-1') as neg_sentiment_words_file:
            pos_words = pos_sentiment_words_file.readlines()
            neg_words = neg_sentiment_words_file.readlines()
            words = pos_words + neg_words
        words = set(word.strip() for word in words)

        return words

    def _get_stopwords(self):
        """
        Returns all the stopwords which excluded from the
        main vocab to form the BoW vocab
        """
        nltk_stopwords = set(stopwords.words('english'))
        sklearn_stopwords = stop_words.ENGLISH_STOP_WORDS

        all_stopwords = set()
        # The '|' operator on sets in python acts as a union operator
        all_stopwords |= spacy_stopwords
        all_stopwords |= nltk_stopwords
        all_stopwords |= sklearn_stopwords

        return all_stopwords


if __name__ == "__main__":
    config = GeneralConfig()
    vocab = Vocab(config)
    vocab.create_vocab()
