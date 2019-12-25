from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords

from linguistic_style_transfer_model.config import global_config


def get_sentiment_words():
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
    nltk_stopwords = set(stopwords.words('english'))
    sklearn_stopwords = stop_words.ENGLISH_STOP_WORDS

    all_stopwords = set()
    all_stopwords |= spacy_stopwords
    all_stopwords |= nltk_stopwords
    all_stopwords |= sklearn_stopwords

    return all_stopwords
