"""
This script performs the following functions:
    1. Downloads and preprocesses the training data.
    2. Downloads lexicon data.
    3. Creates a vocabulary.
    4. Trains word2vec model to generate word embeddings.
"""

import subprocess
import os
from linguistic_style_transfer_pytorch.utils import preprocess, train_w2v, vocab
from linguistic_style_transfer_pytorch.config import GeneralConfig

gconfig = GeneralConfig()
# Download yelp train data from https://github.com/shentianxiao/language-style-transfer
YELP_DATA_NEG = 'https://raw.githubusercontent.com/shentianxiao/language-style-transfer/master/data/yelp/sentiment.train.0'
YELP_DATA_POS = 'https://raw.githubusercontent.com/shentianxiao/language-style-transfer/master/data/yelp/sentiment.train.1'
# Make raw data folder
os.mkdir(gconfig.data_path+'/raw')
subprocess.call('wget '+YELP_DATA_NEG, shell=True)
subprocess.call('wget '+YELP_DATA_POS, shell=True)
# Rename the raw data files and move them to the raw data folder
subprocess.call('mv sentiment.train.0 ' +
                gconfig.data_path+'/raw/yelp_train_neg.txt')
subprocess.call('mv sentiment.train.1 ' +
                gconfig.data_path+'/raw/yelp_train_pos.txt')

# Create clean data directory to store the preprocessed data
os.mkdir(gconfig.data_path+'/clean')
# Start preprocessing
preprocessor = preprocess.Preprocessor()
preprocessor.preprocess()

# Train word2vec embeddings
train_w2v.train_word2vec_model(
    gconfig.train_text_file_path, gconfig.word_embedding_text_file_path)

# Create vocabulary
vocab = vocab.Vocab(gconfig)
vocab.create_vocab()

print("Creation of clean data, vocabulary and word embeddings complete!!!")
