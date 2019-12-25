from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords


class Vocab:
    """
    """

    def __init__(self, vocab_size=9200):

        self.vocab_size = vocab_size
        self.unk_token = "<unk>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.predefined_word_index = {
            self.unk_token: 0,
            self.sos_token: 1,
            self.eos_token: 2,
        }

    def _populate_word_blacklist(word_index):
        """

        """
        blacklisted_words = set()
        blacklisted_words |= set(global_config.predefined_word_index.values())
        if global_config.filter_sentiment_words:
            blacklisted_words |= lexicon_helper.get_sentiment_words()
        if global_config.filter_stopwords:
            blacklisted_words |= lexicon_helper.get_stopwords()

        global bow_filtered_vocab_indices
        allowed_vocab = word_index.keys() - blacklisted_words
        i = 0
        for word in allowed_vocab:
            vocab_index = word_index[word]
            bow_filtered_vocab_indices[vocab_index] = i
            i += 1

        self.bow_size = len(allowed_vocab)
        logger.info("Created word index blacklist for BoW")
        logger.info("BoW size: {}".format(global_config.bow_size))

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

    def _get_stopwords(self):
        """
        Returns all the stopwords which excluded from the 
        main vocab to form the BoW vocab
        """
        nltk_stopwords = set(stopwords.words('english'))
        sklearn_stopwords = stop_words.ENGLISH_STOP_WORDS

        all_stopwords = set()
        # The '|' operator on python sets acts as a union operator
        all_stopwords |= spacy_stopwords
        all_stopwords |= nltk_stopwords
        all_stopwords |= sklearn_stopwords

        return all_stopwords
