
class Config:
    def __init__(self):
        self.vocab_size = 9200
        self.vocab_save_path = '../data/'
        self.train_file_path = '../data/clean/yelp_train_data.txt'
        self.train_pos_reviews_file_path = "../data/raw/yelp_train_pos.txt"
        self.train_neg_reviews_file_path = "../data/raw/yelp_train_neg.txt"
        self.train_text_file_path = "../data/clean/yelp_train_data.txt"
        self.train_labels_file_path = "../data/clean/yelp_train_labels.txt"
        self.pos_sentiment_file_path = "../data/lexicon/positive-words.txt"
        self.neg_sentiment_file_path = "../data/lexicon/negative-words.txt"
        self.unk_token = "<unk>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.predefined_word_index = {
            self.unk_token: 0,
            self.sos_token: 1,
            self.eos_token: 2,
        }
        self.filter_sentiment_words = True
        self.filter_stopwords = True
