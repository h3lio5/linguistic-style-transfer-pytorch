
class Config:
    """
    """

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
        self.word_embedding_file_path = "../data/embedding.txt"
        self.embedding_size = 300
        self.predefined_word_index = {
            "<unk>": 0,
            "<sos>": 1,
            "<eos>": 2,
        }
        self.filter_sentiment_words = True
        self.filter_stopwords = True


class ModelConfig:
    """
    """

    def __init__(self):
        self.embedding_size = 300
        self.max_seq_len = 15
        self.hidden_dim = 256
        self.style_hidden_dim = 8
        self.content_hidden_dim = 128
        self.num_style = 2
        self.content_bow_dim = 7526
        self.emb_dropout = 0.8
        self.label_smoothing = 0.1
