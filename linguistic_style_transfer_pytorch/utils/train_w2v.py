from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging
from linguistic_style_transfer_pytorch.config import Config


def train_word2vec_model(text_file_path, model_file_path):
    # define training data
    # train model
    logging.info("Loading input file and training mode ...")
    model = Word2Vec(sentences=LineSentence(text_file_path),
                     min_count=1, size=config.embedding_size)
    # summarize the loaded model
    logging.info("Model Details: {}".format(model))
    # save model
    model.wv.save_word2vec_format(model_file_path, binary=False)
    logging.info("Model saved")


if __name__ == "__main__":
    config = Config()
    train_word2vec_model(config.train_file_path,
                         config.word_embedding_file_path)
