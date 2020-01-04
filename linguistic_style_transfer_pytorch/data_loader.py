import os
import torch
from torch.utils.data import Dataset
import numpy as np
from linguistic_style_transfer_pytorch.config import GeneralConfig, ModelConfig
import json

gconfig = GeneralConfig()
mconfig = ModelConfig()


class TextDataset(Dataset):
    """
    Dataset for text data
    """

    def __init__(self, mode='train'):
        super(TextDataset, self).__init__()
        # load train data
        with open(gconfig.train_text_file_path) as f:
            self.train_data = f.readlines()
        # load train labels
        with open(gconfig.train_labels_file_path) as f:
            self.train_labels = f.readlines()
        # load word2index
        with open(gconfig.w2i_file_path) as f:
            self.word2index = json.load(f)
        # load bow vocab
        with open(gconfig.bow_file_path) as f:
            self.bow_filtered_vocab_indices = json.load(f)
        self.label2index = {'neg': [0, 1], 'pos': [1, 0]}

    def _padding(self, token_ids):
        """
        Utility function to add padding to and trim the sentence
        """
        if len(token_ids) > mconfig.max_seq_len:
            return token_ids[:mconfig.max_seq_len]
        token_ids = token_ids + \
            (mconfig.max_seq_len-len(token_ids))*[gconfig.pad_token]

        return token_ids

    def _sentence_tokenid(self, sentence):
        """
        Returns token ids of individual words of the sentence
        """
        token_ids = [self.word2index.get(word, gconfig.unk_token)
                     for word in sentence.split()]
        padded_token_ids = self._padding(token_ids)
        return padded_token_ids, len(token_ids)

    def _get_bow_representations(self, text_sequence):
        """
        Returns BOW representation of every sequence of the batch
        """

        sequence_bow_representation = np.zeros(
            shape=gconfig.bow_hidden_dim, dtype=np.float32)
        # Iterate over each word in the sequence
        for index in text_sequence:
            if index in self.bow_filtered_vocab_indices:
                bow_index = self.bow_filtered_vocab_indices[index]
                sequence_bow_representation[bow_index] += 1
        sequence_bow_representation /= np.max(
            [np.sum(sequence_bow_representation), 1])

        return np.asarray(sequence_bow_representation)

    def __len__(self):
        """
        Returns the total number of samples
        """
        return len(self.train_labels)

    def __getitem__(self, index):
        """
        Returns:
            token_ids: token ids of the sentence
            seq_len : length of the sentence before padding
            label   : label of the sentence
            bow_rep : Bag of Words representation of the sentence
        """
        sentence = self.train_data[index]
        label = self.label2index[self.train_labels[index].strip()]
        token_ids, seq_len = self._sentence_tokenid(sentence)
        bow_rep = self._get_bow_representations(sentence)
        return (torch.LongTensor(token_ids), torch.LongTensor([seq_len]), torch.LongTensor(label), torch.FloatTensor(bow_rep))
