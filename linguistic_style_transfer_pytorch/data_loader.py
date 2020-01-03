import torch
from torch.utils.data import Dataset
import numpy as np
from linguistic_style_transfer_pytorch.config import GeneralConfig

gconfig = GeneralConfig()


class TextDataset(Dataset):
    """
    """

    def __init__(self, mode='train'):
        super(self, TextDataset).__init__()
        with open(gconfig.train_text_file_path) as f:
            train_data = f.readlines()
        with open(gconfig.train_labels_file_path) as f:
            train_labels = f.readlines()
        with open(gconfig.word_embedding_file_path) as f:

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
