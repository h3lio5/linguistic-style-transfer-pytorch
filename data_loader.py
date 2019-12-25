import torch
from torch.utils.data import Dataset
import numpy as np


class TextDataset(Dataset):
    """
    """

    def __init__(self, batch_size=32, bow_size=):
        super(self, TextDataset).__init__()

        self.bow_size = bow_size

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
