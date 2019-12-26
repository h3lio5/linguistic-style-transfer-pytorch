import torch
import torch.nn as nn
from linguistic_style_transfer_pytorch.config import ModelConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


config = ModelConfig()


class AutoEncoder(nn.Module):
    """
    """

    def __init__(self, inference=False):
        """
        Initialize networks
        """
        # Inference mode or training mode
        self.inference = inference
        # Encoder model
        self.encoder = nn.GRU(
            config.embedding_size, config.hidden_dim, batch_first=True, bidirectional=True)
        # content latent embedding
        self.content_emb = nn.Linear(
            config.hidden_dim, config.content_hidden_dim)
        # style latent embedding
        self.style_emb = nn.Linear(config.hidden_dim, config.style_hidden_dim)
        # Discriminator/adversary
        self.content_disc = nn.Linear(
            config.content_hidden_dim, config.num_style)
        self.style_disc = nn.Linear(
            config.style_hidden_dim, config.content_bow_dim)
        # classifier
        self.content_classifier = nn.Linear(
            config.content_hidden_dim, config.content_bow_dim)
        self.style_classifier = nn.Linear(
            config.style_hidden_dim, config.num_style)
        # Decoder
        self.decoder = nn.GRUCell(
            config.embedding_size, config.hidden_dim, batch_first=True, bidirectional=True)
        # dropout
        self.dropout = config.emb_dropout
        self.init()

    def forward(self, sequences, seq_lengths, labels, content_bow):
        """
        """
