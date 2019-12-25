import torch
import torch.nn as nn
from linguistic_style_transfer_pytorch.config import ModelConfig


config = ModelConfig()


class AutoEncoder(nn.Module):
    """
    """

    def __init__(self, inference=False):
        """
        """
        self.inference = inference
        self.encoder = nn.GRU(config.embedding_size, config.hidden_dim)
        self.decoder = nn.GRU(config.embedding_size, config.hidden_dim)
        self.content_disc = nn.Linear(
            config.content_hidden_dim, config.num_style)
        self.content_mul = nn.Linear(
            config.content_hidden_dim, config.content_bow_dim)
        self.style_disc = nn.Linear(
            config.style_hidden_dim, config.content_bow_dim)
        self.style_mul = nn.Linear(config.style_hidden_dim, config.num_style)
