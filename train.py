import torch
from torch.utils.data import DataLoader
from linguistic_style_transfer_pytorch.config import GeneralConfig, ModelConfig
from linguistic_style_transfer_pytorch.data_loader import TextDataset
from linguistic_style_transfer_pytorch.model import AdversarialVAE
from tqdm import tqdm, trange
import argparse

use_cuda = True if torch.cuda.is_available() else False


if __name__ == "__main__":

    mconfig = ModelConfig()
    gconfig = GeneralConfig()
    model = AdversarialVAE(inference=False)
    train_dataset = TextDataset(mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=mconfig.batch_size)

    for epoch in trange(mconfig.epochs, desc="Epoch"):
