import torch
from torch.utils.data import DataLoader
from linguistic_style_transfer_pytorch.config import GeneralConfig, ModelConfig
from linguistic_style_transfer_pytorch.data_loader import TextDataset
from linguistic_style_transfer_pytorch.model import AdversarialVAE
from tqdm import tqdm, trange
import argparse
import os

use_cuda = True if torch.cuda.is_available() else False


if __name__ == "__main__":

    mconfig = ModelConfig()
    gconfig = GeneralConfig()
    model = AdversarialVAE(inference=False)
    if use_cuda:
        model = model.cuda()

    #=============== Define dataloader ================#
    train_dataset = TextDataset(mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=mconfig.batch_size)
    content_discriminator_params, style_discriminator_params, vae_and_classifier_params = model.get_params()
    #============== Define optimizers ================#
    # content discriminator/adversary optimizer
    content_disc_opt = torch.optim.RMSprop(
        content_discriminator_params, lr=mconfig.content_adversary_lr)
    # style discriminaot/adversary optimizer
    style_disc_opt = torch.optim.RMSprop(
        style_discriminator_params, lr=mconfig.style_adversary_lr)
    # autoencoder and classifiers optimizer
    vae_and_cls_opt = torch.optim.Adam(
        vae_and_classifier_params, lr=mconfig.autoencoder_lr)
    print("Training started!")
    for epoch in trange(mconfig.epochs, desc="Epoch"):

        for iteration, batch in enumerate(tqdm(train_dataloader)):

            # unpacking
            sequences, seq_lens, labels, bow_rep = batch
            if use_cuda:
                sequences = sequences.cuda()
                seq_lens = seq_lens.cuda()
                labels = labels.cuda()
                bow_rep = bow_rep.cuda()

            content_disc_loss, style_disc_loss, vae_and_cls_loss = model(sequences.squeeze(
                1), seq_lens.squeeze(1), labels.squeeze(1), bow_rep.squeeze(1), iteration+1)

            #============== Update Adversary/Discriminator parameters ===========#
            # update content discriminator parametes
            content_disc_loss.backward()
            content_disc_opt.step()
            # update style discriminator parameters
            style_disc_loss.backward()
            style_disc_opt.step()
            #=============== Update VAE and classifier parameters ===============#
            vae_and_cls_loss.backward()
            vae_and_cls_opt.step()

        print("Saving states")
        #================ Saving states ==========================#
        if not os.path.exists(gconfig.model_save_path):
            os.mkdir(gconfig.model_save_path)
        # save model state
        torch.save(model.state_dict(), gconfig.model_save_path +
                   f'/model_epoch_{epoch}.pt')
        # save optimizers states
        torch.save({'c_disc': content_disc_opt.state_dict(
        ), 's_disc': style_disc_opt.state_dict(), 'ae_cls': vae_and_cls_opt.state_dict()}, gconfig.model_save_path+'/opt_epoch_{epoch}.pt')

    print("Training completed!!!")
