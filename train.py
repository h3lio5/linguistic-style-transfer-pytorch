import torch
from torch.utils.data import DataLoader
from linguistic_style_transfer_pytorch.config import GeneralConfig, ModelConfig
from linguistic_style_transfer_pytorch.data_loader import TextDataset
from linguistic_style_transfer_pytorch.model import AdversarialVAE
from tqdm import tqdm, trange
import os
import numpy as np

use_cuda = False
device = torch.device('cpu')
if torch.cuda.is_available():
    use_cuda = True
    device = torch.device('cuda:0')
print('using backend(',device,')')

if __name__ == "__main__":

    mconfig = ModelConfig()
    gconfig = GeneralConfig()
    weights = torch.tensor(np.load(gconfig.word_embedding_path), device=device, dtype=torch.float)
    model = AdversarialVAE(inference=False, weight=weights, device=device)
    if use_cuda:
        model = model.cuda()

    #=============== Define dataloader ================#
    train_dataset = TextDataset(mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=mconfig.batch_size, drop_last=True, pin_memory=True)
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
    for epoch in range(mconfig.epochs):
        content_loss = 0.0
        style_loss = 0.0
        vae_cls_loss = 0.0
        t = tqdm(train_dataloader, desc=f'Epoch {epoch+1}', unit=' batch', leave=False)
        for iteration, batch in enumerate(t):

            # unpacking
            sequences, seq_lens, labels, bow_rep = batch
            if use_cuda:
                sequences = sequences.cuda()
                seq_lens = seq_lens.cuda()
                labels = labels.cuda()
                bow_rep = bow_rep.cuda()
            content_disc_loss, style_disc_loss, vae_and_cls_loss = model(
                sequences, seq_lens.squeeze(1), labels, bow_rep, iteration+1)
            content_loss = (content_loss*iteration + content_disc_loss.item())/(iteration+1)
            style_loss = (style_loss*iteration + style_disc_loss.item())/(iteration+1)
            vae_cls_loss = (vae_cls_loss*iteration + vae_and_cls_loss.item())/(iteration+1)
            t.set_postfix({'Content loss' : content_loss, 'Style loss': style_loss, 'VAE loss': vae_cls_loss})
            t.update()

            #============== Update Adversary/Discriminator parameters ===========#
            # update content discriminator parametes
            # we need to retain the computation graph so that discriminator predictions are
            # not freed as we need them to calculate entropy.
            # Note that even even detaching the discriminator branch won't help us since it
            # will be freed and delete all the intermediary values(predictions, in our case).
            # Hence, with no access to this branch we can't backprop the entropy loss
            content_disc_loss.backward(retain_graph=True)
            content_disc_opt.step()
            content_disc_opt.zero_grad()

            # update style discriminator parameters
            style_disc_loss.backward(retain_graph=True)
            style_disc_opt.step()
            style_disc_opt.zero_grad()

            #=============== Update VAE and classifier parameters ===============#
            vae_and_cls_loss.backward()
            vae_and_cls_opt.step()
            vae_and_cls_opt.zero_grad()

        print(f"Epoch {epoch+1} completed")
        print(f"Losses - Content: {content_loss} Style: {style_loss}, VAE : {vae_cls_loss}")
        #================ Saving states ==========================#
        if not os.path.exists(gconfig.model_save_path):
            os.mkdir(gconfig.model_save_path)
        # save model state
        torch.save(model.state_dict(), gconfig.model_save_path +
                   f'/model_epoch_{epoch+1}.pt')
        # save optimizers states
        torch.save({'content_disc': content_disc_opt.state_dict(
        ), 'style_disc': style_disc_opt.state_dict(), 'vae_and_cls': vae_and_cls_opt.state_dict()}, gconfig.model_save_path+'/opt_epoch_{epoch}.pt')

    print("Training completed!!!")
