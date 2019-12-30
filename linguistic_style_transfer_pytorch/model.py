import torch
import torch.nn as nn
from linguistic_style_transfer_pytorch.config import ModelConfig, Config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


mconfig = ModelConfig()
gconfig = Config()


class AutoEncoder(nn.Module):
    """
    """

    def __init__(self, inference=False):
        """
        Initialize networks
        """
        # Inference mode or training mode
        self.inference = inference
        # word embeddings
        self.embedding = nn.Embedding(
            gconfig.vocab_size, mconfig.embedding_size)
        # Encoder model
        self.encoder = nn.GRU(
            mconfig.embedding_size, mconfig.hidden_dim, batch_first=True, bidirectional=True)
        # content latent embedding
        self.get_content_emb = nn.Linear(
            mconfig.hidden_dim, mconfig.content_hidden_dim)
        # style latent embedding
        self.get_style_emb = nn.Linear(
            mconfig.hidden_dim, mconfig.style_hidden_dim)
        # Discriminator/adversary
        self.style_disc = nn.Linear(
            mconfig.content_hidden_dim, mconfig.num_style)
        self.content_disc = nn.Linear(
            mconfig.style_hidden_dim, mconfig.content_bow_dim)
        # classifier
        self.content_classifier = nn.Linear(
            mconfig.content_hidden_dim, mconfig.content_bow_dim)
        self.style_classifier = nn.Linear(
            mconfig.style_hidden_dim, mconfig.num_style)
        # Decoder
        # Note: input embeddings are concatenated with the sampled latent vector at every step
        self.decoder = nn.GRUCell(
            mconfig.embedding_size + mconfig.content_hidden_dim + mconfig.style_hidden_dim, mconfig.hidden_dim)
        # dropout
        self.dropout = nn.Dropout(mconfig.emb_dropout)

    def sample_prior(self, mu, log_sigma):
        """
        Returns samples drawn from the latent space constrained to 
        follow diagonal Gaussian
        """
        epsilon = torch.randn(mu.size(1))
        return mu + epsilon*torch.exp(log_sigma)

    def get_content_disc_loss(self, style_emb, content_bow):
        """
        It essentially quantifies the amount of information about content  
        contained in the style space
        Returns:
        cross entropy loss of content discriminator
        """
        # predictions
        # Note: detach the style embedding since when don't want the gradient to flow
        #       all the way to the encoder. content_disc_loss is used only to change the
        #       parameters of the discriminator network
        preds = nn.Softmax(self.content_disc(self.dropout(style_emb.detach())))
        # label smoothing
        smoothed_content_bow = content_bow * \
            (1-mconfig.label_smoothing) + \
            mconfig.label_smoothing/mconfig.content_bow_dim
        # calculate cross entropy loss
        content_disc_loss = nn.BCELoss(preds, smoothed_content_bow)

        return content_disc_loss

    def get_style_disc_loss(self, content_emb, style_labels):
        """
        It essentially quantifies the amount of information about style  
        contained in the content space
        Returns:
        cross entropy loss of style discriminator
        """
        # predictions
        # Note: detach the content embedding since when don't want the gradient to flow
        #       all the way to the encoder. style_disc_loss is used only to change the
        #       parameters of the discriminator network
        preds = nn.Softmax(self.style_disc(self.dropout(content_emb.detach())))
        # label smoothing
        smoothed_style_labels = style_labels * \
            (1-mconfig.label_smoothing) + \
            mconfig.label_smoothing/mconfig.num_style
        # calculate cross entropy loss
        style_disc_loss = nn.BCELoss(preds, smoothed_style_labels)

        return style_disc_loss

    def get_content_mul_loss(self, content_emb, content_bow):
        """
        This loss quantifies the amount of content information preserved 
        in the content space
        Returns:
        cross entropy loss of the content classifier
        """
        # predictions
        preds = nn.Softmax(self.content_classifier(self.dropout(content_emb)))
        # label smoothing
        smoothed_content_bow = content_bow * \
            (1-mconfig.label_smoothing) + \
            mconfig.label_smoothing/mconfig.content_bow_dim
        # calculate cross entropy loss
        content_mul_loss = nn.BCELoss(preds, smoothed_content_bow)

        return content_mul_loss

    def get_style_mul_loss(self, style_emb, style_labels):
        """
        This loss quantifies the amount of content information preserved 
        in the content space
        Returns:
        cross entropy loss of the content classifier
        """
        # predictions
        preds = nn.Softmax(self.content_classifier(self.dropout(style_emb)))
        # label smoothing
        smoothed_style_labels = style_labels * \
            (1-mconfig.label_smoothing) + \
            mconfig.label_smoothing/mconfig.num_style
        # calculate cross entropy loss
        style_mul_loss = nn.BCELoss(preds, smoothed_style_labels)

        return style_mul_loss

    def forward(self, sequences, seq_lengths, style_labels, content_bow):
        """
        Returns the logits
        """
        embedded_seqs = self.dropout(self.embedding(sequences))
        # pack the sequences to reduce unnecessary computations
        packed_seqs = pack_padded_sequence(
            embedded_seqs, lengths=seq_lengths, batch_first=True)
        packed_output, (final_hidden_state,
                        final_cell_state) = self.encoder(packed_seqs)
        # get content and style embeddings from the sentence embeddings,i.e. final_hidden_state
        content_emb_mu, content_emb_sigma = self.get_content_emb(
            final_hidden_state)
        style_emb_mu, style_emb_sigma = self.get_style_emb(final_hidden_state)
        # sample content and style embeddings from their respective latent spaces
        sampled_content_emb = self.sample_prior(
            content_emb_mu, content_emb_sigma)
        sampled_style_emb = self.sample_prior(style_emb_mu, style_emb_sigma)

        #### Losses on content space ###
        # Discriminator Loss
        content_disc_loss = self.get_content_disc_loss(
            sampled_style_emb, content_bow)
        # Multitask loss
        content_mul_loss = self.get_content_mul_loss(
            sampled_content_emb, content_bow)
