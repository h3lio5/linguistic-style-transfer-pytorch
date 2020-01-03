import torch
import torch.nn as nn
from linguistic_style_transfer_pytorch.config import ModelConfig, GeneralConfig
from torch.nn.utils.rnn import pack_padded_sequence
import math

mconfig = ModelConfig()
gconfig = GeneralConfig()


class AdversarialVAE(nn.Module):
    """
    Model architecture defined according to the paper 
    'Disentangled Representation Learning for Non-Parallel Text Style Transfer'
    https://www.aclweb.org/anthology/P19-1041.pdf

    """

    def __init__(self, inference=False):
        """
        Initialize networks
        """
        super(AdversarialVAE, self).__init__()
        # Inference mode or training mode
        self.inference = inference
        # word embeddings
        self.embedding = nn.Embedding(
            mconfig.vocab_size, mconfig.embedding_size)
        #================ Encoder model =============#
        self.encoder = nn.GRU(
            mconfig.embedding_size, mconfig.hidden_dim, batch_first=True, bidirectional=True)
        # content latent embedding
        self.get_content_emb = nn.Linear(
            mconfig.hidden_dim, mconfig.content_hidden_dim)
        # style latent embedding
        self.get_style_emb = nn.Linear(
            mconfig.hidden_dim, mconfig.style_hidden_dim)
        #=============== Discriminator/adversary============#
        self.style_disc = nn.Linear(
            mconfig.content_hidden_dim, mconfig.num_style)
        self.content_disc = nn.Linear(
            mconfig.style_hidden_dim, mconfig.content_bow_dim)
        #=============== Classifier =============#
        self.content_classifier = nn.Linear(
            mconfig.content_hidden_dim, mconfig.content_bow_dim)
        self.style_classifier = nn.Linear(
            mconfig.style_hidden_dim, mconfig.num_style)
        #=============== Decoder =================#
        # Note: input embeddings are concatenated with the sampled latent vector at every step
        self.decoder = nn.GRUCell(
            mconfig.embedding_size + mconfig.generative_emb_dim, mconfig.hidden_dim)
        self.projector = nn.Linear(mconfig.hidden_dim, mconfig.vocab_size)
        # dropout
        self.dropout = nn.Dropout(mconfig.dropout)

    def forward(self, sequences, seq_lengths, style_labels, content_bow, iteration):
        """
        Args:
            sequences : token indices of input sentences of shape = (batch_size,max_seq_length)
            seq_lengths: actual lengths of input sentences before padding, shape = (batch_size,1)
            style_labels: labels of sentiment of the input sentences, shape = (batch_size,1)
            content_bow: Bag of Words representations of the input sentences, shape = (batch_size,bow_hidden_size)
            iteration: number of iterations completed till now; used for KL annealing

        Returns:
            content_disc_loss: loss incurred by content discriminator/adversary
            style_disc_loss  : loss incurred by style discriminator/adversary
            vae_and_classifier_loss : consists of loss incurred by autoencoder, content and style
                               classifiers
        """
        embedded_seqs = self.dropout(self.embedding(sequences))
        # pack the sequences to reduce unnecessary computations
        packed_seqs = pack_padded_sequence(
            embedded_seqs, lengths=seq_lengths, batch_first=True)
        packed_output, (final_hidden_state,
                        final_cell_state) = self.encoder(packed_seqs)
        # get content and style embeddings from the sentence embeddings,i.e. final_hidden_state
        content_emb_mu, content_emb_log_sigma = self.get_content_emb(
            final_hidden_state)
        style_emb_mu, style_emb_log_sigma = self.get_style_emb(
            final_hidden_state)
        # sample content and style embeddings from their respective latent spaces
        sampled_content_emb = self.sample_prior(
            content_emb_mu, content_emb_log_sigma)
        sampled_style_emb = self.sample_prior(
            style_emb_mu, style_emb_log_sigma)
        # Generative embedding
        generative_emb = torch.cat(
            sampled_style_emb, sampled_content_emb, axis=1)

        #=========== Losses on content space =============#
        # Discriminator Loss
        content_disc_preds = self.get_content_disc_preds(sampled_style_emb)
        content_disc_loss = self.get_content_disc_loss(
            content_disc_preds, content_bow)
        # adversarial entropy
        content_entropy_loss = self.get_entropy_loss(content_disc_preds)
        # Multitask loss
        content_mul_loss = self.get_content_mul_loss(
            sampled_content_emb, content_bow)

        #============ Losses on style space ================#
        # Discriminator loss
        style_disc_preds = self.get_style_disc_preds(sampled_content_emb)
        style_disc_loss = self.get_style_disc_loss(
            style_disc_preds, style_labels)
        # adversarial entropy
        style_entropy_loss = self.get_entropy_loss(style_disc_preds)
        # Multitask loss
        style_mul_loss = self.get_style_mul_loss(
            sampled_style_emb, style_labels)

        #============== KL losses ===========#
        # Style space
        style_kl_loss = self.get_kl_loss(
            style_emb_mu, style_emb_log_sigma)
        if iteration < mconfig.kl_anneal_iterations:
            style_kl_loss = self.get_annealed_weight(
                iteration, mconfig.style_kl_lambda) * style_kl_loss
        # Content space
        content_kl_loss = self.get_kl_loss(
            content_emb_mu, content_emb_log_sigma)
        if iteration < mconfig.kl_anneal_iterations:
            content_kl_loss = self.get_annealed_weight(
                iteration, mconfig.content_kl_lambda) * content_kl_loss

        #=============== reconstruction ================#
        reconstructed_sentences = self.generate_sentences(
            sequences, generative_emb)
        reconstruction_loss = self.get_recon_loss(
            reconstructed_sentences, sequences)
        #================ total weighted loss ==========#
        vae_and_classifier_loss = mconfig.content_adversary_loss_weight * content_entropy_loss + \
            mconfig.style_adversary_loss_weight * style_entropy_loss + \
            mconfig.style_multitask_loss_weight * style_mul_loss + \
            mconfig.content_multitask_loss_weight * content_mul_loss + \
            reconstruction_loss

        return content_disc_loss, style_disc_loss, vae_and_classifier_loss

    def get_params(self):
        """
        Returns:
            content_disc_params: parameters of the content discriminator/adversary
            style_disc_params  : parameters of the style discriminator/adversary
            other_params       : parameters of the vae and classifiers
        """

        content_disc_params = self.content_disc.parameters()
        style_disc_params = self.style_disc.parameters()
        other_params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + \
            list(self.style_classifier.parameters()) + \
            list(self.content_classifier.parameters())

        return content_disc_params, style_disc_params, other_params

    def sample_prior(self, mu, log_sigma):
        """
        Returns samples drawn from the latent space constrained to
        follow diagonal Gaussian
        """
        epsilon = torch.randn(mu.size(1))
        return mu + epsilon*torch.exp(log_sigma)

    def get_content_disc_preds(self, style_emb):
        """
        Returns predictions about the content using style embedding
        as input
        output shape : [batch_size,content_bow_dim]
        """
        # predictions
        # Note: detach the style embedding since when don't want the gradient to flow
        #       all the way to the encoder. content_disc_loss is used only to change the
        #       parameters of the discriminator network
        preds = nn.Softmax(self.content_disc(self.dropout(style_emb.detach())))

        return preds

    def get_content_disc_loss(self, content_disc_preds, content_bow):
        """
        It essentially quantifies the amount of information about content
        contained in the style space
        Returns:
        cross entropy loss of content discriminator
        """
        # label smoothing
        smoothed_content_bow = content_bow * \
            (1-mconfig.label_smoothing) + \
            mconfig.label_smoothing/mconfig.content_bow_dim
        # calculate cross entropy loss
        content_disc_loss = nn.BCELoss(
            content_disc_preds, smoothed_content_bow)

        return content_disc_loss

    def get_style_disc_preds(self, content_emb):
        """
        Returns predictions about style using content embeddings
        as input
        output shape: [batch_size,num_style]
        """
        # predictions
        # Note: detach the content embedding since when don't want the gradient to flow
        #       all the way to the encoder. style_disc_loss is used only to change the
        #       parameters of the discriminator network
        preds = nn.Softmax(self.style_disc(self.dropout(content_emb.detach())))

        return preds

    def get_style_disc_loss(self, style_disc_preds, style_labels):
        """
        It essentially quantifies the amount of information about style
        contained in the content space
        Returns:
        cross entropy loss of style discriminator
        """
        # label smoothing
        smoothed_style_labels = style_labels * \
            (1-mconfig.label_smoothing) + \
            mconfig.label_smoothing/mconfig.num_style
        # calculate cross entropy loss
        style_disc_loss = nn.BCELoss(style_disc_preds, smoothed_style_labels)

        return style_disc_loss

    def get_entropy_loss(self, preds):
        """
        Returns the entropy loss: negative of the entropy present in the
        input distribution
        """
        return torch.mean(torch.sum(preds * torch.log(preds + mconfig.epsilon), dim=1))

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
        This loss quantifies the amount of style information preserved
        in the style space
        Returns:
        cross entropy loss of the style classifier
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

    def get_annealed_weight(self, iteration, lambda_weight):
        """
        Args:
            iteration(int): Number of iterations compeleted till now
            lambda_weight(float): KL penalty weight
        Returns:
            Annealed weight(float)
        """
        return (math.tanh(
            (iteration - mconfig.kl_anneal_iterations * 1.5) /
            (mconfig.kl_anneal_iterations / 3))
            + 1) * lambda_weight

    def get_kl_loss(self, mu, log_sigma):
        """
        Args:
            mu: batch of means of the gaussian distribution followed by the latent variables
            log_sigma: batch of log variances(log_sigma) of the gaussian distribution followed by the latent variables
        Returns:
            total loss(float)
        """
        kl_loss = torch.mean((-0.5*torch.sum(1+log_sigma -
                                             log_sigma.exp()-mu.pow(2), dim=1)))
        return kl_loss

    def generate_sentences(self, input_sentences, latent_emb):
        """
        Args:
           input_sentences: batch of token indices of input sentences, shape = (batch_size,max_seq_length)
           latent_emb: generative embedding formed by the concatenation of sampled style and 
                       content latent embeddings, shape = (batch_size,mconfig.)
        Returns:
            output_sentences: batch of token indices or logits of generated sentences based on the
            mode of operation. 
            modes:
                train: shape = (max_seq_len,batch_size,vocab_size)
                inference: shape = (max_seq_len,batch_size)
        """
        # Training mode
        if not self.inference:
            # Prepend the input sentences with <sos> token
            sos_token_tensor = torch.LongTensor(
                [gconfig.predefined_word_index['<sos>']]).unsqueeze(0).repeat(mconfig.batch_size, 1)
            input_sentences = torch.cat(
                (sos_token_tensor, input_sentences), dim=1)
            sentence_embs = self.dropout(self.embedding(input_sentences))
            # Make the latent embedding compatible for concatenation
            latent_emb = latent_emb.unsqueeze(1).repeat(
                1, mconfig.max_seq_len, 1)
            gen_sent_embs = torch.cat(
                sentence_embs, latent_emb, dim=2)
            # Delete latent embedding and sos token tensor to reduce memory usage
            del latent_emb, sos_token_tensor
            output_sentences = torch.zeros(
                mconfig.max_seq_len, mconfig.batch_size, mconfig.vocab_size)
            # initialize hidden state
            hidden_states = torch.zeros(mconfig.batch_size, mconfig.hidden_dim)
            # generate sentences one word at a time in a loop
            for idx in range(mconfig.max_seq_len):
                # get words at the index idx from all the batches
                words = gen_sent_embs[:, idx, :]
                hidden_states = self.decoder(words, hidden_states)
                # project over vocab space
                next_word_logits = self.projector(hidden_states)
                output_sentences[idx] = next_word_logits
        # if inference mode is on
        else:

            sos_token_tensor = torch.LongTensor(
                [gconfig.predefined_word_index['<sos>']]).unsqueeze(0).repeat(mconfig.batch_size, 1)
            word_embs = self.dropout(self.embedding(sos_token_tensor))
            hidden_states = torch.zeros(mconfig.batch_size, mconfig.hidden_dim)
            # Store output sentences
            output_sentences = torch.zeros(
                mconfig.max_seq_len, mconfig.batch_size)
            with torch.no_grad:
                # Greedily generate new words at a time
                for idx in range(mconfig.max_seq_len):
                    hidden_states = self.decoder(word_embs, hidden_states)
                    next_word_probs = nn.Softmax(self.projector(hidden_states))
                    next_words = next_word_probs.argmax(1)
                    output_sentences[idx] = next_words
                    word_embs = self.dropout(self.embedding(next_words))

        return output_sentences

    def get_recon_loss(self, output_logits, input_sentences):
        """
        Args:
            output_logits: logits of output sentences at each time step, shape = (max_seq_length,batch_size,vocab_size)
            input_sentences: batch of token indices of input sentences, shape = (batch_size,max_seq_length)

        Returns:
            reconstruction loss calculated using cross entropy loss function
        """

        loss = nn.CrossEntropyLoss(ignore_index=0)
        recon_loss = loss(
            output_logits.view(-1, mconfig.vocab_size), input_sentences.view(-1))

        return recon_loss
