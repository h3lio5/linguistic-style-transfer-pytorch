## Overview
 * Map the sentences to a latent space using VAE framework.
 * The latent space is artificially divided into style space and content space, and the model is encouraged to disentangle
    the latent space with respect to the above two features,namely, style and content.
 * To accomplish this, the VAE loss (ELBO) is augmented with two auxiliary losses,namely, multitask loss and adversary loss.
 * Multitask loss:
    * It operates on the latent space to ensure that it does contain the information we wish to encode,    i.e. it encourages    the style space and content space to preserve the style and content information respectively.
    * The style style classifier is trained to predict the style label label .
    * The content classifier is trained to predict the Bag of Words (BoW) representation of the sentence.
  * Adversarial Loss:
    * The adversarial loss, on the contrary, minimizes the predictability of information that should not be contained
      in a given latent space.
    * The disentanglement of style space and content space is accomplished by adversarial learning procedure.
    * Adversarial procedure is similar to that of the original GAN, where discriminator is trained to correctly classify 
      the samples and the generator is trained to fool the discriminator by producing samples indistinguishable from 
      the original data samples.
    * In this setting, for style space, the style discriminator is trained to predict the style label and the style generator
      is trained to increase the entropy of the predictions/softmax output since higher entropy corresponds to lesser
      information. Similarly, the same procedure is repeated for content space.
   * To address the posterior collapse issue that usually occurs when powerful decoders like LSTMs are used, sigmoid KL 
     annealing is used during training. Also, the latent embedding is concatenated to word embeddings at every time step of
     the decoder.
   * During the last epoch of training, average of style embeddings over whole train data is calculated for both the styles(positive and negative sentiment). These average positive and negative style embeddings are approximated to be 
     estimates of positive and negative style embeddings.
   * During inference phase, deteremine the content embedding of the sentence and concatenate the estimated style embedding
     of the opposite sentiment to it. Use this latent embedding for decoding/transfering style.
