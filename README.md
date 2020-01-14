# Linguistic Style Transfer 
Implementation of the paper `Disentangled Representation Learning for Non-Parallel Text Style Transfer`[(link)](https://www.aclweb.org/anthology/P19-1041.pdf) in Pytorch

## Abstract
  This paper tackles the problem of disentangling the latent representations of style and content in language models.
  We propose a simple yet effective approach, which incorporates auxiliary multi-task and adversarial objectives, for 
  style prediction and bag-of-words prediction, respectively. We show, both qualitatively and quantitatively, that the 
  style and content are indeed disentangled in the latent space. This disentangled latent representation learning can be                  applied to style transfer on non-parallel corpora. We achieve high performance in terms of transfer accuracy, content     preservation, and language fluency, in comparision to various previous approaches.

## Overview
* ### Introduction
  * Map the sentences to a latent space using VAE framework.
  * The latent space is artificially divided into style space and content space, and the model is encouraged to disentangle
    the latent space with respect to the above two features,namely, style and content.
  * To accomplish this, the VAE loss (ELBO) is augmented with two auxiliary losses,namely, multitask loss and adversary loss.
* ### Model
  * <strong><ins>Multitask Loss</ins>:</strong>
