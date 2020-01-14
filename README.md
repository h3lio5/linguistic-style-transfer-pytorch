# Linguistic Style Transfer 
Implementation of the paper `Disentangled Representation Learning for Non-Parallel Text Style Transfer`[(link)](https://www.aclweb.org/anthology/P19-1041.pdf) in Pytorch

## Abstract
  This paper tackles the problem of disentangling the latent representations of style and content in language models.
  We propose a simple yet effective approach, which incorporates auxiliary multi-task and adversarial objectives, for 
  style prediction and bag-of-words prediction, respectively. We show, both qualitatively and quantitatively, that the 
  style and content are indeed disentangled in the latent space. This disentangled latent representation learning can be                  applied to style transfer on non-parallel corpora. We achieve high performance in terms of transfer accuracy, content     preservation, and language fluency, in comparision to various previous approaches.

## Overview
* ### <ins>Introduction</ins>
  * Map the sentences to a latent space using VAE framework.
  * The latent space is artificially divided into style space and content space, and the model is encouraged to disentangle
    the latent space with respect to the above two features,namely, style and content.
  * To accomplish this, the VAE loss (ELBO) is augmented with two auxiliary losses,namely, multitask loss and adversary loss.
* ### <ins>Model</ins>
  * <strong><ins>VAE Loss</ins>:</strong>
    * J<sub>AE</sub>(θ<sub>E</sub>, θ<sub>D</sub>) = − E<sub>q<sub>E</sub>(z|x)</sub>[log p(x|z)] + λ<sub>kl</sub>        KL(q(z|x)||p(z)),   
     where λkl is the hyperparameter balancing the reconstruction loss and the KL term.
  * <strong><ins>Multitask Loss</ins>:</strong>
    * It operates on the latent space to ensure that the space does contain the information we wish to encode,i.e. 
      it encourages the style space and content space to preserve the style and content information respectively.
    * <strong><ins>Style Loss</ins></strong>:
      * A softmax classifier on the style space <strong>s</strong> is trained to predict the style label, given by      
      <i><strong>y</strong></i><sub>s</sub> = softmax(W<sub>mul(s)</sub><strong>s</strong> + b<sub>mul(s)</sub>)       
      * θ<sub>mul(s)</sub> = [W<sub>mul(s)</sub>;b<sub>mul(s)</sub>] are parameters of the style classifier in the 
        multitask setting, and <em>y<sub>s</sub></em> is the output of softmax layer.
      * The classifier is trained with cross-entropy loss against ground truth distribution t<sub>s</sub>(.) by
        J<sub>mul(s)</sub>(θ<sub>E</sub>;θ<sub>mul(s)</sub>)=-𝝨<sub>l∊labels</sub>t<sub>s</sub>(l)logy<sub>s</sub>(l)
