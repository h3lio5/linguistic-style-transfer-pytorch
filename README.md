# Linguistic Style Transfer 
Implementation of the paper `Disentangled Representation Learning for Non-Parallel Text Style Transfer`[(link)](https://www.aclweb.org/anthology/P19-1041.pdf) in Pytorch

## Abstract
  This paper tackles the problem of disentangling the latent representations of style and content in language models.
  We propose a simple yet effective approach, which incorporates auxiliary multi-task and adversarial objectives, for 
  style prediction and bag-of-words prediction, respectively. We show, both qualitatively and quantitatively, that the 
  style and content are indeed disentangled in the latent space. This disentangled latent representation learning can be                  applied to style transfer on non-parallel corpora. We achieve high performance in terms of transfer accuracy, content     preservation, and language fluency, in comparision to various previous approaches.
     
 ## Training and Inference
  Illustration of training and inference.    
  ![training_and_inference](images/resized_training_inference.png)
  
  ## Documents
  ### Dependencies
  To download all the dependencies, run the following command -    
  ` pip3 install -r requirements.txt`
  ### Directory Description
  Run the following command from the root directory to download, preprocess data, create vocab and word embeddings.          
  ` python3 setup_data.py `           
  The repository should like this after running the above command -         
  <pre><code>
root
├──  images
├──  linguistic_style_transfer_pytorch
│    ├── data
│    │   ├── raw
│    │   │   ├── sentiment.train.0
│    │   │   └── sentiment.train.1
│    │   ├── clean
│    │   │    ├── yelp_train_data.txt
│    │   │    └── yelp_train_labels.txt
│    │   ├── lexicon
│    │   │   ├── positive-words.txt
│    │   │   └── negative-words.txt
│    │   ├── bow.json
│    │   ├── embedding.txt
│    │   ├── index2word.json
│    │   ├── word2index.json
│    │   └── word_embeddings.npy
│    ├── utils 
│    │    ├── __init__.py
│    │    ├── preprocess.py
│    │    ├── train_w2v.py
│    │    └── vocab.py
│    ├── __init__.py
│    ├── model.py
│    ├── data_loader.py
│    ├── config.py
│    └── checkpoints/
├──  README.md
├──  setup.py
├──  setup_data.py
├──  train.py
└──  generate.py
</code></pre>
To train the model, run the following command in the root folder-         
`python3 train.py`       
### Resources
* Original paper `Disentangled Representation Learning for Non-Parallel Text Style Transfer` [(link)](https://www.aclweb.org/anthology/P19-1041.pdf)
* tensorflow implementation by the author [link](https://github.com/vineetjohn/linguistic-style-transfer)
