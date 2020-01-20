# Linguistic Style Transfer 
Implementation of the paper `Disentangled Representation Learning for Non-Parallel Text Style Transfer`[(link)](https://www.aclweb.org/anthology/P19-1041.pdf) in Pytorch

## Abstract
  This paper tackles the problem of disentangling the latent representations of style and content in language models.
  We propose a simple yet effective approach, which incorporates auxiliary multi-task and adversarial objectives, for 
  style prediction and bag-of-words prediction, respectively. We show, both qualitatively and quantitatively, that the 
  style and content are indeed disentangled in the latent space. This disentangled latent representation learning can be                  applied to style transfer on non-parallel corpora. We achieve high performance in terms of transfer accuracy, content     preservation, and language fluency, in comparision to various previous approaches.

To get a basic overview of the paper, read the [summary](summary.md).
 ## Training and Inference
  Illustration of training and inference.    
  ![training_and_inference](images/resized_training_inference.png)
  
  ## Documents
  ### Dependencies
  To download all the dependencies, run the following command -    
  ` pip3 install -r requirements.txt`
  ### Directory Description
   Overview of the repository.       
  <pre><code>
root
├──  images
├──  linguistic_style_transfer_pytorch
│    ├── data
│    │   ├── raw/*
│    │   ├── clean/*  
│    │   ├── lexicon/*
│    │   └──  embedding.txt     
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
<strong> Note:</strong> Run all the commands from the root directory.      
To train the model,                
`python3 train.py`        
To generate style transfered sentence,       
`python3 generate.py`      
The user will be prompted to enter the source sentence and the target style on running the above command:       
Example:           
<pre><code>
Enter the source sentence: the book is good
Enter the target style: pos or neg: neg
Style transfered sentence: the book is boring
</code></pre>
### Resources
* Original paper `Disentangled Representation Learning for Non-Parallel Text Style Transfer` [(link)](https://www.aclweb.org/anthology/P19-1041.pdf)
* tensorflow implementation by the author [link](https://github.com/vineetjohn/linguistic-style-transfer)
