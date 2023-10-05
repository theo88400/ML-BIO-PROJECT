# ML-BIO-PROJECT
ML-BIO Project : Study impact of low data available for federated learning

# Installation
``python3 -m pip install -r requirements.txt``

### Dataset
https://www.kaggle.com/competitions/siim-isic-melanoma-classification/data

Should be put in data/ folder but as it's really heavy (30GB), we will load the data and train our models directly on Kaggle.

# Project's structure

## MyFedProx
This is custom code to simulated federated learning. The code is inspired by the TP5 on federated learning and has been adapted to work with the SIIM_ISIC dataset. It contains methods to either do a classical training on a whole dataset that will be used as witness and methods to simulate federate learning used to answer our question.

## Notebooks
There is several notebooks used to run the training and analyse the results.
    - One will serve as a witness and perform a classical training
    - One for federated learning with a fair split between clients
    - One for federated learning with biais introduced in some client dataset

# Results

TODO