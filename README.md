# Epistatic-Net (EN) regularizarion
Epistatic Net is an algorithm to regularize deep neural networks (DNNs) by promoting sparsity in their spectral transform (i.e., the Walsh-Hadamard transfrom for binary input). EN reduces the sample complexity (i.e., the number of training data points to acheive a certian prediction accuracy) of DNNs and provides an scalable method to explain them in terms of higher-order epistatic interactions.

Details of EN regularization is provided in the paper ["Sparse Epistatic Regularization of Deep Neural Networks for Inferring Fitness Functions"](<https://www.biorxiv.org/content/10.1101/2020.11.24.396994v3.abstract>), where we have demonstrated the perforamnce of EN in protein function prediction on various real-world biological landscapes including: 

1- Four canonical bacterial fitness landscapes available at [this link](<https://github.com/harmslab/notebooks-nonlinear-high-order-epistasis>)

2- *E. quadricolor* fluorescent protein available at [this link](<https://doi.org/10.1038/s41467-019-12130-8>)

3- *Aequorea victoria* (avGFP) landscape available at [this link](<https://doi.org/10.6084/m9.figshare.3102154.v1>)

4- available at [this link](<https://elifesciences.org/articles/16965/figures >) 



The paper is currenly under final round of revision at *Nature Communications*.

This repository provides the EN package which includes the scripts required to predict the DNA repair outcome as well as a complete set of trained SPROUT models. The trained models are stored in the `model` folder of the repository. The main script of the package is the `SPROUT_predict` script which loads the pretrained models, asks for input query depending on the mode selected, and outputs the predicted statistics of the repair outcome.  a


# Quick start
