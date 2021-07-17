# Epistatic Net (EN) regularizarion
Epistatic Net (EN) is an algorithm to regularize deep neural networks (DNNs) by promoting sparsity in their spectral transform (i.e., the Walsh-Hadamard transfrom for binary input). EN reduces the sample complexity (i.e., the number of training data points to acheive a certian prediction accuracy) of DNNs and provides an scalable method to explain them in terms of higher-order epistatic interactions. The basic version of EN simply evaluates the output of the DNN to every enumerations of input sequence (called the DNN landscape) at every iteration and backpropagates the L1 norm of the spectral transform of the DNN landscape. At the hearth of the scalabel verion of EN algorithm (EN-S) is an ADMM optimization algorithm which employs a fast sparse WHT recovery algorothm which evaluates the spectral transfrom of the DNN logarithmiclly faster using techniques from coding theory.

Details of EN regularization is provided in the paper ["Sparse Epistatic Regularization of Deep Neural Networks for Inferring Fitness Functions"](<https://www.biorxiv.org/content/10.1101/2020.11.24.396994v3.abstract>), where we have demonstrated the perforamnce of EN in protein function prediction on various real-world biological landscapes including: 

1- Four canonical bacterial fitness landscapes, available at [this link](<https://github.com/harmslab/notebooks-nonlinear-high-order-epistasis>),

2- *E. quadricolor* fluorescent protein, available at [this link](<https://doi.org/10.1038/s41467-019-12130-8>),

3- Green fluorescent protein from *Aequorea victoria* (avGFP), available at [this link](<https://doi.org/10.6084/m9.figshare.3102154.v1>),

4- Immunoglobulin-binding domain of protein G (GB1) available at [this link](<https://elifesciences.org/articles/16965/figures>).



The paper is currenly under final round of revision at *Nature Communications*.

This repository provides the EN package which includes the scripts required to implement the EN regularization algorithm at `Epistatic Net (EN).ipynb`, the scalable EN-S algorithm at `Scalable Epistatic Net (EN-S).ipynb`, and the dependencies such as the sparse WHT recovrey algorithm called [SPRIGHT](<https://arxiv.org/abs/1508.06336>).

# Quick start
The EN software is conviniently available at `Epistatic Net (EN).ipynb`. The software is simple and self-explanatory. 
