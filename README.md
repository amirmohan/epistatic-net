# Epistatic Net (EN) regularization
Epistatic Net (EN) is an algorithm to regularize deep neural networks (DNNs) by promoting sparsity in their spectral transform (i.e., the Walsh-Hadamard transform for binary inputs). EN reduces the sample complexity (i.e., the number of training data points required to achieve a certain prediction accuracy) of DNNs and provides an scalable method to explain them in terms of higher-order epistatic interactions in massive landscapes. The basic version of EN simply evaluates the output of the DNN to all the enumerations of input sequence (called the DNN landscape) at every iteration and backpropagates the L1 norm of the spectral transform of the resulting DNN landscape. At the heart of the scalable version of EN algorithm (EN-S) is an ADMM optimization algorithm which employs a fast sparse WHT recovery algorithm which evaluates the spectral transform of the DNN logarithmically faster using techniques from coding theory.

Details of EN regularization is provided in the paper ["Sparse Epistatic Regularization of Deep Neural Networks for Inferring Fitness Functions"](<https://www.biorxiv.org/content/10.1101/2020.11.24.396994v3.abstract>), where we have demonstrated the performance of EN in protein function prediction on various real-world biological landscapes including: 

1- Four canonical bacterial fitness landscapes, available at [this link](<https://github.com/harmslab/notebooks-nonlinear-high-order-epistasis>),

2- *E. quadricolor* fluorescent protein, available at [this link](<https://doi.org/10.1038/s41467-019-12130-8>),

3- Green fluorescent protein from *Aequorea victoria* (avGFP), available at [this link](<https://doi.org/10.6084/m9.figshare.3102154.v1>),

4- Immunoglobulin-binding domain of protein G (GB1) available at [this link](<https://elifesciences.org/articles/16965/figures>).



The paper is currently under final round of revision at *Nature Communications*.

This repository provides the EN package which includes the scripts required to implement the EN regularization algorithm at `Epistatic Net (EN).ipynb`, the scalable EN-S algorithm at `Scalable Epistatic Net (EN-S).ipynb`, and the dependencies such as the sparse WHT recovery algorithm called [SPRIGHT](<https://arxiv.org/abs/1508.06336>).

# Quick start
The EN software is conveniently available at `Epistatic Net (EN).ipynb`. The software is simple and self-explanatory. 

The scalable version of the EN algorithm (EN-S) is available at `Scalable Epistatic Net (EN-S).ipynb`. In order to run the code follow these steps:

1- Run `wht-sampling.ipynb` after setting the values for the number of mutations (n), the anticipated sparsity level (2^m), and number of delays per single bit in SPRIGHT (higher values increase robustness of the algorithm to noise, d={3,5,7} are good default values). Generate the uniform sampling patterns of proteins and store them in a folder. Currently the default values are `n=13`, `m=4`, and `d=3` for the *E. quadricolor* fluorescent protein.

2- Set `num_bits = n` and `num_delays_per_bit=d` in the `utils.py file`.

3- Run `Scalable Epistatic Net (EN-S).ipynb` with the appropriate links to the folder of uniform sampling patterns. Note that hyperparameters such as the regularization level and ADMM parameter's should be tuned based on the problem using cross validation.



