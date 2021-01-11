NBGNet - NeuroBondGraph Network
====================================================

This code implements the model from the paper "[Bi-directional Modeling Between Cross-Scale Neural Activity](https://www.biorxiv.org/content/10.1101/2020.11.30.404244v1)". It is a sparsely-connected recurrent neural network designed for investigating multi-scale neural data, but can be applied to any multi-scale time series data. The NBGNet is able to capture the dynamics and thus infer the neural activity from different scales. For example, in the paper, local field potentials and screw electrocorticography are brain activity recorded in different scales. The NBGNet is able to reconstruct the screw electrocorticography from local field potentials, and vice versa.   

Prerequisites
----------------------------------------------------

The code is written in Python 3.7.4. You will also need:
- **Keras** version 2.2.4 or above

Getting started
----------------------------------------------------

Before starting, you need


## Reference

- Y. J. Chang, Y. I. Chen, H. C. Yeh, J. M. Carmena, & S. R. Santacruz (2020). [Bi-directional Modeling Between Cross-Scale Neural Activity](https://www.biorxiv.org/content/10.1101/2020.11.30.404244v1), bioRxiv.
