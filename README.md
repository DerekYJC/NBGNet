NBGNet - NeuroBondGraph Network
====================================================

This code implements the model from the paper "[Inferring system-level brain communication through multi-scale neural activity](https://www.biorxiv.org/content/10.1101/2020.11.30.404244v2)". It is a sparsely-connected recurrent neural network designed for investigating multi-scale neural data, but can be applied to any multi-scale time series data. The NBGNet is able to capture the dynamics and thus infer the neural activity from different scales. For example, in the paper, local field potentials and screw electrocorticography are brain activity recorded in different scales. The NBGNet is able to reconstruct the screw electrocorticography from local field potentials, and vice versa. Furthermore, this work provides a simple framework to implement any dynamical equation with customized recurrent neural network layer.

Prerequisites
----------------------------------------------------

The code is written in Python 3.7.4. You will also need:
- **Keras** version 2.2.4 or above

Getting started
----------------------------------------------------

Before starting, you need to build the dynamical systems based on the data types you are working with. Given different system dynamics equations you may need to modify the update function codes in the recurrent layers. The components, in other words, model sub-units, identified from the dynamical systems will be fed into the NBGNet as inputs when creating the model. Please refer to the reference for more details. 

Training a NBGNet model
----------------------------------------------------

Now given we have the training datasets, we can train the models. The dataset is a dictionary where the neural data are the values stored with the corresponding keys (which are the IDs of the trial). 
```
training_history = train_model(model, X_train, y_train)
```
- Check the function to customize the training protocol.

Evaluating a trained model
----------------------------------------------------

Once your model is finished training, the next step is to evalute it.
```
evaluation_results = evaluate_model(model, X_test, y_test, evaluation_functions)
```
`evaluation_functions` can be a function instance or a list of several function instances. For example, it can be either `mean_squared_error_func` or `[mean_squared_error_func, cross_correlation_func, ...]`. 

Contact
----------------------------------------------------

File any issues with the [issue tracker](https://github.com/DerekYJC/NBGNet/issues). For acy questions or problems, this code is maintained by [@DerekYJC](https://github.com/DerekYJC).

## Reference

- Y. J. Chang, Y. I. Chen, H. C. Yeh, J. M. Carmena, & S. R. Santacruz (2020). [Inferring system-level brain communication through multi-scale neural activity](https://www.biorxiv.org/content/10.1101/2020.11.30.404244v2), bioRxiv.
