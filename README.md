# Lambda Neural Network
>**Note:** Work in progress
This repository comtains various branches that depict leveraging AWS Lambda Functions as Neurons for various Neural Networks. The objective to learn how to build Deep Neural Networks from scratch without leveraging frameworks like TensorFlow, Caffe, Theano etc. Additionally the object is to ensure a better , more advanced, understanding about what is happenning within each Layer and each Hidden Unit.

Each version is mean to enhance the functionality of the implementation to start from a basic Perceptron and evolve into more compleicated funcitonality that includes Regularization, Optmization and Gradient Checking techniques as well as deeper network architectures. The version branches are as follows:

- Version 0.0: Single Neuron Logistic Regression - DynamoDB. (**Obsolete**)
    - DynamoDB does not offer sufficient flexability to store Numpy Arrays, thus forced to store the data on S3.
    - DynamoDB does not offer sufficient flexability to store the network settings as it does not serialize JSON files or dictionaries very well.
    - DynamoDB does not store float data types and nhave to serialize ductionary content to `decimal` which cases significant programming complexity.
- Version 0.1: Single Neuron Logistic Regression - Elastiocache. (**Currently under Investigation**)
- Version 0.2: 2-Layer Logistic Regression. (**On Hold due to Issues with Backprop**)
- Version 0.3: L-Layer Logistic Regression.
- Version 0.44: Optimization.
    - Batch Gradient Decent
    - Adam Optmization
    - Hold-out

The various versions are loosely based on the [deepelarning.ai](https://www.coursera.org/specializations/deep-learning) Coursera specialization.
