# Lambda Neural Network
>**Note:** Work in progress
This repository comtains various branches that depict leveraging AWS Lambda Functions as Neurons for various Neural Networks. The objective to learn how to build Deep Neural Networks from scratch without leveraging frameworks like TensorFlow, Caffe, Theano etc. 

Each version is mean to enhance the functionality of the implementation to start from a basic Perceptron and evolve into more compleicated funcitonality that includes Regularization, Optmization and Gradient Checking techniques as well as deeper network architectures. The version branches are as follows:

- Version 0.0: Single Neuron Logistic Regression with DynamoDB. (**Obsolete**)
    - DynamoDB does not offer sufficient flexability to store Numpy Arrays, thus forced to store the data on S3.
    - DynamoDB does not offer sufficient flexability to store the network settings as it does not serialize JSON files or dictionaries very well.
    - DynamoDB does not store sloat data types and nhave to serialize ductionary content to `decimal` which cases significant programming complexity.
- Version 0.1: single Neuron Logistic Regression with Elastiocache. (**Currently Under Investigation**)
- Version 1.0: Single Neuron Logistic Regression.
- Version 2.0: Single Hidden Layer Logistic Regression.
- Version 3.0: Deep Neural Network.
- Version 4.0: Optimization.

The various versions are loosely based on the [deepelarning.ai](https://www.coursera.org/specializations/deep-learning) Coursera specialization.
