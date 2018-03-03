# Import Libraries
import time
import h5py
import numpy as np
import os
from os import environ
import json
from json import dumps, loads
from boto3 import client, resource, Session
import botocore
import sys

# Global Variables
rgn = 'us-east-1'
s3_client = client('s3', region_name=rgn) # S3 access
s3_resource = resource('s3')

def load_data():
    # Load main dataset
    dataset = h5py.File('/tmp/datasets.h5', 'r')

    # Create numpy arrays from the various h5 datasets
    train_set_x_orig = np.array(dataset["train_set_x"][:]) # train set features
    train_set_y_orig = np.array(dataset["train_set_y"][:]) # train set labels
    test_set_x_orig = np.array(dataset["test_set_x"][:]) # test set features
    test_set_y_orig = np.array(dataset["test_set_y"][:]) # test set labels
    
    # Reshape labels
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig

def lambda_handler(event, context):
    # Retrieve datasets and setting from S3
    input_bucket = s3_resource.Bucket(str(event['Records'][0]['s3']['bucket']['name']))
    dataset_key = str(event['Records'][0]['s3']['object']['key'])
    try:
        input_bucket.download_file(dataset_key, '/tmp/datasets.h5')
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print("Error downloading input data from S3, S3 object does not exist")
        else:
            raise
        
    np.random.seed(1)
    train_x_orig, train_y, test_x_orig, test_y = load_data()

    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    # Debug Statements
    #print ("train_x's shape: " + str(train_x.shape))
    #print ("test_x's shape: " + str(test_x.shape))

    from DeepNeuralNetwork import DeepNeuralNetwork
    layers_dims = (12288, 20, 7, 5, 1)
    activations = ['relu', 'relu', 'relu', 'sigmoid']
    num_iter = 30
    learning_rate = 0.0075

    clf, params = DeepNeuralNetwork(layers_dims, activations)\
                .fit(train_x, train_y, num_iter, learning_rate, True, 100)
    #print('train accuracy: {:.2f}%'.format(clf.accuracy_score(train_x, train_y)*100))
    print('Model Accuracy: {:.2f}%'.format(clf.accuracy_score(test_x, test_y)*100))
    
    # Create params file
    with h5py.File('/tmp/params.h5', 'w') as h5file:
        for key in params:
            h5file['/' + key] = params[key]
    
    # Upload model parameters file to S3
    s3_resource.Object('itsacat-local', 'predict_input/params.h5').put(Body=open('/tmp/params.h5', 'rb'))