#!/usr/bin/python
"""
TBD
"""

"""
Parameter for reference:
{
    "activations": {
        "layer1": "sigmoid"
    },
    "bias": 0,
    "data_dimensions": {
        "test_set_x": [
            12288,
            50
        ],
        "test_set_y": [
            1,
            50
        ],
        "train_set_x": [
            12288,
            209
        ],
        "train_set_y": [
            1,
            209
        ]
    },
    "data_keys": {
        "bias": "bias|int",
        "test_set_x": "test_set_x|float64#12288#50",
        "test_set_y": "test_set_y|int64#1#50",
        "train_set_x": "train_set_x|float64#12288#209",
        "train_set_y": "train_set_y|int64#1#209",
        "weights": "weights|float64#12288#1"
    },
    "epoch": 1,
    "epochs": 1,
    "input_data": [
        "train_set_x",
        "train_set_y",
        "test_set_x",
        "test_set_y"
    ],
    "layer": 1,
    "layers": 1,
    "learning_rate": 0.005,
    "neurons": {
        "layer1": 1
    },
    "weight": 0
}
"""

# Import Libraries needed by the Lambda Function
import numpy as np
import h5py
import scipy
import os
from os import environ
import json
from json import dumps
from boto3 import client, resource, Session
import botocore
import uuid
import io
from redis import StrictRedis as redis

# Global Variables
s3_client = client('s3', region_name='us-west-2') # S3 access
s3_resource = resource('s3')
redis_client = client('elasticache', region_name='us-west-2')
lambda_client = client('lambda', region_name='us-west-2') # Lambda invocations
# Retrieve the Elasticache Cluster endpoint
cc = redis_client.describe_cache_clusters(ShowCacheNodeInfo=True)
endpoint = cc['CacheClusters'][0]['CacheNodes'][0]['Endpoint']['Address']
cache = redis(host=endpoint, port=6379, db=0)
results = from_cache(endpoint=endpoint, key='results')

# Helper Functions

def lambda_handler(event, context):
    """

    """
    
    #############################################
    # Must have the following event variables:  #
    # 1. parameter_key.                         #
    # 2. state/direction.                       #
    # 3. Epoch.                                 #
    # 4. layer.                                 #
    # 5. Is it the final neuron?                #
    #############################################
    
    #TBD

   
    
    # Get the Neural Network paramaters from Elasticache
    parameter_key = event.get('parameters')
    global parameters 
    parameters = from_cache(endpoint, parameters_key)
    num_hidden_units = parameters['neurons']['layer' + str(layer)]
       
    # Get the current state
    state = event.get('state')
    epoch = event.get('epoch')
    layer = event.get('layer')
    ID = event.get('id') # To be used when multiple activations
    activation = event.get('activation')
    # Determine is this is the last Neuron in the layer
    last = event.get('last')

    # Get data to process
    w = from_cache(endpoint=endpoint, key=parameters['data_keys']['weights'])
    b = from_cache(endpoint=endpoint, key=parameters['data_keys']['bias'])
    X = from_cache(endpoint=endpoint, key=parameters['data_keys']['train_set_x'])
    Y = from_cache(endpoint=endpoint, key=parameters['data_keys']['train_set_y'])
    m = X.shape[1] 

    if state == 'forward':
        # Forward propogation from X to Cost
        # Note: Cost is calculate because this is the last "layer", and only layer
        if activation == 'sigmoind':
            A = sigmoid(np.dot(w.T, X) + b)
        else: #Some opther function to be test later
            pass
        
        # Compute the Cost
        """
        ###################################################################################
        #                             ISSUES!!!!!                                         #
        #                                                                                 #
        #   Backprop need A, therefore it somehow has to be stored in order to use it.    #
        #                                                                                 #
        #               SOOOOO I need to think about this!!!!!                            #
        #                                                                                 #
        #                                                                                 #
        #cost = (-1 / m) * np.sum(Y * (np.log(A)) + ((1 - Y) * np.log(1 - A)))            #
        #results['epoch' + str(epoch)]['cost'] = cost                                     #
        #                                                                                 #
        # Update results in Elasticache                                                   #
        #to_cache(endpoint=endpoint, obj=results, name='results')                         #
        ###################################################################################        
        """

        # Add some Assertion statements
        #TBD

        if last:
            
            ###################################################
            # Launch TrainerLambda with last payload, which   #
            # includes the following:                         #
            # 1. parameter_key.                               #
            # 2. state/direction.                             #
            # 3. epoch.                                       #
            # 4. layer.                                       #
            # 5. TBD                                          #
            ###################################################
            
            # build the state payload
            payload = {}
            payload['parameter_key'] = parameter_key
            payload['state'] = 'forward'
            payload['epoch'] = epoch
            payload['layer'] = layer + 1

            # Invoke TrainerLambda

            #return

    elif state == 'backward':

        """
        Note: Need to get the A from the `TrainerLambda` 
        """
        A = event.get('A')

        # Backward propogation to determine gradients
        dw = (1 / m) * np.dot(X, (A - Y).T)
        db = (1 / m) * np.sum(A - Y)
