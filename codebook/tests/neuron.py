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
from json import dumps, loads
from boto3 import client, resource, Session
import botocore
import uuid
import io
import redis
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
def sigmoid(z):
    """
    Computes the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(z)
    """

    s = 1 / (1 + np.exp(-z))

    return s

def to_cache(endpoint, obj, name):
    """
    Serializes multiple data type to ElastiCache and returns
    the Key.
    
    Arguments:
    endpoint -- The ElastiCache endpoint.
    obj -- the object to srialize. Can be of type:
            - Numpy Array.
            - Python Dictionary.
            - String.
            - Integer.
    name -- Name of the Key.
    
    Returns:
    key -- For each type the key is made up of {name}|{type} and for
           the case of Numpy Arrays, the Length and Widtch of the 
           array are added to the Key.
    """
    
    # Test if the object to Serialize is a Numpy Array
    if 'numpy' in str(type(obj)):
        array_dtype = str(obj.dtype)
        length, width = obj.shape
        # Convert the array to string
        val = obj.ravel().tostring()
        # Create a key from the name and necessary parameters from the array
        # i.e. {name}|{type}#{length}#{width}
        key = '{0}|{1}#{2}#{3}'.format(name, array_dtype, length, width)
        # Store the binary string to Redis
        cache = redis(host=endpoint, port=6379, db=0)
        cache.set(key, val)
        return key
    # Test if the object to serialize is a string
    elif type(obj) is str:
        key = '{0}|{1}'.format(name, 'string')
        val = obj
        cache = redis(host=endpoint, port=6379, db=0)
        cache.set(key, val)
        return key
    # Test if the object to serialize is an integer
    elif type(obj) is int:
        key = '{0}|{1}'.format(name, 'int')
        # Convert to a string
        val = str(obj)
        cache = redis(host=endpoint, port=6379, db=0)
        cache.set(key, val)
        return key
    # Test if the object to serialize is a dictionary
    elif type(obj) is dict:
        # Convert the dictionary to a String
        val = json.dumps(obj)
        key = '{0}|{1}'.format(name, 'json')
        cache = redis(host=endpoint, port=6379, db=0)
        cache.set(key, val)
        return key

def from_cache(endpoint, key):
    """
    De-serializes binary object from ElastiCache by reading
    the type of object from the name and converting it to
    the appropriate data type.
    
    Arguments:
    endpoint -- ElastiCache endpoint.
    key -- Name of the Key to retrieve the object.
    
    Returns:
    obj -- The object converted to specifed data type.
    """
    
    # Check if the Key is for a Numpy array containing
    # `float64` data types
    if 'float64' in key:
        cache = redis(host=endpoint, port=6379, db=0)
        val = cache.get(key)
        # De-serialize the value
        array_dtype, length, width = key.split('|')[1].split('#')
        obj = np.fromstring(val, dtype=array_dtype).reshape(int(length), int(width))
        return obj
    # Check if the Key is for a Numpy array containing
    # `int64` data types
    elif 'int64' in key:
        cache = redis(host=endpoint, port=6379, db=0)
        val = cache.get(key)
        # De-serialize the value
        array_dtype, length, width = key.split('|')[1].split('#')
        obj = np.fromstring(val, dtype=array_dtype).reshape(int(length), int(width))
        return obj
    # Check if the Key is for a json type
    elif 'json' in key:
        cache = redis(host=endpoint, port=6379, db=0)
        obj = cache.get(key)
        return json.loads(obj)
    # Chec if the Key is an integer
    elif 'int' in key:
        cache = redis(host=endpoint, port=6379, db=0)
        obj = cache.get(key)
        return int(obj)
    # Check if the Key is a string
    elif 'string' in key:
        cache = redis(host=endpoint, port=6379, db=0)
        obj = cache.get(key)
        return obj

def lambda_handler(event, context):
    """

    """
    
    # Get the Neural Network paramaters from Elasticache
    parameter_key = event.get('parameters')
    global parameters 
    parameters = from_cache(endpoint, parameter_key)
    #num_hidden_units = parameters['neurons']['layer' + str(layer)]
       
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
    m = from_cache(endpoint=endpoint, key=parameters['data_keys']['m'])

    if state == 'forward':
        # Forward propogation from X to Cost
        if activation == 'sigmoind':
            a = sigmoid(np.dot(w.T, X) + b) # Single Neuron activation
        else: # Some other function to be test later like tanh or ReLU
            pass
        
        # Compute the Cost on TrainerLambda by caching it
        to_cache(endpoint=endpoint, obj=a, name='a_'+str(ID))

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
            # Build the state payload
            payload = {}
            payload['parameter_key'] = parameter_key
            payload['state'] = 'forward'
            payload['epoch'] = epoch
            payload['layer'] = layer + 1
            payloadbytes = dumps(payload)

######################################################################################################            
#            # Invoke NeuronLambdas for next layer
#            try:
#                response = lambda_client.invoke(
#                    FunctionName=environ['NeuronLambda'], #ENSURE ARN POPULATED BY CFN
#                    InvocationType='Event',
#                    Payload=payloadbytes
#                )
#            except botocore.exceptions.ClientError as e:
#                print(e)
#                raise
#            print(response)
######################################################################################################

        return

    elif state == 'backward':

        """
        Note: Need to get the A from the `TrainerLambda` 
        """
        A = event.get('A')

        # Backward propogation to determine gradients
        dw = (1 / m) * np.dot(X, (A - Y).T)
        db = (1 / m) * np.sum(A - Y)
