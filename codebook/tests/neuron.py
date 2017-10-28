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

# Helper Functions
def to_cache(endpoint, obj, name):
    """
    Serializes multiple data type to ElastiCache and returns
    the Key.
    
    Arguments:
    endpoint -- The ElastiCache endpoint
    obj -- the object to srialize. Can be of type:
            - Numpy Array
            - Python Dictionary
            - String
            - Integer
    name -- Name of the Key
    
    Returns:
    key -- For each type the key is made up of {name}|{type} and for
           the case of Numpy Arrays, the Length and Widtch of the 
           array are added to the Key.
    """
    
    # Test if the object to Serialize is a Numpy Array
    if 'numpy' in str(type(obj)):
        array_dtype = str(obj.dtype)
        if len(obj.shape) == 0:
            length = 0
            width = 0
        else:
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
    else:
        print(str(type(obj)) + "is not a supported serialization type")

def from_cache(endpoint, key):
    """
    De-serializes binary object from ElastiCache by reading
    the type of object from the name and converting it to
    the appropriate data type
    
    Arguments:
    endpoint -- ElastiCacheendpoint
    key -- Name of the Key to retrieve the object
    
    Returns:
    obj -- The object converted to specifed data type
    """
    
    # Check if the Key is for a Numpy array containing
    # `float64` data types
    if 'float64' in key:
        cache = redis(host=endpoint, port=6379, db=0)
        val = cache.get(key)
        # De-serialize the value
        array_dtype, length, width = key.split('|')[1].split('#')
        if int(length) == 0:
            obj = np.float64(np.fromstring(val))
        else:
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
    else:
        print(str(type(obj)) + "is not a supported serialization type")

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

def lambda_handler(event, context):
    """

    """
    
    # Get the Neural Network paramaters from Elasticache
    parameter_key = event.get('parameter_key')
    parameters = from_cache(endpoint, key=parameter_key)
       
    # Get the current state
    state = event.get('state')
    epoch = event.get('epoch')
    layer = event.get('layer')
    ID = event.get('id') # To be used when multiple activations
    # Determine is this is the last Neuron in the layer
    last = event.get('last')

    # Get data to process
    w = from_cache(endpoint=endpoint, key=parameters['data_keys']['weights'])
    b = from_cache(endpoint=endpoint, key=parameters['data_keys']['bias'])
    X = from_cache(endpoint=endpoint, key=parameters['data_keys']['train_set_x'])
    Y = from_cache(endpoint=endpoint, key=parameters['data_keys']['train_set_y'])
    m = from_cache(endpoint=endpoint, key=parameters['data_keys']['m'])

    # Forward propogation from X to Cost
    activation = event.get('activation')
    if activation == 'sigmoid':
        a = sigmoid(np.dot(w.T, X) + b) # Single Neuron activation
    else: # Some other function to be test later like tanh or ReLU
        pass

    if state == 'forward':
        # Capture activations
        A_key = parameters['data_keys']['A']
        # Load the activation object
        A = from_cache(endpoint=endpoint, key=A_key) # Should be empty dictionary
        # Update the activation object for this Neuron at this layer
        A['layer' + str(layer)]['a_' + str(ID)] = to_cache(endpoint=endpoint, obj=a, name='a_'+str(ID))
        # Update loacal parameter
        parameters['data_keys']['A'] = A
        # Upload to ElastiCache
        parameter_key = to_cache(endpoint=endpoint, obj=parameters, name='parameters')
        
        # Compute the Cost on TrainerLambda by caching it
        #to_cache(endpoint=endpoint, obj=a, name='a_'+str(ID))

        if last == "True":
            # Build the state payload
            payload = {}
            payload['parameter_key'] = parameter_key
            payload['state'] = 'forward'
            payload['epoch'] = epoch
            payload['layer'] = layer + 1
            payloadbytes = dumps(payload)
            print("Payload to be sent to TrainerLambda: \n" + dumps(payload, indent=4, sort_keys=True))

######################################################################################################            
#            # Invoke TrainerLambda to process activations
#            try:
#                response = lambda_client.invoke(
#                    FunctionName=environ['TrainerLambda'], #ENSURE ARN POPULATED BY CFN
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
        # Get the results of the forwardprop activation
        # Determine the correct notation for the layer and get the Matrix
        # of activations calculated at the end of forward propogation
        A_name = 'A'+str(layer)
        A_key = parameters.get('data_keys')[A_name]
        A = from_cache(endpoint=endpoint, key=A_key)

        # Backward propogation to determine gradients of current layer
        dw = (1 / m) * np.dot(X, (A - Y).T)
        print("Partial Derivatives - Weights for Neuron" + str(ID) + ":\n" + dw)
        db = (1 / m) * np.sum(A - Y)
        print("Partial Derivatives - Bias for Neuron" + str(ID) + ":\n" + dw)

        # Capture gradients
        grads_key = parameters['data_keys']['grads']
        # Load the grads object
        grads = from_cache(endpoint, key=grads_key) # Should be empty dictionary
        # Update the grads object with the calculated derivatives
        grads['layer' + str(layer)]['dw_' + str(ID)] = to_cache(endpoint, obj=dw, name='dw_'+str(ID))
        grads['layer' + str(layer)]['db_' + str(ID)] = to_cache(endpoint, obj=db, name='db_'+str(ID))
        # Update the pramaters (local)
        parameters['data_keys']['grads'] = grads
        # Upload to ElastiCache
        parameter_key = to_cache(endpoint, obj=parameters, name='parameters')

        if last == "True":
            # Build the state payload
            payload = {}
            payload['parameter_key'] = parameter_key
            payload['state'] = 'backward'
            payload['epoch'] = epoch
            payload['layer'] = layer - 1
            payloadbytes = dumps(payload)
            print("Payload to be sent to TrainerLambda: \n" + dumps(payload, indent=4, sort_keys=True))

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

    else:
        raise