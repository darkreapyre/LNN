#!/usr/bin/python
"""
Lambda Fucntion that tracks state, launches Neurons to process
forward and backward propogation.
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
from redis import StrictRedis as redis

# Global Variables
s3_client = client('s3', region_name='us-west-2') # S3 access
s3_resource = resource('s3')
lambda_client = client('lambda', region_name='us-west-2') # Lambda invocations
redis_client = client('elasticache', region_name='us-west-2')
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
        obj = np.fromstring(data, dtype=array_dtype).reshape(int(length), int(width))
        return obj
    # Check if the Key is for a Numpy array containing
    # `int64` data types
    elif 'int64' in key:
        cache = redis(host=endpoint, port=6379, db=0)
        data = cache.get(key)
        # De-serialize the value
        array_dtype, length, width = key.split('|')[1].split('#')
        obj = np.fromstring(data, dtype=array_dtype).reshape(int(length), int(width))
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

def start_epoch(epoch, layer):
    """
    Starts a new epoch and configures the necessary state tracking objcts.
    
    Arguments:
    epoch -- Integer representing the "current" epoch.
    layer -- Integer representing the current hidden layer.
    """
    
    #TBD
    pass

    # Initialize the results onbject for the new epoch
    #results['epoch' + str(epoch)] = {}
    
    # Start forwardprop
    #propogate(direction='forward', epoch=epoch)

def finish_epoch(direction, epoch, layer):
    """
    Closes out the current epoch and updates the necessary information to the results object.

    Arguments:
    direction -- The current direction of the propogation, either `forward` or `backward`.
    epoch -- Integer representing the "current" epoch to close out.
    """

    #TBD
    pass

def propogate(direction, epoch, layer):
    """
    Determines the amount of "hidden" units based on the layer and loops
    through launching the necessary `NeuronLambda` functions with the 
    appropriate state. Each `NeuronLambda` implements the cost function 
    OR the gradients depending on the direction.

    Arguments:
    direction -- The current direction of the propogation, either `forward` or `backward`.
    epoch -- Integer representing the "current" epoch to close out.
    layer -- Integer representing the current hidden layer.
    """
    
    ###########################################################
    # When launching Neuron, the following must be added      #
    # to the payload:                                         #
    # 1. parameter_key.                                       #
    # 2. state/direction.                                     #
    # 3. epoch.                                               #
    # 4. layer.                                               #
    # 5. final. (Is it the last neuron? True|False).          #
    ###########################################################

    # Build the NeuronLambda payload
    #payload = {}

    # Add the parameters to the payload
    #payload['state'] = direction
    #payload['parameter_key] = parameter_key
    #payload['epoch'] = epoch
    #payload['layer'] = layer


    #TBD
    if direction == 'forward':
        # Launch Lambdas to propogate forward
        pass
    elif direction == 'backward':
        # Launch Lambdas to propogate backward
        pass

    """
    Note:
    When launching NeuronLambda with multiple hidden unit,
    remember to assign an ID
    """

def optimize(epoch, layer, params, grads):
    """
    Optimizes `w` and `b` by running Gradient Descent to get the `cost`.

    Arguments:
    epoch -- Integer representing the "current" epoch to close out.
    layer -- Integer representing the current hidden layer.
    params -- Dictionary containing the gradients of the weights and 
                bias.
    grads -- Dictionary containing the gardients of the wights and
                bias with respect to the cost function.
    
    Returns:
    TBD
    """

    #TBD
    #Get the learning rate
    #learning_rate = parameters['learning_rate']

    """
    Note:
    Probably have to get the cost from the output of the NeuronLambdas
    OR
    Get data from the NeuronLambdas and calculate the cost here
    """

    # Get the grads and params
    
    # Perform the update rule
    #w = w - learning_rate * grads['dw']
    #b = b - learning_rate * grads['db']

    pass

def end():
    """
    Finishes out the process and launches the next state mechanisms for prediction.
    """

    #TBD
    pass

def lambda_handler(event, context):
    """

    """
       
    # Get the Neural Network paramaters from Elasticache
    global parameter_key
    parameter_key = event.get('parameters')
    global parameters 
    parameters = from_cache(parameter_key)
    
    # Get the current state from the invoking lambda
    state = event.get('state')
    
    # Execute appropriate action based on the the current state
    if state == 'forward'
        # Get important state variables
        epoch = event.get('epoch')
        layer = event.get('layer')
        
        # Determine the location within forwardprop
        if layer > layers:
            # Location is at the end of forwardprop
            # Caculate the Loss function
            
            # Update the Loss function to the results object
            loss = cacl_loss()
            results['epoch' + str(epoch)]['loss'] = loss
            
            # Start backprop
            #propogate(direction='backward', layer=layer-1)
            
            pass
            
        else:
            # Move to the next hidden layer
            #propogate(direction='forward', layer=layer+1, activations=activations)
            
            pass
        
    elif state == 'backward':
        # Get important state variables
        
        # Determine the location within backprop
        if epoch == epochs and layer == 0:
            # Location is at the end of the final epoch
            
            # Caculate derivative?????????????????????????\
            
            # Caclulate the absolute final weight
            
            # Update the final weights and results (cost) to DynamoDB
            
            # Finalize the the process and clean up
            #end()
            
            pass
            
        elif epoch < epochs and layer == 0:
            # Location is at the end of the current epoch and backprop is finished
            # Calculate the derivative?????????????????????????

            # Get the Cost and store it to the results object
            
            # Calculate the weights for this epoch
            
            # Update/get the weights and bias from the results object
            
            # Start the next epoch
            #epoch = epoch + 1
            #start_epoch(epoch)
            
            pass
            
        else:
            # Move to the next hidden layer
            #propogate(direction='backward', layer=layer-1)
            
            pass
            
    elif state == 'start':
        # Start of a new run of the process
        # Initialize the results tracking object
        global results
        results = {}
        
        # Create initial parameters
        epoch = 1
        layer = 1
        start_epoch(epoch=epoch, layer=layer)
       
    else:
        print("No state informaiton has been provided.")
        raise

###### Add code to clean up if this is `epochs + 1`