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
from json import dumps
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
# Initialize state tracking object, as the event payload
payload = {}


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
    elif type(obj) is dict
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

    """
    
    # Update the results onbject for the new epoch
    exec('epoch' + str(epoch)) = {}
    
    
    # Start forwardprop
    #layer = layer + 1 # Shuould equate to 0+1
    #propogate(direction='forward', layer=layer+1, activations=activations)

def finish_epoch():
    """

    """

def propogate(direction):
    """

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
    

def optimize():
    """

    """

def calc_loss():
    """

    """

def update_state():
    """

    """

def end():
    """
    
    """

def lambda_handler(event, context):
    """

    """
       
    # Get the Neural Network paramaters from Elasticache
    parameter_key = event.get('parameter_key')
    global parameters 
    parameters = cache.get(parameter_key)
    
#    # Get the results object from ElastiCache
#    # Since this might have
#    global results
#    results = cache.get('results')
    
    # Get the current state from the invoking lambda
    state = event.get('state')
   
    # Start tracking state
    payload['parameter_key'] = parameter_key
    
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
        
    elif current_state == 'backward':
        # Get important state variables
        
        # Determine the location within backprop
        if epoch == epochs and layer == 0:
            # Location is at the end of the final epoch
            
            # Caculate derivative?????????????????????????\
            
            # Caclulate the absolute final weight
            
            # Update the final weights and results (cost) to DynamoDB
            
            # Finalize the the process and clean up\n",
            #finish_epoch()
            
            pass
            
        elif epoch < epochs and layer == 0:
            # Location is at the end of the current epoch and backprop is finished
            # Calculate the derivative?????????????????????????
            
            # Calculate the weights for this epoch
            
            # Update the weights and results (cost) to DynamoDB
            
            # Start the next epoch
            #epoch = epoch + 1
            #start_epoch(epoch)
            
            pass
            
        else:
            # Move to the next hidden layer
            #propogate(direction='backward', layer=layer-1)
            
            pass
            
    elif current_state == 'start':
        # Start of a new run of the process
        # Create the results object and initialize the correct structure
        # in order to get the correct key naming structure.
        template = {'epoch': 1, }
        
        # Create initial parameters
        epoch = 1
        layer = 1
        start_epoch(epoch=epoch, layer=layer)
       
    else:
        print("No state informaiton has been provided.")
        raise

###### Add code to clean up if this is `epochs + 1`