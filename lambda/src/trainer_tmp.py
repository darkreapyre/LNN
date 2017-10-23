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
# Get the results object from ElastiCache
results = cache.get('results')
# Initialize state tracking object, as the event payload
payload = {}


# Helper Functions
def start_epoch():
    """

    """

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
    

def sgd():
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
    global parameters = cache.get(parameters_key)
    
    # Get the current state
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
        # Get important event variables from event triggerd from `LaunchLambda`
        S3_bucket = event['s3_bucket']
        state_table = event['state_table']
        learning_rate = event['learning_rate']
        #weights = event['w']
        #bias = event['b']
        epochs = event['epochs']
        epoch = event['epoch']
        ayers = event['layers']
        layer = event['layer']
        activations = event['activations']
        #neurons = event.get('neurons')['layer' + str(layer)]
        #current_activation = event.get('activations')['layer' + str(current_layer)]

        # Create a epoch 1 in DynamoDB with ALL initial parameters
        
        # Start forwardprop
        #layer = layer + 1 # Shuould equate to 0+1
        #propogate(direction='forward', layer=layer+1, activations=activations)
        
    else:
        print("No state informaiton has been provided.")
        raise

###### Add code to clean up if this is `epochs + 1`