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
            
            
            \n",
    "        #propogate(direction='backward', layer=layer-1)\n",
    "        \n",
    "        pass\n",
    "\n",
    "    else:\n",
    "        # Move to the next hidden layer\n",
    "        #propogate(direction='forward', layer=layer+1, activations=activations)\n",
    "        \n",
    "        pass\n",
    "\n",
    "elif current_state == 'backward':\n",
    "    # Get important state variables\n",
    "    \n",
    "    # Determine the location within backprop\n",
    "    if epoch == epochs and layer == 0:\n",
    "        # Location is at the end of the final epoch\n",
    "        \n",
    "        # Caculate derivative?????????????????????????\n",
    "        \n",
    "        # Caclulate the absolute final weight\n",
    "        \n",
    "        # Update the final weights and results (cost) to DynamoDB\n",
    "        \n",
    "        # Finalize the the process and clean up\n",
    "        #finish()\n",
    "        \n",
    "        pass\n",
    "    \n",
    "    elif epoch < epochs and layer == 0:\n",
    "        # Location is at the end of the current epoch and backprop is finished\n",
    "        # Calculate the derivative?????????????????????????\n",
    "        \n",
    "        # Calculate the weights for this epoch\n",
    "        \n",
    "        # Update the weights and results (cost) to DynamoDB\n",
    "        \n",
    "        # Start the next epoch\n",
    "        #epoch = epoch + 1\n",
    "        #start(epoch)\n",
    "        \n",
    "        pass\n",
    "        \n",
    "    else:\n",
    "        # Move to the next hidden layer\n",
    "        #propogate(direction='backward', layer=layer-1)\n",
    "        \n",
    "        pass\n",
    "    \n",
    "elif current_state == 'start':\n",
    "    # Start of a new run of the process\n",
    "    # Get important event variables from event triggerd from `LaunchLambda`\n",
    "    s3_bucket = event['s3_bucket']\n",
    "    state_table = event['state_table']\n",
    "    learning_rate = event['learning_rate']\n",
    "    #weights = event['w']\n",
    "    #bias = event['b']\n",
    "    epochs = event['epochs']\n",
    "    epoch = event['epoch']\n",
    "    layers = event['layers']\n",
    "    layer = event['layer']\n",
    "    activations = event['activations']\n",
    "    #neurons = event.get('neurons')['layer' + str(layer)]\n",
    "    #current_activation = event.get('activations')['layer' + str(current_layer)]\n",
    "    \n",
    "    # Initialize Weights\n",
    "    #if weights == 0: # Initial weights to dimensions of input data\n",
    "    #    dims = event.get('dimensions')['train_set_x'][0]\n",
    "    #    w = np.zeros((dims, 1))\n",
    "    #    # Store the initial Weights data to S3 for Neurons\n",
    "    #    numpy2s3(w, name='weights', bucket=s3_bucket)\n",
    "        \n",
    "    #else:\n",
    "    #    #placeholder for random initialization of weights\n",
    "    #    pass\n",
    "    \n",
    "    # Initialize Bias\n",
    "    #if bias != 0:\n",
    "    #    #placeholder for other bias initialization\n",
    "    #    pass\n",
    "    #else:\n",
    "    #    b = bias\n",
    "    #    # Store the initial Bias data to S3 for Neurons\n",
    "    #    numpy2s3(b, name='bias', bucket=s3_bucket)\n",
    "    \n",
    "    # Create a epoch 1 in DynamoDB with ALL initial parameters\n",
    "    table = dynamodb.Table('state')\n",
    "    table.put_item(\n",
    "        Item = {\n",
    "            'epoch': epoch,\n",
    "            'epochs': epochs,\n",
    "            'layer': layer+1,\n",
    "            'learning_rate': Decimal(learning_rate),\n",
    "            'activations': activations,\n",
    "            'state_table': state_table,\n",
    "            's3_bucket': s3_bucket,\n",
    "            'params': event['params']\n",
    "        }\n",
    "    )\n",
    "    \n",
    "    # Start forwardprop\n",
    "    #layer = layer + 1 # Shuould equate to 0+1\n",
    "    #propogate(direction='forward', layer=layer+1, activations=activations)\n",
    "\n",
    "else:\n",
    "    print(\"No state informaiton has been provided.\")\n",
    "    raise"




###### Add code to clean up if this is `epochs + 1`