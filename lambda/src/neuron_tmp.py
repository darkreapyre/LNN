#!/usr/bin/python

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
# Get the results object from ElastiCache
results = cache.get('results')
# Initialize state tracking object, as the event payload
payload = {}

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
    
   
    
    # Get the Neural Network paramaters from Elasticache
    parameter_key = event.get('parameter_key')
    global parameters = cache.get(parameters_key)
    # Start tracking state
    payload['parameter_key'] = parameter_key
       
    # Get the current state
    state = event.get('state')
    epoch = event.get('epoch')
    layer = event.get('layer')
    final = event.get('final')
   
    if state == 'forward':
        activation = parameters['activations']['layer' + str(layer)]
        pass
    elif state == 'backward':
        pass
    
    if final:
        ###################################################
        # Launch TrainerLambda with FINAL payload, which  #
        # includes the following:                         #
        # 1. parameter_key.                               #
        # 2. state/direction.                             #
        # 3. epoch.                                       #
        # 4. layer.                                       #
        # 5. TBD
        
        pass



