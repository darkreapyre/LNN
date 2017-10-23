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
redis_client = client('elasticache', region_name='us-west-2')
# Retrieve the Elasticache Cluster endpoint
cc = redis_client.describe_cache_clusters(ShowCacheNodeInfo=True)
endpoint = cc['CacheClusters'][0]['CacheNodes'][0]['Endpoint']['Address']
lambda_client = client('lambda', region_name='us-west-2') # Lambda invocations



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

def sgd():
    """

    """

def loss():
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
    
    # Get the Neural Netowkr paramaters from Elasticache
    gobal parameters = event.get('parameters_key')


# Add code to clean up if this is `epochs + 1`