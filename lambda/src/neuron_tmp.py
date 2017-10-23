#!/usr/bin/python

#############################################
# Must have the following event variables:  #
# 1. parameter_key.                         #
# 2. state/direction.                       #
# 3. Epoch.                                 #
# 4. layer.                                 #
# 5. Activation (if direction=forward).     #
#############################################

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
#Retrieve the Elasticache Cluster endpoint
cc = redis_client.describe_cache_clusters(ShowCacheNodeInfo=True)
endpoint = cc['CacheClusters'][0]['CacheNodes'][0]['Endpoint']['Address']
lambda_client = client('lambda', region_name='us-west-2') # Lambda invocations

