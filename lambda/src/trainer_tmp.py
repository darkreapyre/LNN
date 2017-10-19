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
from decimal import Decimal, Inexact, Rounded
from boto3.dynamodb.types import DYNAMODB_CONTEXT

# Global Variables
s3_client = client('s3', region_name='us-west-2') # S3 access
s3_resource = resource('s3')
dynamo_client = client('dynamodb', region_name='us-west-2') # DynamoDB access
dynamodb = resource('dynamodb', region_name='us-west-2')
DYNAMODB_CONTEXT.traps[Inexact] = 0
DYNAMODB_CONTEXT.traps[Rounded] = 0
lambda_client = client('lambda', region_name='us-west-2') # Lambda invocations

# Helper Functions
def start():
    """

    """

def finish():
    """

    """

def initialize_with_zeros(dim):
    """

    """

def propogate(direction):
    """

    """

def initialize_weights():
    """

    """

def optimize():
    """

    """

def calc_loss():
    """

    """

def update_state():
    """

    """

def lambda_handler(event, context):
    """

    """


# Add code to clean up if this is `epochs + 1`