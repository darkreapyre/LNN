# Import Libraries needed by the Lambda Function
import sys
import datetime
import math
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
#rgn = environ['Region']
rgn = 'us-west-2'
s3_client = client('s3', region_name=rgn) # S3 access
s3_resource = resource('s3')
sns_client = client('sns', region_name=rgn) # SNS
lambda_client = client('lambda', region_name=rgn) # Lambda invocations
redis_client = client('elasticache', region_name=rgn) # ElastiCache
# Retrieve the Elasticache Cluster endpoint
cc = redis_client.describe_cache_clusters(ShowCacheNodeInfo=True)
endpoint = cc['CacheClusters'][0]['CacheNodes'][0]['Endpoint']['Address']
# Set redis database to `16` as default for the main parameters
cache = redis(host=endpoint, port=6379, db=15)
dynamo_client = client('dynamodb', region_name=rgn)
dynamo_resource = resource('dynamodb', region_name=rgn)

# Helper Functions
def publish_sns(sns_message):
    """
    Publish message to the master SNS topic.

    Arguments:
    sns_message -- the Body of the SNS Message
    """

    print("Publishing message to SNS topic...")
    sns_client.publish(TargetArn=environ['SNSArn'], Message=sns_message)
    return

def get_arns(function_name):
    """
    Return the ARN for the LNN Functions.
    Note: This addresses circular dependency issues in CloudFormation
    """
    function_list = lambda_client.list_functions()
    function_arn = None
    for function in function_list['Functions']:
        if function['FunctionName'] == function_name:
            function_arn = function['FunctionArn']
            break
    return function_arn

def vectorize(x_orig):
    """
    Vectorize the image data into a matrix of column vectors
    
    Argument:
    x_orig -- Numpy array of image data
    
    Return:
    Reshaped/Transposed Numpy array
    """
    return x_orig.reshape(x_orig.shape[0], -1).T

def standardize(x_orig):
    """
    Standardize the input data
    
    Argument:
    x_orig -- Numpy array of image data
    
    Return:
    Call to `vectorize()`, stndrdized Numpy array of image data
    """
    return vectorize(x_orig) / 255

def initialize_data(parameters):
    """
    Extracts the training and testing data from S3, flattens, 
    standardizes and then initializes a random Weights and
    Bias object for each layer.

    Arguments:
    parameters -- The initial/running parameters dictionary object
    
    Returns:
    X -- Traing set features
    Y - Training set labels
    """
    # Load main dataset
    dataset = h5py.File('/tmp/datasets.h5', "r")

    # Create numpy arrays from the various h5 datasets
    train_set_x_orig = np.array(dataset["train_set_x"][:]) # train set features
    train_set_y_orig = np.array(dataset["train_set_y"][:]) # train set labels
    test_set_x_orig = np.array(dataset["test_set_x"][:]) # test set features
    test_set_y_orig = np.array(dataset["test_set_y"][:]) # test set labels

    # Reshape labels
    Y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

    # Preprocess inputs
    X = standardize(train_set_x_orig)
    
    return X, Y

def to_cache(db, obj, name):
    """
    Serializes multiple data type to ElastiCache and returns
    the Key.
    
    Arguments:
    db -- The ElastiCache database
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
        cache = redis(host=endpoint, port=6379, db=int(db))
        cache.set(key, val)
        return key
    # Test if the object to serialize is a string
    elif type(obj) is str:
        key = '{0}|{1}'.format(name, 'string')
        val = obj
        cache = redis(host=endpoint, port=6379, db=int(db))
        cache.set(key, val)
        return key
    # Test if the object to serialize is an integer
    elif type(obj) is int:
        key = '{0}|{1}'.format(name, 'int')
        # Convert to a string
        val = str(obj)
        cache = redis(host=endpoint, port=6379, db=int(db))
        cache.set(key, val)
        return key
    # Test if the object to serialize is a dictionary
    elif type(obj) is dict:
        # Convert the dictionary to a String
        val = json.dumps(obj)
        key = '{0}|{1}'.format(name, 'json')
        cache = redis(host=endpoint, port=6379, db=int(db))
        cache.set(key, val)
        return key
    else:
        sns_message = "`to_cache` Error:\n" + str(type(obj)) + "is not a supported serialization type"
        publish_sns(sns_message)
        print("The Object is not a supported serialization type")
        raise

def random_minibatches(X, Y, batch_size=64):
    """
    Creates a list of random smaller batches of X and Y
    
    Arguments:
    X -- Training examples input data of size (input size, no. of examples)
    Y -- Training example labels vector of size (1, no. of examples)
    batch_size -- Size of the mini batches
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_x, mini_batch_Y)
    """
    #np.random.seed(seed)
    m = X.shape[1] # no. of training examples
    mini_batches = [] 
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape(1, m)
    
    # Step 2: Partition (shuffled_X, shuffled_Y), minus the end case.
    num_complete_minibatches = math.floor(m / batch_size) # No. of mini batches
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * batch_size : (k + 1) * batch_size]
        mini_batch_Y = shuffled_Y[:, k * batch_size : (k + 1) * batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Step 3: Deal with the end case
    if m % batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def inv_counter(name, invID, task):
    """
    Manages the Counter assigned to a unique Lambda Invocation ID, by
    either setting it to 0, updating it to 1 or querying the value.
   
    Arguments:
    name -- The Name of the function being invoked
    invID -- The unique invocation ID created for the specific invokation
    task -- Task to perfoirm: set | get | update
    """
    table = dynamo_resource.Table(name)
    if task == 'set':
        table.put_item(
            Item={
                'invID': invID,
                'cnt': 0
            }
        )
        
    elif task == 'get':
        task_response = table.get_item(
            Key={
                'invID': invID
            }
        )
        item = task_response['Item'] 
        return int(item['cnt'])
        
    elif task == 'update':
        task_response = table.update_item(
            Key={
                'invID': invID
            },
            UpdateExpression='SET cnt = :val1',
            ExpressionAttributeValues={
                ':val1': 1
            }
        )