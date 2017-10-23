#!/usr/bin/python
"""
Lambda Fucntion that launches Neural Network training from
an S3 training data upload.
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
#Retrieve the Elasticache Cluster endpoint
cc = redis_client.describe_cache_clusters(ShowCacheNodeInfo=True)
endpoint = cc['CacheClusters'][0]['CacheNodes'][0]['Endpoint']['Address']
lambda_client = client('lambda', region_name='us-west-2') # Lambda invocations

# Helper Functions
def dump2cache(endpoint, dump, name):
    """
    Serializes a string or JSON to a binary string and stores it in ElastiCache
    
    Arguments:
    endpoint -- ElastiCache endpoint
    dump -- JSON dump or string to be stored in ElastiCache
    name -- name used as the hash key of the data
    
    Return:
    Hash key for the data
    """
    
    key = str(name)
    cache = redis(host=endpoint, port=6379, db=0)
    val = dump
    cache.set(key, val)
    return key

def cache2numpy(endpoint, key):
    """
    De-serializes binary string from Elasticache to a Numpy array
    
    Arguments:
    endpoint -- ElastiCache endpoint
    key -- name of te hash key of the data to be retrieved
    
    Return:
    Numpy array
    """
    
    # Get array from Redis
    cache = redis(host=endpoint, port=6379, db=0)
    data = cache.get(key)
    # De-serialize the value
    array_dtype, length, width = key.split('|')[1].split('#')
    array = np.fromstring(data, dtype=array_dtype).reshape(int(length), int(width))
    return array

def numpy2cache(endpoint, array, name):
    """
    Seializes a Numpy array to a binary string and stores it in the Redis cache
    
    Arguments:
    endpoint -- ElastiCache endpoint
    array -- Numpy array to stored in ElastiCache
    name -- name used as the hash key of the data
    
    Return:
    Hash key for the array
    """
    
    # Get parameters for the key
    array_dtype = str(array.dtype)
    length, width = array.shape
    # Convert the array to string
    val = array.ravel().tostring()
    # Create a key from the name and necessary parameters from the array
    # i.e. {name}|{type}#{length}#{width} 
    key = '{0}|{1}#{2}#{3}'.format(name, array_dtype, length, width)
    # Store the binary string to Redis
    cache = redis(host=endpoint, port=6379, db=0)
    cache.set(key, val)
    return key

def name2str(obj, namespace):
    """
    Converts the name of the numpy array to string
    
    Arguments:
    obj -- Numpy array object
    namespace -- dictionary of the current global symbol table
    
    Return:
    List of the names of the Numpy arrays
    """
    return [name for name in namespace if namespace[name] is obj]

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

def initialize_data(endpoint, w, b):
    """
    Extracts the training and testing data from S3, flattens, 
    standardizes and then dumps the data to ElastiCache 
    for neurons to process as layer a^0
    """
    
    # Load main dataset
    dataset = h5py.File('/tmp/datasets.h5', "r")
    
    # Create numpy arrays from the various h5 datasets
    train_set_x_orig = np.array(dataset["train_set_x"][:]) # train set features
    train_set_y_orig = np.array(dataset["train_set_y"][:]) # train set labels
    test_set_x_orig = np.array(dataset["test_set_x"][:]) # test set features
    test_set_y_orig = np.array(dataset["test_set_y"][:]) # test set labels
    
    # Reshape labels
    train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    # Preprocess inputs
    train_set_x = standardize(train_set_x_orig)
    test_set_x = standardize(test_set_x_orig)

    # Dump the inputs to the temporary s3 bucket for TrainerLambda
    #bucket = storage_init() # Creates a temporary bucket for the propogation steps
    data_keys = {} # Dictionary for the hask keys of the data set
    dims = {} # Dictionary of data set dimensions
    a_list = [train_set_x, train_set_y, test_set_x, test_set_y]
    a_names = [] # Placeholder for array names
    for i in range(len(a_list)):
        # Create a lis of the names of the numpy arrays
        a_names.append(name2str(a_list[i], locals()))
    for j in range(len(a_list)):
        # Dump the numpy arrays to ElastiCache
        data_keys[str(a_names[j][0])] = numpy2cache(endpoint, array=a_list[j], name=a_names[j][0])
        # Append the array dimensions to the list
        dims[str(a_names[j][0])] = a_list[j].shape
    
    # Initialize weights
    if w == 0: # Initialize weights to dimensions of the input data
        dim = dims.get('train_set_x')[0]
        weights = np.zeros((dim, 1))
        # Store the initial weights as a column vector on S3
        data_keys['weights'] = numpy2cache(endpoint, array=weights, name='weights')
    else:
        #placeholder for random weight initialization
        pass
        
    # Initialize Bias
    if b != 0:
        #placeholder for random bias initialization
        #data_keys['bias'] = numpy2cache(endpoint, array=bias, name='bias')
        pass
    else:
        data_keys['bias'] = dump2cache(endpoint, dump=str(b), name='bias')
    
    # Initialize the results tracking object
    dump2cache(endpoint, dump='', name='results')
        
    return data_keys, [j for i in a_names for j in i], dims

def lambda_handler(event, context):
    # Retrieve datasets and setting from S3
    input_bucket = s3_resource.Bucket(str(event['Records'][0]['s3']['bucket']['name']))
    dataset_key = str(event['Records'][0]['s3']['object']['key'])
    settings_key = dataset_key.split('/')[-2] + '/parameters.json'
    try:
        input_bucket.download_file(dataset_key, '/tmp/datasets.h5')
        input_bucket.download_file(settings_key, '/tmp/parameters.json')
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print("Error downloading input data from S3, S3 object does not exist")
        else:
            raise
    
    # Extract the neural network parameters
    with open('/tmp/parameters.json') as parameters_file:
        parameters = json.load(parameters_file)
    
    # Build in additional parameters from neural network parameters
    parameters['epoch'] = 1
    # Next Layer to process
    parameters['layer'] = 1
    # Input data sets and data set parameters
    parameters['data_keys'], \
    parameters['input_data'], \
    parameters['data_dimensions'] = initialize_data(
        endpoint=endpoint, w=parameters.get('weight'), b = parameters.get('bias')
    )
    
    # Initialize payload to `TrainerLambda`
    payload = {}
    # Initialize the overall state
    payload['state'] = 'start'
    # Dump the parameters to ElastiCache
    payload['parameter_key'] = dump2cache(endpoint, dump=dumps(parameters), name='parameters')
    #payload['endpoint'] = endpoint
    # Prepare the payload for `TrainerLambda`
    payloadbytes = dumps(payload)
    
    print("Complete Neural Network Settings: \n")
    print(dumps(parameters, indent=4, sort_keys=True))
    print("\n" + "Payload to be sent to TrainerLambda")
    print(dumps(payload, indent=4, sort_keys=True))
    
    # Invoke TrainerLambda for next layer
    try:
        response = lambda_client.invoke(
            FunctionName=environ['TrainerLambda'], #ENSURE ARN POPULATED BY CFN OR S3 EVENT
            InvocationType='Event',
            Payload=payloadbytes
            )
    except botocore.exceptions.ClientError as e:
        print(e)
        raise
    print(response)
    return