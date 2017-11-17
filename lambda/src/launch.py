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
from json import dumps, loads
from boto3 import client, resource, Session
import botocore
import uuid
import io
import redis
from redis import StrictRedis as redis

# Global Variables
rgn = environ['Region']
s3_client = client('s3', region_name=rgn) # S3 access
s3_resource = resource('s3')
sns_client = client('sns', region_name=rgn) # SNS
redis_client = client('elasticache', region_name=rgn)
#Retrieve the Elasticache Cluster endpoint
cc = redis_client.describe_cache_clusters(ShowCacheNodeInfo=True)
endpoint = cc['CacheClusters'][0]['CacheNodes'][0]['Endpoint']['Address']
lambda_client = client('lambda', region_name=rgn) # Lambda invocations

# Helper Functions
def get_lambda_function_arn(function_name):
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

def publish_sns(sns_message):
    """
    Publish message to the master SNS topic.

    Arguments:
    sns_message -- the Body of the SNS Message
    """

    print("Publishing message to SNS topic...")
    sns_client.publish(TargetArn=environ['SNSArn'], Message=sns_message)
    return

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
    else:
        sns_message = "`to_cache` Error:\n" + str(type(obj)) + "is not a supported serialization type"
        publish_sns(sns_message)
        print("The Object is not a supported serialization type")
        raise

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
        if int(length) == 0:
            obj = np.float64(np.fromstring(val))
        else:
            obj = np.fromstring(val, dtype=array_dtype).reshape(int(length), int(width))
        return obj
    # Check if the Key is for a Numpy array containing
    # `int64` data types
    elif 'int64' in key:
        cache = redis(host=endpoint, port=6379, db=0)
        val = cache.get(key)
        # De-serialize the value
        array_dtype, length, width = key.split('|')[1].split('#')
        obj = np.fromstring(val, dtype=array_dtype).reshape(int(length), int(width))
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
    else:
        sns_message = "`from_cache` Error:\n" + str(type(obj)) + "is not a supported serialization type"
        publish_sns(sns_message)
        print("The Object is not a supported de-serialization type")
        raise

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

def initialize_data(endpoint, parameters):
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

    # Create necessary keys for the data in ElastiCache
    data_keys = {} # Dictionary for the hask keys of the data set
    dims = {} # Dictionary of data set dimensions
    a_list = [train_set_x, train_set_y, test_set_x, test_set_y]
    a_names = [] # Placeholder for array names
    for i in range(len(a_list)):
        # Create a lis of the names of the numpy arrays
        a_names.append(name2str(a_list[i], locals()))
    for j in range(len(a_list)):
        # Dump the numpy arrays to ElastiCache
        data_keys[str(a_names[j][0])] = to_cache(endpoint=endpoint, obj=a_list[j], name=a_names[j][0])
        # Append the array dimensions to the list
        dims[str(a_names[j][0])] = a_list[j].shape
    
    # Initialize A0 and Y names from `train_setx`` and train_set_y`
    data_keys['A0'] = to_cache(endpoint=endpoint, obj=train_set_x, name='A0')
    data_keys['Y'] = to_cache(endpoint=endpoint, obj=train_set_y, name='Y')
    
    # Initialize weights to zero for single layer
    dim = dims.get('train_set_x')[0]
    weights = np.zeros((dim, 1))
    # Store the initial weights in ElastiCache
    data_keys['weights'] = to_cache(endpoint=endpoint, obj=weights, name='weights')
        
    # Initialize Bias to zero for single layer
    bias = 0
    # Store the bias in ElastiCache
    data_keys['bias'] = to_cache(endpoint=endpoint, obj=bias, name='bias')
   
    # Initialize training example size
    m = train_set_x.shape[1]
    data_keys['m'] = to_cache(endpoint, obj=m, name='m')
    
    # Initialize the results tracking object
    results = {}
    data_keys['results'] = to_cache(endpoint, obj=results, name='results')

    # Initialize gradient tracking object for each layer
    grads = {}
    for l in range(1, parameters['layers']+1):
        layer_name = 'layer' + str(l)
        grads[layer_name] = {}
    data_keys['grads'] = to_cache(endpoint=endpoint, obj=grads, name='grads')
        
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
            sns_message = "Error downloading input data from S3, S3 object does not exist"
            publish_sns(sns_message)
        else:
            raise
    
    # Extract the neural network parameters
    with open('/tmp/parameters.json') as parameters_file:
        parameters = json.load(parameters_file)
    
    # Get the ARNs for the TrainerLambda and NeuronLambda
    parameters['ARNs'] = {
        'TrainerLambda': get_lambda_function_arn('TrainerLambda'),
        'NeuronLambda': get_lambda_function_arn('NeuronLambda')
    }

    # Build in additional neural network parameters
    # Input data sets and data set parameters
    parameters['s3_bucket'] = event['Records'][0]['s3']['bucket']['name']
    parameters['data_keys'],\
    parameters['input_data'],\
    parameters['dims'] = initialize_data(
        endpoint=endpoint,
        parameters=parameters
    )
    
    # Initialize payload to `TrainerLambda`
    payload = {}
    # Initialize the overall state
    payload['state'] = 'start'
    # Dump the parameters to ElastiCache
    payload['parameter_key'] = to_cache(endpoint, obj=parameters, name='parameters')
    #payload['endpoint'] = endpoint
    # Prepare the payload for `TrainerLambda`
    payloadbytes = dumps(payload)
    
    print("Complete Neural Network Settings: \n")
    print(dumps(parameters, indent=4, sort_keys=True))
    print("Payload to be sent to TrainerLambda: \n" + dumps(payload, indent=4, sort_keys=True))

    return

    
    # Invoke TrainerLambda for next layer
    try:
        response = lambda_client.invoke(
            FunctionName=parameters['ARNs']['TrainerLambda'],
            InvocationType='Event',
            Payload=payloadbytes
            )
    except botocore.exceptions.ClientError as e:
        sns_message = "Errors occurred invoking TrainerLambda from LaunchLambda."
        sns_message += "\nError:\n" + e
        sns_message += "\nCurrent Payload:\n" +  dumps(payload, indent=4, sort_keys=True)
        publish_sns(sns_message)
        print(e)
        raise
    print(response)

    return