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

def dict2item(raw):
    """
    Converts a dictionary to the appropriate Key, Value formatting for DynamoDB
    and ensures that `float` values are converted to `Decimal`
    """
    if type(raw) is dict:
        resp = {}
        for k,v in raw.items():
            if type(v) is str:
                resp[k] = {
                    'S': v
                }
            elif type(v) is int:
                resp[k] = {
                    'I': str(v)
                }
            elif type(v) is dict:
                resp[k] = {
                    'M': dict_to_item(v)
                }
            elif type(v) is list:
                resp[k] = []
                for i in v:
                    resp[k].append(dict_to_item(i))
            elif type(v) is float:
                resp[k] = Decimal(v)
                    
        return resp
    elif type(raw) is str:
        return {
            'S': raw
        }
    elif type(raw) is int:
        return {
            'I': str(raw)
        }
    elif type(raw) is float:
        return Decimal(raw)

def numpy2s3(array, name, bucket):
    """
    Write a numpy array to S3 as a file, without using local copy
    
    Arguments:
    array -- Numpy array to save to s3
    name -- file of the saved Numpy array
    """
    f_out = io.BytesIO()
    np.save(f_out, array)
    try:
        s3_client.put_object(Key=name, Bucket=bucket, Body=f_out.getvalue(), ACL='bucket-owner-full-control')
    except botocore.exceptions.ClientError as e:
        print(e)

def dynamo_init():
    """
    Create a DynamoDB table to track stat across iteration/epochs
    """
    try:
        table = dynamodb.create_table(
            TableName = 'state',
            KeySchema = [
                {
                    'AttributeName': 'epoch',
                    'KeyType': 'HASH'
                }
            ],
            AttributeDefinitions = [
                {
                    'AttributeName': 'epoch',
                    'AttributeType': 'N'
                }
            ],
            ProvisionedThroughput = {
                'ReadCapacityUnits': 10,
                'WriteCapacityUnits': 10
            }
        )
        
        # Wait until the table is created
        table.meta.client.get_waiter('table_exists').wait(TableName='state')
    except botocore.exceptions.ClientError as e:
        print(e)
    return table.name

def storage_init():
    """
    Create temporary S3 bucket to store numpy arrays
    
    Return:
    tmp_bucket -- Name of the newly created S3 bucket
    """
    tmp_bucket = 'lnn-{}'.format(uuid.uuid4())
    print('Creating new bucket with name: {}'.format(tmp_bucket))
    try:
        s3_client.create_bucket(Bucket=tmp_bucket, CreateBucketConfiguration={'LocationConstraint': 'us-west-2'})
    except botocore.exceptions.ClientError as e:
        print(e)
    return tmp_bucket

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
    Call to `vectorize()`, standardized Numpy array of image data
    """
    return vectorize(x_orig) / 255

def initialize_data():
    """
    Extracts the training and testing data from S3, flattens, 
    standardizes, dumps back to S3 for neurons to process as layer a^0

    Returns:
    bucket -- name of the S3 bucket of Numpy arrays
    a_names -- list of the Numpy array names
    """

    # Load main dataset
    dataset = h5py.File('/tmp/datasets.h5', "r")
    # Create numpy arrays from the various h5 datasets
    train_set_x_orig = np.array(dataset["train_set_x"][:]) # train set features
    train_set_y_orig = np.array(dataset["train_set_y"][:]) # train set labels
    test_set_x_orig = np.array(dataset["test_set_x"][:]) # test set features
    test_set_y_orig = np.array(dataset["test_set_y"][:]) # test set labels
    classes = np.array(dataset["list_classes"][:]) # the list of classes
    
    # Reshape labels
    train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    # Preprocess inputs
    train_set_x = standardize(train_set_x_orig)
    test_set_x = standardize(test_set_x_orig)
    
    # Dump the inputs to the temporary s3 bucket for TrainerLambda
    bucket = storage_init() # Creates a temporary bucket for the propogation steps
    a_list = [train_set_x, train_set_y, test_set_x, test_set_y, classes] # List of Numpy array names
    a_names = [] # list to store the Numpy array names as strings
    dims = {} # dictionary of input data dimensions
    for i in range(len(a_list)):
        # Create a lis of the names of the numpy arrays
        a_names.append(name2str(a_list[i], locals()))
    for j in range(len(a_list)): 
        # Save the Numpy arrays to S3
        numpy2s3(array=a_list[j], name=a_names[j][0], bucket=bucket)
        dims[str(a_names[j][0])] = a_list[j].shape
    # Return the bucket and Numpy array names
    return bucket, [j for i in a_names for j in i], dims

def lambda_handler(event, context):
    # Retrieve datasets and setting from S3
    input_bucket = s3_resource.Bucket(str(event['Records'][0]['s3']['bucket']['name']))
    dataset_key = str(event['Records'][0]['s3']['object']['key'])
    settings_key = dataset_key.split('/')[-2] + '/parameters.json'
    try:
        response = input_bucket.download_file(dataset_key, '/tmp/datasets.h5')
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(e.response['Error']['Message'])
        else:
            raise
    try:
        response = input_bucket.download_file(settings_key, '/tmp/parameters.json')
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(e.response['Error']['Message'])
        else:
            raise
    
    # Extract the neural network parameters
    with open('/tmp/parameters.json') as parameters_file:
        parameters = json.load(parameters_file)
        
    # Create payload to send to the trainer
    # Create payload parameters from neural network parameters
    payload = {}
    payload['epochs'] = parameters['epochs']
    payload['layers'] = parameters['layers']
    payload['activations'] = parameters['activations']
    payload['neurons'] = parameters['neurons']
    payload['learning_rate'] = parameters['learning_rate']

    # Create payload parameters showing "current state" for TrainerLambda
    # Next epoch to process
    payload['epoch'] = 1
    
    # Next Layer to process
    payload['layer'] = 0

    # Variable to initialize the weights and bias to
    payload['w'] = parameters['weight']
    payload['b'] = parameters['bias']
    
    # Input data sets
    payload['s3_bucket'], \
    payload['input_data'], \
    payload['data_dimensions'] = initialize_data() # Returns S3 Bucket of input data

    # State tracking table
    payload['state_table'] = dynamo_init()

    # State to pass to the TrainerLambda
    payload['state'] = 'start'

    # Create the payload
    payloadbytes = dumps(payload)
    
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