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
rgn = environ['Region']
s3_client = client('s3', region_name=rgn) # S3 access
s3_resource = resource('s3')
sns_client = client('sns', region_name=rgn) # SNS
lambda_client = client('lambda', region_name=rgn) # Lambda invocations
redis_client = client('elasticache', region_name=rgn) # ElastiCache
# Retrieve the Elasticache Cluster endpoint
cc = redis_client.describe_cache_clusters(ShowCacheNodeInfo=True)
endpoint = cc['CacheClusters'][0]['CacheNodes'][0]['Endpoint']['Address']
# Set redis database to `15` as default for the main parameters
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
           the case of Numpy Arrays, the Length and Width of the 
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
        cache = redis(host=endpoint, port=6379, db=db)
        cache.set(key, val)
        return key
    # Test if the object to serialize is a string
    elif type(obj) is str:
        key = '{0}|{1}'.format(name, 'string')
        val = obj
        cache = redis(host=endpoint, port=6379, db=db)
        cache.set(key, val)
        return key
    # Test if the object to serialize is an integer
    elif type(obj) is int:
        key = '{0}|{1}'.format(name, 'int')
        # Convert to a string
        val = str(obj)
        cache = redis(host=endpoint, port=6379, db=db)
        cache.set(key, val)
        return key
    # Test if the object to serialize is a dictionary
    elif type(obj) is dict:
        # Convert the dictionary to a String
        val = json.dumps(obj)
        key = '{0}|{1}'.format(name, 'json')
        cache = redis(host=endpoint, port=6379, db=db)
        cache.set(key, val)
        return key
    else:
        sns_message = "`to_cache` Error:\n" + str(type(obj)) + "is not a supported serialization type"
        publish_sns(sns_message)
        print("The Object is not a supported serialization type")
        raise

def from_cache(db, key):
    """
    De-serializes binary object from ElastiCache by reading
    the type of object from the name and converting it to
    the appropriate data type
    
    Arguments:
    db -- ElastiCache database
    key -- Name of the Key to retrieve the object
    
    Returns:
    obj -- The object converted to specifed data type
    """
    
    # Check if the Key is for a Numpy array containing
    # `float64` data types
    if 'float64' in key:
        cache = redis(host=endpoint, port=6379, db=db)
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
        cache = redis(host=endpoint, port=6379, db=db)
        val = cache.get(key)
        # De-serialize the value
        array_dtype, length, width = key.split('|')[1].split('#')
        obj = np.fromstring(val, dtype=array_dtype).reshape(int(length), int(width))
        return obj
    # Check if the Key is for a json type
    elif 'json' in key:
        cache = redis(host=endpoint, port=6379, db=db)
        obj = cache.get(key)
        return json.loads(obj)
    # Chec if the Key is an integer
    elif 'int' in key:
        cache = redis(host=endpoint, port=6379, db=db)
        obj = cache.get(key)
        return int(obj)
    # Check if the Key is a string
    elif 'string' in key:
        cache = redis(host=endpoint, port=6379, db=db)
        obj = cache.get(key)
        return obj
    else:
        sns_message = "`from_cache` Error:\n" + str(type(obj)) + "is not a supported serialization type"
        publish_sns(sns_message)
        print("The Object is not a supported de-serialization type")
        raise

def sigmoid(z):
    """
    Computes the sigmoid of z.
    
    Arguments:
    z -- A scalar or numpy array of any size
    Return:
    sigmoid(z)
    """
    return 1. / (1. + np.exp(-z))

def relu(z):
    """
    Implement the ReLU function.
    
    Arguments:
    z -- Output of the linear layer, of any shape
    Returns:
    a -- Post-activation parameter, of the same shape as z
    """
    a = np.maximum(0, z)
    # Debug statement
    #assert(a.shape == z.shape)
    return a

def sigmoid_backward(dA, z):
    """
    Implement the derivative of the sigmoid function.
    
    Arguments:
    dA -- Post-activation gradient, of any shape
    z -- Cached linear activation from Forward prop
    
    Returns:
    dZ -- Gradient of the Cost with respect to z
    """
    s = 1. / (1. + np.exp(-z))
    dZ = dA * s * (1 - s)
    # Debug statement
    #assert(dZ.shape == z.shape)
    return dZ

def relu_backward(dA, z):
    """
    Implement the backward propagation for a single ReLU unit.
    
    Arguments:
    dA -- Post-activation gradient, of any shape
    z -- Cached linear activation from Forward propagation
    
    Return:
    dz -- Gradient of the Cost with respect to z
    """
    dz = np.array(dA, copy=True) #converting dz to a correct object
    # When z <= 0, set dz to 0 as well
    dz[z <= 0] = 0
    # Debug statement
    #assert (dz.shape == z.shape)
    return dz

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
    invID -- The unique invocation ID created for the specific invocation
    task -- Task to perform: set | get | update
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

def propogate(direction, batch, layer, parameter_key):
    """
    Determines the amount of "hidden" units based on the layer and loops
    through launching the necessary `NeuronLambda` functions with the 
    appropriate state. Each `NeuronLambda` implements the cost function 
    OR the gradients depending on the direction.

    Arguments:
    direction -- The current direction of the propagation, either `forward` or `backward`.
    epoch -- Integer representing the "current" epoch to close out.
    layer -- Integer representing the current hidden layer.

    Note: When launching NeuronLambda with multiple hidden unit,
    remember to assign an ID, also remember to start at 1
    and not 0. for example:
    num_hidden_units = 5
    for i in range(1, num_hidden_units + 1):
        # Do stuff
    """
    # Get the parameters for the layer
    parameters = from_cache(db=batch, key=parameter_key)
    num_hidden_units = parameters['neurons']['layer'+str(layer)]
    
    # Build the NeuronLambda payload
    payload = {}
    # Add the parameters to the payload
    payload['state'] = direction
    payload['batch'] = batch
    payload['layer'] = layer
    
    # Determine process based on direction
    if direction == 'forward':
        # Launch Lambdas to propagate forward
        # Prepare the payload for `NeuronLambda`
        # Update parameters with this function's updates
        parameters['layer'] = layer
        payload['parameter_key'] = to_cache(db=batch, obj=parameters, name='parameters')

        # Debug Statements
        print("Starting Forward Propagation for batch " + str(batch) + ", layer " + str(layer))

        # Prepare the payload for `NeuronLambda`
        for i in range(1, num_hidden_units + 1):
            payload['id'] = i
            if i == num_hidden_units:
                payload['last'] = "True"
            else:
                payload['last'] = "False"
            payload['activation'] = parameters['activations']['layer' + str(layer)]
            
            # Debug Statements
            #print("Payload to be sent NeuronLambda: \n" + dumps(payload, indent=4, sort_keys=True))
                        
            # Create an Invokation ID to ensure no duplicate funcitons are launched
            invID = str(uuid.uuid4()).split('-')[0]
            name = "NeuronLambda" #Name of the Lambda fucntion to be invoked
            task = 'set'
            inv_counter(name, invID, task)
            payload['invID'] = invID
            payloadbytes = dumps(payload)

            # Invoke NeuronLambdas for next layer
            try:
                response = lambda_client.invoke(
                    FunctionName=parameters['ARNs']['NeuronLambda'],
                    InvocationType='Event',
                    Payload=payloadbytes
                )
            except botocore.exceptions.ClientError as e:
                sns_message = "Errors occurred invoking Neuron Lambda from TrainerLambda."
                sns_message += "\nError:\n" + str(e)
                sns_message += "\nCurrent Payload:\n" +  dumps(payload, indent=4, sort_keys=True)
                publish_sns(sns_message)
                print(e)
                raise
            print(response)
    
    elif direction == 'backward':
        # Launch Lambdas to propagate backward        
        # Prepare the payload for `NeuronLambda`
        # Update parameters with this functions updates
        parameters['layer'] = layer
        payload['parameter_key'] = to_cache(db=batch, obj=parameters, name='parameters')
        
        # Debug Statements
        print("Starting Backward Propagation for batch " + str(batch) + ", layer " + str(layer))

        # Prepare the payload for `NeuronLambda`
        for i in range(1, num_hidden_units + 1):
            payload['id'] = i
            if i == num_hidden_units:
                payload['last'] = "True"
            else:
                payload['last'] = "False"
            payload['activation'] = parameters['activations']['layer' + str(layer)]
            
            # Debug Statements
            #print("Payload to be sent NeuronLambda: \n" + dumps(payload, indent=4, sort_keys=True))

            # Create an Invokation ID to ensure no duplicate funcitons are launched
            invID = str(uuid.uuid4()).split('-')[0]
            name = "NeuronLambda" #Name of the Lambda fucntion to be invoked
            task = 'set'
            inv_counter(name, invID, task)
            payload['invID'] = invID
            payloadbytes = dumps(payload)

            # Debug Statement
            #print("Payload to be sent to NeuronLambda: \n" + dumps(payload, indent=4, sort_keys=True))

            # Invoke NeuronLambdas for next layer
            try:
                response = lambda_client.invoke(
                    FunctionName=parameters['ARNs']['NeuronLambda'],
                    InvocationType='Event',
                    Payload=payloadbytes
                )
            except botocore.exceptions.ClientError as e:
                sns_message = "Errors occurred invoking Neuron Lambda from TrainerLambda."
                sns_message += "\nError:\n" + str(e)
                sns_message += "\nCurrent Payload:\n" +  dumps(payload, indent=4, sort_keys=True)
                publish_sns(sns_message)
                print(e)
                raise
            print(response)

    else:
        sns_message = "Errors processing the `propogate() function."
        publish_sns(sns_message)
        raise


def start_batch(batch, layer, parameter_key):
    """
    Starts a new mini-batch and configures the necessary
    state tracking objects for the batch.
    
    Arguments:
    batch -- Integer representing the "current" mini-batch
    layer -- Integer representing the current hidden layer
    """
    # Configure state parameters for this batch
    parameters = from_cache(db=batch, key=parameter_key)
    parameters['layer'] = layer
    parameter_key = to_cache(
        db=batch,
        obj=parameters,
        name='parameters'
    )
    
    # Start Forward propogation
    propogate(
        direction='forward',
        batch=batch,
        layer=layer+1,
        parameter_key=parameter_key
    )

def vectorizer(Outputs, Layer, batch, parameters):
    """
    Creates a matrix of the individual neuron output for better vectorization.
    
    Arguments:
    Outputs -- ElastiCache key to search for the data from `NeuronLambda`
               e.g. 'a' for activations; 'dw' for Weight Derivatives
    Layer -- Layer to search for neuron output that need to vectorized
    batch -- The current mini-batch
    parameters -- The current mini-batch parameters
    
    Returns:
    result -- Matrix matching the size for the entire layer
    """
    # Use the following Redis command to ensure a pure string is return for the key
    r = redis(host=endpoint, port=6379, db=batch, charset="utf-8", decode_responses=True)
    search_results = []
    # Compile a list of all the neurons in the search layer based on the search criteria
    for n in range(1, parameters['neurons']['layer'+str(Layer)]+1):
        tmp = r.keys('layer'+str(Layer)+'_'+str(Outputs)+'_'+str(n)+'|*')
        search_results.append(tmp)
    # Created an ordered list of neuron data keys
    key_list = []
    for result in search_results:
        key_list.append(result[0])
    # Create a dictionary of neuron data
    Dict = {}
    for data in key_list:
        Dict[data] = from_cache(db=batch, key=data)
    # Number of Neuron Activations for the search layer
    num_neurons = parameters['neurons']['layer'+str(Layer)]
    # Create a numpy array of the results, depending on the number
    # of neurons (a Matrix of Activations)
    matrix = np.array([arr.tolist() for arr in Dict.values()])
    if num_neurons == 1:
        # Single Neuron Activation
        dims = (key_list[0].split('|')[1].split('#')[1:])
        matrix = matrix.reshape(int(dims[0]), int(dims[1]))
    else:
        # Multiple Neuron Activations
        matrix = np.squeeze(matrix)
    
    return matrix

def update_parameters_with_gd(W, b, dW, db, learning_rate, batch, layer, parameters):
    """
    Updates parameters using one step of gradient descent for a layer.
    
    Arguments:
    W -- Matrix of the Weights for the layer
    b -- Vector of the Bias for the layer
    dW -- Matrix of the Derivatives of  Weights for the layer
    db -- Vector of the Derivatives of the Bias for the layer
    learning_rate -- Learning rate for the network, scalar
    batch -- Current mini-batch
    layer -- Current layer being optimized
    
    Returns:
    parameter_key -- Updated parameter key with Weight and Bias data keys
    """
    # Run gradient descent
    W_prime = W - learning_rate * dW
    b_prime = b - learning_rate * db
    
    # Update the current batch ElastiCache with Weights and Bias
    parameters['data_keys']['W'+str(layer+1)] = to_cache(
        db=batch,
        obj=W_prime,
        name='W'+str(layer+1)
    )
    parameters['data_keys']['b'+str(layer+1)] = to_cache(
        db=batch,
        obj=b_prime,
        name='b'+str(layer+1)
    )
    
    # Update parameters for the mini-batch
    parameter_key = to_cache(
        db=batch,
        obj=parameters,
        name='parameters'
    )
    
    return parameter_key