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
from json import dumps, loads
from boto3 import client, resource, Session
import botocore
import uuid
import io
import redis
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
#global parameter_key
#global parameters 
#global results_key
#global results

# Helper Functions
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
        print(str(type(obj)) + "is not a supported serialization type")

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
        print(str(type(obj)) + "is not a supported serialization type")

def start_epoch(epoch, layer, results, parameter_key):
    """
    Starts a new epoch and configures the necessary state tracking objcts.
    
    Arguments:
    epoch -- Integer representing the "current" epoch.
    layer -- Integer representing the current hidden layer.
    """

    # Initialize the results object for the new epoch
    # Note: key = 'results|json'
    results['epoch' + str(epoch)] = {}
    results_key = to_cache(endpoint=endpoint, obj=results, name='results')
    
    # Start forwardprop
    propogate(direction='forward', epoch=epoch, layer=layer+1, parameter_key=parameter_key)

def finish_epoch(direction, epoch, layer):
    """
    Closes out the current epoch and updates the necessary information to the results object.

    Arguments:
    direction -- The current direction of the propogation, either `forward` or `backward`.
    epoch -- Integer representing the "current" epoch to close out.
    """

    #TBD
    pass

def optimize(epoch, layer, params, grads):
    """
    Optimizes `w` and `b` by running Gradient Descent to get the `cost`.

    Arguments:
    epoch -- Integer representing the "current" epoch to close out.
    layer -- Integer representing the current hidden layer.
    params -- Dictionary containing the gradients of the weights and 
                bias.
    grads -- Dictionary containing the gardients of the wights and
                bias with respect to the cost function.
    
    Returns:
    TBD
    """

    #TBD
    #Get the learning rate
    #learning_rate = parameters['learning_rate']

    """
    Note:
    Probably have to get the cost from the output of the NeuronLambdas
    OR
    Get data from the NeuronLambdas and calculate the cost here
    """

    # Get the grads and params
    
    # Perform the update rule
    #w = w - learning_rate * grads['dw']
    #b = b - learning_rate * grads['db']

    pass

def end():
    """
    Finishes out the process and launches the next state mechanisms for prediction.
    """

    #TBD
    pass

def propogate(direction, epoch, layer, parameter_key):
    """
    Determines the amount of "hidden" units based on the layer and loops
    through launching the necessary `NeuronLambda` functions with the 
    appropriate state. Each `NeuronLambda` implements the cost function 
    OR the gradients depending on the direction.

    Arguments:
    direction -- The current direction of the propogation, either `forward` or `backward`.
    epoch -- Integer representing the "current" epoch to close out.
    layer -- Integer representing the current hidden layer.
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

    # Get the parameters for the layer
    num_hidden_units = parameters['neurons']['layer' + str(layer)]
    
    # Build the NeuronLambda payload
    payload = {}
    # Add the parameters to the payload
    payload['state'] = direction
    payload['parameter_key'] = parameter_key
    payload['results_key'] = results_key
    payload['epoch'] = epoch
    payload['layer'] = layer

    # Determine process based on direction
    if direction == 'forward':
        # Launch Lambdas to propogate forward
        # Create the Activation tracking object
        A = {}
        A['layer' + str(layer)] = {}
        # Cache the object
        A_key = to_cache(endpoint=endpoint, obj=A, name='A')
        parameters['data_keys']['A'] = A
        # Update ElastiCache with the latest parameters
        parameter_key = to_cache(endpoint=endpoint, obj=parameters, name='parameters')
        # Prepare the payload for `NeuronLambda`
        payload['parameter_key'] = parameter_key
        # Remember to start the count from 1 as hidden unit indexing
        # starts at 1
        for i in range(1, num_hidden_units + 1):
            # Prepare the payload for `NeuronLambda`
            payload['id'] = i
            if i == num_hidden_units:
                payload['last'] = True
            else:
                payload['last'] = False
            payload['activation'] = parameters['activations']['layer' + str(layer)]
            payloadbytes = dumps(payload)
            print("Payload to be sent NeuronLambda: \n" + dumps(payload, indent=4, sort_keys=True))

######################################################################################################            
#            # Invoke NeuronLambdas for next layer
#            try:
#                response = lambda_client.invoke(
#                    FunctionName=environ['NeuronLambda'], #ENSURE ARN POPULATED BY CFN
#                    InvocationType='Event',
#                    Payload=payloadbytes
#                )
#            except botocore.exceptions.ClientError as e:
#                print(e)
#                raise
#            print(response)
######################################################################################################
        
        return
    
    elif direction == 'backward':
        # Launch Lambdas to propogate backward
        # Create the gradient tracking object
        grads = {}
        grads['layer' + str(layer-1)] = {}
        # Cache the object
        grads_key = to_cache(endpoint=endpoint, obj=grads, name='grads')
        parameters['data_keys']['grads'] = grads_key
        # Update ElastiCache with the latest parameters
        parameter_key = to_cache(endpoint=endpoint, obj=parameters, name='parameters')
        # Prepare the payload for `NeuronLambda`
        payload['parameter_key'] = parameter_key

        for i in range(1, num_hidden_units + 1):
            # Prepare the payload for `NeuronLambda`
            payload['id'] = i
            if i == num_hidden_units:
                payload['last'] = True
            else:
                payload['last'] = False
            payloadbytes = dumps(payload)
            print("Payload to be sent to NeuronLambda: \n" + dumps(payload, indent=4, sort_keys=True))

######################################################################################################            
#            # Invoke NeuronLambdas for next layer
#            try:
#                response = lambda_client.invoke(
#                    FunctionName=environ['NeuronLambda'], #ENSURE ARN POPULATED BY CFN
#                    InvocationType='Event',
#                    Payload=payloadbytes
#                )
#            except botocore.exceptions.ClientError as e:
#                print(e)
#                raise
#            print(response)
######################################################################################################

            return

    else:
        raise

    """
    Note:
    When launching NeuronLambda with multiple hidden unit,
    remember to assign an ID, also remember to start at 1
    and not 0. 
    i.e num_hidden_units = 5
        for i in range(1, num_hidden_units + 1):
        # Do stuff
    """

def lambda_handler(event, context):
    """
    Processes the `event` vaiables from the various Lambda functions that call it, 
    i.e. `TrainerLambda` and `NeuronLambda`. Determines the "current" state and
    then directs the next steps.
    """
       
    # Get the Neural Network parameters from Elasticache
    # This Will fail if this is the first time `TrainerLambda` is called since
    # there is no results object
    try:
        results_key = event.get('results_key')
        results = from_cache(endpoint=endpoint, key=results_key)
    except Exception:
        pass
    
    # Get the current state from the invoking lambda
    state = event.get('state')
    global parameters
#    global parameter_key
    parameter_key = event.get('parameter_key')
    parameters = from_cache(endpoint=endpoint, key=parameter_key)
    
    # Execute appropriate action based on the the current state
    if state == 'forward':
        # Get important state variables
        epoch = event.get('epoch')
        layer = event.get('layer')
        
        # Determine the location within forwardprop
        if layer > parameters['layers']:
            # Location is at the end of forwardprop, therefore calculate Cost
            # First pre-process the Activations
            """
            # Note: May not need to use `scan_iter()` anymore since the exact 
            #keys are available in `data_keys`, so a scan of all keys pertaining
            # to `a_*` may not be necessary as the keys pertaining to the 
            # layer are there. Just need to look through the layers.
            #
            # Try the following for each individual layer:

            num_layers = parameters['layers']
            A_key = parameters['data_keys']['A']
            A = from_cache(endpoint=endpoint, key=A_key)
            
            for l in range(1, num_layers + 1):
               Do the following for the layer without using `scan_iter()`
               key_list = list(A.get('layer' + str(l)).keys()) #<- hope this works
               tmp_dict = {}
               for i in key_list:
                   tmp_dict[i] = from_cache(endpoint=endpoint, key=i)
               num_activations = len(key_list)
               tmp_array = np.array([arr.tolist() for arr in tmp_dict.values()])
            ########   key_list[l] = np.array([arr.tolist() for arr in tmp_dict.values()]) #<- hope this works
               if num_activations == 1:
                   dims = (key_list[0].split('|')[1].split('#')[1:])
                   tmp_array = tmp_array.reshape(int(dims[0]), int(dims[1]))
               else:
                   tmp_array = np.squeeze(tmp_array)
               A_name = 'A' + str(l)
               parameters['data_keys'][A_name] = to_cache(endpoint=endpoint, obj=tmp_array, name=A_name)
            
            # Note: See if this can be built into a function, for example:
            A_list = build_matrix(m_type=(activations | weights | bias), num_layers)
            if len(A_list) == 1:
                cost = (-1 / m) * np.sum(Y * (np.log(A_list[0])) + ((1 - Y) * np.log(1 - A_list[0])))
            else:
                # Multiple layers
                logprobs = np.multiply(np.log(A_list[1]), Y) + np.multiply((1 - Y), np.log(1 - A_list[1]))
                cost = - np.sum(logprobs) / m
            ####################################################################################################
            ###               FFFFFFFFFFFFFFFUUUUUUUUUUUUUCCCCCCCCCCKKKKKKKKKK!!!!!!!!!!!!!!!                ###
            ###                                                                                              ###
            ### Need to think about this, as the cost may only need to be computed on the last activation    ###
            ### and hence the only requirement is to compute the last layer.                                 ###
            ###                                                                                              ###
            ###                                      BUT                                                     ###
            ###                                                                                              ###
            ### I think that the derivative calculations still need A1 as well as A2, so this may still be   ###
            ### usable!                                                                                      ###
            ####################################################################################################

            """

            # Get the activations from the NeuronLambda by using this redis
            # command to ensure that a pure string is returned for the key
            r = redis(host=endpoint, port=6379, db=0, charset="utf-8", decode_responses=True)
            key_list = []
            for key in r.scan_iter(match='a_*'):
                key_list.append(key)
            # Create a dictionary of numpy arrays
            A_dict = {}
            for i in key_list:
                A_dict[i] = from_cache(endpoint=endpoint, key=i)
            # Create the numpy array of activations, depending on the 
            # number of hidden units
            num_activations = len(key_list)
            # Create the matrix of Activations
            A = np.array([arr.tolist() for arr in A_dict.values()])
            # Format the shape depending on the number of activations
            if num_activations == 1:
                dims = (key_list[0].split('|')[1].split('#')[1:])
                A = A.reshape(int(dims[0]), int(dims[1]))
            else:
                A = np.squeeze(A)
            # Add `A{Layer}` to input data for backprop
            A_name = 'A' + str(layer-1)
            parameters['data_keys']['A'] = to_cache(endpoint=endpoint, obj=A, name=A_name)
            
            # Calculate the Cost
            # Get the training examples data
            Y = from_cache(endpoint=endpoint, key=parameters['data_keys']['train_set_y'])
            m = from_cache(endpoint=endpoint, key=parameters['data_keys']['m'])
            # Calculate the Cost
            cost = (-1 / m) * np.sum(Y * (np.log(A)) + ((1 - Y) * np.log(1 - A)))

            # Update results with the Cost
            results['epoch' + str(epoch)]['cost'] = cost
            results_key = to_cache(endpoint=endpoint, obj=results, name='results')

            # Start backprop
            propogate(direction='backward', epoch=epoch, layer=layer-1)
            
        else:
            # Move to the next hidden layer
            #propogate(direction='forward', layer=layer, activations=activations)
            
            pass
        
    elif state == 'backward':
        # Get important state variables
        epoch = event.get('epoch')
        layer = event.get('layer')
        
        # Determine the location within backprop
        if epoch == parameters['epochs'] and layer == 0:
            # Location is at the end of the final epoch
            # First pre-process the Weights
            # Note: Get the activations from the NeuronLambda by using this redis
            # command to ensure that a pure string is returned for the key
            r = redis(host=endpoint, port=6379, db=0, charset="utf-8", decode_responses=True)
            key_list = []
            for key in r.scan_iter(match='dw_*'):
                key_list.append(key)
            # Create a dictionary of the numpy arrays
            dW_dict = {}
            for i in key_list:
                dW_dict[i] = from_cache(endpoint=endpoint, key=i)
            # Create a numpy array of all the Weights, depending on
            # the number of hidden units
            num_weights = len(key_list)
            # Create the matrix of Weights
            dW = np.arra([arr.tolist() for arr in dW_dict.values()])
            # Format the shape depending onthe number of Weights
            if num_weights == 1:
                dims = (key_list[0].split('|')[1].split('#')[1:])
                dW = dW.reshape(int(dims[0]), int(dims1))
            else:
                dW = np.squeeze(dW)
            # Concatenate into matrix of column vectors
            dW = dW.T
            # Add `W{layer}` to input for Gradient Descent
            dW_name = 'dW' + str(layer+1)
            parameters['data_keys'][dW_name] = to_cache(endpoint=endpoint, obj=dW, name=dW_name)

            # Repeat teh above process for the Bias
            key_list = []
            for key in r.scan_iter(match="db_*"):
                key_list.append(key)
            db_dict = {}
            for i in key_list:
                db_dict[i] = from_cache(endpoint=endpoint, key=i)
            num_bias = len(key_list)
            db = np.array([arr.tolist() for arr in db_dict.value()])
            if num_bias ==1:
                db = np.float64(db)
            else:
                db = np.squeeze(db)
            db = db.T
            dB_name = 'db' + str(layer+1)
            parameters['data_keys'][db_name] = to_cache(endpoint=endpoint, obj=db, name=db_name)

            # Run Gadient Descent
            """
            # Note: Need to determine the exact location as SGD only runs at the end of 
            # the epoch and I need to ensure that we truly are and not just calculating
            # Weights and Bias for a hidden layer -> Work through flow and detmine if 
            # this is the right place
            """

            
            # Finalize the the process and clean up
            #end()
            
            pass
            
        elif epoch < parameters['epochs'] and layer == 0:
            # Location is at the end of the current epoch and backprop is finished

            # Run Gradient Descent
            
            # Calculate the weights for this epoch
            
            # Update/get the weights and bias from the results object
            
            # Start the next epoch
            #epoch = epoch + 1
            #start_epoch(epoch)
            
            pass
            
        else:
            # Move to the next hidden layer
            #propogate(direction='backward', layer=layer-1)
            
            pass
            
    elif state == 'start':
        # Start of a new run of the process
        # Initialize the results tracking object
        #global results
        results = {}
        
        # Create initial parameters
        epoch = 1
        layer = 0
        start_epoch(epoch=epoch, layer=layer, results=results, parameter_key=parameter_key)
       
    else:
        print("No state informaiton has been provided.")
        raise

###### Add code to clean up if this is `epochs + 1`