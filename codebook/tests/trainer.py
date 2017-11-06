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
def numpy2s3(array, name, bucket):
    """
    Serialize a Numpy array to S3 without using local copy
    
    Arguments:
    array -- Numpy array of any shape
    name -- filename on S3
    """
    f_out = io.BytesIO()
    np.save(f_out, array)
    try:
        s3_client.put_object(Key=name, Bucket=bucket, Body=f_out.getvalue(), ACL='bucket-owner-full-control')
    except botocore.exceptions.ClientError as e:
        priont(e)

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

def start_epoch(epoch, layer, parameter_key):
    """
    Starts a new epoch and configures the necessary state tracking objcts.
    
    Arguments:
    epoch -- Integer representing the "current" epoch.
    layer -- Integer representing the current hidden layer.
    """

    # Initialize the results object for the new epoch
    parameters = from_cache(endpoint=endpoint, key=parameter_key)
    
    # Add current epoch to results
    epoch2results = from_cache(endpoint=endpoint, key=parameters['data_keys']['results'])
    epoch2results['epoch' + str(epoch)] = {}
    parameters['data_keys']['results'] = to_cache(endpoint=endpoint, obj=epoch2results, name='results')
   
    # Update paramaters with this functions data
    parameters['epoch'] = epoch
    parameters['layer'] = layer
    parameter_key = to_cache(endpoint=endpoint, obj=parameters, name='parameters')
    
    # Start forwardprop
    propogate(direction='forward', epoch=epoch, layer=layer+1, parameter_key=parameter_key)

def end(parameter_key):
    """
    Finishes out the process and launches the next state mechanisms for prediction.
    """
    parameters = from_cache(
        endpoint=endpoint,
        key=parameter_key
    )
    
    # Get the results
    final_results = from_cache(
        endpoint=endpoint,
        key=parameters['data_keys']['results']
    )
    # Upload results to S3
    # TBD
    
    # Get the final Weights and Bias
    weights = from_cache(
        endpoint=endpoint,
        key=parameters['data_keys']['weights']
    )
    bias = from_cache(
        endpoint=endpoint,
        key=parameters['data_keys']['bias']
    )
    bucket = parameters['s3_bucket']
    
    # Put the weights and bias onto S3 for prediction
    numpy2s3(array=weights, name='prediction_input\weights', bucket=bucket)
    numpy2s3(array=bias, name='prediction_input\bias', bucket=bucket)

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

    Note: When launching NeuronLambda with multiple hidden unit,
    remember to assign an ID, also remember to start at 1
    and not 0. for example:
    num_hidden_units = 5
    for i in range(1, num_hidden_units + 1):
        # Do stuff
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
    parameters = from_cache(endpoint=endpoint, key=parameter_key)
    num_hidden_units = parameters['neurons']['layer' + str(layer)]
    
    # Build the NeuronLambda payload
    payload = {}
    # Add the parameters to the payload
    payload['state'] = direction
    payload['parameter_key'] = parameter_key
    payload['epoch'] = epoch
    payload['layer'] = layer

    # Determine process based on direction
    if direction == 'forward':
        # Launch Lambdas to propogate forward
        # Prepare the payload for `NeuronLambda`
        # Update parameters with this function's updates
        parameters['epoch'] = epoch
        parameters['layer'] = layer
        payload['parameter_key'] = to_cache(endpoint=endpoint, obj=parameters, name='parameters')

        print("Starting Forward Propogation for epoch " + str(epoch) + ", layer " + str(layer))

        for i in range(1, num_hidden_units + 1):
            # Prepare the payload for `NeuronLambda`
            payload['id'] = i
            if i == num_hidden_units:
                payload['last'] = "True"
            else:
                payload['last'] = "False"
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
        # Prepare the payload for `NeuronLambda`
        # Update parameters with this functions updates
        parameters['epoch'] = epoch
        parameters['layer'] = layer
        payload['parameter_key'] = to_cache(endpoint=endpoint, obj=parameters, name='parameters')

        print("Starting Backward Propogation for epoch " + str(epoch) + ", layer " + str(layer))

        for i in range(1, num_hidden_units + 1):
            # Prepare the payload for `NeuronLambda`
            payload['id'] = i
            if i == num_hidden_units:
                payload['last'] = "True"
            else:
                payload['last'] = "False"
            payload['activation'] = parameters['activations']['layer' + str(layer)]
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


def lambda_handler(event, context):
    """
    Processes the `event` vaiables from the various Lambda functions that call it, 
    i.e. `TrainerLambda` and `NeuronLambda`. Determines the "current" state and
    then directs the next steps.
    """
       
    """
    # Get the Neural Network parameters from Elasticache
    # This Will fail if this is the first time `TrainerLambda` is called since
    # there is no results object
    try:
        results_key = event.get('results_key')
        results = from_cache(endpoint=endpoint, key=results_key)
    except Exception:
        pass
    """
    
    # Get the current state from the invoking lambda
    state = event.get('state')
    global parameters
#    global parameter_key
#    parameter_key = event.get('parameter_key')
    parameters = from_cache(endpoint=endpoint, key=event.get('parameter_key'))
    
    # Execute appropriate action based on the the current state
    if state == 'forward':
        # Get important state variables
        epoch = event.get('epoch')
        layer = event.get('layer')

        # First pre-process the Activations from the "previous" layer
        # Use the folling Redis command to ensure a pure string is return for the key
        r = redis(host=endpoint, port=6379, db=0, charset="utf-8", decode_responses=True)
        key_list = []
        # Compile a list of activations
        for key in r.scan_iter(match='layer'+str(layer-1)+'_a_*'):
            key_list.append(key)
        # Create a dictionary of activation results
        A_dict = {}
        for i in key_list:
            A_dict[i] = from_cache(endpoint=endpoint, key=i)
        # Number of Neuron Activations
        num_activations = len(key_list)
        # Create a numpy array of the results, depending on the number
        # of hidden units
        A = np.array([arr.tolist() for arr in A_dict.values()])
        if num_activations == 1:
            """
            Note: This assumes a single hidden unit for the last layer
            """
            dims = (key_list[0].split('|')[1].split('#')[1:])
            #debug
            #print("Dimensions to reshape single hidden unit activations: " + str(dims))
            A = A.reshape(int(dims[0]), int(dims[1]))
            assert(A.shape == (parameters['dims']['train_set_y'][0], parameters['dims']['train_set_y'][1]))
        else:
            A = np.squeeze(A)
            assert(A.shape == (parameters['neurons']['layer'+str(layer-1)], parameters['dims']['train_set_x'][1]))
        # Add the `A` Matrix to `data_keys` for later Neuron use
        A_name = 'A' + str(layer-1)
        parameters['data_keys'][A_name] = to_cache(endpoint=endpoint, obj=A, name=A_name)

        # Update ElastiCache with this funciton's data
        parameter_key = to_cache(endpoint=endpoint, obj=parameters, name='parameters')
        
        # Determine the location within forwardprop
        if layer > parameters['layers']:
            # Location is at the end of forwardprop (layer 3), therefore calculate Cost
            # Get the training examples data
            Y = from_cache(endpoint=endpoint, key=parameters['data_keys']['train_set_y'])
            m = Y.shape[1]
            
            # Calculate the Cost
            cost = -1 / m * np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A)))
            cost = np.squeeze(cost)
            assert(cost.shape == ())

            # Update results with the Cost
            # Get the results object
            cost2results = from_cache(endpoint=endpoint, key=parameters['data_keys']['results'])
            # Append the cost to results object
            cost2results['epoch' + str(epoch)]['cost'] = cost
            # Update results key in ElastiCache
            parameters['data_keys']['results'] = to_cache(endpoint=endpoint, obj=cost2results, name='results')

            print("Cost after epoch {0}: {1}".format(epoch, cost))

            # Initialize backprop
            # Calculate the derivative of the Cost with respect to the last activation
            # Ensure that `Y` is the correct shape as the last activation
            Y = Y.reshape(A.shape)
            dZ = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
            dZ_name = 'dZ' + str(layer-1)
            parameters['data_keys'][dZ_name] = to_cache(endpoint=endpoint, obj=dZ, name=dZ_name)

            # Update parameters from theis function in ElastiCache
            parameter_key = to_cache(endpoint=endpoint, obj=parameters, name='parameters')

            # Start Backpropogation
            # This should start with layer (layers = 3-1)
            propogate(direction='backward', epoch=epoch, layer=layer-1, parameter_key=parameter_key)
            
        else:
            # Move to the next hidden layer
            #debug
            print("Propogating forward onto Layer " + str(layer))
            propogate(direction='forward', epoch=epoch, layer=layer, parameter_key=parameter_key)
        
    elif state == 'backward':
        # Get important state variables
        epoch = event.get('epoch')
        layer = event.get('layer')
        
        # Determine the location within backprop
        if epoch == parameters['epochs']-1 and layer == 0:
            # Location is at the end of the final epoch
            # Retieve the "params"
            w = from_cache(
                endpoint=endpoint,
                key=parameters['data_keys']['weights']
            )
            b = from_cache(
                endpoint=endpoint,
                key=parameters['data_keys']['bias']
            )

            # Retrieve the gradients
            grads = from_cache(
                endpoint=endpoint,
                key=parameters['data_keys']['grads']
            )
            dw = from_cache(
                endpoint=endpoint,
                key=grads['layer'+ str(layer + 1)]['dw']
            )
            db = from_cache(
                endpoint=endpoint,
                key=grads['layer'+ str(layer + 1)]['db']
            )

            # Run Gradient Descent
            w = w - learning_rate * dw
            b = b - learning_rate * db

            # Update ElastiCache with the Weights and Bias so be used as the inputs for
            # the next epoch
            parameters['data_keys']['weights'] = to_cache(
                endpoint=endpoint,
                obj=w,
                name='weights'
            )
            parameters['data_keys']['bias'] = to_cache(
                endpoint=endpoint,
                obj=b,
                name='bias'
            )
            
            # Update paramters for the next epoch
            parameter_key = to_cache(
                endpoint=endpoint,
                obj=parameters,
                name='parameters'
            )
                        
            # Finalize the the process and clean up
            end(parameter_key=parameter_key)

            """
            Previos Code

            # First pre-process the Weights
            # Get the activations from the NeuronLambda by using this redis
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

            # Repeat the above process for the Bias
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
            # Note: Need to determine the exact location as SGD only runs at the end of 
            # the epoch and I need to ensure that we truly are and not just calculating
            # Weights and Bias for a hidden layer -> Work through flow and detmine if 
            # this is the right place
            """
            
            pass
            
        elif epoch < parameters['epochs']-1 and layer == 0:
            # Location is at the end of the current epoch and backprop is finished
            # Retieve the "params"
            w = from_cache(
                endpoint=endpoint,
                key=parameters['data_keys']['weights']
            )
            b = from_cache(
                endpoint=endpoint,
                key=parameters['data_keys']['bias']
            )

            # Retrieve the gradients
            grads = from_cache(
                endpoint=endpoint,
                key=parameters['data_keys']['grads']
            )
            dw = from_cache(
                endpoint=endpoint,
                key=grads['layer'+ str(layer + 1)]['dw']
            )
            db = from_cache(
                endpoint=endpoint,
                key=grads['layer'+ str(layer + 1)]['db']
            )

            # Run Gradient Descent
            w = w - learning_rate * dw
            b = b - learning_rate * db

            # Update ElastiCache with the Weights and Bias so be used as the inputs for
            # the next epoch
            parameters['data_keys']['weights'] = to_cache(
                endpoint=endpoint,
                obj=w,
                name='weights'
            )
            parameters['data_keys']['bias'] = to_cache(
                endpoint=endpoint,
                obj=b,
                name='bias'
            )
            
            # Update paramters for the next epoch
            parameter_key = to_cache(
                endpoint=endpoint,
                obj=parameters,
                name='parameters'
            )
                        
            # Start the next epoch
            start_epoch(epoch=epoch+1, layer=0, parameter_key=parameter_key)
            
        else:
            # Move to the next hidden layer
            #propogate(direction='backward', epoch=epoch, layer=layer, parameter_key=parameter_key)
            
            pass
            
    elif state == 'start':
        # Start of a new run of the process        
        # Create initial parameters
        epoch = 0
        layer = 0
        start_epoch(epoch=epoch, layer=layer, parameter_key=event.get('parameter_key'))
       
    else:
        print("No state informaiton has been provided.")
        raise

###### Add code to clean up if this is `epochs + 1`