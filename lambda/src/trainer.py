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
rgn = environ['Region']
s3_client = client('s3', region_name=rgn) # S3 access
s3_resource = resource('s3')
sns_client = client('sns', region_name=rgn) # SNS
lambda_client = client('lambda', region_name=rgn) # Lambda invocations
redis_client = client('elasticache', region_name=rgn) # ElastiCache
# Retrieve the Elasticache Cluster endpoint
cc = redis_client.describe_cache_clusters(ShowCacheNodeInfo=True)
endpoint = cc['CacheClusters'][0]['CacheNodes'][0]['Endpoint']['Address']
cache = redis(host=endpoint, port=6379, db=0)
dynamo_client = client('dynamodb', region_name=rgn)
dynamo_resource = resource('dynamodb', region_name=rgn)

# Helper Functions
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

def publish_sns(sns_message):
    """
    Publish message to the master SNS topic.

    Arguments:
    sns_message -- the Body of the SNS Message
    """

    print("Publishing message to SNS topic...")
    sns_client.publish(TargetArn=environ['SNSArn'], Message=sns_message)

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
        sns_message = "The following error occurred while running `numpy2s3`:\n" + str(e)
        publish_sns(sns_message)
        raise

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
   
    # Update parameters with this functions data
    parameters['epoch'] = epoch
    parameters['layer'] = layer
    parameter_key = to_cache(endpoint=endpoint, obj=parameters, name='parameters')
    
    # Start forwardprop
    propogate(direction='forward', epoch=epoch, layer=layer+1, parameter_key=parameter_key)

def end(parameter_key):
    """
    Finishes the oveall training sequence and saves the "optmized" 
    weights and bias to S3, for the prediction aplication.
    
    Arguments:
    parameter_key -- The ElastiCache key for the current set of state parameters.
    """
    # Get the latest parameters
    parameters = from_cache(
        endpoint=endpoint,
        key=parameter_key
    )

    # Get the results key
    final_results = from_cache(
        endpoint=endpoint,
        key=parameters['data_keys']['results']
    )
    # Upload the final results to S3
    bucket = parameters['s3_bucket']
    results_obj = s3_resource.Object(bucket,'training_results/results.json')
    try:
        results_obj.put(Body=json.dumps(final_results))
    except botocore.exceptions.ClientError as e:
        print(e)
        raise
    
    # Get the final Weights and Bias
    weights = from_cache(
        endpoint=endpoint,
        key=parameters['data_keys']['weights']
    )
    bias = from_cache(
        endpoint=endpoint,
        key=parameters['data_keys']['bias']
    )
    
    # Put the weights and bias onto S3 for prediction
    numpy2s3(array=weights, name='predict_input/weights', bucket=bucket)
    numpy2s3(array=bias, name='predict_input/bias', bucket=bucket)

    sns_message = "Training Completed Successfully!\n" + dumps(final_results)
    publish_sns(sns_message)

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

            # Crate an Invokation ID to ensure no duplicate funcitons are launched
            invID = str(uuid.uuid4()).split('-')[0]
            name = "NeuronLambda" #Name of the Lambda fucntion to be invoked
            task = 'set'
            inv_counter(name, invID, task)
            payload['invID'] = invID
            payloadbytes = dumps(payload)
            print("Payload to be sent NeuronLambda: \n" + dumps(payload, indent=4, sort_keys=True))

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

            # Crate an Invokation ID to ensure no duplicate funcitons are launched
            invID = str(uuid.uuid4()).split('-')[0]
            name = "NeuronLambda" #Name of the Lambda fucntion to be invoked
            task = 'set'
            inv_counter(name, invID, task)
            payload['invID'] = invID
            payloadbytes = dumps(payload)
            print("Payload to be sent to NeuronLambda: \n" + dumps(payload, indent=4, sort_keys=True))

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
        sns_message = "Errors processing `propogate()` function."
        publish_sns(sns_message)


def lambda_handler(event, context):
    """
    Processes the `event` vaiables from the various Lambda functions that call it, 
    i.e. `TrainerLambda` and `NeuronLambda`. Determines the "current" state and
    then directs the next steps.
    """
    # Ensure that this is not a duplicate invokation
    invID = event.get('invID')
    name = "TrainerLambda" #Name of the current Lambda function
    task = 'get'
    cnt = inv_counter(name, invID, task) #should be 0 for a new function invoked
    if cnt == 0:
        task = 'update'
        inv_counter(name, invID, task)
    else:
        sns_message = "Invocation ID Already Exists: " + str(invID)
        publish_sns(sns_message)

    # If this is an origional invocation,
    # Get the current state from the invoking lambda
    state = event.get('state')
    global parameters
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

        """
        Note: The following is code to differentiate between S-Layer and L-Layer. For now (11/18/17),
        only S-Layer will be implemented
        if num_activations == 1:
            dims = (key_list[0].split('|')[1].split('#')[1:])
            #debug
            #print("Dimensions to reshape single hidden unit activations: " + str(dims))
            A = A.reshape(int(dims[0]), int(dims[1]))
            assert(A.shape == (parameters['dims']['train_set_y'][0], parameters['dims']['train_set_y'][1]))
        else:
            A = np.squeeze(A)
            assert(A.shape == (parameters['neurons']['layer'+str(layer-1)], parameters['dims']['train_set_x'][1]))
        """
        dims = (key_list[0].split('|')[1].split('#')[1:])
        A = A.reshape(int(dims[0]), int(dims[1]))

        # Add the `A` Matrix to `data_keys` for later Neuron use
        A_name = 'A' + str(layer-1)
        parameters['data_keys'][A_name] = to_cache(endpoint=endpoint, obj=A, name=A_name)

        # Update ElastiCache with this funciton's data
        parameter_key = to_cache(endpoint=endpoint, obj=parameters, name='parameters')
        
        # Determine the location within forwardprop
        if layer > parameters['layers']:
            # Location is at the end of forwardprop (layer 3), therefore calculate Cost
            # Get the training examples data
            #Y = from_cache(endpoint=endpoint, key=parameters['data_keys']['train_set_y'])
            #m = Y.shape[1]
            Y = from_cache(
                endpoint=endpoint,
                key=parameters['data_keys']['Y']
            )
            m = from_cache(
                endpoint=endpoint,
                key=parameters['data_keys']['m']
            )
            
            # Calculate the Cost
            #cost = -1 / m * np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A)))
            cost = (-1 / m) * np.sum(Y * (np.log(A)) + ((1 - Y) * np.log(1 - A)))
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

            """
            Note: For the S-Layer implementation, the error is now calculated on NeuronLambda and hence
            Backprop is initialized there. SO the following code will be used for L-Layer testing to verify
            that Backprop should be initialized here.

            # Initialize backprop
            # Calculate the derivative of the Cost with respect to the last activation
            # Ensure that `Y` is the correct shape as the last activation
            Y = Y.reshape(A.shape)
            dZ = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
            dZ_name = 'dZ' + str(layer-1)
            parameters['data_keys'][dZ_name] = to_cache(endpoint=endpoint, obj=dZ, name=dZ_name)

            # Update parameters from this function in ElastiCache
            parameter_key = to_cache(endpoint=endpoint, obj=parameters, name='parameters')

            # Start Backpropogation on NeuronLambda
            propogate(direction='backward', epoch=epoch, layer=layer-1, parameter_key=parameter_key)
            """

            #Update parameters from this function in ElastiCache
            parameter_key = to_cache(endpoint=endpoint, obj=parameters, name='parameters')

            # Start Backpropogation on NeuronLambda
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
            learning_rate = parameters['learning_rate']
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
            
        elif epoch < parameters['epochs']-1 and layer == 0:
            # Location is at the end of the current epoch and backprop is finished
            # Retieve the "params"
            learning_rate = parameters['learning_rate']
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
            propogate(direction='backward', epoch=epoch, layer=layer, parameter_key=parameter_key)
            
    elif state == 'start':
        # Start of a new run of the process        
        # Create initial parameters
        epoch = 0
        layer = 0
        start_epoch(epoch=epoch, layer=layer, parameter_key=event.get('parameter_key'))
       
    else:
        sns_message = "General error processing TrainerLambda handler!"
        publish_sns(sns_message)
        raise