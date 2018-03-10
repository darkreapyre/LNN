#!/usr/bin/python
"""
Lambda Function that simulates a single Neuron for both forward and backward propogation.
If the state is `forward` then the function simulates forward propogation for `X` to the `Cost`.
If the state is backward, then the function calculates the gradient of the derivative of the 
activation function for the current layer.
"""

# Import Libraries and Global variables needed by the Lambda Function
from Utils import *

# Lambda Handler
def lambda_handler(event, context):
    """
    This Lambda Function simulates a single Perceptron for both 
    forward and backward propogation.
    """
    # Ensure that this is not a duplicate invokation
    invID = event.get('invID')
    name = "NeuronLambda" #Name of the current Lambda function
    task = 'get'
    cnt = inv_counter(name, invID, task) #should be 0 for a new function invoked
    if cnt == 0:
        task = 'update'
        inv_counter(name, invID, task)
    else:
        sys.exit()
    
    # Get the current state
    state = event.get('state')
    batch = event.get('batch')
    layer = event.get('layer')
    # Get the ID of this individual neuron
    ID = event.get('id')
    # Determine is this is the last Neuron in the layer
    last = event.get('last')
    activation = event.get('activation')
    
    # Get the mini-batch parameters from Elasticache
    parameters = from_cache(db=batch, key=event.get('parameter_key'))

    # Debug Statement
    print("Starting {} propagation on Neuron: {}, for Batch {} and Layer {}".format(state, str(ID), str(batch), str(layer)))
    
    # Forward propagation from A0 to Cost
    if state == 'forward':
        # Activations from the previous layer
        A_prev = from_cache(db=batch, key=parameters['data_keys']['A'+str(layer - 1)])
        # Get the MASTER weights for this neuron
        w = from_cache(db=15, key=parameters['data_keys']['W'+str(layer)])[ID-1, :]
        # Convert weights to a row vector
        w = w.reshape(1, w.shape[0])
        # Get the MASTER bias for this neuron as row vector
        b = from_cache(db=15, key=parameters['data_keys']['b'+str(layer)])[ID-1, :]
        # Perform the linear part of the layer's forward propagation
        z = np.dot(w, A_prev) + b
        # Upload the linear transformation results to batch ElastiCache for use with Backprop
        to_cache(db=batch, obj=z, name='layer'+str(layer)+'_z_'+str(ID))
        
        # Perform non-linear activation based on the activation function
        if activation == 'sigmoid':
            a = sigmoid(z)
        elif activation == 'relu':
            a = relu(z)
        else:
            # No other activation functions supported at this time
            pass
        # Upload the results to batch ElastiCache for `TrainerLambda` to vectorize
        to_cache(db=batch, obj=a, name='layer'+str(layer)+'_a_'+str(ID))

        # Debug Statements
        print("Completed Forward Propogation for batch {}, layer {}.".format(str(batch), str(layer)))
        
        if last == 'True':
            # Update parameters with this Neuron's data
            parameters['layer'] = layer + 1
            # Build the state payload
            payload = {}
            payload['parameter_key'] = to_cache(db=batch, obj=parameters, name='parameters')
            payload['state'] = 'forward'
            payload['batch'] = batch
            payload['layer'] = layer + 1
            
            # Debug Statement
            #print("Payload to be sent to TrainerLambda: \n" + dumps(payload, indent=4, sort_keys=True))
            
            # Create an Invokation ID to ensure no duplicate funcitons are launched
            invID = str(uuid.uuid4()).split('-')[0]
            name = "TrainerLambda" #Name of the Lambda fucntion to be invoked
            task = 'set'
            inv_counter(name, invID, task)
            payload['invID'] = invID
            payloadbytes = dumps(payload)

            # Invoke TrainerLambda to process activations
            try:
                response = lambda_client.invoke(
                    FunctionName=parameters['ARNs']['TrainerLambda'],
                    InvocationType='Event',
                    Payload=payloadbytes
                )
            except botocore.exceptions.ClientError as e:
                sns_message = "Errors occurred invoking Trainer Lambd from NeuronLambdaa."
                sns_message += "\nError:\n" + str(e)
                sns_message += "\nCurrent Payload:\n" +  dumps(payload, indent=4, sort_keys=True)
                publish_sns(sns_message)
                print(e)
                raise
        
        return
    
    # Backprop from Cost to X (A0)
    elif state == 'backward':
        # Get necessary parameters for the mini-batch
        r = redis(host=endpoint, port=6379, db=batch, charset="utf-8", decode_responses=True)
        z_key = []
        for z in r.scan_iter(match='layer'+str(layer)+'_z_'+str(ID)+'*'):
            z_key.append(z)
        z = from_cache(db=batch, key=z_key[0])
        m = from_cache(db=batch, key=parameters['data_keys']['m'])
        A_prev = from_cache(
            db=batch,
            key=parameters['data_keys']['A'+str(layer-1)]
        )
        
        # Get the derivative of the current layer's activation,
        # based on the size of the layer.
        if layer == parameters['layers']:
            # This is the last layer in the network, then assume
            # a single neuron
            dA = from_cache(db=batch, key=parameters['data_keys']['dA'+str(layer)])
            # Get the MASTER weights for this layer
            W = from_cache(db=15, key=parameters['data_keys']['W'+str(layer)])
        else:
            dA = from_cache(db=batch, key=parameters['data_keys']['dA'+str(layer)])[ID-1, :]
            dA = dA.reshape(1, dA.shape[0])
            # Get the MASTER weights for this layer
            W = from_cache(db=15, key=parameters['data_keys']['W'+str(layer)])[ID-1, :]
            W = W.reshape(1, W.shape[0])
        
        # Calculate the deivative of the activations
        if activation == 'sigmoid':
            dZ = sigmoid_backward(dA, z)
        elif activation == 'relu':
            dZ = relu_backward(dA, z)
        else:
            # No other activations supported at this time
            pass
        
        # Upload the derivative of the activations for the 
        # mini-batch to ElastiCache, to be used by `TrainerLambda`.
        to_cache(db=batch, obj=dZ, name='layer'+str(layer)+'_dZ_'+str(ID))
        
        # Calculate the derivative of the Weights for this neuron
        dw = 1 / m * np.dot(dZ, A_prev.T)
        # Upload the derivative of the weight for the 
        # neuron to ElastiCache, to be used by `TrainerLambda`.
        to_cache(db=batch, obj=dw, name='layer'+str(layer)+'_dw_'+str(ID))
        
        # Debug statement
        assert(dw.shape == W.shape)
        
        # Calculate the derivate of the bias for this neuron
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        # Upload the derivative of the bias for the neuron
        # to ElastiCache, to be used by `TrainerLambda`.
        to_cache(db=batch, obj=db, name='layer'+str(layer)+'_db_'+str(ID))
        
        # Debug Statements
        print("Completed Back Propogation for batch {}, layer {}".format(str(batch), str(layer)))
        
        if last == "True":
            # Update parameters with this Neuron's data
            parameters['layer'] = layer - 1
            # Build the state payload
            payload = {}
            payload['parameter_key'] = to_cache(db=batch, obj=parameters, name='parameters')
            payload['state'] = 'backward'
            payload['batch'] = batch
            payload['layer'] = layer - 1
            
            # Debug Statement
            #print("Payload to be sent to TrainerLambda: \n" + dumps(payload, indent=4, sort_keys=True))

            # Create an Invokation ID to ensure no duplicate funcitons are launched
            invID = str(uuid.uuid4()).split('-')[0]
            name = "TrainerLambda" #Name of the Lambda fucntion to be invoked
            task = 'set'
            inv_counter(name, invID, task)
            payload['invID'] = invID
            payloadbytes = dumps(payload)

            # Invoke TrainerLambda to process activations
            try:
                response = lambda_client.invoke(
                    FunctionName=parameters['ARNs']['TrainerLambda'],
                    InvocationType='Event',
                    Payload=payloadbytes
                )
            except botocore.exceptions.ClientError as e:
                sns_message = "Errors occurred invoking Trainer Lambd from NeuronLambdaa."
                sns_message += "\nError:\n" + str(e)
                sns_message += "\nCurrent Payload:\n" +  dumps(payload, indent=4, sort_keys=True)
                publish_sns(sns_message)
                print(e)
                raise
            
        return

    else:
        sns_message = "General error processing TrainerLambda handler!"
        publish_sns(sns_message)
        raise