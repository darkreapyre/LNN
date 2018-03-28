#!/usr/bin/python
"""
Lambda Fucntion that tracks state, launches Neurons to process
forward and backward propogation.
"""

# Import Libraries and Global variables needed by the Lambda Function
from Utils import *

# Lambda Handler
def lambda_handler(event, context):
    """
    1. Processes the `event` variables from the various Lambda functions that call it, 
        i.e. `TrainerLambda` and `NeuronLambda`.
    2. Determines the "current" state and then directs the next steps.
    3. Performs Vectorization from the NeuronLambda forward propagation outputs.
    4. Calculates the Cost.
    5. Performs Gradient Descent given the gradients from the backward propagation outputs.
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
        sys.exit()

    # Get the current state from the invoking lambda function
    state = event.get('state')
    batch = event.get('batch')
    layer = event.get('layer')
    parameters = from_cache(db=batch, key=event.get('parameter_key'))
    
    # Execute appropriate action based on the current state
    if state == 'forward':
        # Get the Vectorized matrix of Activations from `NeuronLambda` output
        A = vectorizer(Outputs='a', Layer=layer-1, batch=batch, parameters=parameters)
        
        # Add the `A` Matrix to `data_keys` for later Neuron use
        A_name = 'A' + str(layer-1)
        parameters['data_keys'][A_name] = to_cache(db=batch, obj=A, name=A_name)
        
        # Update ElastiCache with this function's data
        parameter_key = to_cache(db=batch, obj=parameters, name='parameters')
        
        # Determine the location within Forwardprop
        if layer > parameters['layers']:
            # Location is at the end of forwardprop, therefore calculate Cost
            # Get the training examples data and no. examples (`Y` and `m`)
            Y = from_cache(db=batch, key=parameters['data_keys']['Y'])
            m = from_cache(db=batch, key=parameters['data_keys']['m'])
            
            # Calculate the Cross-Entropy Cost
            cost = (1. / m) * (-np.dot(Y, np.log(A).T) - np.dot(1 - Y, np.log(1 - A).T))
            cost = np.squeeze(cost)
            assert(cost.shape == ())            
            # Add batch cost to DynamoDB Costs tracking object
            table = dynamo_resource.Table('Costs')
            table.update_item(
                Key={
                    'epoch': parameters['epoch']
                },
                UpdateExpression="Set #batch = :costval",
                ExpressionAttributeValues={
                    ':costval': str(cost)
                },
                ExpressionAttributeNames={
                    '#batch': 'batch'+str(batch)
                }
            )

            print("Cost after Epoch {}, Batch {}: {}".format(parameters['epoch'],batch,cost))
            
            # Initialize Backprop
            # Calculate the derivative of the Cost with respect to the last
            # activation and ensure that `Y` is the correct shape as the
            # last activation.
            Y = Y.reshape(A.shape)
            dA = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
            dA_name = 'dA' + str(layer-1)
            parameters['data_keys'][dA_name] = to_cache(db=batch, obj=dA, name=dA_name)
            
            # Update parameters from this function in ElastiCache
            parameter_key = to_cache(db=batch, obj=parameters, name='parameters')
            
            # Start Backpropogation on NeuronLambda
            propogate(direction='backward', batch=batch, layer=layer-1, parameter_key=parameter_key)
            
        else:
            # Move to the next hidden layer for multiple layer networks
            # Debug statement
            print("Propogating forward onto Layer " + str(layer))
            propogate(direction='forward', batch=batch, layer=layer, parameter_key=parameter_key)
            
    elif state == 'backward':
        # Vectorize the derivatives
        dZ = vectorizer(Outputs='dZ', Layer=layer+1, batch=batch, parameters=parameters)
        
        # Pre-process the derivatives of the Weights
        dW = vectorizer(Outputs='dw', Layer=layer+1, batch=batch, parameters=parameters)
        
        # Pre-process the derivatives of the Bias
        db = vectorizer(Outputs='db', Layer=layer+1, batch=batch, parameters=parameters)
        db = vectorizer(Outputs='db', Layer=layer+1, batch=batch, parameters=parameters)
        db = db.reshape(db.shape[0], 1)
        
        # Get the Learning rate
        learning_rate = parameters['learning_rate']
        
        # Determine the location within Backprop
        if batch == (parameters['num_batches'] - 1) and (layer == 0):
            # Location is at the end of the final mini-batch, therefore
            # get the necessary MASTER parameters for optimization and close
            # out the parallel run by launching `LaunchLambda`.
            W = from_cache(db=15, key=parameters['data_keys']['W'+str(layer+1)])
            b = from_cache(db=15, key=parameters['data_keys']['b'+str(layer+1)])
            
            # Update parameters for current layer to the mini-batch ElastiCache
            parameter_key = update_parameters_with_gd(
                W, b, dW, db,
                learning_rate,
                batch, layer,
                parameters
            )
            
            # Debug Statement
            print("Training complete for batch {}, epoch {}".format(batch, parameters['epoch']))
            
            # Launch `LaunchLambda` to close out the mini-batches
            # Initialize the payload for initial batch to `LaunchLambda`
            payload = {}
            payload['state'] = 'next' # Initialize overall state
            payload['parameter_key'] = parameter_key
            # Create the invocation ID to ensure no duplicate functions
            # are launched
            invID = str(uuid.uuid4()).split('-')[0]
            name = 'LaunchLambda'
            task = 'set'
            inv_counter(name, invID, task)
            payload['invID'] = invID

            # Prepare the payload for `LaunchLambda`
            payloadbytes = dumps(payload)

            # Debug Statements
            #print("Complete Neural Network Settings for batch: {}\n".format(batch))
            #print(dumps(batch_parameters, indent=4, sort_keys=True))
            #print("\n"+"Payload to be sent to LaunchLambda: \n")
            #print(dumps(payload))

            # Invoke LaunchLambda to start the training process for
            # the current batch
            try:
                response = lambda_client.invoke(
                    FunctionName=parameters['ARNs']['LaunchLambda'],
                    InvocationType='Event',
                    Payload=payloadbytes
                )
            except botocore.exceptions.ClientError as e:
                sns_message = "Errors occurred invoking LaunchLambda from TrainerLambda."
                sns_message += "\nError:\n" + str(e)
                sns_message += "\nCurrent Payload:\n" +  dumps(payload, indent=4, sort_keys=True)
                publish_sns(sns_message)
                print(e)
                raise
            print(response)
            
        elif (batch < parameters['num_batches'] - 1) and (layer == 0):
            # Location is at the end of the current mini-batch and
            # backprop is finished, therefore get the necessary 
            # MASTER parameters for optimization and simply close
            # out the mini-batch.
            W = from_cache(db=15, key=parameters['data_keys']['W'+str(layer+1)])
            b = from_cache(db=15, key=parameters['data_keys']['b'+str(layer+1)])
            
            # Update parameters for current layer to the mini-bacth ElastiCache
            parameter_key = update_parameters_with_gd(
                W, b, dW, db,
                learning_rate,
                batch, layer,
                parameters
            )
            # Debug Statement
            print("Training complete for batch {}, epoch {}".format(batch, parameters['epoch']))
                        
        else:
            # Location is still within the backprop process, therefore 
            # get the necessary MASTER parameters for optimization and 
            # continue backprop for the mini-batch.
            W = from_cache(db=15, key=parameters['data_keys']['W'+str(layer+1)])
            b = from_cache(db=15, key=parameters['data_keys']['b'+str(layer+1)])
            
            # Calculate the current layer's activations with respect to the
            # Cost/Error.
            dA = np.dot(W.T, dZ)
            dA_name = 'dA'+str(layer)
            parameters['data_keys'][dA_name] = to_cache(
                db=batch,
                obj=dA,
                name=dA_name
            )
            
            # Update parameters for current layer to the mini-bacth ElastiCache
            parameter_key = update_parameters_with_gd(
                W, b, dW, db,
                learning_rate,
                batch, layer,
                parameters
            )
            
            # Move onto the the next hidden layer for multiple layer networks
            propogate(direction='backward', batch=batch, layer=layer, parameter_key=parameter_key)
            
    elif state == 'start':
        # Initialize the start of a mini-batch execution
        layer = 0
        start_batch(batch=batch, layer=layer, parameter_key=event.get('parameter_key'))
    else:
        sns_message = "General error processing TrainerLambda handler!"
        publish_sns(sns_message)
        raise