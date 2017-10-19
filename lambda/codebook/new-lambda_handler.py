# Execute appropriate action based on the the current state
# Get the current state
current_state = event.get('state')

# Determine next steps based on the state
if current_state == 'forward':
    # Get important state variables
    
    # Determine the location within forwardprop
    if layer > layers:
        # Location is at the end of forwardprop
        # Caculate the Loss function
        
        # Update the Loss function to Dynamo
        
        # Start backprop
        #propogate(direction='backward', layer=layer-1)
        
        pass
    else:
        # Continue packprop and move to the next hidden layer
        #propogate(direction='forward', layer=layer+1, activations=activations)
        
        pass
elif current_state == 'backward':
    # Get important state variables
    
    # Determine the location within backprop
    if epoch == epochs and layer == 0:
        # Location is at the end of the final epoch
        
        # Caculate derivative?????????????????????????
        
        # Caclulate the absolute final weight
        
        # Update the final weights and results (cost) to DynamoDB
        
        # Finalize the the process and clean up
        #finish()
        
        pass
    
    elif epoch < epochs and layer == 0:
        # Location is at the end of the current epoch and backprop is finished
        
        # Calculate the derivative?????????????????????????
        
        # Calculate the weights for this epoch
        
        # Update the weights and results (cost) to DynamoDB
        
        # Start the next epoch
        #epoch = epoch + 1
        #start(epoch)
        
        pass
        
    else:
        # Move to the next hidden layer
        #propogate(direction='backward', layer=layer-1)
        
        pass
        
elif current_state == 'start':
    # Start of a new run of the entire process
    # Get important event variables from event triggerd from `LaunchLambda`
    s3_bucket = event['s3_bucket']
    state_table = event['state_table']
    learning_rate = event['learning_rate']
    epochs = event['epochs']
    epoch = event['epoch']
    layers = event['layers']
    layer = event['layer']
    activations = event['activations']
    #neurons = event.get('neurons')['layer' + str(layer)]
    #current_activation = event.get('activations')['layer' + str(current_layer)]
    params = event['params']
       
    # Create a epoch 1 in DynamoDB with ALL initial parameters
    table = dynamodb.Table(state_table)
    table.put_item(
        Item = {
            'epoch': epoch,
            'epochs': epochs,
            'layer': layer+1,
            'learning_rate': Decimal(learning_rate),
            'activations': activations,
            'state_table': state_table,
            's3_bucket': s3_bucket,
            'params': params
        }
    )
    
    # Start forwardprop
    #propogate(direction='forward')

else:
    print("No state informaiton has been provided.")
    raise