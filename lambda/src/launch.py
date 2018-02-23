#!/usr/bin/python
"""
Lambda Fucntion that launches Neural Network training from
an S3 training data upload and managed training Epochs.
"""

# Import Libraries and Global variablesneeded by the Lambda Function
from Utils import *

# Lambda Handler
def lambda_handler(event, context):
    # Determine if this is the initial launch of the funciton
    if not event.get('state') == 'next':
        # This is the first invocation, therefore set up the initial parameters
        # Retrieve datasets and parmeters from S3
        input_bucket = s3_resource.Bucket(str(event['Records'][0]['s3']['bucket']['name']))
        dataset_key = str(event['Records'][0]['s3']['object']['key'])
        settings_key = dataset_key.split('/')[-2] + '/parameters.json'
        try:
            input_bucket.download_file(dataset_key, '/tmp/datasets.h5')
            input_bucket.download_file(settings_key, '/tmp/parameters.json')
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                sns_message = "Error downloading input data from S3, S3 object does not exist"
                #publish_sns(sns_message)
                print(sns_message)
            else:
                raise
        
        # Extract the neural network paramaters
        with open('/tmp/parameters.json') as parameters_file:
            parameters = json.load(parameters_file)
        
        # Start building the Master parameters
        # Get the ARNs for the Lambda Functions
        parameters['ARNs'] = {
            'LaunchLambda': get_arns('LaunchLambda'),
            'TrainerLambda': get_arns('TrainerLambda'),
            'NeuronLambda': get_arns('NeuronLambda')
        }
        # Input data set and parameter bucket
        parameters['s3_bucket'] = event['Records'][0]['s3']['bucket']['name']
        # Initial epoch for tracking
        parameters['epoch'] = 0
        
        # Initialize and pre-process the training set
        X, Y = initialize_data(parameters)
        
        # Upload the pre-processed data sets to the Master ElastiCache DB (db=15)
        # and update the parameters with the keys
        parameters['data_keys'] = {}
        parameters['data_keys']['X'] = to_cache(db=15, obj=X, name='X')
        parameters['data_keys']['Y'] = to_cache(db=15, obj=Y, name='Y')
        
        # Initialize the Weights and Bias using Xavier Initialization
        # for the ReLU neurons
        for l in range(1, parameters['layers']+1):
            if l == 1:
                """
                Note: This assumes Layer 1 uses the ReLU Activation.
                """
                # Standard Weight initialization for ReLU
                W = np.random.randn(
                    parameters['neurons']['layer'+str(l)],
                    X.shape[0]
                ) * np.sqrt((2.0 / X.shape[0]))
            else:
                if parameters['activations']['layer'+str(l)] == 'sigmoid':
                    # Standard Weight initialization
                    W = np.random.randn(
                        parameters['neurons']['layer'+str(l)],
                        parameters['neurons']['layer'+str(l-1)]
                    ) / np.sqrt(parameters['neurons']['layer'+str(l-1)])
                else:
                    # Xavier Weight initialization for ReLu
                    W = np.random.randn(
                        parameters['neurons']['layer'+str(l)],
                        parameters['neurons']['layer'+str(l-1)]
                    ) * np.sqrt((2.0 / parameters['neurons']['layer'+str(l-1)]))
            # Standard Bias initialization
            b = np.zeros((parameters['neurons']['layer'+str(l)], 1))
            # Upload the Weights and Bias to ElastiCache Master DB
            parameters['data_keys']['W'+str(l)] = to_cache(db=15, obj=W, name='W'+str(l))
            parameters['data_keys']['b'+str(l)] = to_cache(db=15, obj=b, name='b'+str(l))
        
        # Initialize mini-batches
        batch_size = parameters['batch_size']
        batches = random_minibatches(X, Y, batch_size)
        parameters['num_batches'] = len(batches)
        
        # Update Master parameters to ElastiCache
        parameter_key = to_cache(db=15, obj=parameters, name='paremeters')
        
        # Initialize DynamoDB tables for treacking Lambda invocations
        table_list = ['LaunchLambda','TrainerLambda', 'NeuronLambda']
        for t in table_list:
            # Check to see if the table already exists
            list_response = dynamo_client.list_tables()
            if t in list_response['TableNames']:
                # Delete the existing table
                dynamo_client.delete_table(TableName=t)
                waiter = dynamo_client.get_waiter('table_not_exists')
                waiter.wait(TableName=t)
            
            # Create a "fresh" table
            table = dynamo_resource.create_table(
                TableName=t,
                KeySchema=[
                    {
                        'AttributeName': 'invID',
                        'KeyType': 'HASH'
                    },
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'invID',
                        'AttributeType': 'S'
                    },
                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 20,
                    'WriteCapacityUnits': 50
                }
            )
            table.meta.client.get_waiter('table_exists').wait(TableName=t)
        
        # Invoke the training for each batch by invoking the `TrainerLambda`
        current_batch = -1
        for batch in batches:
            # Create parameters that are specific to the batch, `batch_parameters`
            current_batch += 1
            (batch_X, batch_Y) = batch
            m = batch_X.shape[1]
            batch_parameters = parameters
            batch_parameters['batch_ID'] = current_batch
            batch_parameters['data_keys']['A0'] = to_cache(
                db=current_batch,
                obj=batch_X,
                name='A0'
            )
            batch_parameters['data_keys']['Y'] = to_cache(
                db=current_batch,
                obj=batch_Y,
                name='Y'
            )
            batch_parameters['data_keys']['m'] = to_cache(
                db=current_batch,
                obj=m,
                name='m'
            )
            # Debug Statements
            #print("\n"+"\n"+"Batch {} Parameters: ".format(current_batch))
            #print(dumps(batch_parameters, indent=4, sort_keys=True))
            
        # Initialize the payload for initial batch to `TrainerLambda`
        payload = {}
        payload['state'] = 'start' # Initialize overall state
        payload['batch_ID'] = 0 # Append the batch ID
        """
        Note: Need to think about this
        payload['parameter_key'] = to_cache(
            db=0,
            obj=batch_parameters,
            name='parameters'
        )
        """
            
        # Create the invocation ID to ensure no duplicate functions
        # are launched.
        invID = str(uuid.uuid4()).split('-')[0]
        name = 'TrainerLambda'
        task = 'set'
        inv_counter(name, invID, task)
        payload['invID'] = invID
            
        # Prepare the initial batch payload for `TrainerLambda`
        payloadbytes = dumps(payload)
            
        # Debug Statements
        #print("Complete Neural Network Settings for the initial batch:\n")
        #print(dumps(batch_parameters, indent=4, sort_keys=True))
        #print("\n"+"Payload to be sent to TrainerLambda: \n")
        #print(dumps(payload))
            
        # Invoke TrainerLambda to start the training process for
        # the current batch.
        try:
            response = lambda_client.invoke(
                FunctionName=batch_parameters['ARNs']['TrainerLambda'],
                InvocationType='Event',
                Payload=payloadbytes
            )
        except botocore.exceptions.ClientError as e:
            sns_message = "Errors occurred invoking TrainerLambda from LaunchLambda."
            sns_message += "\nError:\n" + str(e)
            sns_message += "\nCurrent Payload:\n" +  dumps(payload, indent=4, sort_keys=True)
            publish_sns(sns_message)
            print(e)
            raise
        print(response)
            
    """
    To Do: Complete next stage if this is not the first time the function is called
    """