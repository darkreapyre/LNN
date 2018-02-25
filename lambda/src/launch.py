#!/usr/bin/python
"""
Lambda Fucntion that launches Neural Network training from
an S3 training data upload and managed training Epochs.
"""

# Import Libraries and Global variables needed by the Lambda Function
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
        parameters['datasets_key'] = event['Records'][0]['s3']['object']['key']
        # Initial epoch for tracking
        parameters['epoch'] = 0
        # Initialize hash key tracking object
        parameters['data_keys'] = {}
        
        # Initialize and pre-process the training set
        X, Y = initialize_data(parameters)
        
        # Initialize the Weights and Bias using Xavier Initialization
        # for the ReLU neurons
        for l in range(1, parameters['layers']+1):
            if l == 1:
                """
                Note: This assumes Layer 1 uses the ReLU Activation.
                """
                # Standard Wieght initialization for ReLU
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
        
        # Initialize the optimizer
        if parameters['optimizer'] == "gd":
            pass # No initialization required for gradient descent
        elif parameters['optimizer'] == "adam":
            pass # Future use
        
        # Upload the pre-processed data sets to the Master ElastiCache DB (db=15)
        # and update the parameters with the keys
        parameters['data_keys']['X'] = to_cache(db=15, obj=X, name='X')
        parameters['data_keys']['Y'] = to_cache(db=15, obj=Y, name='Y')
        
        # Initialize mini-batches
        batch_size = parameters['batch_size']
        batches = random_minibatches(X, Y, batch_size)
        parameters['num_batches'] = len(batches)
        
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
                    'WriteCapacityUnits': 20
                }
            )
            table.meta.client.get_waiter('table_exists').wait(TableName=t)
        
        # Initialize the Results tracking object
        results = {}
        results['Start'] = str(datetime.datetime.now())
        # Initialize tracking for first epoch
        results['epoch0'] = {}
        parameters['data_keys']['results'] = to_cache(
            db=15, obj=results, name='results'
        )
        
        # Finalize master parameters to ElastiCache
        master_parameter_key = to_cache(
            db=15, obj=parameters, name='parameters'
        )
                
        # Configure batch specific parameters
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
            
            # Upload Batch parameters to respecitve ElastiCache database
            batch_parameter_key = to_cache(
                db=current_batch,
                obj=batch_parameters,
                name='parameters'
            )
            
            # Confirm Batch key matches Master key
            assert(str(batch_parameter_key) == str(master_parameter_key))
        
        # Initialize the payload for initial batch to `TrainerLambda`
        payload = {}
        payload['state'] = 'start' # Initialize overall state
        payload['parameter_key'] = master_parameter_key # Master parameter key
            
        # Create the invocation ID to ensure no duplicate functions
        # are launched
        invID = str(uuid.uuid4()).split('-')[0]
        name = 'TrainerLambda'
        task = 'set'
        inv_counter(name, invID, task)
        payload['invID'] = invID
        
        # Prepare the payload for `TrainerLambda`
        payloadbytes = dumps(payload)
        
        # Debug Statements
        #print("Complete Neural Network Settings for batch: {}\n".format(current_batch))
        #print(dumps(batch_parameters, indent=4, sort_keys=True))
        #print("\n"+"Payload to be sent to TrainerLambda: \n")
        #print(dumps(payload))
            
        # Invoke TrainerLambda to start the training process for
        # the current batch
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
        
    else:
        # Ensure that this is not a duplicate invokation
        invID = event.get('invID')
        name = "LaunchLambda" #Name of the current Lambda function
        task = 'get'
        cnt = inv_counter(name, invID, task) #should be 0 for a new function invoked
        if cnt == 0:
            task = 'update'
            inv_counter(name, invID, task)
        else:
            sys.exit()

        # Determine current Epoch status
        parameters = from_cache(db=15, key=event.get('parameter_key'))
        epoch = parameters['epoch']
        
        if epoch == parameters['epochs'] - 1:
            # This is the final epoch, therefore close out training
            # and clean up.
            # Get the results key
            final_results = from_cache(db=15, key=parameters['data_keys']['results'])

            # Add the end time to the results
            final_results['End'] = str(datetime.datetime.now())

            # Upload the final results to S3
            bucket = parameters['s3_bucket']
            results_obj = s3_resource.Object(bucket,'training_results/results.json')
            try:
                results_obj.put(Body=json.dumps(final_results))
            except botocore.exceptions.ClientError as e:
                print(e)
                raise

            # Create dictionary of model parameters for prediction app
            params = {}
            for l in range(1, parameters['layers']+1):
                params['W'+str(l)] = from_cache(db=15, key=parameters['data_keys']['W'+str(l)])
                params['b'+str(l)] = from_cache(db=15, key=parameters['data_keys']['b'+str(l)])
            # Create a model parameters file for use by prediction app
            with h5py.File('/tmp/params.h5', 'w') as h5file:
                for key in params:
                    h5file['/' + key] = params[key]
            # Upload model parameters file to S3
            s3_resource.Object(bucket, 'predict_input/params.h5').put(Body=open('/tmp/params.h5', 'rb'))

            # Get the last results entry to publish to SNS
            final_cost = final_results['epoch' + str(parameters['epochs']-1)]
            sns_message = "Training Completed Successfully!\n" + dumps(final_cost)
            publish_sns(sns_message)
            
            print("Training Completed successfully!\n" + dumps(final_cost))
        
        else:
            # Close off current Epoch and move onto the next
            # Send status updates for epochs every 100 epochs
            current_results = from_cache(db=15, key=parameters['data_keys']['results'])
            current_cost = current_results['epoch'+str(epoch)]['cost']
            if epoch % 100 == 0:
                sns_message = "Training update!\n Cost after epoch {0} = {1}".format(epoch, float(current_cost))
                publish_sns(sns_message)
            
            # Debug Statements
            print("Cost after epoch {} = {}".format(epoch, float(current_cost)))
            
            # Retrieve "fresh" datasets and parmeters from S3
            input_bucket_name = parameters['s3_bucket']
            #datasets_key = parameters['datasets_key']
            datasets_key = 'training_input/datasets.h5'

            try:
                s3_resource.Bucket(input_bucket_name).download_file(datasets_key, '/tmp/datasets.h5')
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    sns_message = "Error downloading input data from S3, S3 object does not exist"
                    publish_sns(sns_message)
                    print(sns_message)
                else:
                    raise
            
            # Initialize new mini-batches
            batch_size = parameters['batch_size']
            #X = from_cache(db=15, key=parameters['data_keys']['X'])
            #Y = from_cache(db=15, key=parameters['data_keys']['Y'])
            X, Y = initialize_data(parameters)
            batches = random_minibatches(X, Y, batch_size)
            parameters['num_batches'] = len(batches)
            
            # Initialize tracking for next epoch
            epoch += 1
            parameters['epoch'] = epoch
            current_results['epoch'+str(epoch)] = {}
            parameters['data_keys']['results'] = to_cache(
                db=15, obj=current_results, name='results'
            )
            
            # Finalize master parameters to ElastiCache
            master_parameter_key = to_cache(
                db=15, obj=parameters, name='parameters'
            )
            
            # Configure batch specific parameters
            current_batch = -1
            for batch in batches:
                # Create parameters that are specific to the batch, `batch_parameters`
                current_batch += 1
                # Create "fresh" ElastiCache mini-batch databases
                # Debug statements
                print("Flushing Bacth {} Database.".format(current_batch))
                batch_cache = redis(host=endpoint, port=6379, db=int(current_batch))
                batch_cache.flushdb()
                # Update new batch database
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

                # Upload Batch parameters to respecitve ElastiCache database
                batch_parameter_key = to_cache(
                    db=current_batch,
                    obj=batch_parameters,
                    name='parameters'
                )

                # Confirm Batch key matches Master key
                assert(str(batch_parameter_key) == str(master_parameter_key))

            # Initialize the payload for initial batch to `TrainerLambda`
            payload = {}
            payload['state'] = 'start' # Initialize overall state
            payload['parameter_key'] = master_parameter_key # Master parameter key

            # Create the invocation ID to ensure no duplicate functions
            # are launched
            invID = str(uuid.uuid4()).split('-')[0]
            name = 'TrainerLambda'
            task = 'set'
            inv_counter(name, invID, task)
            payload['invID'] = invID

            # Prepare the payload for `TrainerLambda`
            payloadbytes = dumps(payload)

            # Debug Statements
            #print("Complete Neural Network Settings for batch: {}\n".format(current_batch))
            #print(dumps(batch_parameters, indent=4, sort_keys=True))
            #print("\n"+"Payload to be sent to TrainerLambda: \n")
            #print(dumps(payload))

            # Invoke TrainerLambda to start the training process for
            # the current batch
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
