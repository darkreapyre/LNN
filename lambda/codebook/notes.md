# General Notes
1. Add the SNS Topic Arn to the Environmental variables to that the 1Finish()` function can use it.
2. Add the necessary SNS permissions to the CloudFormation Template.
3. Make sure to set Lambda loggin for a maximum of 7 days so as to not have lagacy Lambda Logs from previouys runs.
4. Make sure to add the ARNs of the various Lambda Functions to the environmental variables in the CloudFormation Template.
5. Ensure to always include the `state` in the `event` every time the `TrainerLambda` is invoked.

---

# Notes on DynamoDB
1. DynamoDB supports the following data types:
    - **Scalar**: Number, String, Binary, Boolean and Null.
    - **Multi-value**: String Set, Number Set and Binary Set. Multi-valued types are sets, which means that the values in this data type are unique. 
    - **Document**: List and Map.
2. Key Concepts:
    - **Partition Key**: This is a unique identifier for an entry.
    - **Sort Key**: Each row can be thought of as a hashmap ordered by keys as ech row can be broken up and have additional structure.
    - **Provisioned Throughput**: The amount of readds and writes expected to occur.
3. In the case of LNN and DynamoDB, the `PartitionKey` is **epoch** and should be of `KeyType` **HASH**. 
    > **Note**: there is no need for a **Sort Key** in the case of the LNN, since we are simply updating the table of results to be sent via SNS, but this may change.
    
    The schema should look as follows:

    ```python
    import boto3
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.create_table(
        TableName = 'results',
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
        ProvishionedThroughput = {
            'ReadCapacityUnits': 5,
            'WriteCapacityUnits': 5
        }
    )
    # Wait until the table exists
    table.meta.client.get_waiter('table_exists').wait(TableName='results')

    # Print out some data about the table
    print(table.name) # Expected Value 'results'
    ```
4. Creating a new item (part of the `Start()` function) in the table, in this case a new epoch:
    ```python
    # Import additional libraries needed by DynamoDB
    dynamodb = resource('dynamodb', region_name='us-west-2')
    from decimal import Decimal, Inexact, Rounded
    from boto3.dynamodb.types import DYNAMODB_CONTEXT
    DYNAMODB_CONTEXT.traps[Inexact] = 0
    DYNAMODB_CONTEXT.traps[Rounded] = 0

    # Simulated event parameters
    epoch = 1 # Can't be of type float
    cost = 0.
    learning_rate = 0.

    # Create a new item in DynamoDB
    table.put_item(
        Item = {
            'epoch': epoch,
            'cost': Decimal(cost),
            'learning_rate': Decimal(learning_rate)
        }
    )
    ```
5. To track the state across the various invocations of the `TrainerLambda` it would be preferred to also capture the Weights as well as Bias across epochs. The initial plan was to dump the data to S3 along with the input data, but it would be good to have visibility on what the actual weights and bias are across epochs. However, this causes an issue because DynamnoDB does not support numpy arrays and it does not support float data types. Thus, the data needs to be searialzed  to a DynamoDB List and converted from float to Decimal. Should this course be pursed, the following is sample code to ingest a numpy array into DynamoDB and de-serialize it back out:
    ```python
    # Retrieve the Item for epoch 2
    from boto3.dynamodb.conditions import Key, Attr
    from botocore.exceptions import ClientError

    # Helper class to convert a DynamoDB item to JSON.
    class DecimalEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, Decimal):
                if o % 1 > 0:
                    return float(o)
                else:
                    return int(o)
            return super(DecimalEncoder, self).default(o)

    # Simulate dimensions of input data dimensions
    dims = (12288, 209)
    # Initialize weights to zero
    w = np.zeros((dim[0], 1))

    # Function to serialize numpy arrays to DynamoDB List
    def ndarray2item(raw):
        """
        Converts a numpy array to a list formatting for DynamoDB
        and ensures that `float` values are converted to `Decimal`
        """
        if type(raw) is np.ndarray:
            resp = []
            for v in np.nditer(raw):
                resp.append(float(v))
            return ndarray2item(resp)
        elif type(raw) is list:
            resp = []
            for v in  raw:
                resp.append(Decimal(v))
            return resp
    
    # Test inserting the float data to dunamoDB
    epoch = 1
    cost = 6.00006477319
    learning_rate = 0.009
    
    # Create a new item in DynamoDB
    table.put_item(
        Item={
            'epoch': epoch,
            'cost': Decimal(cost),
            'learning_rate': Decimal(learning_rate),
            'weights': ndarray2item(w),
            'bias': b
        }
    )

    # Retrieve the Item for epoch 1
    try:
        response = table.get_item(
            Key={
                'epoch': epoch
            }
        )
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        item = response['Item']
        print("GetItem succeeded:")
        print(json.dumps(item, indent=4, cls=DecimalEncoder))
    
    # Function to serialize DynamoDB List to Numpy array
    def item2ndarray(raw):
        """
        Converts a DynamoDB Item to numpy array and ensures
        that `float` values are converted from `Decimal`
        """
        if type(raw) is list:
            resp = []
            for v in raw:
                resp.append(float(v))
            return np.array(resp)
    
    # Generate new weights from DynamoDB result
    w = item2ndarray(item['weights])

    # Reshape to match input data
    w = w.reshape((w.shape[0], 1))
    ```

---

# Notes on TrainerLambda.py

1. The TrainerLambda has three stages, `start`, `forward` and `backward`. The algorithm with the `lambda_handler()` function, to implement this is as follows:
    ```python
    # Get event variables first

    if state == 'start':
        Start(next_epoch)
        # Initialize Weights

        # Initialize Bias

        # Instialize new item in DynamoDB for the epoch

        # Call Propogate to start Forwardprop
        Propogate(direction='forward')

    elif state == 'forward':
        # Determine location in Forwardprop
        if layer > layers:
            # Location is at the end of forwardprop
            # Calculate Loss Function
            
            # Upadate epoch these items in DynamoDB
            
            # Start Backprop
            Propogate(direction='backward')
        else:
            # move onto the next hidden layer
            Propogate(direction='forward')

    elif state == 'backward':
        # Determine location within backprop
        if epoch == epochs and layer == 0:
            # Location is at the finish
            # Calculate Deriviative

            # Calculate final Weights

            # Update epoch these items in DynamoDB

            # Finalize training run
            Finish()

        elif epoc < epochs and layer == 0:
            # Location is at the end of the epoch
            # Calculate Derivative

            # Calculate weights for epoch

            # Start new epoch
            Start(next_epoch)

        else:
            # Move to the next hidden layer
            Propogate(direction='backward')
    ```
2. When running the *Trainer* Lambda Function, the function will need to know the current layer as well the number of hidden units (neurons) to use, plus the activation function for the current layer (if it's not layer 0). The following code show the code to use:
    ```python
    # Make "fake" layer 2 data for dimulation
    # Activations
    payload['activations']['layer' + str(current_layer)] = 'sigmoid'
    # Hidden units
    payload['neurons']['layer' + str(current_layer)] = 4

    # Process "alyer 2"
    current_layer = int(payload.get('layer')) + 1
    current_activation = payload.get('activations')['layer' + str(current_layer)] # Get activations
    num_hidden_units = payload.get('neurons')['layer' + str(current_layer)] # Get hidden units


---

# Notes on Building the Lambda Deployment Package

## Overview
This document outlinnes the proceedure to create the Lambda deployment packagew ith the intent to leverage the proceedures for other LNN Versions.

## Adding the necessary Python Libraries
```text
pip install virtualenv
virtualenv -p /var/lang/bin/python \
--always-copy \
--no-site-packages \
lambda_build

source lambda_build/bin/activate

pip install --upgrade pip wheel
pip install numpy
pip install scipy
pip install -U scikit-learn
pip install multiprocess

libdir="$VIRTUAL_ENV/lib/python3.6/site-packages/lib/"
mkdir -p $VIRTUAL_ENV/lib/python3.6/site-packages/lib || true
echo "venv original size $(du -sh $VIRTUAL_ENV | cut -f1)"
find $VIRTUAL_ENV/lib/python3.6/site-packages/ -name "tests" | xargs rm -r
find $VIRTUAL_ENV/lib/python3.6/site-packages/ -name "dataset" | xargs rm -r

Can't remove tests files from pandas

pip install pandas

find $VIRTUAL_ENV/lib/python3.6/site-packages/ -name ".pyc" -delete

find $VIRTUAL_ENV/lib/python3.6/site-packages/ -type d -empty -delete

find $VIRTUAL_ENV/lib/python3.6/site-packages/ -name ".so" | xargs strip

pushd $VIRTUAL_ENV/lib/python3.6/site-packages/

Only select pkgs that needed for your project
zip -r -9 -q /lambda.zip scipy numpy sklearn pandas pytz multiprocess dill

popd
```

## Makefile

## CloudFormation Template