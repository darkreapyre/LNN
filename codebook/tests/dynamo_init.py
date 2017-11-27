# Global Variables
dynamo_client = client('dynamodb', region_name=rgn)
dynamo_resource = boto3.resource('dynamodb', region_name=rgn) # Ensure to add `from boto3 import resource`

# Initialize DynamoDB Tables
table_list = ['TrainerLambda', 'NeuronLambda']
for t in table_list:
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
            {
                'AttributeName': 'Count',
                'AttributeType': 'N'
            },
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 10,
            'WriteCapacityUnits': 10
        }
    )
    table.meta.client.get_waiter('table_exists').wait(TableName=t)