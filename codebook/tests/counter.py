# Libraries
import uuid
from boto3 import resource

# Global Variables
dynamo_client = client('dynamodb', region_name=rgn)
dynamo_resource = resource('dynamodb', region_name=rgn)

# 
def inv_counter(name, invID, task):
    """
    Manages the Counter assigned to a unique Lambda Invocation ID, by
    either setting it to 0, updating it to 1 or querying the value.
   
    Arguments:
    name -- The Name of the function being invoked
    invID -- The unique invocation ID created for the specific invokation
    task -- Task to perfoirm: set | get | update
    """
    table = dynamo_resouce.Table(name)
    if task == 'set':
        table.put_item(
            Item={
                'invID': invID,
                'Count': 0
            }
        )
        
    elif task == 'get':
        task_response = table.get_item(
            Key={
                'invID': invID
            }
        )
        
        item = task_response['Item']
        
        return int(item['Count'])
        
    elif task == 'update':
        task_response = table.update_item(
            Key={
                'invID': invID
            },
            UpdateExpression='SET Count = :val1',
            ExpressionAttributeValues={
                ':val1': 1
            }
        )


# Sample Command to run just befor invoking a new Lambda
invID = str(uuid.uuid4()).split('-')[0]
name = str(TrainerLambda) #Name of the Lambda fucntion to be invoked
task = 'set'
inv_counter(name, invID, task)
payload['invID'] = InvID

# Sammple Command to run at the beginning of every Lambda Handler
invID = event.get('InvID')
name = str(TrainerLambda) #Name of the current Lambda function that was invoked
task = 'get'
cnt = inv_counter(name, invID, task) #should be 0 for a new function invoked
if cnt == 0:
    task = 'update'
    inv_counter(name, invID, task)
else:
    break