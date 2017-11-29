#!bin/bash/python
'''
Lambda Function that applies a Trigger Event to the S3 Bucket. this funciton is used to the 
LNN CloudFormation depployments.
'''

from uuid import uuid4
from json import dumps
import boto3
import requests
import botocore

def lambda_handler(event, context):
    '''
    main handler
    '''
    print "Event JSON: \n" + dumps(event) # Dump Event Data for troubleshooting
    response_status = 'SUCCESS'
    properties = event['ResourceProperties']
    bucket_name = properties['Bucket']
    rgn = properties['Region']
    s3_client = client('s3', region_name=rgn)
    lambda_client = client('lambda', region_name=rgn)
    configuration = {}
    
    # If the CloudFormation Stack is being deleted, delete the limits and roles created
    if event['RequestType'] == 'Delete':
        try:
            s3_response = s3_client.put_bucket_notification_configuration(
                Bucket=bucket_name,
                NotificationConfiguration=configuration
            )
            print(s3_response)
        except Exception as e:
            print(str(e))
        
        send_respinse(event, context, response_status)

    # If the CloudFormation Stack is being updated, do nothing, exit with success
    if event['RequestType'] == 'Update':
        send_response(event, context, response_status)

    # If the Cloudformation Stack is being created, create the trigger
    if event['RequestType'] == 'Create':
        configuration['LambdaFunctionConfigurations'] = [
            {
                'LambdaFunctionArn': properties['FunctionArn'],
                'Events': [
                    's3:ObjectCreated:Put'
                ],
                'Filter': {
                    'Key': {
                        'FilterRules': [
                            {
                                'Name': 'suffix',
                                'Value': 'h5'
                            },
                            {
                                'Name': 'prefix',
                                'Value': 'training_input/'
                            }
                        ]
                    }
                }
            }
        ]
        
        # Create Permission to trigger the LaunchLambda
        try:
            lambda_response = lambda_client.add_permission(
                FunctionName=properties['FunctionArn'],
                StatementId = str(uuid.uuid4()),
                Action='lambda:InvokeFunction',
                Principal='s3.amazonaws.com',
                SourceArn='arn:aws:s3:::' + bucket_name,
                SourceAccount=properties['AccountNumber'],
            )
            #print(lambda_response)
        except Exception as e:
            print(str(e))
        try:
            s3_response = s3_client.put_bucket_notification_configuration(
                Bucket=bucket_name,
                NotificationConfiguration=configuration
            )
            #print(s3_response)
        except Exception as e:
            print(str(e))
        
        #Send response to CloudFormation to signal success so stack continues.  If there is an error, direct user at CloudWatch Logs to investigate responses
        send_response(event, context, response_status)

def send_response(event, context, response_status):
    '''
    sends UUID
    '''
    #Generate UUID for deployment
    try:
        UUID = uuid4()
    except:
        UUID = 'Failed'

    #Build Response Body
    responseBody = {'Status': response_status,
                    'Reason': 'See the details in CloudWatch Log Stream: ' + context.log_stream_name,
                    'PhysicalResourceId': context.log_stream_name,
                    'StackId': event['StackId'],
                    'RequestId': event['RequestId'],
                    'LogicalResourceId': event['LogicalResourceId'],
                    'Data': {'UUID': str(UUID)}}
    print('RESPONSE BODY:\n' + dumps(responseBody))

    try:
        #Put response to pre-signed URL
        req = requests.put(event['ResponseURL'], data=dumps(responseBody))
        if req.status_code != 200:
            print(req.text)
            raise Exception('Recieved non 200 response while sending response to CFN.')
        return str(event)
    except requests.exceptions.RequestException as e:
        print(req.text)
        print(e)
        raise

if __name__ == '__main__':
    lambda_handler('event', 'handler')