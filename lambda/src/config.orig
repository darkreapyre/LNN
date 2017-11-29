#!/usr/bin/python
"""
Lambda Function that applies a Trigger Event to the S3 Bucket. this funciton is used to the 
LNN CloudFormation depployments.
"""

# Import necessary Libraries
#import os
#from os import environ
import json
from json import dumps, loads
from boto3 import client, resource, Session
import botocore
import uuid
import cfnresponse

def lambda_handler(event, context):
    properties = event['ResourceProperties']
    bucket_name = properties['Bucket']
    rgn = properties['Region']
    s3_client = client('s3', region_name=rgn)
    lambda_client = client('lambda', region_name=rgn)

    # Define notification configuration
    configuration = {}
    if event['RequestType'] != 'Delete':
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
        lambda_response = lambda_client.add_permission(
            FunctionName=properties['FunctionArn'],
            StatementId = str(uuid.uuid4()),
            Action='lambda:InvokeFunction',
            Principal='s3.amazonaws.com',
            SourceArn='arn:aws:s3:::' + bucket_name,
            SourceAccount=properties['AccountNumber'],
        )
        print(lambda_response)
    
    # Create Notification
    try:
        s3_response = s3_client.put_bucket_notification_configuration(
            Bucket=bucket_name,
            NotificationConfiguration=configuration
        )
        print(s3_response)
        responseData = {'Data': 'OK'}
        return cfnresponse.send(event, context, cfnresponse.SUCCESS, responseData)
    except Exception as e:
        print(str(e))
        return cfnresponse.send(event, context, cfnresponse.FAILED, responseData)