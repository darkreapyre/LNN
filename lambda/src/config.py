#!/usr/bin/python
"""
Lambda Function that applies a Trigger Event to the S3 Bucket. this funciton is used to the 
LNN CloudFormation depployments.
"""

# Import necessary Libraries
import os
from os import environ
import json
from json import sumps, loads
from boto3 import client, resource, Session
import botocore
import uuid

def lambda_handler(event, context):
    rgn = environ['Region']
    s3_cleint = client('s3', region_name=rgn)
    lambda_client = client('lambda', region_name=rgn)

    # Define notification confugration
    configuration = {}
    configuration['LambdaFunctionConfigurations'] = [
        {
            'LambdaFunctionArn': environ['FunctionArn'],
            'Events': [
                's3:ObjectCreated:*'
            ],
            'Filter': {
                'Key': {
                    'FilterRules': [
                        {
                            'Name': 'suffix',
                            'Value': 'h5'
                        }
                    ]
                }
            }
        }
    ]

    # Create Permission to trigger the LaunchLambda
    lambda_response = lambda_client.add_permission(
        FunctionName=environ['FunctionArn'],
        StatementId = str(uuid.uuid4()),
        Action='Lambda:InvokeFunction',
        Principal='s3.amazonaws.com',
        SourceArn='arn:aws:s3:::' + environ['Bucket'],
        SourceAccount=environ['Account'],
    )
    print(lambda_response)
    
    # Create Notification
    s3_response = s3_client.put_bucket_notification_configuration(
        Bucket=environ['Bucket'],
        NotificationConfiguration=configuration
    print(s3_response)