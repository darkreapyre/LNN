# Import Libraries
from __future__ import print_function
import os
import sys
import time
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.mxnet import MXNet

# Specify the Traiing Job to build against
# Default: training_job = 0
training_job = 0
if training_job == 0:
    print("No Training job defined, exiting ...")
    sys.exit()

# Global Variables
sagemaker_client = boto3.client('sagemaker')
iam_client = boto3.client('iam')
build_id = str(os.environ['CODEBUILD_RESOLVED_SOURCE_VERSION'])
model_name = build_id[:7]

# Create IAM Role for SageMaker Session
role_response = iam_client.create_role(
    RoleName='sagemaker-'+model_name+'-Role',
    AssumeRolePolicyDocument='{ "Version": "2012-10-17", "Statement": [ { "Effect": "Allow", "Principal": { "Service": "sagemaker.amazonaws.com" }, "Action": "sts:AssumeRole" } ] }'
)
time.sleep(5)
print("Created IAM Role for SageMaker Session.")

# Attach Managed Role Policy
iam_client.attach_role_policy(
    PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
    RoleName=role_response['Role']['RoleName']
)
print("Attached Managed Policy to SageMaker Session.")
time.sleep(5)

# Create a model using the Session API
# by attaching to the training job
training_job_info = sagemaker_client.describe_training_job(TrainingJobName=training_job)
training_job_name = str(training_job_info['HyperParameters']['sagemaker_job_name'].split('"')[1])
sagemaker_role = role_response['Role']['Arn']
print("Attaching estimator to training job: {}".format(training_job_name))
estimator = MXNet.attach(training_job_name)
session = sagemaker.Session()
time.sleep(5)
model = estimator.create_model()
container_def = model.prepare_container_def(instance_type='ml.m4.xlarge')
session.create_model(model_name, sagemaker_role, container_def)

# Create endpoint config using the Session API
endpoint_config_name = session.create_endpoint_config(
    name=model_name,
    model_name=model_name,
    initial_instance_count=1,
    instance_type='ml.m4.xlarge'
)

# Create endpoint using the boto3 API
print("Creating Endpoint ...")
create_endpoint_response = sagemaker_client.create_endpoint(
    EndpointName=model_name,
    EndpointConfigName=endpoint_config_name
)

# Wait until the status has changed
sagemaker_client.get_waiter('endpoint_in_service').wait(EndpointName=model_name)

# Print the final status of the endpoint
endpoint_response = sagemaker_client.describe_endpoint(EndpointName=model_name)
status = endpoint_response['EndpointStatus']
print('Endpoint status: {}'.format(status))

if status != 'InService':
    raise Exception('Endpoint creation failed.')