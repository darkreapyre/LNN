#!/usr/bin/python

import boto3
from cfnresponse import send, SUCCESS, FAILED

AVAILABLE_CONFIGURATIONS = (
    'LambdaFunctionConfigurations'
)

def register_permission(self, template):
        template.add_resource(
            awslambda.Permission(
                utils.valid_cloudformation_name(
                    self.bucket_notification_configuration.name,
                    self.id,
                    'permission'
                ),
                Action="lambda:InvokeFunction",
                FunctionName=self.get_destination_arn(),
                Principal="s3.amazonaws.com",
                SourceAccount=troposphere.Ref(troposphere.AWS_ACCOUNT_ID),
                SourceArn=self.bucket_notification_configuration.get_bucket_arn()
            )
        )
def get_destination_arn(self):
    return troposphere.Ref(
        self.bucket_notification_configuration.project.reference(
        utils.lambda_friendly_name_to_grn(
            self.settings['lambda']
        )
    )
)




def handler(event, context):
    """
    Bucket notifications configuration.
    """
    properties = event['ResourceProperties']
    buckent_name = properties['Bucket']
    physical_resource_id = '{}-bucket-notification-configuration'.format(buckent_name)

    client = boto3.client('s3')
    existing_notifications = client.get_bucket_notification_configuration(
        Bucket=buckent_name
    )

    # Clean existing configurations
    configuration = {}
    if event['RequestType'] != 'Delete':

        arn_name_map = {
            'LambdaFunctionConfigurations': 'LambdaFunctionArn'
        }

        for _type in AVAILABLE_CONFIGURATIONS:
            configuration[_type] = []
            for notification in properties.get(_type, []):
                data = {
                    arn_name_map.get(_type): notification['DestinationArn'],
                    'Events': notification['Events'],
                }
                if notification['KeyFilters']:
                    data['Filter'] = {
                        'Key': {
                            'FilterRules': notification['KeyFilters']
                        }
                    }
                configuration[_type].append(data)

    client.put_bucket_notification_configuration(
        Bucket=buckent_name,
        NotificationConfiguration=configuration
    )

    send(event, context, SUCCESS, physical_resource_id=physical_resource_id)