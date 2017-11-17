#!/usr/bin/python

import boto3
from cfnresponse import send, SUCCESS, FAILED

AVAILABLE_CONFIGURATIONS = (
    'LambdaFunctionConfigurations',
    'TopicConfigurations',
    'QueueConfigurations'
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

    # It doesn't matter how big you put this on the doc... people wil
    # always put bucket's arn instead of name... and it would be a shame
    # to fail because this stupid error.
    buckent_name = properties['Bucket'].replace('arn:aws:s3:::', '')
    physical_resource_id = '{}-bucket-notification-configuration'.format(buckent_name)

    client = boto3.client('s3')
    existing_notifications = client.get_bucket_notification_configuration(
        Bucket=buckent_name
    )

    # Check if there is any notification-id which doesn't start with gordon-
    # If so... fail.
    for _type in AVAILABLE_CONFIGURATIONS:
        for notification in existing_notifications.get(_type, []):
            if not notification.get('Id', '').startswith('gordon-'):
                send(
                    event,
                    context,
                    FAILED,
                    physical_resource_id=physical_resource_id,
                    reason=("Bucket {} contains a notification called {} "
                            "which was not created by gordon, hence the risk "
                            "of trying it to add/modify/delete new notifications. "
                            "Please check the documentation in order to understand "
                            "why gordon refuses to proceed.").format(
                                buckent_name,
                                notification.get('Id', '')
                     )
                )
                return

    # For Delete requests, we need to simply send an empty dictionary.
    # Again - this have bad implications if the user has tried to configure
    # notification manually, because we are going to override their
    # configuration. There is no much else we can do.
    configuration = {}
    if event['RequestType'] != 'Delete':

        arn_name_map = {
            'LambdaFunctionConfigurations': 'LambdaFunctionArn',
            'TopicConfigurations': 'TopicArn',
            'QueueConfigurations': 'QueueArn',
        }

        for _type in AVAILABLE_CONFIGURATIONS:
            configuration[_type] = []
            for notification in properties.get(_type, []):
                data = {
                    'Id': notification['Id'],
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