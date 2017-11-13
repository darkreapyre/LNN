You need a NotificationConfiguration property in your CloudFormation template. Unfortunately, it seems to require the bucket to already exist. To get around this, you can create an initial stack, then update it with the NotificationConfiguration. For example:

template1.json
```json
{
  "AWSTemplateFormatVersion": "2010-09-09",
  "Parameters": {
    "mylambda": {
      "Type": "String"
    }
  },
  "Resources": {
    "bucketperm": {
      "Type": "AWS::Lambda::Permission",
      "Properties" : {
        "Action": "lambda:InvokeFunction",
        "FunctionName": {"Ref": "mylambda"},
        "Principal": "s3.amazonaws.com",
        "SourceAccount": {"Ref": "AWS::AccountId"},
        "SourceArn": { "Fn::Join": [":", [
            "arn", "aws", "s3", "" , "", {"Ref" : "mybucket"}]]
        }
      }
    },
    "mybucket": {
      "Type": "AWS::S3::Bucket"
    }
  }
}
```

template2.json -- adds the ConfigurationNotification
```json
{
  "AWSTemplateFormatVersion": "2010-09-09",
  "Parameters": {
    "mylambda": {
      "Type": "String"
    }
  },
  "Resources": {
    "bucketperm": {
      "Type": "AWS::Lambda::Permission",
      "Properties" : {
        "Action": "lambda:InvokeFunction",
        "FunctionName": {"Ref": "mylambda"},
        "Principal": "s3.amazonaws.com",
        "SourceAccount": {"Ref": "AWS::AccountId"},
        "SourceArn": { "Fn::Join": [":", [
            "arn", "aws", "s3", "" , "", {"Ref" : "mybucket"}]]
        }
      }
    },
    "mybucket": {
      "Type": "AWS::S3::Bucket",
      "Properties": {
        "NotificationConfiguration": {
          "LambdaConfigurations": [
            {
              "Event" : "s3:ObjectCreated:*",
              "Function" : {"Ref": "mylambda"}
            }
          ]
        }
      }
    }
  }
}
```

You can use the aws CLI tool to create the stack like this:
```bash
$ aws cloudformation create-stack --stack-name mystack --template-body file://template1.json --parameters ParameterKey=mylambda,ParameterValue=<lambda arn>
# wait until stack is created
$ aws cloudformation update-stack --stack-name mystack --template-body file://template2.json --parameters ParameterKey=mylambda,ParameterValue=<lambda arn>
```