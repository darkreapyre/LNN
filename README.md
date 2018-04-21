# Serverless Neural Network for Image Classification - Training/Prediction Pipeline with Logging Demo
<!-- Note to Self: Change the picture and text below to reflect when adding to CodeCommit Repository -->
![alt text](https://github.com/darkreapyre/LNN/blob/1.0/artifacts/images/Prediction_Architecture-2.png "Architecture")

## Pre-Requisites
>**Note:** This Demo must be run the **us-east-1** AWS Region as it leverages the **AWS Fargate** Service.
1. This demo uses the [AWS CLI](http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html). If the AWS CLI isn't installed,  follow [these](http://docs.aws.amazon.com/cli/latest/userguide/installing.html) instructions. The CLI [configuration](http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html) needs `PowerUserAccess` and `IAMFullAccess` [IAM policies](http://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html) associated. To verify that the AWS CLI is installed and up to date, run the following:
```console
    $ aws --version
```
2. Clone the repo.
```console
    $ git checkout 1.0
``` 

## Deployment
1. Make any changes to the Neural Network configuraiton parameters file (`parameters.json`) before running the deployment.
2. To deploy the environment, an easy to use deployment script has been created to automatically deploy the environment. Start the process by running `bin/deploy`. You will be prompted for the following information:
```console
    Enter the S3 bucket to create > <<UNIQUE S3 BUCKET>>
    Enter the name of the Stack to deploy > <<UNIQUE CLOUDFOMRATION STACK NAME>>
    Enter GitHub User > <<GitHub Username>>
    Enter GitHubToken > <<GitHub Token>>
    Enter the e-mail address to send training update > <<E-MAIL ADDRESS>>
```
>**Note:** For more information on creating a GitHub [Token](https://github.com/settings/tokens).

3. The `deploy.sh` script creates the an *S3* Bucket; copies the necessary *CloudFormation* templates to the Bucket; creates the *Lambda* deployment package and uploads it to the Bucket and lastly, it creates the CloudFormation *Stack*. Once completed, the following message is displayed:
```console
    "Successfully created/updated stack - <<Stack Name>>"
```
>**Note:** During the deployment, and e-mail will be sent to the specified address. Make sure to confirm the subscription to the SNS Topic.

## Integration with SageMaker Notebook Instance
Once the stack has been deployed, integrate [Amazon SageMaker](https://aws.amazon.com/sagemaker/) into the stack to start reviewing the Demo content by using the following steps:
1. Open the SageMaker [console](https://console.aws.amazon.com/sagemaker).
2. Create notebook instance.
3. Notebook instance settings.
    - Notebook instance name.
    - Notebook instance type -> ml.t2.medium.
    - IAM Role -> Create new role.
    - Specific S3 Bucket -> "UNIQUE BUCKET NAME" -> Create Role.
    - VPC -> "UNIQUE STACK NAME" .
    - Subnet -> Select the subnet marked "Private Subnet (AZ1)".
    - Security group(s) -> ComputeSecurityGroup.
    - Direct Internet Access -> Enable.
    - Create notebook instance.
3. You should see Status -> Pending.
4. Configure Service Access role.
    - IAM Console -> Role -> "AmazonSageMaker-ExecutionRole-...".
    - Policy name -> "AmazonSageMaker-ExecutionPolicy-..." -> Edit policy.
    - Visual editor tab -> Add additional permissions.
        - Service -> Choose a service -> ElastiCache.
        - Action -> Select and action
            - All ElastiCache actions (elasticache:*).
        - Repeat the above step to add All DynamoDB actions (dynamoDB:*).
        - Review Policy.
        - Save changes.
    - The final Policy should look similar to this:
    ```json
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "VisualEditor0",
                    "Effect": "Allow",
                    "Action": "s3:ListBucket",
                    "Resource": "arn:aws:s3:::<<UNIQUE BUCKET NAME>>"
                },
                {
                    "Sid": "VisualEditor1",
                    "Effect": "Allow",
                    "Action": [
                        "s3:PutObject",
                        "s3:GetObject",
                        "s3:DeleteObject"
                    ],
                    "Resource": "arn:aws:s3:::<<UNIQUE BUCKET NAME>>/*"
                },
                {
                    "Sid": "VisualEditor2",
                    "Effect": "Allow",
                    "Action": [
                        "dynamodb:*",
                        "elasticache:*"
                    ],
                    "Resource": "*"
                }
            ]
        }
    ```
5. Return to the SageMaker console and confirm the Notebook instance is created. Under "Actions" -> Select "Open".
6. After the Jupyter Interface has opened, Configure the Python Libraries:
    - Select the "Conda" tab -> under "Conda environments" -> select "python3".
    - Under "Available packages" -> click the "Search" box -> enter "redis". The following two "redis" packages should be available. 
        - redis
        - redis-py
    - Select both packages and click the "->" button to install the packages.
    - Confirm to "Install" on the pop-up.
    >**Note:** Ignore any error messages that may pop-up.
7. Clone the GitHuib Repository.
    - Under the "Files" tab -> Click "New" -> "Terminal".
    - Under the Shell run the following commands:
    ```shell
        $ cd SageMaker
        $ git clone https://github.com/<<Repository Name>>/LNN
        $ git checkout 1.0
        $ exit
    ```
    - Go back to the "Files" tab -> click "`<<Repository Nane>>`" -> click "Notebooks" -> select `Introduction.ipynb`

## Demo Process Flow
To follow the Machine Learning Pipeline process flow the four steps listed below:

### Step 1. Jupyter Notebooks
Three Jupyter Notebooks have been created to explain and simulate the *Data Scientist's* and *DevOps Engineer's* role witin the Machine Learning Pipeline process.
- To run the notebook document step-by-step (one cell a time) by pressing shift + enter.
- To run the whole notebook in a single step by clicking on the menu Cell -> Run All.
- To restart the kernel (i.e. the computational engine), click on the menu Kernel -> Restart. This can be useful to start over a computation from scratch (e.g. variables are deleted, open files are closed, etc…).
- More information on editing a notebook: [Notebook Basics](http://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)

#### `Introduction.ipynb` Notebook
The **Introduction** provides an overview of the *Architecture*, *Why* and *How* the *Serverless Neural Network* is implemented.

#### `Codebook.ipynb` Notebook
The **Codebook** provides an overview of the various *Python Libraries*, *Helper Functions* and the *Handler* functions that are integrated into each of the Lambda Functions. It also provides a sample implementation of the Neural Network, to get an understanding of how the full training process will be executed. After executing the various cells withing the *CodeBook*, the results from the *10 Epoch* sample will trigger the production deployment pipeline. To view the unoptmized version the Prediction API in the QA/Staging environment, see the [Prediction API](#step-4-prediction-api) step of the process flow.
>**Note:** In order to avoid any conflict with the Parameter Server (ElastiCache) and to avoid unnecessary serivce charges, a good practice is to Stop the SageMaker Notebook Instance before proceeding to the next step.

### Step 2. Training the Classifier
To train the full classification model on the *SNN* framework, simply upload the `datasets.h5` file found in the `artifacts\datasets` directory to the `training_iput` folder of the S3 bucket that has already been created by the deployment process.
>**Note:** A pre-configured `parameters.json` file has already been created. To change the Neural Network configuration parameters, before running the training process, change this file and upload it to the `training_input` folder of the S3 Bucket before uploading the data set.

Once the data file has been uploaded, an S3 Event will automatically trigger the training process, and in-depth insight to the training process can be viewed through the **CloudWatch** console. When the model training is complete (typically after 22 hours), the final optimized parameters will be copied to the `prediction_input` folder of the S3 bucket. This will trigger the production deployment pipeline and a message will be sent to review the Prediction API in the QA/Staging environment.
>**Note:** Refer to the [Troubleshooting](#troubleshooting) section for any errors that may appear in the **CloudWatch** Console.

### Step 3. Analyzing the Results
Once the training process has successfully completed, an e-mail will be sent to the address configured during the deployment. To analyze the results of the testing and to determine if the trained model is "production-worthy", using the same *SageMaker* instance used for the *Codebook*, navigate to the `artifacts` directory and launch the `Analysis.ipynb` notebook.

Work through the various code cells to see:
1. Results fo the training.
2. How well the model performs against the **Test** dataset.
3. How well the model performs against new images.

>**Note:** Ensure to add the name of the S3 Bucket and AWS Region, used during deployment, to get the correct results files created during the training process.

### Step 4. Prediction API
The deployment pipeline for the production application is triggered at two separate stages within the Demo Process Flow:
- After executing the `Codebook.ipynb` in [Step 1.](#step-1-jupyter-notebooks), the parameters are written to the `predict_input` folder of the S3 bucket. Since this is a Source for CodePipeline to trigger the deployment. At this stage, since the parameters have only been trained for 10 iterations, they are not fully optmized, so the Prediction API will not fully predict a "cat" picture.
- After the model has been optimally trained in [Step 2.](#step-2-training-the-classifier), the parameters once again written to the `predict_input` folder fo the S3 bucket and thius the deployment pipeline is triggered. Since the model has been optmially trained, the Prediction API should fully predict a "cat" picture.

During either of the above stages, an e-mail will be sent to the address configured during deployment similar to the following (stripped for brevity):
```text
Hello,

The following Approval action is waiting for your response:

--Pipeline Details--

...
```
Included is the e-mail is a link to the *CodePipeline* Service Console to approve the deployment from QA to Production. To view and test the Prediction API in the QA stage, execute the following:
1. Open the *CloudFormation* Service Console and select the nested Stack for the Elastic Container Service (ECS). e.g. **`<<Stack Name>>-DeploymentPipeline-...-ecs-cluster`**.
2. Click on the CloudFormation Outputs tab.
3. The *ApplicationURL** Value provides a link to the **Prediction API URL for Production (Blue)**. Clicking on this link will open a browser page to the Prediciton API. Successful connection to the API will display the **"Ping Successfull!"** message.
4. To view the production (Blue) API, find the URL of a "cat" picture (e.g.[Grumpy Cat](http://i0.kym-cdn.com/entries/icons/facebook/000/011/365/GRUMPYCAT.jpg)) and add it to the URL as follows:
    
    `http://<<GitHub Repo Name>>.us-east-1.elb.amazonaws.com/image?image=http://i0.kym-cdn.com/entries/icons/facebook/000/011/365/GRUMPYCAT.jpg`

5. To view the Test/Staging (Green) API, simply change the port to **8080** as follows:

    `http://<<GitHub Repo Name>>.us-east-1.elb.amazonaws.com:8080/image?image=http://i0.kym-cdn.com/entries/icons/facebook/000/011/365/GRUMPYCAT.jpg`

Accessing the (Greep) API after [Step 2.](#step-2-training-the-classifier)) should correctly predict a "cat" image and thus the **Manual-Approval** stage in CodePipeline can be *Approved*. This in turn will swap the (Green) API to production (Blue), wich can be accessued using **Prediction API URL for Production (Blue)**.

It is at this point that a successull integration of a **Machine Learning Pipeline** into a production **DevOps Pipeline** has been successfully demonstrated. To avoid additional charges for AWS resources, refer to the [Cleanup](#cleanup) Section.

## Troubleshooting
Since the framework launches a significant amount of Asynchronous Lambda functions without any pre-warming, the **CloudWatch** logs may display an error similar to the following:  
**Streams for /aws/lambda/TrainerLambda**
```python
    list index out of range: IndexError
    Traceback (most recent call last):
    File "/var/task/trainer.py", line 484, in lambda_handler
    A = vectorizer(Outputs='a', Layer=layer-1)
    File "/var/task/trainer.py", line 235, in vectorizer
    key_list.append(result[0])
    IndexError: list index out of range
```
To address this, simply delete all the DynamoDB Tables as well as data set (`datasets.h5`) from the S3 Bucket and re-upload data set to re-launch the training process.

## Cleanup
1. Delete the SageMaker Notebook instance.
    - Open the SageMaker Service console.
    - Select the Notebook Instance and Stop it, if the instance is still running.
    - Select the Notebook Instance -> click Actions -> Delete.
2. After the SageMaker Notebook Instance is deleted, delete the CloudFormnation Stack.
    - Open the CloudFormation Service console.
    - Ensure all nested stacks have a **CREATE_COMPLETE** or **UPDATE_COMPLETE** Status. If not, wait for any stack updates to complete.
    - Select the Elastic Container Service (ECS) Stack created by CodePipeline. e.g. **`<<Stack Name>>-DeploymentPipeline-...-ecs-cluster`**
    - Click Actions -> Delete Stack -> "Yes, Delete".
    - Select the stack created by the initial deployment and repeat the above step.
3. Delete DynamoDB Tables.
    - Open DynamoDB Service console.
    - Select "Tables" in the navigation panel.
    - Check **NeuronLambda** -> Delete table -> Delete.
    - Repeat the above process for the **Costs**, **TrainerLambda** and **LaunchLambda** tables.
4. Delete the CloudWatch Logs.
    - Open the CloudWatch Service console.
    - Select "Logs" in the navigation panel.
    - Check */aws/lambda/LaunchLambda* -> Actions -> Delete log group -> Yes, Delete.
    - Repeat the above process for **NeuronLambda**, **S3TriggerLambda**, **LaunchLambda**, **TrainerLambda**  and any of the logs created by CodePipeline.
5. Delete the S3 Bucket.
    - Open the S3 Service console.
    - Highlite the bucket created at deployment time -> Delete bucket.
    - Perform the same actions for the bucket used by CodePipeline for artifacts.
6. Delete any Elastic Container Repositories (ECR) created by CodePipeline.
    - Open the amazon ECS Service Console.
    - Select Amazon ECR -> Repositories.
    - Check the relevant ERC Repositories -> Click Delete Repository.