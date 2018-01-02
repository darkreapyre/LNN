# Lambda Neural Network
>**Note:** Work in progress
This repository comtains various branches that depict leveraging AWS Lambda Functions as Neurons for various Neural Networks. The objective to learn how to build Deep Neural Networks from scratch without leveraging frameworks like TensorFlow, Caffe, Theano etc. Additionally the object is to ensure a better , more advanced, understanding about what is happenning within each Layer and each Hidden Unit.

Each version is mean to enhance the functionality of the implementation to start from a basic Perceptron and evolve into more compleicated funcitonality that includes Regularization, Optmization and Gradient Checking techniques as well as deeper network architectures. The version branches are as follows:

- Version 0.0: Single Neuron Logistic Regression - DynamoDB. (**Obsolete**)
    >**Notes:**
    - DynamoDB does not offer sufficient flexability to store Numpy Arrays, thus forced to store the data on S3.
    - DynamoDB does not offer sufficient flexability to store the network settings as it does not serialize JSON files or dictionaries very well.
    - DynamoDB does not store float data types and thus we have to serialize ductionary content to `decimal` which cases significant programming complexity.
- Version 0.1: Single Neuron Logistic Regression - Elasticache. (**Obsolete**)
    >**Notes:**
    - After testing, close to 1000 epochs, the TrainerLambda and NeuronLambda both re-invoke, thus causing epochs to repeat. Fortunately Gradient Descent seems to function correclty and the *Cost* continues to decrease, but the epochs repeat infinitely. After 8 - 10 hours, multiple interations of epochs are visible with no end in sight.
- Version 0.1.1: Single Neuron Logistic Regression - ElastiCache/Batches (**Obsolete**)
    >**Notes:**
    - Multiple Techniques were applied, with little to no effect.:
        1. Increasing the `connect_timeout` and `read_timeout` by applying the following `Boto` configuration parameter:
        ```python
        config = botocore.config.Config(connect_timeout=300, read_timeout=300)
        Lambda_client = boto3.client(‘lambda’, region_name=rgn, config=config)
        ```
        2. Disabling the `retry` value so that should a timeout occur, the Lambda function will not execute. This was done by changing the meta data of the lambda client as follows:
        ```python
        lambda_client.meta.events._unique_id_handlers[‘retry-config-lambda’][‘handler’]._checker.__dict__[‘_max_attempts’] = 0
        ```
        3. Switching from *Asynchronous* to *Synchronous* Lambda invocations, in the hopes that no duplicate Lambda functions would be spawned, by changing the `InvocationType=‘Event’` to `InvocationType=‘RequestResponse’` on each Lambda invocation.
    - Disabling the `retry` value didn't seem to have the desired effect,  not only on the "mutant" Lambda Functions, but if an error occured, the Lambda Funcitons still tried to retry, except on random errors that couldnb't be reproduced.
    - Ching the invocation type had an unexpected side effect in that new Lambda functions were spawned as opposed to re-used, thus causing each to require a dedicated **ENI**. Unfortunately, the limit for ENI’s* on the VPC is 300, therefore the processes halted as ENI’s limits were saturated quickly.
- Version 0.1.2: Single Neuron Logistic Regression - ElastiCache/Batches using CloudWatch Scheduled Events (**Obsolete**)
    >**Notes:**
    - After executing 100 epochs, a *CloudWatch* scheduled event is created to wait *30* minutes and then execurte the next 100 epochs.
    - The event didn't triogger ons schedule some times and other times, the event was not even created and training process simply continued.
- Version 0.1.3: Single Neuron Logistic Regression - ElastiCache/Recursive Checking/DynamoDB (**Successfulew Completion**)
    >**Notes:**
    - After doing significant research, it seems that other users had found similar issues with Lambda Functions spawning duplicate Lambda invocations.
    - [To Add](https://cloudonaut.io/your-lambda-function-might-execute-twice-deal-with-it/)
- Version 0.2: L-Layer Logistic Regression. (**On Hold due to Issues with Backprop**)
    >**Notes:**
    - **MORE THEN 2500 EPOCHS??????, therefore separation of training and *CI/CD Pipeline*.**
    - **`np.sqrt`**
    - **`bias * 0.075`**
    - **Incorrect Cost function --> 0.30319607531996434**
    - **Vectorization ordering ??? --> 0.07576201861014191**
- Version 0.2.1: L-Layer Logistic Regression - Batch Gradient Decent Optimization (**TBD**)
    >**Side Note:**  Should they succeed, the training process will integrate with the *CI/CD Pipeline*.
    >**Notes:**
    - In order to improve the overall error **without** inreasing the number of Epochs, *Regularization* and *Mini-Batch* Gradient Descent is tested, with the goal of integrating the training process with the final *CI/CD* Pipeline from Version 0.2.x.
- Version 0.2.2:  L-Layer Logistic Regression - Adam Optmization (**TBD**)
- Version 0.3.0: L-Layer Logistic Regression - Introduction of Blue/Green Pipeline with dedicated ELB (**TBD**)
- Version 0.3.1: L-Layer Logistic Regression - Introduction of Blue/Green Pipleline with Fargate (**TBD**)
- Version 0.3.2: L-Layer Logistic Regression - Introduction of Blue/Green Pipeline with API Gateway (**TBD**)
    >**Notes:**
    - [To test](https://www.96cloudshiftstrategies.com/flasklambdalab.html)

The various versions are loosely based on the [deepelarning.ai](https://www.coursera.org/specializations/deep-learning) Coursera specialization.
