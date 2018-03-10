# Serverless Neural Network (Work in progress)
This repository comtains various branches that depict leveraging AWS Lambda Functions as Neurons for various Neural Networks. The objective to learn how to build Deep Neural Networks from scratch without leveraging frameworks like TensorFlow, Caffe, Theano etc. Additionally the object is to ensure a better , more advanced, understanding about what is happenning within each Layer and each Hidden Unit. The various versions (branches) are loosely based on the [deepelarning.ai](https://www.coursera.org/specializations/deep-learning) Coursera specialization.

Each version is meant to enhance the functionality of the implementation to start from a basic Perceptron and evolve into more compleicated funcitonality that includes Regularization, Optmization and Gradient Checking techniques as well as deeper network architectures. The version branches are as follows:

### Version 0.0: Single Neuron Logistic Regression - DynamoDB. (**Obsolete**)
>**Notes:**
- DynamoDB does not offer sufficient flexability to store Numpy Arrays, thus forced to store the data on S3.
- DynamoDB does not offer sufficient flexability to store the network settings as it does not serialize JSON files or dictionaries very well.
- DynamoDB does not store float data types and thus we have to serialize ductionary content to `decimal` which cases significant programming complexity.
### Version 0.1: Single Neuron Logistic Regression - Elasticache. (**Obsolete**)
>**Notes:**
- After testing, close to 1000 epochs, the TrainerLambda and NeuronLambda both re-invoke, thus causing epochs to repeat. Fortunately Gradient Descent seems to function correclty and the *Cost* continues to decrease, but the epochs repeat infinitely. After 8 - 10 hours, multiple interations of epochs are visible with no end in sight.
### Version 0.1.1: Single Neuron Logistic Regression - ElastiCache/Batches. (**Obsolete**)
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
- Changing the invocation type had an unexpected side effect in that new Lambda functions were spawned as opposed to re-used, thus causing each to require a dedicated **ENI**. Unfortunately, the limit for ENI’s* on the VPC is 300, therefore the processes halted as ENI’s limits were saturated quickly.
### Version 0.1.2: Single Neuron Logistic Regression - ElastiCache/Batches using CloudWatch Scheduled Events. (**Obsolete**)
>**Notes:**
- After executing 100 epochs, a *CloudWatch* scheduled event is created to wait *30* minutes and then execute the next 100 epochs.
- The event didn't triogger on schedule some times and other times, the event was not even created and training process simply continued.
### Version 0.1.3: Single Neuron Logistic Regression - ElastiCache/Recursive Checking with DynamoDB. (**Complete**)
>**Notes:**
- After doing research, it seems that other users had found similar issues with Lambda Functions spawning duplicate Lambda invocations, see [here](https://cloudonaut.io/your-lambda-function-might-execute-twice-deal-with-it/) for more information. To address this, the code now initializes [AWS DynamoDB](https://aws.amazon.com/dynamodb/) Tables (one for each Lambda Function) and ensure that each invocation is assigned a unique ID. If any duplicate functions are spawned, there is a conflict with the unique ID and the duplicate funciton immediatley terminates.
### Version 0.2: L-Layer Logistic Regression. (**Complete**)
>**Notes:**
- Unlike  `Version 0.1.3`, where **20,000** Epochs produces a good set of optimal paramaters, **2,500** Epochs is a good startong point for training iterations on `Version 0.2`. But due to the fact that it takes a significant amnount of time to train an L-Layer Network (approx. 40 hours), the development infrastructure has been removed from the *CloudFormation* deployment.
- Initially, the code (`LaunchLambda`) initialized the **Weights** and **Bias** with a `0.075` constraint. This significantly impacted the **Cost** function. It was further realized that the intitialization should be $\frac{1}{\sqrt{n}}$ and the **Bias** initialized to zero.
- Based on the above, it was further realized that the **Cost Function** calculation was incorrect, **0.30319607531996434** after **2,500** Epochs. After applying the correct [cross-entropy cost function](http://neuralnetworksanddeeplearning.com/chap3.html#introducing_the_cross-entropy_cost_function), this was significantly improved upon.
- Another issue that was discovered was the fact that when the various Activations a compiled into a single Matrix after eacch layer invocation, by the `TrainerLambda`, the sequence retuned by *ElastiCache* is **not** in numerical order. This basically means that the vectorized implementaiton of how the *Weights* apply to the output is out of order. After ensuring the correct numerical sequence was returned to construct the correct Matrix, the final **Cost** was **0.07576201861014191** after **3,000** Epochs.
- Additionally it is noted that the ordering of the individual **Liner Activations** that are returned from *ElastiCache* and used in the backward propogation step were also out of sequence.
- To address both of the above issues, a `vectorizer()` function was created to returned an ordered list of individual neuron outputs in both the forward and backward propogation steps. Applying this resulted in a final **Cost** of **0.011273672815000354** after **2,500** Epochs.
- This Branch was used to create the [itsacat](https://github.com/darkreapyre/itsacat) demo, and prototyped the usage of [Xavier Glorot](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) initialization as well as the [ReLU](https://arxiv.org/pdf/1502.01852v1.pdf) initializations.
### Version 0.2.1: L-Layer Logistic Regression - Xavier Initialization with L2 Regularization. (**Complete**)
>**Notes:**
- Should this or subsequent versions succeed, the training process will integrate with the *CI/CD Pipeline* (`Version 0.3.x`). Currently the Pipeline has been de-coupled to only lerage the services required by the Lambda functions, namely:
    - VPC
    - Security Groups
    - ElastiCache
- Although introducing *Xavier* initialization for the **Weights** shows significant improvement (final **Cost** of **0.00658138016613162** after **2,500** Epochs), adding **L2 Regularization** does not solve the **Exploding Gradient** problem and in fact produces final **Cost** that is worse, **0.12014227975111075** after **2,500** Epochs. 
- It is also important to note that runnning the network without *Xavier* initialization produces an overall Accuracy Score of **$78%$** after **3,000** Epochs. Using **Xavier** initialization (without **L2 Regularization**) produces an Accuracy Score of **$74%$** after **2,500** Epochs. It is the hope to see an improvement by increasing this to **3,000** Epochs. As a side note, the network leveraging **Xavier** initialization as well as **L2 Regularization** only produces a **$68%$** Accuracy Score.
- To try and reduce the exploding gradient issue, the next release will investigate different optimizers and mini-batch gradient descent.
### Version 0.2.2: L-Layer Logistic Regression - Mini-Batch Gradient Decent Optimization. (**Under Investigation**)
>**Notes:**
- In order to improve the overall error **without** inreasing the number of Epochs, *Mini-Batch* Gradient Descent is tested. This process requires a complete reworking of the architecture:
        - The `LaunchLambda` now controls the overall iterations/epochs.
        - Mini-batch training is controlled by the `TrainerLambda`.
- The re-worked architecture specifies a dedicated ElastiCache database for each mini-batch. The "master" database for paramaters is by default hard-coded to **15** as a single Redis server can only support **16** databases. This means that for the solution to work, no more than **15** mini-batches can be used.
- After reworking the framework to accommodate mini-batches, the following error occurred intermitently, but mostly during the inital epochs:
```python
    list index out of range: IndexError
    Traceback (most recent call last):
    File "/var/task/trainer.py", line 40, in lambda_handler
    A = vectorizer(Outputs='a', Layer=layer-1, batch=batch, parameters=parameters)
    File "/var/task/Utils.py", line 556, in vectorizer
    key_list.append(result[0])
    IndexError: list index out of range
```
- A similar error was seen before in some of the previous Versions and related to latency in the ElastiCache datababase. TYhis was resolved by re-submitting the `datasets.h5` file to S3 as it happenned on the first Epoch. However the error above accurs anywhere between **7** and **23** Epochs. It was assumed that that the `NeuronLandler` was launching the `TrainerLambda` before the other Neurons had updated ElstiCache and therefore the index was out of range. To address this, the following was tried:
    - Tried **10** second latency before launching the `TrainerLambda`. --> This approach was terminated due to the fact that the overall latency of adding would defeat the purpose of trying mini-batches to optimize quicker and reduce training time.
    - Tried 2 second latency before launching `TrainerLambda`. --> Faileden after **90** Epochs.
    - Tried 1 second latency before launching `TrainerLambda`. --> Failed after **50** Epochs.
    - Tried bigger Lambda memory size. --> Failed after **3** Epochs.
    - Tried to constrain the Lambda function and ElastiCache to a single Availability Zone in case there were Lambda invocations in the second AZ of the Region. --> No effect.
- After trying to debug the above error by analysing ElastiCache data, the error was re-produced. The index is out of range because the `result` array was empty. In fact there was no data from ElastiCache when searning. It was later determined that ElastyiCache REDIS engine deleted the database when `flushdb()` is called, in the background. This command was used within `LaunchLambda` to "refresh" the database between Epochs. Since this didn't happen synchronously, the data was bening delated between the updated from `NauronLambda` and `TrainerLambda`. The `flushdb()` command was removed from `LaunchLambda`, which resolved the issue.
- After testing with **750** Epochs, the training time was significantly betters, **18** hours. However, event thoguh final cost was **0.0012926892893857303**, the overall model accuracy was only **$74%$**.
- After testing with **625** Epcohs, it was hoped that since each mini-batch is approximately a quarter of the training data, that the number of epochs could be reduced by a factor of **4** to produce a similar result, the final cost being **0.0025214069754894153** after **15** hours with an Accuracy score of **$78%$**. SO it seems the theory was correct, **BUT** the model did not accurately predict "non-cat" images, while the **750** epoch model had lower scores, it predicted the "non-cat" model accurately. This shows that the model is geared to "cat" images sepcifically.
### Version 0.2.3:  L-Layer Logistic Regression - Adam Optmization. (**Failed**)
- Adam Optmization failed at **750** and **2500** Epochs, producing **NaN** for the cost. 
- However it succedded twice running **625** Epochs, but failed to converge.
### Version 0.2.4: L-Layer Logistic Regression - Average Parameters Mini-batch Gradient Descent Optmization. (**Under Investigation**)
>**Notes:**
- The idea behind this version is to create a "custom" i.e. don't think this has been done before, metahdology to optimize the parameters. This is done by running each mini-batch in parallel and taking the average of the **Weights** and **Bias** (optmized by running mini-batch gradient descent) over the various mini-batches.

    >**Issues:** Between 30 and 90 Epochs
    ```python
        'cost': KeyError
        Traceback (most recent call last):
        File "/var/task/launch.py", line 349, in lambda_handler
        Costs.append(from_cache(db=b, key=batch_parameters['data_keys']['cost']))
        KeyError: 'cost'
    ```




### Version 0.2.5: L-Layer Logistic Regression - Lowest Error Mini-batch Gradient Descent Optimization. (**TBD**)
- The idea behind this version is to create a "custom" i.e. don't think this has been done before, metahdology to optimize the parameters. This is done by running each mini-batch in parallel and taking the mini-batch with the lowest error/cost over the various mini-batches.
### Version 0.3.0: L-Layer Logistic Regression - Introduction of Blue/Green Pipeline with Fargate. (**Complete**)
>**Notes:**
- This version merges [itsacat](https://github.com/darkreapyre/itsacat) demo (derived from Version 0.2.1 **without** L2 Regularization) with the Prediction API *CI/CD Pipeline*. 
- In order for the Pipeline to leverage [AWS Fargate](https://aws.amazon.com/fargate/), the solution is tested using the `us-east-1` Region and therefore levergaes a dedicated *S3* Bucket (**lnn**) in that Region.
- As a side note, the Training Pipeline produced an Accuracy Score of **$78%$** using [2500](https://github.com/darkreapyre/LNN/blob/0.3.0/artifacts/Analysis-MMD.ipynb) Epochs and an Accuracy Score of **$82%$** using [3000](https://github.com/darkreapyre/LNN/blob/0.3.0/artifacts/Analysis-MMM.ipynb) Epochs.
### Version 0.3.1: L-Layer Logistic Regression - Introduction of Blue/Green Pipeline with API Gateway. (**TBD**)
>**Notes:**
- [To test](https://www.96cloudshiftstrategies.com/flasklambdalab.html)