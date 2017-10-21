# General Notes
1. Add the SNS Topic Arn to the Environmental variables to that the `finish()` function can use it.
2. Add the necessary SNS permissions to the CloudFormation Template.
3. Add necessary permissions for Elasticache to CloudFormation template.
4. Push the security group name to the `LaunchLambda` Cloudformation template to ensure that the Redis cluster is deployed to the correct security group.
>**Note**: This may mean that the Lamabda functions need to run in the VPX and therfore the CloudFormation template needs to reflect this.
5. May need to create  a *Subnet Group* in the CloudFormation template.
5. Ensure that the correct port (**6379**) is open on the security group and access from the within the VPC is allowed.
5. Make sure to set Lambda loggin for a maximum of 7 days so as to not have lagacy Lambda Logs from previouys runs.
6. Make sure to add the ARNs of the various Lambda Functions to the environmental variables in the CloudFormation Template.
7. Ensure to always include the `state` in the `event` every time the `TrainerLambda` is invoked.

---

# Notes on Elasticache

---

# Notes on TrainerLambda.py

1. The TrainerLambda has three stages, `start`, `forward` and `backward`. The algorithm with the `lambda_handler()` function, to implement this is as follows:
    ```python
    # Get event variables first

    if state == 'start':
        Start(next_epoch)
        # Initialize Weights

        # Initialize Bias

        # Instialize new item in DynamoDB for the epoch

        # Call Propogate to start Forwardprop
        Propogate(direction='forward')

    elif state == 'forward':
        # Determine location in Forwardprop
        if layer > layers:
            # Location is at the end of forwardprop
            # Calculate Loss Function
            
            # Upadate epoch these items in DynamoDB
            
            # Start Backprop
            Propogate(direction='backward')
        else:
            # move onto the next hidden layer
            Propogate(direction='forward')

    elif state == 'backward':
        # Determine location within backprop
        if epoch == epochs and layer == 0:
            # Location is at the finish
            # Calculate Deriviative

            # Calculate final Weights

            # Update epoch these items in DynamoDB

            # Finalize training run
            Finish()

        elif epoc < epochs and layer == 0:
            # Location is at the end of the epoch
            # Calculate Derivative

            # Calculate weights for epoch

            # Start new epoch
            Start(next_epoch)

        else:
            # Move to the next hidden layer
            Propogate(direction='backward')
    ```
2. When running the *Trainer* Lambda Function, the function will need to know the current layer as well the number of hidden units (neurons) to use, plus the activation function for the current layer (if it's not layer 0). The following code show the code to use:
    ```python
    # Make "fake" layer 2 data for dimulation
    # Activations
    payload['activations']['layer' + str(current_layer)] = 'sigmoid'
    # Hidden units
    payload['neurons']['layer' + str(current_layer)] = 4

    # Process "alyer 2"
    current_layer = int(payload.get('layer')) + 1
    current_activation = payload.get('activations')['layer' + str(current_layer)] # Get activations
    num_hidden_units = payload.get('neurons')['layer' + str(current_layer)] # Get hidden units


---

# Notes on Building the Lambda Deployment Package

## Overview
This document outlinnes the proceedure to create the Lambda deployment packagew ith the intent to leverage the proceedures for other LNN Versions.

## Adding the necessary Python Libraries
```text
pip install virtualenv
virtualenv -p /var/lang/bin/python \
--always-copy \
--no-site-packages \
lambda_build

source lambda_build/bin/activate

pip install --upgrade pip wheel
pip install numpy
pip install scipy
pip install -U scikit-learn
pip install multiprocess

libdir="$VIRTUAL_ENV/lib/python3.6/site-packages/lib/"
mkdir -p $VIRTUAL_ENV/lib/python3.6/site-packages/lib || true
echo "venv original size $(du -sh $VIRTUAL_ENV | cut -f1)"
find $VIRTUAL_ENV/lib/python3.6/site-packages/ -name "tests" | xargs rm -r
find $VIRTUAL_ENV/lib/python3.6/site-packages/ -name "dataset" | xargs rm -r

Can't remove tests files from pandas

pip install pandas

find $VIRTUAL_ENV/lib/python3.6/site-packages/ -name ".pyc" -delete

find $VIRTUAL_ENV/lib/python3.6/site-packages/ -type d -empty -delete

find $VIRTUAL_ENV/lib/python3.6/site-packages/ -name ".so" | xargs strip

pushd $VIRTUAL_ENV/lib/python3.6/site-packages/

Only select pkgs that needed for your project
zip -r -9 -q /lambda.zip scipy numpy sklearn pandas pytz multiprocess dill

popd
```

## Makefile

## CloudFormation Template