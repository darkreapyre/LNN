# Image Prediction API, DevOps Pipeline Demo using SAM Local

![alt text](https://github.com/darkreapyre/itsacat-local/blob/master/assets/images/SAM_Local_Architecture.png "Architecture")

## Pre-requisites
1. Install AWS CLI.
2. Install SAM Local and Docker.

## Pre-demo Setup
1. Clone the Repository
```terminal
$ git clone https://github.com/darkreapyre/itsacat-local.git
```
2. Build the CI/CD Pipeline
```terminal
$ bin/deploy
```
>**Note:** The deployment should fail getting the parameters source.
3. Generate "bad" parameters.
    - Run *Trainer* using SAM Local.
    ```terminal
        $ cd lambda
        $ docker run -v $PWD:/var/task -it lambci/lambda:build-python3.6 /bin/bash -c './create_package.sh'
        $ sam local generate-event s3 --bucket itsacat-local --key training_input/datasets.h5 > event.json
        $ sam local invoke "Trainer" -e event.json
    ```
    - Cleanup the environment.
    ```terminal
        $ rm -rf venv
        $ rm event.json
        $ rm package.zip
    ```
    - Comment the last line of `trainer.py`.
    - Switch from "Blue" to "Green" in `src/templates/results.html` and save.
    - Commit changes to GitHub.
    - Reject any requests for pipeline approval.
    - Comment out the last line of `trainer.py` **BUT** do not push the changes to GitHub.
    - Clear out all CloudWatch Event logs.    
5. Prepare to run the demo.
    - Open the "Blue" prediction API in a Web Browser.
    - Open a browser tab of the GitHub Repository.
    - Open a browser tab of appropriate cat imaages from Google images.
    - Open a browser tab to the AWS CloudFormation Console.

## Run the demo.
1. Update `num_interations` to `3000` and comment out last line, then save.
2. Update deployment package.
```terminal
    $ cd lambda
    $ docker run -v $PWD:/var/task -it lambci/lambda:build-python3.6 /bin/bash -c './create_package.sh'
```
3. Generate the `events.json`.
```terminal
    $ sam local generate-event s3 --bucket itsacat-local --key training_input/datasets.h5 > event.json
```
4. Invoke the function locally.
```terminal
    $ sam local invoke "Demo" -e event.json
```
5. **Uncomment last line of `trainer.py`.**
6. Update deployment package.
```terminal
    $ docker run -v $PWD:/var/task -it lambci/lambda:build-python3.6 /bin/bash -c './create_package.sh'
```
7. Package the function for the cloud.
```terminal
    $ sam package --template-file template.yaml --s3-bucket itsacat-local --output-template-file packaged.yaml
```
8. Deploy to produciton.
```terminal
    $ sam deploy --template-file packaged.yaml --stack-name demo-lambda --capabilities CAPABILITY_IAM
```
9. Trigger event by uploading `datasets.h5` to the created S3 Bucket and monitor.