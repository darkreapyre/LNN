#!/usr/bin/python
import os
import io
import json
import boto3
import tempfile
import urllib3
import base64
import sagemaker
import json
#import mxnet as mx
import numpy as np
#from mxnet import gluon, nd
import matplotlib.pyplot as plt
from flask import Flask, Response, request, jsonify, render_template
from PIL import Image
from skimage import transform

build_id = str(os.environ['BUILD_ID'])[:7]

def process_url(url):
    """
    Retrieves image from a URL and converts the image
    to a Numpy Array as the Payload for the SageMaker
    hosted endpoint.
    
    Arguments:
    url -- Full URL of the image
    
    Returns:
    payload -- Preprocessed image as a numpy array and returns a list
    """
    http = urllib3.PoolManager()
    req = http.request('GET', url)
    image = np.array(Image.open(io.BytesIO(req.data)))
    result = transform.resize(image, (64, 64), mode='constant').reshape((1, 64 * 64 * 3))
    return image, result.tolist()

app = Flask(__name__)

@app.route('/')
def index():
    resp = Response(response="Ping Successfull!",
         status=200, \
         mimetype="application/json")
    return (resp)

@app.route('/image')
def image():

    sagemaker_client = boto3.client('sagemaker')
    list_results = sagemaker_client.list_endpoints(
        SortBy='Name',
        NameContains=build_id,
        MaxResults=1,
        StatusEquals='InService'
    )
    endpoint_name = str(list_results.get('Endpoints')[0]['EndpointName'])

    print('api')
    url = request.args.get('image')
    print(url)

    # Process the URL
    image, payload = process_url(url)

    # Invoke the SageMaker endpoint
    runtime = boto3.client(service_name='runtime.sagemaker')
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )

    # Format the prediction from SgeMaker Endpoint
    classes = ['non-cat', 'cat']
    prediction = classes[int(json.loads(response['Body'].read().decode('utf-8')))]

    # Return prediction and image
    figfile = io.BytesIO()
    plt.imsave(figfile, image, format='png')
    figfile.seek(0)
    figfile_png = base64.b64encode(figfile.getvalue()).decode('ascii')
    return render_template('results.html', image=figfile_png, prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)