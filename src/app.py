#!/usr/bin/python
import os
import json
import boto3
import tempfile
import urllib3
import scipy
import botocore
import base64
import h5py
import sagemaker
import json
import mxnet as mx
import numpy as np
from mxnet import gluon, nd
import matplotlib.pyplot as plt
from io import BytesIO
from cStringIO import StringIO
from flask import Flask, Response, request, jsonify, render_template
from PIL import Image
from skimage import transform

build_id = str(os.environ['BUILD_ID'])
sagemaker_client = boto3.client('sagemaker')
list_results = sagemaker_client.list_endpoints(
    SortBy='Name',
    NameContains=build_id,
    MaxResults=1,
    StatusEquals='InService'
)
if not list_results['Endpoints']:
    endpoint_name = 0
else:
    endpoint_name = str(list_results.get('Endpoints')[0]['EndpointName'])

def local_predict(data):
    """
    Runs the Gluon network if SageMaker Endpoint is not available

    Arguments:
    data -- Input image data as string.

    Returns:
    response_body -- Predition respons as string.
    """
    # Load the saved Gluon model
    symbol = mx.sym.load('model.json')
    outputs = mx.sym.sigmoid(data=symbol, name='sigmoid_label')
    inputs = mx.sym.var('data')
    param_dict = gluon.ParameterDict('model_')
    net = gluon.SymbolBlock(outputs, inputs, param_dict)
    net.load_params('model.params', ctx=mx.cpu())
    # Parse the data
    parsed = json.loads(data)
    # Convert input to MXNet NDArray
    nda = mx.nd.array(parsed)
    output = net(nda)
    prediction = nd.argmax(output, axis=1)
    response_body = json.dumps(prediction.asnumpy().tolist()[0])
    return response_body

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
    image = np.array(Image.open(StringIO(req.data)))
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
    print('api')
    url = request.args.get('image')
    print(url)

    # Process the URL
    image, payload = process_url(url)

    # Determine if SageMaker Endpoint or local is used
    if endpoint != 0:
        # Invoke the SageMaker endpoint
        response = runtime.invoke_endpoint(
            EndpointName=model_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
    else:
        response = local_predict(data=json.dumps(payload))

    # Format the prediction from SgeMaker Endpoint
    classes = ['non-cat', 'cat']
    prediction = classes[int(json.loads(response['Body'].read().decode('utf-8')))]

    # Return prediction and image
    figfile = BytesIO()
    plt.imsave(figfile, image, format='png')
    figfile.seek(0)
    figfile_png = base64.b64encode(figfile.getvalue()).decode('ascii')
    return render_template('results.html', image=figfile_png, prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)