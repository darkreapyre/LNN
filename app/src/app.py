#!/usr/bin/python
import os
import json
import boto3
import tempfile
import urllib
import scipy
import botocore
import base64
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, Response, request, jsonify, render_template
from json import dumps, loads
from boto3 import client, resource, Session
from PIL import Image
from scipy import ndimage, misc


# Global variables
bucket = 'lnn'
s3_client = client('s3', region_name='us-west-2') # S3 access
s3_resource = resource('s3')

# Helper Functions
def s3numpy(bucket, key):
    """
    Retrieves a saved numpy array from S3 without using a local copy
    
    Arguments:
    bucket -- Name of the S3 Bucket where the array is stored
    key -- name of the saved numpy file. Must be in a string format
    
    Returns:
    array -- Numpy array
    """
    file = BytesIO(
        s3_client.get_object(
            Bucket=bucket,
            Key='predict_input/'+key)['Body'].read()
        )
    content = file.getvalue()
    array = np.load(BytesIO(content))
    return array

def sigmoid(z):
    """
    Computes the sigmoid of z
    
    Arguments:\
    z -- A scalar or numpy array of any size
    
    Returns:
    s -- sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))
    
    return s

# Prediciton funciton
def predict(w, b, X):
    """
    Predict wether the label is 1 or 0 (cat vs. non-cat)
    
    Arguments:
    w -- weights from trained model on S3, (num_px * num_px, 3, 1)
    b -- scalar representing the bias from trained model from S3
    X -- Input reshaped image (1, num_px * num_px * 3).T
    
    Returns:
    Y -- Numpy array containing predeictions of 1 or 0
    """
    m = X.shape[1]
    Y_pred = np.zeros((1, m))
    classes = ("non-cat", "cat")
    
    # Compute the vector `A` predicting the probabilities of a cat
    A = sigmoid(np.dot(w.T, X) + b)
    
    # Convert probabilities to predictions
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_pred[0, i] = 1
        else:
            Y_pred[0, i] = 0
    assert(Y_pred.shape == (1, m))

    output = str(classes[int(np.squeeze(Y_pred))])

    return output

app = Flask(__name__)

@app.route('/')
def index():
    resp = Response(response="Success",
         status=200, \
         mimetype="application/json")
    return (resp)

@app.route('/image')
def image():
    print('api')
    url = request.args.get('image')
    print(url)

    # Get data from S3
    w = s3numpy(bucket, 'weights')
    b = s3numpy(bucket, 'bias')

    # Pre-process the image
    req = urllib.request.Request(url)
    res = urllib.request.urlopen(req).read()
    fname = BytesIO(res)
    img = np.array(ndimage.imread(fname, flatten=False))
    #img = np.array(ndimage.imread(fname, mode='RGB', flatten=False))
    #img = np.array(misc.imread(fname, flatten=False))
    image = misc.imresize(img, size=(64, 64)).reshape((1, 64*64*3)).T

    # Prediction
    prediction = predict(w, b, image)

    # Return prediction and image
    figfile = BytesIO()
    plt.imsave(figfile, img, format='png')
    figfile.seek(0)
    figfile_png = base64.b64encode(figfile.getvalue()).decode('ascii')
    # Remove byst string formatting
    #result = str(figfile_png)[2:-1]

    return render_template('results.html', image=figfile_png, prediction=prediction)

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)