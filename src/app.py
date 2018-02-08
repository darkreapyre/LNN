#!/usr/bin/python
import os
import json
import boto3
import tempfile
import urllib
import scipy
import botocore
import base64
import h5py
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from flask import Flask, Response, request, jsonify, render_template
from json import dumps, loads
from boto3 import client, resource, Session
from PIL import Image
from scipy import ndimage, misc
from skimage import transform

def sigmoid(z):
    """
    Computes the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size

    Return:
    sigmoid(z)
    """
    return 1. / (1. + np.exp(-z))

def relu(z):
    """
    Implement the ReLU function.

    Arguments:
    z -- Output of the linear layer, of any shape

    Returns:
    a -- Post-activation parameter, of the same shape as z
    """
    a = np.maximum(0, z)
    # Debug statement
    #assert(a.shape == z.shape)
    return a

def forward_prop(X, params, layers, activations):
    """
    Executes the Forward Propogation step.
    
    Arguments:
    input -- Input data of shape (No. features, No. training examples)
    params -- Trained Weights and Bias
    layers -- Numer of hidden layers in the trained neural network
    activations -- Activation function for each layer
    """
    A_prev = X
    results = []
    results.append({'A': A_prev})
    for l in range(1, layers+1):
        W = params['W'+str(l)]
        b = params['b'+str(l)]
        Z = np.dot(W, A_prev) + b
        if activations['layer'+str(l)] == 'relu':
            A_prev = relu(Z)
        if activations['layer'+str(l)] == 'sigmoid':
            A_prev = sigmoid(Z)
        cache = {'A': A_prev}
    results.append(cache)
    output = results[-1]['A']
    return output

def predict(X, NN_parameters, trained_parameters):
    """
    Applies the Forward Propogation step with optmized paramters to the input data.
    Compares the output to the labeled data to determine classification with a
    decision boundary of 0.5.
    
    Arguments:
    X -- Input data of any shape
    NN_parameters -- Neural Network configuration
    trained_parameters -- Numpy array of optmized weights and bias
    
    Returns:
    Y_pred -- The predicted label
    """
    # Run Forward Propagation on the input data
    layers = NN_parameters['layers']
    activations = NN_parameters['activations']
    output = forward_prop(X, trained_parameters, layers, activations)
    decision_boundry = np.vectorize(lambda x: 1 if x > 0.5 else 0)
    Y_pred = decision_boundry(output)
    return Y_pred

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

    y = [1] # Truth Label for cat image
    classes = ("NON-CAT", "CAT")
    # Open Neural Network parameters file
    with open("/app/parameters.json","r") as f:
        NN_parameters = json.load(f)

    # Open Model parameters file
    with h5py.File('/app/params.h5', 'r') as h5file:
        trained_parameters = {}
        for key, item in h5file['/'].items():
            trained_parameters[key] = item.value

    # Pre-process the image
    req = urllib.request.Request(url)
    res = urllib.request.urlopen(req).read()
    fname = BytesIO(res)
    """
    Note: This code is the tested code on Jupyter. Switching to 
    original version to work with Flask.
   
    #img = np.array(ndimage.imread(fname, flatten=False))
    img = plt.imread(fname)
    #image = misc.imresize(img, size=(64, 64)).reshape((1, 64*64*3)).T
    image = transform.resize(image, (64, 64), mode='constant').reshape((64 * 64 * 3, 1))
    """
    ### Origional Code ###
    img = np.array(ndimage.imread(fname, flatten=False))
    image = misc.imresize(img, size=(64, 64)).reshape((1, 64 * 64 * 3)).T

    # Prediction
    Y_pred = predict(image, NN_parameters, trained_parameters)
    prediction = str(classes[int(np.squeeze(Y_pred))])

    # Return prediction and image
    figfile = BytesIO()
    plt.imsave(figfile, img, format='png')
    figfile.seek(0)
    figfile_png = base64.b64encode(figfile.getvalue()).decode('ascii')
    # Remove byst string formatting
    #result = str(figfile_png)[2:-1]

    return render_template('results.html', image=figfile_png, prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)