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
from skimage import transform
from model import *

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
    classes = ("non-cat", "cat")
    # Open Model parameters file
    with h5py.File('params.h5', 'r') as h5file:
        parameters = {}
        for key, item in h5file['/'].items():
            parameters[key] = item.value

    # Pre-process the image
    req = urllib.request.Request(url)
    res = urllib.request.urlopen(req).read()
    fname = BytesIO(res)
    #img = np.array(ndimage.imread(fname, flatten=False))
    img = plt.imread(fname)
    #image = misc.imresize(img, size=(64, 64)).reshape((1, 64*64*3)).T
    image = transform.resize(image, (num_px, num_px), mode='constant').reshape((num_px * num_px * 3, 1))

    # Prediction
    Y_pred = predict(image, y, parameters)
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
    app.run('0.0.0.0', debug=True)