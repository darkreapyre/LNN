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