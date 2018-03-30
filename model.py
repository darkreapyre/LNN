# Import necessary libraries
from __future__ import print_function
import boto3
import os
import io
import logging
import datetime
import json
import mxnet as mx
import numpy as np
from json import dumps, loads
from mxnet import nd, autograd, gluon

# ---------------------------------------------------------------------------- #
#                            Training functions                                #
# ---------------------------------------------------------------------------- #

def train(channel_input_dirs, hyperparameters, hosts, num_gpus, **kwargs):
    epochs = hyperparameters.get('epochs', 2500)
    optmizer = hyperparameters.get('optmizer', 'sgd')
    lr = hyperparameters.get('learning_rate', 75e-4)
    batch_size = hyperparameters.get('batch_size', 64)
    # Set logging
    logging.getLogger().setLevel(logging.DEBUG)
    # Set Local vs. Distributed training
    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync' if num_gpus > 0 else 'dist_sync'
    # Set Context based on provided parameters
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    #print(channel_input_dirs)
    #print(os.listdir(channel_input_dirs['training']))
    # Load Training/Testing Data
    f_path = channel_input_dirs['training']
    train_X, train_Y = get_data(f_path)
    num_examples = train_X.shape[0]
    
    # Create Training and Test Data Iterators
    train_data = mx.gluon.data.DataLoader(
        mx.gluon.data.ArrayDataset(
            train_X,
            train_Y
        ),
        shuffle=True,
        batch_size=batch_size
    )
    
    # Initialize the network
    net = build_network()
    # Parameter Initialization (He .et al)
    net.collect_params().initialize(mx.init.MSRAPrelu())
    # Optimizer
    trainer = gluon.Trainer(net.collect_params(), optmizer, {'learning_rate': lr})
    # Cross Entropy Loss Function
    binary_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    
    # Start the Training loop
    results = {} # Track Loss function
    results['Start'] = str(datetime.datetime.now())
    for epoch in range(epochs):
        cumulative_loss = 0
        # Enumerate batches
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # Record for calculating derivatives for forward pass
            with autograd.record():
                output = net(data)
                loss = binary_ce(output, label)
            # Run backward pass
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
        results['epoch'+str(epoch)] = cumulative_loss/num_examples
        if epoch % 100 == 0:
            print("Epoch: {}; Loss: {}".format(epoch,cumulative_loss/num_examples))
        elif epoch == epochs-1:
            print("Epoch: {}; Loss: {}".format(epoch,cumulative_loss/num_examples))
            results['end'] = str(datetime.datetime.now())
    # Return the model for saving
    return net, results
                
def build_network():
    """
    Defines and Returns the Gluon Network Structure.
    """
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(20, activation='relu'))
        net.add(gluon.nn.Dense(7, activation='relu'))
        net.add(gluon.nn.Dense(5, activation='relu'))
        net.add(gluon.nn.Dense(1, activation='sigmoid'))
    net.hybridize()
    return net

def transform(x, y):
    """
    Pre-Processes the image data.
    
    Arguments:
    x -- Numpy Array of input images
    y -- Numpy Array of labels
    
    Returns:
    x -- Vectorized and scaled Numpy Array as a 32-bit float.
    y -- Numpy Array as a 32-bit float.
    """
    x = x.reshape((x.shape[0], (x.shape[1] * x.shape[2]) * x.shape[3]))
    return x.astype(np.float32) / 255, y.astype(np.float32)

def save(net, results, model_dir):
    """
    Saves the trained model to S3.
    
    Arguments:
    model -- The model returned from the `train()` function.
    model_dir -- The model directory location to save the model.
    """
    print("Saving the model in {}".format(model_dir))
    y = net(mx.sym.var('data'))
    y.save('%/model.json' % model_dir)
    net.collect_params().save('%s/model.params' % model_dir)
    with io.open(str(model_dir)+'/results.json', 'w', encoding='utf-8') as f:
        f.write(dumps(results, ensure_ascii=False))

def get_data(f_path):
    """
    Retrieves and loads the training/testing data from S3.
    
    Arguments:
    f_path -- Location for the training/testing input dataset.
    
    Returns:
    Pre-processed training and testing data along with training and testing labels.
    """
    X = np.load(f_path+'/train_X.npy')
    Y = np.load(f_path+'/train_Y.npy')
    train_X, train_Y = transform(X, Y)
    return train_X, train_Y

# ---------------------------------------------------------------------------- #
#                           Hosting functions                                  #
# ---------------------------------------------------------------------------- #

def model_fn(model_dir):
    """
    Load the Gluon model for hosting.

    Arguments:
    model_dir -- SAgeMaker model directory.

    Retuns:
    Gluon model
    """
    # Load the saved Gluon model
    symbol = mx.sym.load('%s/model.json' % model_dir)
    outputs = mx.sym.sigmoid(data=symbol, name='sigmoid_label')
    inputs = mx.sym.var('data')
    param_dict = gluon.ParameterDict('model_')
    net = gluon.SymbolBlock(outputs, inputs, param_dict)
    net.load_params('%s/model.params' % model_dir, ctx=mx.cou())
    return net

def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform input data into prediction result.

    Argument:
    net -- Gluon model loaded from `model_fn()` function.
    data -- Input data from the `InvokeEndpoint` request.
    input_content_type -- Content type of the request (JSON).
    output_content_type -- Disired content type (JSON) of the repsonse.
    
    Returns:
    JSON paylod of the prediction result and content type.
    """
    # Parse the data
    parsed = loads(data)
    # Convert input to MXNet NDArray
    nda = mx.nd.array(parsed)
    output = net(nda)
    prediction = (nd.sign(output) + 1) / 2
    response_body = dumps(prediction.asnumpy().tolist()[0])
    return response_body, output_content_type