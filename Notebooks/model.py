# Import necessary libraries
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

# Set logging
logging.getLogger().setLevel(logging.INFO)

# ---------------------------------------------------------------------------- #
#                            Training functions                                #
# ---------------------------------------------------------------------------- #

def train(channel_input_dirs, hyperparameters, hosts, num_gpus, output_data_dir, **kwargs):
    epochs = hyperparameters.get('epochs', 2500)
    optmizer = hyperparameters.get('optmizer', 'sgd')
    lr = hyperparameters.get('learning_rate', 75e-4)
    batch_size = hyperparameters.get('batch_size', 64)
    
    # Set Local vs. Distributed training
    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync' if num_gpus > 0 else 'dist_sync'

    # Set Context based on provided parameters
    ctx = mx.gpu() if num_gpus > 0 else mx.cpu()
    # Load Training/Testing Data
    f_path = channel_input_dirs['training']
    train_X, train_Y, test_X, test_Y = get_data(f_path)
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
    test_data = mx.gluon.data.DataLoader(
        mx.gluon.data.ArrayDataset(
            test_X,
            test_Y
        ),
        shuffle=False,
        batch_size=batch_size
    )
    
    # Initialize the neural network structure
    net = create_graph()
    
    # Parameter Initialization (Xavier)
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.34))
    
    # Optimizer
    trainer = gluon.Trainer(net.collect_params(), optmizer, {'learning_rate': lr})
    
    # Cross Entropy Loss Function
    softmax_ce = gluon.loss.SoftmaxCrossEntropyLoss()
    
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
                loss = softmax_ce(output, label)
                # Run backward pass
                loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
        # Accuracy Score
        val_accuracy = eval_acc(test_data, net, ctx)
        train_accuracy = eval_acc(train_data, net, ctx)
        results['epoch'+str(epoch)] = {}
        if epoch % 100 == 0:
            print("Epoch: {}; Loss: {}; Train-accuracy = {}; Validation-accuracy = {}"\
            .format(epoch,cumulative_loss/num_examples,train_accuracy,val_accuracy))
            results['epoch'+str(epoch)]['cost'] = cumulative_loss/num_examples
            results['epoch'+str(epoch)]['val_acc'] = val_accuracy
            results['epoch'+str(epoch)]['train_acc'] = train_accuracy
        elif epoch == epochs-1:
            print("Epoch: {}; Loss: {}; Train-accuracy = {}; Validation-accuracy = {}"\
            .format(epoch,cumulative_loss/num_examples,train_accuracy,val_accuracy))
            results['epoch'+str(epoch)]['cost'] = cumulative_loss/num_examples
            results['epoch'+str(epoch)]['val_acc'] = val_accuracy
            results['epoch'+str(epoch)]['train_acc'] = train_accuracy
            results['End'] = str(datetime.datetime.now())
            
    # Save the results
    print("Saving the training results ...")
    with open(str(output_data_dir)+'/results.json', 'w') as f:
        json.dump(results, f)

    # Return the model for saving
    return net
                
def create_graph():
    """
    Defines and Returns the Gluon Network Structure.
    """
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(56, activation='relu'))
        net.add(gluon.nn.Dense(20, activation='relu'))
        net.add(gluon.nn.Dense(7, activation='relu'))
        net.add(gluon.nn.Dense(5, activation='relu'))
        net.add(gluon.nn.Dense(2, activation='relu'))
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

def save(net, model_dir):
    """
    Saves the trained model to S3.
    
    Arguments:
    model -- The model returned from the `train()` function.
    model_dir -- The model directory location to save the model.
    """
    print("Saving the trained model ...")
    y = net(mx.sym.var('data'))
    y.save('%s/model.json' % model_dir)
    net.collect_params().save('%s/model.params' % model_dir)

def get_data(f_path):
    """
    Retrieves and loads the training/testing data from S3.
    
    Arguments:
    f_path -- Location for the training/testing input dataset.
    
    Returns:
    Pre-processed training and testing data along with training and testing labels.
    """
    train_x = np.load(f_path+'/train_X.npy')
    train_y = np.load(f_path+'/train_Y.npy')
    train_X, train_Y = transform(train_x, train_y)
    test_x = np.load(f_path+'/test_X.npy')
    test_y = np.load(f_path+'/test_Y.npy')
    test_X, test_Y = transform(test_x, test_y)
    return train_X, train_Y, test_X, test_Y

# Evlauation metric
def eval_acc(data_iterator, net, ctx):
    """
    Evaluates the Accuracy Score based on the input data and the results of the network.

    Arguments:
    data_iterator -- input data and associated label
    net -- Gluon model
    ctx -- Gluon memory context

    Returns:
    Accuracy score
    """
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

# ---------------------------------------------------------------------------- #
#                           Hosting functions                                  #
# ---------------------------------------------------------------------------- #

def model_fn(model_dir):
    """
    Load the Gluon model for hosting.

    Arguments:
    model_dir -- SageMaker model directory.

    Retuns:
    Gluon model
    """
    # Load the saved Gluon model
    symbol = mx.sym.load('%s/model.json' % model_dir)
    outputs = mx.sym.sigmoid(data=symbol, name='sigmoid_label')
    inputs = mx.sym.var('data')
    param_dict = gluon.ParameterDict('model_')
    net = gluon.SymbolBlock(outputs, inputs, param_dict)
    net.load_params('%s/model.params' % model_dir, ctx=mx.cpu())
    return net

def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform input data into prediction result.

    Argument:
    net -- Gluon model loaded from `model_fn()` function.
    data -- Input data from the `InvokeEndpoint` request.
    input_content_type -- Content type of the request (JSON).
    output_content_type -- Desired content type (JSON) of the repsonse.
    
    Returns:
    JSON payload of the prediction result and content type.
    """
    # Parse the data
    parsed = loads(data)
    # Convert input to MXNet NDArray
    nda = mx.nd.array(parsed)
    output = net(nda)
    prediction = nd.argmax(output, axis=1)
    response_body = dumps(prediction.asnumpy().tolist()[0])
    return response_body, output_content_type