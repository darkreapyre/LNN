# Import necessary libraries
import os
import logging
import h5py
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------- #
#                            Training functions                                #
# ---------------------------------------------------------------------------- #

def train(channel_input_dirs, hyperparameters, hosts):
    epochs = hyperparameters.get('epochs', 2500)
    learning_rate = hyperparameters.get('learning_rate', 75e-4)
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
    # Load Training Data
    f_path = channel_input_dirs

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

def save(model, model_dir):
    """
    Saves the trained model to S3.
    
    Arguments:
    model -- The model returned from the `train()` function.
    model_dir -- The model directory location to save the model.
    """
    # Create aplaceholder
    n = net(mx.sym.var('data'))
    # Save the model
    n.save('%s/model.json' % model_dir)
    # Save the Optimized Parameters
    net.collect_params().save('%s/model.params' % model_dir)

def accuracy(data_iterator, net, Y):
    """
    Evaluates the prediction against a test label.
    
    Arguments:
    data_iterator -- Training data.
    net -- Gluon network.
    Y -- True label.
    
    Returns:
    Accuracy Score.
    """
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        output = net(data)
        decision_boundary = np.vectorize(lambda x: 1 if x > 0.5 else 0)
        y_pred = list(decision_boundary(output.asnumpy()).flat)
        Y = list(Y.flat)
    return accuracy_score(Y, y_pred)

"""
def get_data(f_path, channel):
    """
    Retrieves and loads the training data from S3.
    
    Arguments:
    f_path -- Location for the training input dataset.
    
    Returns:
    Pre-processed training and testing data along with training and testing labels.
    """
    # Import the Datasets
    dataset = h5py.File(f_path, 'r')
    train_set_x_orig = np.array(dataset['train_set_x'][:])
    train_set_y_orig = np.array(dataset['train_set_y'][:])
    test_set_x_orig = np.array(dataset['test_set_x'][:])
    test_set_y_orig = np.array(dataset['test_set_y'][:])
    # Pre-Process the data
    train_X, train_Y = transform(train_set_x_orig, train_set_y_orig)
    test_X, test_Y = transform(test_set_x_orig, test_set_y_orig)
    return train_X, train_Y, test_X, test_Y
"""
    
    
    
    
    
    