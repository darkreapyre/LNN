# Import necessary libraries
from __future__ import print_function
import os
import logging
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
"""
Note: There is no sklearn and h5py support

import h5py
from sklearn.metrics import accuracy_score
"""

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
    print(channel_input_dirs)
    print(os.listdir(channel_input_dirs['training']))
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
    # Initialize the network
    net = build_network()
    # Parameter Initialization
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24))
    # Optimizer
    trainer = gluon.Trainer(net.collect_params(), optmizer, {'learning_rate': lr})
    # Cross Entropy Loss Function
    binary_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    
    # Start the Training loop
    costs = [] # Track Loss function
    for epoch in range(epochs):
        cumulative_loss = 0
        # Enumerate batches
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx)
            label = label.as_in_context(model_ctx)
            # Record for calculating derivatives for forward pass
            with autograd.record():
                output = net(data)
                loss = binary_ce(output, label)
            # Run backward pass
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
        #test_accuracy = accuracy(test_data, net, test_Y)
        costs.append(cumulative_loss/num_examples)
        if epoch % 100 == 0:
            #print("Epoch: {}; Loss: {}; Test Set Accuracy: {}"\
            #      .format(epoch,cumulative_loss/num_examples,test_accuracy))
            print("Epoch: {}; Loss: {}".format(epoch,cumulative_loss/num_examples))
        elif epoch == epochs-1:
            #print("Final Epoch: {}; Final Loss: {}; Final Test Set Accuracy: {}"\
            #      .format(epoch,cumulative_loss/num_examples,test_accuracy))
            print("Epoch: {}; Loss: {}".format(epoch,cumulative_loss/num_examples))
    # Return the model for saving
    return net
                

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

def save(net, model_dir):
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

def accuracy(data_iterator, net, Y):
    """
    Evaluates overall accuracy the prediction against a test label.
    
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


def get_data(f_path):
    """
    Retrieves and loads the training/testing data from S3.
    
    Arguments:
    f_path -- Location for the training/testing input dataset.
    
    Returns:
    Pre-processed training and testing data along with training and testing labels.
    """
    train_X = np.load(os.path.join(f_path,'train/train_X.npy'))
    train_Y = np.load(os.path.join(f_path,'train/train_Y.npy'))
    train_X, train_Y = transform(train_X, train_Y)
    test_X = np.load(os.path.join(f_path,'test/test_X.npy'))
    test_Y = np.load(os.path.join(f_path,'test/vtest_Y.npy'))
    test_X, test_Y = transform(test_X, test_Y)
    return train_X, train_Y, test_X, test_Y
    
    
    
    