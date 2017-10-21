# Calling function
payload['s3_bucket'], \
payload['input_data'], \
payload['data_dimentions'], \
payload['params'] = initialize_data(w=parameters['weight'], b=parameters['bias'])

def initialize_data(w, b):
    """
    Extracts the training and testing data from S3, flattens,
    standardizes, dumps back to S3 for neurons to process as layer a^0
    
    Returns:
    bucket -- name of the S3 bucket of Numpy arrays
    a_names -- list of the Numpy array names
    dims -- dimensions of each of the data sets
    params -- dictionary of the weight and bias parameters
    """
    
    # Load main dataset\n",
    dataset = h5py.File('/tmp/datasets.h5', 'r')
    
    # Create numpy arrays from the various h5 datasets
    train_set_x_orig = np.array(dataset['train_set_x'][:]) # train set features
    train_set_y_orig = np.array(dataset['train_set_y'][:]) # train set labels
    test_set_x_orig = np.array(dataset['test_set_x'][:]) # test set features
    test_set_y_orig = np.array(dataset['test_set_y'][:]) # test set labels
    classes = np.array(dataset['list_classes'][:]) # the list of classes
    
    # Reshape labels
    train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    # Preprocess inputs
    train_set_x = standardize(train_set_x_orig)
    test_set_x = standardize(test_set_x_orig)
    
    # Dump the inputs to the temporary s3 bucket for TrainerLambda
    bucket = storage_init() # Creates a temporary bucket for the propogation steps
    a_list = [train_set_x, train_set_y, test_set_x, test_set_y, classes] # List of Numpy array names
    a_names = [] # list to store the Numpy array names as strings
    dims = {} # Dictionary of input data dimensions
    for i in range(len(a_list)):
        # Create a lis of the names of the numpy arrays
        a_names.append(name2str(a_list[i], locals()))
    for j in range(len(a_list)):
        # Save the Numpy arrays to S3
        numpy2s3(array=a_list[j], name=a_names[j][0], bucket=bucket)
        dims[str(a_names[j][0])] = a_list[j].shape
    
    # Initialize weights and bias data
    if w == 0: # Initialize weights to dimensions of the input data
        dim = dims.get('train_set_x')[0]
        weights = np.zeros((dim, 1))
        # Store the initial weights as a column vector on S3
        numpy2s3(weights, name='weights', bucket=bucket)
    else:
        #placeholder for random weight initialization
        pass
        
    # Initialize Bias
    if b != 0:
        #placeholder for random bias initialization
        pass
    else:
        numpy2s3(b, name='bias', bucket=bucket)
    
    # Create the initial paramaters for `TrainerLambda`
    params = {
        'w': w,
        'b': b
    }   
    
    # Return the bucket, numpy array, dimensions and params
    return bucket, [j for i in a_names for j in i], dims, params