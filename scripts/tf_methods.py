import tensorflow as tf
import numpy as np

def read_data(filename):
    """
    Reads from a CSV file,
    A row contains d+1 Attributes,
    d=feature vector dimension, last column=class label
    Also, does one hot encoding of the labels.
    Outputs a tuple of features and labels
    """
    with open(filename) as F:
        ip_data = [lines.rstrip() for lines in F]
    "Split by comma and save it in an nparray"
    ip_data = np.array([list(map(float,ip_data[i].split(","))) for i in range(len(ip_data))])
    M,N = np.shape(ip_data)
    feats = np.array([ip_data[i][0:N-1] for i in range(M)])
    labels = np.array([int(ip_data[i][N-1]) for i in range(M)])
    sess = tf.Session()
    NUM_CLASSES = len(np.unique(labels))
    labels = sess.run(tf.one_hot(labels,NUM_CLASSES))
    #Return a tuple containing features and corresponding labels
    return (feats,labels)


def splice_feats(features, NUM_CONTEXT):
    """
    Adds (left and right) context to the input features.
    The number of frames to include is
    defined by NUM_CONTEXT.
    Returns an nparray of features with context
    """
    if NUM_CONTEXT==0:
        return features
    else:
        extended_features = np.array([])
        n,m = np.shape(features)
        features_with_context = np.zeros((n,m*(2*NUM_CONTEXT+1)))
        temp = np.zeros((NUM_CONTEXT*m)).reshape(NUM_CONTEXT,m)
        extended_features = np.concatenate((temp,features,temp))

        for i in range(n):
            features_with_context[i] = extended_features[i:i+(2*NUM_CONTEXT+1)].reshape(1,(2*NUM_CONTEXT+1)*m)

        return features_with_context

def get_classification_accuracy(network_outputs, true_labels):
    """
    Computes the accuracy_score.
    Returns an accuracy percentage value
    """
    number_correct=0.0
    for label_ind in range(0, len(true_labels)):
        if true_labels[label_ind][np.argmax(network_outputs[label_ind],0)]==1:
            number_correct = number_correct+1
    return (100*(number_correct/len(true_labels)))


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def gaussian_noise_layer(input_tensor, std):
    """ Additive white gaussian_noise_layer to input_tensor
    with standard deviation = std"""
    noise = tf.random_normal(shape=tf.shape(input_tensor), mean=0.0, stddev=std, dtype=tf.float32)
    return input_tensor+noise

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.sigmoid):
    """
    Creates a layer with given dimensions and outputs an activation.
    Does matrix multiply (input tensor and weights), add (biases) and
    uses sigmoid to non linearize.
    It uses name scopes to make the graph readable, also writes a lot of
    summaries.
    """
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            weights = tf.Variable(tf.random_normal([input_dim, output_dim]))
            variable_summaries(weights)
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([output_dim]))
            variable_summaries(biases)
        with tf.name_scope("activations"):
            activations = act(tf.matmul(input_tensor,weights) + biases)
            tf.summary.histogram('activations', activations)
    return activations

def auto_encoder(dimensions,tied_weights=True, NUM_CONTEXT=0, optimizer = tf.train.AdamOptimizer(0.01), pretrain=False):
    """
    Creates an auto_encoder,
    number of layers (in encoding part)
    given by the dimensions(list).
    By default, it has tied weights.
    The function returns a dictionary of
    input tensor, weights and biases of the layers, loss
    """

    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
    stddev = np.random.uniform(0,1.5)
    current_input = gaussian_noise_layer(x, stddev)
    x_corrupt = current_input

    #Encoder
    encoder=[]
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        W = tf.Variable(tf.random_normal([n_input, n_output]), tf.float32)  #Weights
        b = tf.Variable(tf.zeros([n_output]), tf.float32)                   #Biases
        encoder.append([W,b])
        output = tf.nn.sigmoid(tf.matmul(current_input, W) +b)
        current_input = output

    # latent representation
    z = current_input
    encoder.reverse()

    #Decoder
    decoder = []
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.Variable(tf.random_normal((encoder[layer_i][0]).get_shape().as_list()), tf.float32)
        b = tf.Variable(tf.zeros(n_output), tf.float32)
        if (tied_weights):
            W = tf.transpose(encoder[layer_i][0])
        decoder.append([W,b])
        output = tf.nn.sigmoid(tf.matmul(current_input, W) + b)
        current_input = output

    # Reconstructed output
    y = current_input
    encoder.reverse()

    I = tf.placeholder(tf.float32,[None, dimensions[0]*(2*NUM_CONTEXT+1)], name='I')
    dec_W = tf.placeholder(tf.float32)
    if pretrain:
        O = tf.nn.sigmoid(tf.matmul(y, dec_W))

    #loss function
    with tf.name_scope("loss"):
        split_inputs = tf.split(x, 2*NUM_CONTEXT+1, 1)
        split_outputs = tf.split(y, 2*NUM_CONTEXT+1, 1)
        if pretrain:
            split_inputs = tf.split(I, 2*NUM_CONTEXT+1, 1)
            split_outputs = tf.split(O, 2*NUM_CONTEXT+1, 1)
        center_frame_x = split_inputs[NUM_CONTEXT]
        center_frame_y = split_outputs[NUM_CONTEXT]
        loss = tf.reduce_mean(tf.square(center_frame_y-center_frame_x)) # Reduce mean squared error

    train = optimizer.minimize(loss)

    return ({'x': x, 'I': I, 'dec_W': dec_W, 'encoder':encoder, 'decoder': decoder, "y": y, "loss":loss, "train":train, "z":z})


def auto_encoder_gain_mask(dimensions,tied_weights=True, NUM_CONTEXT=0, optimizer = tf.train.AdamOptimizer(0.01), pretrain=False, lamb=0, binarize_gain=False):
    """
    Creates an auto_encoder,
    number of layers (in encoding part)
    given by the dimensions(list).
    By default, it has tied weights.
    layer_wise training: Given the encoder weights,
    keep them non trainable and add a hidden layer
    that is trainable.
    The function returns a dictionary of
    input tensor, weights and biases of the layers, loss
    """

    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
    stddev = np.random.uniform(0,1.5)
    noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=stddev, dtype=tf.float32)
    x_corrupt = x+noise
    current_input = x_corrupt
    #Encoder
    encoder=[]
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        W = tf.Variable(tf.random_normal([n_input, n_output]), tf.float32)  #Weights
        b = tf.Variable(tf.zeros([n_output]), tf.float32)                   #Biases
        encoder.append([W,b])
        output = tf.nn.sigmoid(tf.matmul(current_input, W) +b)
        current_input = output

    # latent representation
    z = current_input
    encoder.reverse()

    #Decoder
    decoder = []
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.Variable(tf.random_normal((encoder[layer_i][0]).get_shape().as_list()), tf.float32)
        b = tf.Variable(tf.zeros(n_output), tf.float32)
        if (tied_weights):
            W = tf.transpose(encoder[layer_i][0])
        decoder.append([W,b])
        output = (tf.matmul(current_input, W) + b)
        current_input = output

    # Reconstructed output
    y = current_input
    encoder.reverse()

    gain = tf.abs(x)/(tf.abs(x)+tf.abs(noise))  #Oracle gain
    if binarize_gain:
        gain_thres = 0.7
        gain = (gain>gain_thres)
        gain = tf.cast(gain,tf.float32)

    #loss function
    with tf.name_scope("loss"):
        split_inputs = tf.split(x, 2*NUM_CONTEXT+1, 1)
        split_gain = tf.split(gain, 2*NUM_CONTEXT+1, 1)
        split_outputs = tf.split(y, 2*NUM_CONTEXT+1, 1)
        if pretrain:
            split_inputs = tf.split(I, 2*NUM_CONTEXT+1, 1)
            split_outputs = tf.split(O, 2*NUM_CONTEXT+1, 1)
        center_frame_x = split_inputs[NUM_CONTEXT]
        center_frame_y = split_outputs[NUM_CONTEXT]
        center_frame_gain = split_gain[NUM_CONTEXT]
        loss = tf.reduce_mean(tf.abs(center_frame_y-center_frame_gain) ) # Reduce mean squared error

        regularizers = 0
        for i in range(len(encoder)):
            regularizers = regularizers + tf.nn.l2_loss(encoder[i][0])
        for i in range(len(decoder)):
            regularizers = regularizers + tf.nn.l2_loss(decoder[i][0])

        loss = tf.reduce_mean(loss + lamb*regularizers)

    train = optimizer.minimize(loss)

    return ({'x': x,'gain': center_frame_gain, 'encoder':encoder, 'decoder': decoder, "y": y, "loss":loss, "train":train, "z":z})

def fine_tune_layers(input_dim, encoder_val, decoder_val, tied_weights=True, NUM_CONTEXT=0, act=tf.nn.sigmoid, optimizer = tf.train.AdamOptimizer(0.01)):
    """
    Construct a graph based on the weights of the encoder and decoder
    and fine tune them.
    Returns a dictionary of input tensor, latent_representation and Reconstructed output
    """
    x = tf.placeholder(tf.float32, [None, input_dim], name='x')
    current_input = x
    encoder = []
    for i in range(len(encoder_val)):
        W = tf.Variable(encoder_val[i][0])
        b = tf.Variable(encoder_val[i][1])
        encoder.append([W,b])
        print("Encoder", i, [W.get_shape(), b.get_shape()])
        hidden_output = act(tf.matmul(current_input, W) + b)
        current_input = hidden_output

    # latent_representation
    z = current_input
    encoder.reverse()

    decoder = []
    for i in range(len(decoder_val)):
        #W = tf.Variable(decoder_val[i][0].T)

        if (tied_weights):
            W = tf.transpose(encoder[i][0])
        else:
            W = tf.Variable(tf.random_normal(np.shape(decoder_val[i][0].T)))
        b = tf.Variable(decoder_val[i][1])
        print("Decoder", i, [W.get_shape(), b.get_shape()])
        hidden_output = act(tf.matmul(current_input, W)+b)
        decoder.append([W,b])
        current_input = hidden_output

    #output
    y = current_input
    encoder.reverse()

    #loss
    split_inputs = tf.split(x, 2*NUM_CONTEXT+1, 1)
    center_frame = split_inputs[NUM_CONTEXT]
    split_outputs = tf.split(y, 2*NUM_CONTEXT+1, 1)
    center_frame_y = split_outputs[NUM_CONTEXT]
    loss = tf.reduce_mean(tf.square(center_frame_y-center_frame)) # Reduce mean squared error

    #optimizer
    train = optimizer.minimize(loss)


    return ({"x":x, "loss":loss, "train": train, "encoder":encoder, "decoder":decoder, "z": z, "y":y})

def latent_representation(encoder, act=tf.nn.sigmoid):
    """
    Calculates the intermediate latent_representation given the encoder weights.
    Useful when doing layer wise pretraining.
    Returns a numpy array which can be given as input for training the
    next hidden layer.
    """

    x = tf.placeholder(tf.float32)
    current_input = x
    sess = tf.Session()
    for i in range(len(encoder)):
        W, b = encoder[i]
        output = act(tf.matmul(W, current_input) + b)
        current_input = output

    return ({'x': x, 'latent':sess.run(current_input)})              # Returns a numpy ndarray

def restore_graph(experiment_folder, model, key):
    """
    Restores the weights and biases from a model.
    It is assumed that the model is stored inside the experiment_folder/saved_models/
    folder.
    Arguments:
    experiment_folder - a string containing absolute path to the current experiment_folder
    model - string having the model name (eg: model-100)
    key - String corresponding to the layer name (eg: Layer1)
    """
    model_path = experiment_folder+"/saved_models/"+model
    with tf.Session() as sess:
        #Restore weights
        print ('Restore '+ key+' weights')
        saver = tf.train.import_meta_graph(model_path+'.meta')
        saver.restore(sess,model_path)
        W1 = tf.get_collection(key+'_E')[0] #Encoder weights
        b1 = tf.get_collection(key+'_E')[1] #Encoder biases
        Wd1 = tf.get_collection(key+'_D')[0] #Decoder weights
        bd1 = tf.get_collection(key+'_D')[1]    #Decoder weights
        [W1,b1, Wd1, bd1] = sess.run([W1,b1, Wd1, bd1])
        return [W1,b1,Wd1,bd1]
