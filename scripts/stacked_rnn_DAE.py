"""
Implements a stacked LSTM based DAE.
Input: vocal + background
Output: vocals (or an ideal binary mask)

"""
import tensorflow as tf
import numpy as np
import os, sys
from tf_methods import *
from tensorflow.contrib import rnn
import pdb

# Network Parameters
n_input = 513 # Spectrogram
num_context = 5
n_steps = 2*num_context + 1 # timesteps
n_hidden = 512 # hidden layer num of nodes
n_classes = n_input # reconstruct input without background accompaniment
learning_rate = 0.001
num_epoch = 10
batch_size = 128

experiments_folder='/Users/RupakVignesh/Desktop/fall17/7100/Singing-Voice-Separation/experiments/expt1'
summaries_dir=experiments_folder+'/summaries'

# tf Graph input
print("Building computational graph")
x = tf.placeholder("float", [None, n_steps, n_input])
background = tf.placeholder("float")
clean_vocals = tf.placeholder("float")

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}



def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, weights, biases)

# Time-frequency mask
with tf.name_scope("Gain_mask"):
    gain = tf.abs(clean_vocals)/(tf.abs(clean_vocals)+tf.abs(background))
    current_frame_gain = tf.split(gain, 2*num_context+1, 1)[num_context]
# Define loss and optimizer
with tf.name_scope("Loss"):
    loss = tf.reduce_mean(tf.square(pred - current_frame_gain))

loss1 = tf.summary.scalar("loss1", loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# Initializing the variables
init = tf.global_variables_initializer()

print("Reading Data")
train_feats, _ = read_data(sys.argv[1])         # Clean Vocals
noise_feats_train, _ = read_data(sys.argv[2])   # Background track
# valid_feats, _ = read_data(sys.argv[3])
# noise_feats_valid, _ = read_data(sys.argv[4])

print("Add context")
train_feats = splice_feats(train_feats, num_context)
noise_feats_train = splice_feats(noise_feats_train, num_context)


print "Training Neural Network"
num_train_instance, _= np.shape(train_feats)

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summaries_dir+'/train', sess.graph)
    # valid_writer = tf.summary.FileWriter(summaries_dir+'/valid')
    tf.global_variables_initializer().run()

    for i in range(num_epoch):
        sess_loss = 0.0
        #sess_output = np.zeros([num_train_instance, num_classes])

        for batch_ind in range(0,num_train_instance,batch_size):
            batch_x = train_feats[batch_ind:batch_ind+batch_size]+noise_feats_train[batch_ind:batch_ind+batch_size] #clean vocals + background
            batch_x = batch_x.reshape((np.shape(batch_x)[0], n_steps, n_input))
            _, batch_loss, train_summary = sess.run([train, loss, merged], feed_dict={x: batch_x, clean_vocals: train_feats[batch_ind:batch_ind+batch_size], background: noise_feats_train[batch_ind:batch_ind+batch_size]})
            sess_loss += batch_loss

        train_writer.add_summary(train_summary, i)
        # valid_summary = sess.run(merged, feed_dict={inputs:valid_feats, ground_truth_labels: valid_labels})
        # valid_writer.add_summary(valid_summary,i)

        if ((i+1)%10==0):
            saver.save(sess, experiments_folder+'/saved_models/model', global_step=i+1)

        print("Epoch "+str(i)+ " loss", sess_loss/(num_train_instance/batch_size))
