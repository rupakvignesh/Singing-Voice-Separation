"""
Implements a feed forward multilayer perceptron (3 hidden layers
with 1024 hidden units each). Input is polyphonic musical mix and outputs
are vocals and background accompaniment.
Author: Rupak Vignesh Swaminathan

"""
import tensorflow as tf
import numpy as np
import os, sys
from tf_methods import *
import pdb

# Network Parameters
n_input = 513 # Spectrogram
num_context = 1
n_steps = 2*num_context + 1 # timesteps
n_hidden = 1024 # hidden layer num of nodes
n_hidden2 = 1024
n_hidden3 = 1024
n_classes = n_input*n_steps # reconstruct input without background accompaniment
learning_rate = 0.01
num_epoch = 130
batch_size = 512

experiments_folder='/home/rvignesh/singing_voice_separation/experiments/expt11'
summaries_dir=experiments_folder+'/summaries'

# tf Graph input
print("Building computational graph")
background = tf.placeholder("float", [None, n_input*(2*num_context+1)], name='background')
clean_vocals = tf.placeholder("float", [None, n_input*(2*num_context+1)], name='clean_vocals')
x_mix = tf.placeholder("float", [None, n_input*(2*num_context+1)], name='mix') 

#first layer
weights1 = tf.Variable(tf.random_normal([n_input*n_steps, n_hidden]), tf.float32, name='weights1')
biases1 = tf.Variable(tf.zeros([n_hidden]), tf.float32, name='biases1')
hidden1_output = tf.nn.relu(tf.add(tf.matmul(x_mix ,weights1),biases1), name='hidden1_output')

#Second layer
weights2 = tf.Variable(tf.random_normal([n_hidden, n_hidden2]), tf.float32, name='weights2')
biases2 = tf.Variable(tf.zeros([n_hidden2]), tf.float32, name='biases2')
hidden2_output = tf.nn.relu(tf.add(tf.matmul(hidden1_output, weights2), biases2), name='hidden2_output')

#Third layer
weights3 = tf.Variable(tf.random_normal([n_hidden2, n_hidden3]), tf.float32, name='weights3')
biases3 = tf.Variable(tf.zeros([n_hidden3]), tf.float32, name='biases3')
hidden3_output = tf.nn.relu(tf.add(tf.matmul(hidden2_output, weights3), biases3), name='hidden3_output')

# output layer
weights4 = tf.Variable(tf.random_normal([n_hidden3, n_classes]), tf.float32, name='weights4')
biases4 = tf.Variable(tf.zeros([n_classes]), tf.float32, name='biases4')
weights5 = tf.Variable(tf.random_normal([n_hidden3, n_classes]), tf.float32, name='weights5')
biases5 = tf.Variable(tf.zeros([n_classes]), tf.float32, name='biases5')

voc_tf_pred = tf.nn.relu(tf.add(tf.matmul(hidden3_output,weights4), biases4))		#Time freq mask predictions for vocals
back_tf_pred = tf.nn.relu(tf.add(tf.matmul(hidden3_output,weights5), biases5))		#Time freq mask predictions for background

est_vocals = (voc_tf_pred/(voc_tf_pred + back_tf_pred + 0.001)) * x_mix			#Deterministic layer for computing magnitude spectra of vocals
est_background = (back_tf_pred/(voc_tf_pred + back_tf_pred + 0.001)) * x_mix		#Deterministic layer for computing magnitude spectra of background

est_vocals = tf.identity(est_vocals, name='estimated_vocals')
est_background = tf.identity(est_background, name='estimated_background')

with tf.name_scope("Loss"):
    loss = tf.reduce_mean(tf.square(est_vocals - clean_vocals) + tf.square(est_background - background))
    #print(current_frame_gain.get_shape(), "Current Frame Gain Shape")


loss1 = tf.summary.scalar("loss1", loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

print("Reading Data")
train_feats, _ = read_data(sys.argv[1])         # Clean Vocals
noise_feats_train, _ = read_data(sys.argv[2])   # Background track
valid_feats, _ = read_data(sys.argv[3])
noise_feats_valid, _ = read_data(sys.argv[4])

print("Add context")
train_feats_context = splice_feats(train_feats, num_context)
noise_feats_train_context = splice_feats(noise_feats_train, num_context)
valid_feats_context = splice_feats(valid_feats, num_context)
noise_feats_valid_context = splice_feats(noise_feats_valid, num_context)
print(np.shape(train_feats_context), "Train feats shape after slice feats")

print ("Training Neural Network")
num_train_instance, _= np.shape(train_feats_context)
saver = tf.train.Saver(max_to_keep=10)
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summaries_dir+'/train', sess.graph)
    valid_writer = tf.summary.FileWriter(summaries_dir+'/valid')
    tf.global_variables_initializer().run()

    for i in range(num_epoch):
        sess_loss = 0.0

	shuffle = np.random.permutation(len(noise_feats_train_context))			#Randomize background to virtually create more data (augmentation technique)
        for batch_ind in range(0,num_train_instance,batch_size):
            batch_noise = noise_feats_train_context[shuffle[batch_ind:batch_ind+batch_size]]
            batch_vocals = train_feats_context[batch_ind:batch_ind+batch_size]
            batch_mix = batch_noise + batch_vocals
            
            _, batch_loss, train_summary = sess.run([train, loss, merged], feed_dict={clean_vocals: batch_vocals, background: batch_noise, x_mix: batch_mix})
            sess_loss += batch_loss
        train_writer.add_summary(train_summary, i)
        valid_summary, valid_loss = sess.run([merged, loss], feed_dict={clean_vocals: valid_feats_context, background: noise_feats_valid_context, x_mix: valid_feats_context+noise_feats_valid_context})
        valid_writer.add_summary(valid_summary,i)

        saver.save(sess, experiments_folder+'/saved_models/model', global_step=i+1)

        print("Epoch "+str(i)+ " Train loss", sess_loss/(num_train_instance/batch_size), "Valid loss", valid_loss)

