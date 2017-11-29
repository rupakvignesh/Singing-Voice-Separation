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
n_hidden = 1024 # hidden layer num of nodes
n_hidden2 = 1024
n_hidden3 = 1024
n_classes = n_input # reconstruct input without background accompaniment
learning_rate = 0.01
num_epoch = 25
batch_size = 128

experiments_folder='/home/rvignesh/singing_voice_separation/experiments/expt7'
summaries_dir=experiments_folder+'/summaries'

# tf Graph input
print("Building computational graph")
#x = tf.placeholder("float", [None, n_steps, n_input])
x = tf.placeholder(tf.float32, [None, n_steps*n_input])
background = tf.placeholder("float")
clean_vocals = tf.placeholder("float")

# Define weights
#weights = {
#    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
#}
#biases = {
#    'out': tf.Variable(tf.zeros([n_classes]))
#}



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
    return tf.nn.softmax(tf.matmul(outputs[-1], weights['out']) + biases['out'])


#first layer
weights1 = tf.Variable(tf.random_normal([n_input*n_steps, n_hidden]))
biases1 = tf.Variable(tf.zeros([n_hidden]))
hidden1_output = tf.nn.relu(tf.matmul(x ,weights1) + biases1)

#Second layer
weights2 = tf.Variable(tf.random_normal([n_hidden, n_hidden2]))
biases2 = tf.Variable(tf.zeros([n_hidden2]))
hidden2_output = tf.nn.relu(tf.matmul(hidden1_output, weights2) + biases2)

# output layer
weights4 = tf.Variable(tf.random_normal([n_hidden2, n_classes]))
biases4 = tf.Variable(tf.zeros([n_classes]))
pred = (tf.matmul(hidden2_output,weights4)+biases4)

#pred = RNN(x, weights, biases)
print(pred.get_shape().as_list(),"RNN Output shape")
# Time-frequency mask
with tf.name_scope("Gain_mask"):
    gain = tf.abs(clean_vocals)/(tf.abs(clean_vocals)+tf.abs(background))
    current_frame_gain = tf.split(gain, n_steps, 1)[num_context]
# Define loss and optimizer
with tf.name_scope("Loss"):
    loss = tf.reduce_mean(tf.square(pred - current_frame_gain))

loss1 = tf.summary.scalar("loss1", loss)
train = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
print("Reading Data")
train_feats, _ = read_data(sys.argv[1])         # Clean Vocals
noise_feats_train, _ = read_data(sys.argv[2])   # Background track
mix_feats = train_feats + noise_feats_train
# valid_feats, _ = read_data(sys.argv[3])
# noise_feats_valid, _ = read_data(sys.argv[4])

#print("apply log")
#train_feats = np.log(1+train_feats)
#noise_feats_train = np.log(1+noise_feats_train)
#mix_feats = np.log(1+mix_feats)

#print("Normalize[0-1]")


#print("Normalizing")
#train_mean = np.mean(train_feats,axis=0)
#train_std = np.std(train_feats,axis=0)
#train_feats = (train_feats - train_mean)/(train_std)
#back_mean = np.mean(noise_feats_train,axis=0)
#back_std = np.std(noise_feats_train,axis=0)
#noise_feats_train = (noise_feats_train - back_mean)/(back_std)
print(np.shape(train_feats), "Train feats shape")
print("Add context")
train_feats_context = splice_feats(train_feats, num_context)
noise_feats_train_context = splice_feats(noise_feats_train, num_context)
mix_feats_train_context = splice_feats(mix_feats, num_context)
print(np.shape(train_feats_context), "Train feats shape after slice feats")

print ("Training Neural Network")
num_train_instance, _= np.shape(train_feats_context)
saver = tf.train.Saver()
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summaries_dir+'/train', sess.graph)
    # valid_writer = tf.summary.FileWriter(summaries_dir+'/valid')
    tf.global_variables_initializer().run()

    for i in range(num_epoch):
        sess_loss = 0.0

        for batch_ind in range(0,num_train_instance,batch_size):
            batch_noise = noise_feats_train_context[batch_ind:batch_ind+batch_size]
            batch_x = mix_feats_train_context[batch_ind:batch_ind+batch_size] #clean vocals + background
            #batch_x = batch_x.reshape((np.shape(batch_x)[0], n_steps, n_input))
            _, batch_loss, train_summary = sess.run([train, loss, merged], feed_dict={x: batch_x, clean_vocals: train_feats_context[batch_ind:batch_ind+batch_size], background: batch_noise})
            sess_loss += batch_loss

        train_writer.add_summary(train_summary, i)
        # valid_summary = sess.run(merged, feed_dict={inputs:valid_feats, ground_truth_labels: valid_labels})
        # valid_writer.add_summary(valid_summary,i)

        if ((i+1)%10==0):
            saver.save(sess, experiments_folder+'/saved_models/model', global_step=i+1)

        print("Epoch "+str(i)+ " loss", sess_loss/(num_train_instance/batch_size))
    batch_x = train_feats_context[0:300]
    batch_noise = noise_feats_train_context[0:300]
    batch_x = batch_x + batch_noise
    #batch_xi = batch_x.reshape((np.shape(batch_x)[0], n_steps, n_input))
    outputs = sess.run(pred, feed_dict={x: batch_x})
    center_batch_x = np.split(batch_x, n_steps,axis = 1)[num_context]
    outputs = center_batch_x*outputs
    print(np.shape(outputs),"Output shape")
    with open("outputs.csv", "w") as csv_file:
        for line in outputs:
            for num in line:
                csv_file.write(str(num)+',')
            csv_file.write('\n')
    csv_file.close()
    #with open("batch_noise.csv", "w") as csv_file:
    #    for line in noise_feats_train[0:128]:
    #        for num in line:
    #            csv_file.write(str(num)+',')
    #        csv_file.write('\n')
    #csv_file.close() 
