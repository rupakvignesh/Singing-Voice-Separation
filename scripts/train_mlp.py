"""
Implements a stacked LSTM based DAE.
Input: vocal + background
Output: vocals (or an ideal binary mask)

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
#x = tf.placeholder("float", [None, n_steps, n_input])
background = tf.placeholder("float", [None, n_input*(2*num_context+1)])
clean_vocals = tf.placeholder("float", [None, n_input*(2*num_context+1)])
x_mix = tf.placeholder("float", [None, n_input*(2*num_context+1)]) # tf.placeholder("float", [None, n_input*(2*num_context+1)]) 

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

voc_tf_pred = tf.nn.relu(tf.add(tf.matmul(hidden3_output,weights4), biases4))
back_tf_pred = tf.nn.relu(tf.add(tf.matmul(hidden3_output,weights5), biases5))

est_vocals = (voc_tf_pred/(voc_tf_pred + back_tf_pred + 0.001)) * x_mix 
est_background = (back_tf_pred/(voc_tf_pred + back_tf_pred + 0.001)) * x_mix

# Estimating vocals from Time-frequency masks
with tf.name_scope("Loss"):
    loss = tf.reduce_mean(tf.square(est_vocals - clean_vocals) + tf.square(est_background - background))
    #print(current_frame_gain.get_shape(), "Current Frame Gain Shape")


loss1 = tf.summary.scalar("loss1", loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

print("Reading Data")
train_feats, _ = read_data(sys.argv[1])         # Clean Vocals
noise_feats_train, _ = read_data(sys.argv[2])   # Background track
mix_feats_train, _ = read_data(sys.argv[3])
valid_feats, _ = read_data(sys.argv[4])
noise_feats_valid, _ = read_data(sys.argv[5])
mix_feats_valid, _ = read_data(sys.argv[6])

#print("Normalizing")
#train_mean = np.mean(train_feats,axis=0)
#train_std = np.std(train_feats,axis=0)
#train_feats = (train_feats - train_mean)/(train_std)
#back_mean = np.mean(noise_feats_train,axis=0)
#back_std = np.std(noise_feats_train,axis=0)
#noise_feats_train = (noise_feats_train - back_mean)/(back_std)
#print(np.shape(train_feats), "Train feats shape")

print("Add context")
train_feats_context = splice_feats(train_feats, num_context)
noise_feats_train_context = splice_feats(noise_feats_train, num_context)
mix_feats_context = splice_feats(mix_feats_train, num_context)
valid_feats_context = splice_feats(valid_feats, num_context)
noise_feats_valid_context = splice_feats(noise_feats_valid, num_context)
mix_feats_valid_context = splice_feats(mix_feats_valid, num_context)
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

        for batch_ind in range(0,num_train_instance,batch_size):
            batch_noise = noise_feats_train_context[batch_ind:batch_ind+batch_size]
            batch_vocals = train_feats_context[batch_ind:batch_ind+batch_size]
            batch_mix = mix_feats_context[batch_ind:batch_ind+batch_size]
            
            _, batch_loss, train_summary = sess.run([train, loss, merged], feed_dict={clean_vocals: batch_vocals, background: batch_noise, x_mix: batch_mix})
            sess_loss += batch_loss
        train_writer.add_summary(train_summary, i)
        valid_summary, valid_loss = sess.run([merged, loss], feed_dict={clean_vocals: valid_feats_context, background: noise_feats_valid_context, x_mix: mix_feats_valid_context})
        valid_writer.add_summary(valid_summary,i)

        #saver.save(sess, experiments_folder+'/saved_models/model', global_step=i+1)

        print("Epoch "+str(i)+ " Train loss", sess_loss/(num_train_instance/batch_size), "Valid loss", valid_loss)
    
    batch_x = valid_feats_context[0:300]
    batch_noise = noise_feats_valid_context[0:300]
    batch_mix = mix_feats_valid_context[0:300]
    vocal_outputs, background_outputs = sess.run([est_vocals, est_background], feed_dict={clean_vocals: batch_x, background: batch_noise, x_mix: batch_mix})
    vocal_outputs = np.split(vocal_outputs, n_steps,axis = 1)[num_context]
    background_outputs = np.split(background_outputs, n_steps, axis=1)[num_context]

    #print(np.shape(outputs),"Output shape")
    with open("vocal_outputs.csv", "w") as csv_file:
        for line in vocal_outputs:
            for num in line:
                csv_file.write(str(num)+',')
            csv_file.write('\n')
    csv_file.close()
    with open("background_outputs.csv", "w") as csv_file:
        for line in background_outputs:
            for num in line:
                csv_file.write(str(num)+',')
            csv_file.write('\n')
    csv_file.close()
