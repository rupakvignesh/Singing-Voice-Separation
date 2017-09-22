"""
Implements an Autoencoder
Author: Rupak Vignesh Swaminathan
"""
import tensorflow as tf
import numpy as np
import os, sys
from tf_methods import *
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# Model Parameters
FEAT_DIM = 48                                  # input feature dimension
NUM_CLASSES = 2                                # number of classes
NUM_CONTEXT = 10                                 # frames to add left and right
NUM_OUTPUT = FEAT_DIM
NUM_HIDDEN_LAYERS = 1
NUM_HIDDEN1_NODES = 64
NUM_HIDDEN2_NODES = FEAT_DIM
NUM_HIDDEN3_NODES = FEAT_DIM
NUM_EPOCH = 150
ALPHA = 0.01     # Learning Rate
BATCH_SIZE = 1000
FEAT_DIM = FEAT_DIM + 2*NUM_CONTEXT*FEAT_DIM #Recompute dimension based on context
OPTIMIZER = 'AdamOptimizer'
MAX_TO_KEEP = 10
NOTES = ''

# Global variables
experiment_folder = "C:/Users/swaminr/Documents/Project/experiments/DAE/expt5"
summaries_dir = experiment_folder+"/summaries"
examples_to_show = 5

# Write Model Parameters to file
with open(experiment_folder+'/Params.txt','w') as F:
    F.write("FEAT_DIM = " + str(FEAT_DIM) + '\n')
    F.write("NUM_CLASSES = " + str(NUM_CLASSES) + '\n')
    F.write("NUM_CONTEXT = " + str(NUM_CONTEXT) + '\n')
    F.write("NUM_OUTPUT = " + str(NUM_OUTPUT) + '\n')
    F.write("NUM_HIDDEN_LAYERS = " + str(NUM_HIDDEN_LAYERS) + '\n')
    F.write("NUM_HIDDEN1_NODES = " + str(NUM_HIDDEN1_NODES) + '\n')
    if NUM_HIDDEN_LAYERS>1:
        F.write("NUM_HIDDEN2_NODES = " + str(NUM_HIDDEN2_NODES) + '\n')
    if NUM_HIDDEN_LAYERS>2:
        F.write("NUM_HIDDEN3_NODES = " + str(NUM_HIDDEN3_NODES) + '\n')
    F.write("NUM_EPOCH = " + str(NUM_EPOCH) + '\n')
    F.write("ALPHA = " + str(ALPHA) + '\n')
    F.write("BATCH_SIZE = " + str(BATCH_SIZE) + '\n')
    F.write("OPTIMIZER = " + OPTIMIZER + '\n')
    F.write("Notes = " + NOTES+'\n')

F.close()

print ("Building Computational graph")

#First layer
ae1 = auto_encoder([FEAT_DIM, NUM_HIDDEN1_NODES, NUM_HIDDEN2_NODES], NUM_CONTEXT=NUM_CONTEXT)
loss1 = tf.summary.scalar("loss1", ae1["loss"])
[W1, b1] = ae1["encoder"][0]
[Wd1, bd1] = ae1["decoder"][0]
[W2, b2] = ae1["encoder"][1]
[Wd2, bd2] = ae1["decoder"][1]
tf.add_to_collection("Layer1_E", W1)
tf.add_to_collection("Layer1_E", b1)
tf.add_to_collection("Layer1_D", Wd1)
tf.add_to_collection("Layer1_D", bd1)
tf.add_to_collection("Layer2_E", W2)
tf.add_to_collection("Layer2_E", b2)
tf.add_to_collection("Layer2_D", Wd2)
tf.add_to_collection("Layer2_D", bd2)



print ("Reading data")
######################################################
train_feats, _ = read_data(sys.argv[1])         # Clean Vocals
noise_feats_train, _ = read_data(sys.argv[2])   # Background track
valid_feats, _ = read_data(sys.argv[3])
noise_feats_valid, _ = read_data(sys.argv[4])

print ("Z-score norm")                          # Implement Batch norm instead of Z-Score in the future
train_mu = np.mean(train_feats,axis=0)
train_std = np.std(train_feats,axis=0)
train_feats = (train_feats - train_mu)/train_std
valid_feats = (valid_feats - train_mu)/train_std

noise_train_mu = np.mean(noise_feats_train, axis=0)
noise_train_std = np.std(noise_feats_train, axis=0)
noise_feats_train = (noise_feats_train - noise_train_mu)/noise_train_std
noise_feats_valid = (noise_feats_valid - noise_train_mu)/noise_train_std
with open(experiment_folder+'/train_mu.txt','w') as F:
    for i in range(len(train_mu)):
        F.write(str(train_mu[i])+' ')
        F.write('\n')
F.close()
with open(experiment_folder+'/train_std.txt','w') as F:
    for i in range(len(train_std)):
        F.write(str(train_std[i])+' ')
        F.write('\n')
F.close()
with open(experiment_folder+'/noise_train_mu.txt','w') as F:
    for i in range(len(noise_train_mu)):
        F.write(str(train_mu[i])+' ')
        F.write('\n')
F.close()
with open(experiment_folder+'/noise_train_std.txt','w') as F:
    for i in range(len(noise_train_std)):
        F.write(str(train_std[i])+' ')
        F.write('\n')
F.close()


print ("Splice feats")
train_feats = splice_feats(train_feats, NUM_CONTEXT)
noise_feats_train = splice_feats(noise_feats_train, NUM_CONTEXT)
valid_feats = splice_feats(valid_feats, NUM_CONTEXT)
noise_feats_valid = splice_feats(noise_feats_valid, NUM_CONTEXT)
#####################################################
#try the network on MNIST data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# train_feats = mnist.train.images

print ("Train model")
NUM_TRAIN_INSTANCES, _ = np.shape(train_feats)
shuffle = np.random.permutation(len(train_feats))
saver = tf.train.Saver(max_to_keep=MAX_TO_KEEP) # Save all variables
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summaries_dir+'/train',sess.graph)
    valid_writer = tf.summary.FileWriter(summaries_dir+'/valid')
    tf.global_variables_initializer().run()

    for i in range(NUM_EPOCH):
        sess_loss = 0.0
        sess_output = np.zeros([NUM_TRAIN_INSTANCES, NUM_CLASSES])
        # Batch shuffle and train
        for batch_ind in range(0,NUM_TRAIN_INSTANCES, BATCH_SIZE):
            # Add context
            _,batch_loss, train_summary = sess.run([ae1['train'], ae1['loss'], merged], feed_dict={ae1['x']: train_feats[shuffle[batch_ind:batch_ind + BATCH_SIZE]], \
                                            ae1['n']: noise_feats_train[shuffle[batch_ind:batch_ind + BATCH_SIZE]]})
            sess_loss += batch_loss

        #Write summary
        train_writer.add_summary(train_summary, i)
        _, valid_summary = sess.run([ae1['loss'], merged], feed_dict={ae1['x']:valid_feats, ae1['n']: noise_feats_valid})
        valid_writer.add_summary(valid_summary, i)
        #Save every 10th model, test validation features and write summary
        if ((i+1)%10==0):
            saver.save(sess, experiment_folder+'/saved_models/model', global_step=i+1)

        print("Epoch "+str(i)+" loss", sess_loss)


    print ("Training complete")

    #Close file writer
    train_writer.close()
    valid_writer.close()
