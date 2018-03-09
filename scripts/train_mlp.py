"""
Implements a stacked LSTM based DAE.
Input: vocal + background
Output: vocals (or an ideal binary mask)

"""
import tensorflow as tf
import librosa
import numpy as np
import os, sys
from tf_methods import *
import pdb

if len(sys.argv)==1:
    print("Arg1: train vocal feats, Argv2: train background feats, Arg3: path to valid wav")
    sys.exit()

# Network Parameters
n_input = 513 # Spectrogram
num_context = 1
n_steps = 2*num_context + 1 # timesteps
n_hidden = 1024 # hidden layer num of nodes
n_hidden2 = 1024
n_hidden3 = 1024
n_classes = n_input*n_steps # reconstruct input without background accompaniment
learning_rate = 0.01
num_epoch = 1000
batch_size = 512
eps = 1e-10

#Audio  parameters
src_sr = 16000
tgt_sr = 16000
win_size = 640
hop_size = 320
fft_size = 1024

experiments_folder='/home/rvignesh/singing_voice_separation/experiments/expt21'
summaries_dir=experiments_folder+'/summaries'

# tf Graph input
print("Building computational graph")
#x = tf.placeholder("float", [None, n_steps, n_input])
background = tf.placeholder("float", [None, n_input*(2*num_context+1)], name='background')
clean_vocals = tf.placeholder("float", [None, n_input*(2*num_context+1)], name='clean_vocals')
x_mix = tf.placeholder("float", [None, n_input*(2*num_context+1)], name='mix') # tf.placeholder("float", [None, n_input*(2*num_context+1)]) 

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

est_vocals = (voc_tf_pred/(voc_tf_pred + back_tf_pred + eps)) * x_mix 
est_background = (back_tf_pred/(voc_tf_pred + back_tf_pred + eps)) * x_mix

est_vocals = tf.identity(est_vocals, name='estimated_vocals')
est_background = tf.identity(est_background, name='estimated_background')

# Loss function
with tf.name_scope("Loss"):
    loss = tf.reduce_mean(tf.square(est_vocals - clean_vocals) + tf.square(est_background - background))
    #print(current_frame_gain.get_shape(), "Current Frame Gain Shape")

loss1 = tf.summary.scalar("loss1", loss)

#Train op
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

print("Reading Data")
train_feats, _ = read_data(sys.argv[1])         # Clean Vocals
noise_feats_train, _ = read_data(sys.argv[2])   # Background track
valid_back, valid_voc, valid_mixed = get_wav_from_path(sys.argv[3], src_sr, tgt_sr)

#Compute STFT
valid_back_stft, valid_voc_stft, valid_mix_stft = list(map(lambda x: get_stft_from_wav(x, fft_size, hop_size, win_size), [valid_back, valid_voc, valid_mixed]))

#Remove first and last frames for valid
remove_frames = lambda x: [feat[:,1:-1] for feat in x]
[valid_back_stft, valid_voc_stft, valid_mix_stft] = list(map(remove_frames, [valid_back_stft, valid_voc_stft, valid_mix_stft]))

#Resynthesise waveforms from truncated stfts
pred_phase = [np.angle(mix) for mix in valid_mix_stft]
pred_phase_back = [np.angle(back) for back in valid_back_stft]
pred_phase_voc = [np.angle(voc) for voc in valid_voc_stft]
valid_back = get_wav_from_stft([np.abs(stft.T) for stft in valid_back_stft], pred_phase_back, hop_size, win_size)
valid_voc = get_wav_from_stft([np.abs(stft.T) for stft in valid_voc_stft], pred_phase_voc, hop_size, win_size)
valid_mixed = get_wav_from_stft([np.abs(stft.T) for stft in valid_mix_stft], pred_phase, hop_size, win_size)


#print("Normalizing")
#train_mean = np.mean(train_feats,axis=0)
#train_std = np.std(train_feats,axis=0)
#train_feats = (train_feats - train_mean)/(train_std)
#back_mean = np.mean(noise_feats_train,axis=0)
#back_std = np.std(noise_feats_train,axis=0)
#noise_feats_train = (noise_feats_train - back_mean)/(back_std)
#valid_voc_stft = [ (np.abs(feat.T) - train_mean)/(train_std) for feat in valid_voc_stft]
#valid_back_stft = [ (np.abs(feat.T) - back_mean)/(back_std) for feat in valid_back_stft]
#valid_mix_stft = [ (valid_voc_stft[i]+valid_back_stft[i]) for i in range(len(valid_voc))]

print("Add context")
train_feats_context = splice_feats(train_feats, num_context)
noise_feats_train_context = splice_feats(noise_feats_train, num_context)
valid_feats_back_context = [splice_feats(np.abs(valid_feats.T), num_context) for valid_feats in valid_back_stft]
valid_feats_voc_context = [splice_feats(np.abs(valid_feats.T), num_context) for valid_feats in valid_voc_stft]
valid_feats_mix_context = [valid_feats_voc_context[i] + valid_feats_back_context[i] for i in range(len(valid_feats_voc_context))]
print(np.shape(train_feats_context), "Train feats shape after slice feats")

print ("Training Neural Network")
num_train_instance, _= np.shape(train_feats_context)
saver = tf.train.Saver(max_to_keep=15)
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summaries_dir+'/train', sess.graph)
    valid_writer = tf.summary.FileWriter(summaries_dir+'/valid')
    tf.global_variables_initializer().run()

    for i in range(num_epoch):
        sess_loss = 0.0

        shuffle = np.random.permutation(len(noise_feats_train_context))			#Shuffling the background to virtually create more data
        for batch_ind in range(0,num_train_instance,batch_size):
            batch_noise = noise_feats_train_context[shuffle[batch_ind:batch_ind+batch_size]]
            batch_vocals = train_feats_context[batch_ind:batch_ind+batch_size]
            batch_mix = batch_noise + batch_vocals
            
            _, batch_loss, train_summary = sess.run([train, loss, merged], feed_dict={clean_vocals: batch_vocals, background: batch_noise, x_mix: batch_mix})
            sess_loss += batch_loss
        train_writer.add_summary(train_summary, i)
        
        pred_vocal_mag = []
        pred_back_mag = []
        valid_loss = 0
        for val_ind in range(len(valid_voc)):
                [a1, a2, batch_loss] = sess.run([est_vocals, est_background, loss], feed_dict={x_mix: valid_feats_mix_context[val_ind], clean_vocals: valid_feats_voc_context[val_ind], background: valid_feats_back_context[val_ind]})
                pred_vocal_mag.append((np.split(a1, n_steps, axis=1)[num_context]))
                pred_back_mag.append((np.split(a2, n_steps, axis=1)[num_context]))
                valid_loss += batch_loss
        pred_vocal = get_wav_from_stft(pred_vocal_mag, pred_phase, hop_size, win_size)
        pred_back = get_wav_from_stft(pred_back_mag, pred_phase, hop_size, win_size)
        GNSDR, GSIR, GSAR = global_bss_metrics(valid_back, valid_voc, valid_mixed, pred_back, pred_vocal)
        
        if ((i+1)%10==0):
            saver.save(sess, experiments_folder+'/saved_models/model', global_step=i+1)
            for val_ind in range(len(valid_voc)):
                librosa.output.write_wav('predicted_valid'+str(val_ind)+'.wav', pred_vocal[val_ind], tgt_sr)

        print("Epoch "+str(i)+ " Train loss", sess_loss/(num_train_instance/batch_size), "GNSDR, GSIR, GSAR ",str(GNSDR[1]), str(GSIR[1]),str(GSAR[1]), " Valid loss", valid_loss/len(valid_voc))
    

