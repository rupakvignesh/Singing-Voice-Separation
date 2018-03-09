"""
Takes in a model and list of (test) files
outputs the estimate time freq mask for each file.
arg1 - model name (without the .meta extension)
arg2 - list of test files (filepath/filename)
"""

import tensorflow as tf
from tf_methods import *
import sys, os
import pdb

#Audio  parameters
src_sr = 16000
tgt_sr = 16000
win_size = 640
hop_size = 320
fft_size = 1024

model_name = sys.argv[1]
num_context = 1
n_steps = 2*num_context + 1
experiments_folder = '/home/rvignesh/singing_voice_separation/experiments/expt21/'
test_output_path = experiments_folder + 'test_out/'

print("Restoring models")
sess = tf.Session()
saver = tf.train.import_meta_graph(experiments_folder+'saved_models/'+model_name+'.meta')
saver.restore(sess,experiments_folder+'saved_models/'+model_name)

# Get placeholders
graph = tf.get_default_graph()
x_mix = graph.get_tensor_by_name("mix:0")
background = graph.get_tensor_by_name("background:0")
clean_vocals = graph.get_tensor_by_name("clean_vocals:0")
est_vocals = graph.get_tensor_by_name("estimated_vocals:0")
est_background = graph.get_tensor_by_name("estimated_background:0")

import timeit
start = timeit.default_timer()

print("Extract STFT")
back_wav, voc_wav, mixed_wav = get_wav_from_path(sys.argv[2], src_sr, tgt_sr)	# argv[2] -> path to test files
back_stft, voc_stft, mix_stft = list(map(lambda x: get_stft_from_wav(x, fft_size, hop_size, win_size), [back_wav, voc_wav, mixed_wav]))    

#Remove first and last frames for valid
remove_frames = lambda x: [feat[:,1:-1] for feat in x]
[back_stft, voc_stft, mix_stft] = list(map(remove_frames, [back_stft, voc_stft, mix_stft]))

#Resynthesise waveforms from truncated stfts
pred_phase = [np.angle(mix) for mix in mix_stft]
pred_phase_back = [np.angle(back) for back in back_stft]
pred_phase_voc = [np.angle(voc) for voc in voc_stft]
back_wav = get_wav_from_stft([np.abs(stft.T) for stft in back_stft], pred_phase_back, hop_size, win_size)
voc_wav = get_wav_from_stft([np.abs(stft.T) for stft in voc_stft], pred_phase_voc, hop_size, win_size)
mixed_wav = get_wav_from_stft([np.abs(stft.T) for stft in mix_stft], pred_phase, hop_size, win_size)

#Add context
print("Adding context")
feats_back_context = [splice_feats(np.abs(feats.T), num_context) for feats in back_stft]
feats_voc_context = [splice_feats(np.abs(feats.T), num_context) for feats in voc_stft]
feats_mix_context = [feats_voc_context[i] + feats_back_context[i] for i in range(len(feats_voc_context))]

print("Estimate Masks")
pred_vocal_mag = []
pred_back_mag = []
test_loss = 0
for val_ind in range(len(voc_wav)):
    [a1, a2] = sess.run([est_vocals, est_background], feed_dict={x_mix: feats_mix_context[val_ind], clean_vocals: feats_voc_context[val_ind], background: feats_back_context[val_ind]})
    pred_vocal_mag.append((np.split(a1, n_steps, axis=1)[num_context]))
    pred_back_mag.append((np.split(a2, n_steps, axis=1)[num_context]))

print("Reconstruct")
pred_vocal = get_wav_from_stft(pred_vocal_mag, pred_phase, hop_size, win_size)
pred_back = get_wav_from_stft(pred_back_mag, pred_phase, hop_size, win_size)

print("Compute BSSeval")
GNSDR, GSIR, GSAR = global_bss_metrics(back_wav, voc_wav, mixed_wav, pred_back, pred_vocal)

print("GNSDR: ", GNSDR, "GSIR: ", GSIR, "GSAR: ", GSAR) 

stop = timeit.default_timer()
print ("Total time for testing:",stop - start)
