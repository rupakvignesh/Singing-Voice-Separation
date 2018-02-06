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

model_name = sys.argv[1]
test_file_list = sys.argv[2]
num_context = 1
experiments_folder = '/home/rvignesh/singing_voice_separation/experiments/expt15/'
test_output_path = experiments_folder + 'test_out/'

print("Restoring models")
sess = tf.Session()
saver = tf.train.import_meta_graph(experiments_folder+'saved_models/'+model_name+'.meta')
saver.restore(sess,experiments_folder+'saved_models/'+model_name)

with open(test_file_list,'r') as F:
    file_list_parsed = [lines.strip() for lines in F]



# Get placeholders
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("mix:0")
est_vocals = graph.get_tensor_by_name("estimated_vocals:0")
est_background = graph.get_tensor_by_name("estimated_background:0")

print("Estimate Masks")
import timeit
start = timeit.default_timer()

for f in file_list_parsed:
    input_feats, _ = read_data(f)
    print(f)
    input_feats = splice_feats(input_feats, num_context)
    [vocal_out, background_out] = sess.run([est_vocals, est_background], feed_dict={x: 2*input_feats})
    test_filename = os.path.basename(f)
    vocal_out = np.split(vocal_out, 2*num_context+1, axis=1)[num_context]
    background_out = np.split(background_out, 2*num_context+1, axis=1)[num_context]
    with open(test_output_path +'vocal_'+ test_filename, "w") as csv_file:
        for line in vocal_out:
            for num in line:
                csv_file.write(str(num)+',')
            csv_file.write('\n')
    csv_file.close()
    with open(test_output_path +'background_'+ test_filename, "w") as csv_file:
        for line in background_out:
            for num in line:
                csv_file.write(str(num)+',')
            csv_file.write('\n')
    csv_file.close()

stop = timeit.default_timer()
print (stop - start)
