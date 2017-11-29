"""
Takes in a model and list of (test) files
outputs the estimate time freq mask for each file.
arg1 - model name (without the .meta extension)
arg2 - list of test files (filepath/filename)
"""

import tensorflow as tf
from tf_methods import *



model_name = sys.argv[1]
test_file_list = sys.argv[2]
num_context = 5
experiments_folder = '/Users/RupakVignesh/Documents/HomeDepot/fall2017/conversion_prediction/'
test_output_path = experiments_folder + ''


sess = tf.Session()
saver = tf.train.import_meta_graph(experiments_folder+model_name+'.meta')
saver.restore(sess,experiments_folder+model_name)

with open(test_file_list,'r') as F:
    file_list_parsed = [lines.strip() for lines in F]



# Get placeholders
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("mix:0")
pred = graph.get_tensor_by_name("predictions:0")

for f in file_list_parsed:
    input_feats, _ = read_data(f)
    input_feats = splice_feats(input_feats, num_context)
    output_mask = sess.run(pred, feed_dict={x: input_feats})
    test_filename = os.path.basename(f)
    with open(test_output_path + test_filename, "w") as csv_file:
        for line in output_mask:
            for num in line:
                csv_file.write(str(num)+',')
            csv_file.write('\n')
    csv_file.close()
