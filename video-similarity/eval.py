#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
#from tensorflow.contrib import learn
from eval_helper import InputHelper, compute_distance
from scipy import misc
# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 1)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
#tf.flags.DEFINE_string("model", "/data4/abhijeet/gta/runs/1504276069/checkpoints/model-3542", "Load trained model checkpoint (Default: None)")
#tf.flags.DEFINE_string("model", "/data4/abhijeet/gta/runs/1504326119/checkpoints/model-3289", "Load trained model checkpoint (Default: None)")
# Intersection only (test only negative)
#tf.flags.DEFINE_string("model", "/data4/abhijeet/gta/runs/1504356159/checkpoints/model-1895", "Load trained model checkpoint (Default: None)")
#tf.flags.DEFINE_string("model", "/home/tushar/abhijeet/gta/runs/14a/checkpoints/model-4932", "Load trained model checkpoint (Default: None)")
tf.flags.DEFINE_string("model", "/home/tushar/abhijeet/gta/runs/temp/checkpoints/model-5754", "Load trained model checkpoint (Default: None)")

#Alderly
#tf.flags.DEFINE_string("model", "/data4/abhijeet/gta/runs/1504139288/checkpoints/model-189", "Load trained model checkpoint (Default: None)")
tf.flags.DEFINE_string("eval_filepath", "/home/tushar/Heavy_dataset/gta_data/final/", "testing folder (default: /home/halwai/gta/final)")
tf.flags.DEFINE_integer("max_frames", 20, "Maximum Number of frame (default: 20)")
tf.flags.DEFINE_string("loss", "contrastive", "Type of Loss functions:: contrastive/AAAI(default: contrastive)")
tf.flags.DEFINE_string("name", "result", "Name of the folder where images with incorrect results are stored")

tf.flags.DEFINE_string("filename", "./annotation_files/negative_annotations_test_all_intersections_only.txt", "Name of the file to be tested upon")
tf.flags.DEFINE_integer("label", 0, "Label of the files (default: 0)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.eval_filepath==None or FLAGS.model==None :
    print("Eval or Vocab filepaths are empty.")
    exit()

# load data and map id-transform based on training time vocabulary
inpH = InputHelper()
x1_test,x2_test,y_test,video_lengths_test = inpH.getTestDataSet(FLAGS.eval_filepath, FLAGS.max_frames, FLAGS.filename, FLAGS.label)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = FLAGS.model
print(checkpoint_file)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_imgs = graph.get_operation_by_name("input_imgs").outputs[0]
        input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
        input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
        video_lengths = graph.get_operation_by_name("video_lengths").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
        conv_output = graph.get_operation_by_name("conv/output").outputs[0]
        predictions = graph.get_tensor_by_name("distance:0")

        print(conv_output, predictions)
        # Generate batches for one epoch
        batches = inpH.batch_iter(x1_test,x2_test,y_test,video_lengths_test, 1, 1, [[104, 114, 124], (227, 227)] ,shuffle=False, is_train=False)
        # Collect the predictions here
        all_predictions = []
        all_dist=[]
        num = 0
        for (x1_dev_b,x2_dev_b,y_dev_b,v_len_b) in batches:
            num = num+1
            [x1] = sess.run([conv_output], {input_imgs: x1_dev_b})
            [x2] = sess.run([conv_output], {input_imgs: x2_dev_b})
            [dist] = sess.run([predictions], {input_x1: x1, input_x2: x2, input_y:y_dev_b, dropout_keep_prob: 1.0, video_lengths: v_len_b})
            d = compute_distance(dist, FLAGS.loss)
            correct = np.sum(y_dev_b==d)
            print(dist, y_dev_b, d)
            all_dist.append(dist)
            all_predictions.append(correct)
            #if ~correct:
            #misc.imsave(FLAGS.name + '/'+ FLAGS.filename + '/' + str(num)+'.png', np.vstack([np.hstack(x1_dev_b),np.hstack(x2_dev_b)]))


        #for ex in all_predictions:
        #    print(ex)
        correct_predictions = np.sum(all_predictions)*1.0/ len(all_predictions)
        print(len(all_predictions), np.sum(all_predictions))
        print("Accuracy: {:g}".format(correct_predictions))

