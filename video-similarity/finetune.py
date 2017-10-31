#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import re
import os
import time
import datetime
import gc
from helper import InputHelper, save_plot, compute_distance
from siamese_network import SiameseLSTM
import gzip
from random import random
from amos import Conv

# Parameters
# ==================================================

tf.flags.DEFINE_integer("embedding_dim", 1000, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_string("training_file_path", "/home/tushar/abhijeet/gta/final/", "training folder (default: /home/halwai/gta_data/final)")
tf.flags.DEFINE_integer("max_frames", 20, "Maximum Number of frame (default: 20)")
tf.flags.DEFINE_string("name", "result", "prefix names of the output files(default: result)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 4, "Batch Size (default: 10)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many epochs (default: 100)")
tf.flags.DEFINE_integer("num_lstm_layers", 1, "Number of LSTM layers(default: 1)")
tf.flags.DEFINE_integer("hidden_dim", 50, "Number of LSTM layers(default: 2)")
tf.flags.DEFINE_string("loss", "contrastive", "Type of Loss functions:: contrastive/AAAI(default: contrastive)")
tf.flags.DEFINE_boolean("projection", False, "Project Conv Layers Output to a Lower Dimensional Embedding (Default: True)")
tf.flags.DEFINE_boolean("conv_net_training", False, "Training ConvNet (Default: False)")
tf.flags.DEFINE_float("lr", 0.0000001, "learning-rate(default: 0.00001)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", False, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("return_outputs", 1, "Outpust from LSTM, 0=>Last LSMT output, 2=> Cell-State Output. 1=> Hidden-State Output (default: 2)")
tf.flags.DEFINE_string("summaries_dir", "/home/tushar/abhijeet/gta/summaries/", "Summary storage")

#Conv Net Parameters
tf.flags.DEFINE_string("conv_layer", "pool6", "CNN features from AMOSNet(default: pool6)")

#Model-Restore Parameters
tf.flags.DEFINE_string("model", "/home/tushar/abhijeet/gta/runs/15a/checkpoints/model-10998", "Load trained model checkpoint (Default: None)")

tf.flags.DEFINE_string("train_file_positive", "./annotation_files/alderly_positives_train.txt", "Positive_training_file")
tf.flags.DEFINE_string("train_file_negative", "./annotation_files/alderly_negatives_train.txt", "Negative_training_file")
tf.flags.DEFINE_integer("train_val_ratio", 10, "learning-rate(default:10%)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

checkpoint_file = FLAGS.model
print(checkpoint_file)

if FLAGS.training_file_path==None:
    print("Input Files List is empty. use --training_file_path argument.")
    exit()

inpH = InputHelper()
train_set, dev_set, sum_no_of_batches = inpH.getDataSets(FLAGS.training_file_path, FLAGS.max_frames, FLAGS.train_val_ratio, FLAGS.train_file_positive, FLAGS.train_file_negative, FLAGS.batch_size)

# Training
# ==================================================
print("starting graph def")
graph = tf.Graph()
with graph.as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=gpu_options,
      )
    sess = tf.Session(config=session_conf)
    print("started session")
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.lr, name="Adam_finetune")


        convModel_input_imgs = graph.get_operation_by_name("input_imgs").outputs[0]
        convModel_features =  graph.get_operation_by_name("conv/output").outputs[0]
        siameseModel_input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
        siameseModel_input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
        siameseModel_input_y = graph.get_operation_by_name("input_y").outputs[0]
        siameseModel_video_lengths = graph.get_operation_by_name("video_lengths").outputs[0]
        siameseModel_out1 = graph.get_operation_by_name("output/concat").outputs[0]
        siameseModel_out2 = graph.get_operation_by_name("output/concat_1").outputs[0]
        siameseModel_distance = graph.get_operation_by_name("distance").outputs[0]
        siameseModel_loss = graph.get_operation_by_name("loss/div_1").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        print("loaded all variables")
    grads_and_vars=optimizer.compute_gradients(siameseModel_loss)
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join("/home/tushar/abhijeet/gta/", "runs", FLAGS.name))
    print("Writing to {}\n".format(out_dir))

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    lstm_savepath="/home/tushar/abhijeet/gta/lstm_outputs"
    if not os.path.exists(lstm_savepath):
            os.makedirs(lstm_savepath)

    #saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    #print all trainable parameters
    tvar = tf.trainable_variables()
    for i, var in enumerate(tvar):
        print("{}".format(var.name))

    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', graph=tf.get_default_graph())
    val_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/val' , graph=tf.get_default_graph())

    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
        f.write(graphpb_txt)


    sess.run(tf.global_variables_initializer())

    saver.restore(sess, checkpoint_file)
    print("Restored Model")

    def train_step(x1_batch, x2_batch, y_batch, video_lengths):

        #A single training step

        [x1_batch] = sess.run([convModel_features],  feed_dict={convModel_input_imgs: x1_batch})
        [x2_batch] = sess.run([convModel_features],  feed_dict={convModel_input_imgs: x2_batch})

        feed_dict = {
                         siameseModel_input_x1: x1_batch,
                         siameseModel_input_x2: x2_batch,
                         siameseModel_input_y: y_batch,
                         dropout_keep_prob: FLAGS.dropout_keep_prob,
                         siameseModel_video_lengths: video_lengths,
        }

        out1, out2, _, step, loss, dist = sess.run([siameseModel_out1, siameseModel_out2, tr_op_set, global_step, siameseModel_loss, siameseModel_distance],  feed_dict)
        time_str = datetime.datetime.now().isoformat()
        d=compute_distance(dist, FLAGS.loss)
        correct = y_batch==d
        #print(out1, out2)
        #print(video_lengths)
        #print("TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, correct))
        print(y_batch, dist, d)
        return  np.sum(correct), loss

    def dev_step(x1_batch, x2_batch, y_batch, video_lengths, dev_iter, epoch):

        #A single training step

        [x1_batch] = sess.run([convModel_features],  feed_dict={convModel_input_imgs: x1_batch})
        [x2_batch] = sess.run([convModel_features],  feed_dict={convModel_input_imgs: x2_batch})

        feed_dict = {
                         siameseModel_input_x1: x1_batch,
                         siameseModel_input_x2: x2_batch,
                         siameseModel_input_y: y_batch,
                         dropout_keep_prob: FLAGS.dropout_keep_prob,
                         siameseModel_video_lengths: video_lengths,
        }

        step, loss, dist, out1, out2 = sess.run([global_step, siameseModel_loss, siameseModel_distance, siameseModel_out1,siameseModel_out2],  feed_dict)
        time_str = datetime.datetime.now().isoformat()
        d=compute_distance(dist, FLAGS.loss)
        correct = y_batch==d
        #print("DEV {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, correct))
        #print(y_batch, dist, d)
        return np.sum(correct), loss, correct

    # Generate batches
    batches=inpH.batch_iter(
                train_set[0], train_set[1], train_set[2], train_set[3], FLAGS.batch_size, FLAGS.num_epochs, [[104, 114, 124], (227, 227)], shuffle=True, is_train=False)

    ptr=0
    max_validation_correct=0.0
    start_time = time.time()
    train_accuracy, val_accuracy = [] , []
    train_loss, val_loss = [], []
    train_batch_loss_arr, val_batch_loss_arr = [], []


    for nn in xrange(FLAGS.num_epochs):
        # Evaluate on Validataion Data for every epoch
        sum_val_correct=0.0
        val_epoch_loss=0.0
        val_results = []
        print("\nEvaluation:")
        dev_batches = inpH.batch_iter(dev_set[0],dev_set[1],dev_set[2],dev_set[3], FLAGS.batch_size, 1, [[104, 114, 124], (227, 227)], shuffle=False , is_train=False)
        dev_iter=0
        for (x1_dev_b,x2_dev_b,y_dev_b, dev_video_lengths) in dev_batches:
            if len(y_dev_b)<1:
                continue
            dev_iter += 1
            batch_val_correct , val_batch_loss, batch_results = dev_step(x1_dev_b, x2_dev_b, y_dev_b, dev_video_lengths, dev_iter,nn)
            val_results = np.concatenate([val_results, batch_results])
            sum_val_correct = sum_val_correct + batch_val_correct
            current_step = tf.train.global_step(sess, global_step)
            #val_writer.add_summary(summary, current_step)
            val_epoch_loss = val_epoch_loss + val_batch_loss*len(y_dev_b)
            val_batch_loss_arr.append(val_batch_loss*len(y_dev_b))
        print("val_loss ={}".format(val_epoch_loss/len(dev_set[2])))
        print("total_val_correct={}/total_val={}".format(sum_val_correct, len(dev_set[2])))
        val_accuracy.append(sum_val_correct*1.0/len(dev_set[2]))
        val_loss.append(val_epoch_loss/len(dev_set[2]))

        print("Epoch Number: {}".format(nn))
        epoch_start_time = time.time()
        sum_train_correct=0.0
        train_epoch_loss=0.0
        for kk in xrange(sum_no_of_batches):
            x1_batch, x2_batch, y_batch, video_lengths = batches.next()
            if len(y_batch)<1:
                continue
            train_batch_correct, train_batch_loss =train_step(x1_batch, x2_batch, y_batch, video_lengths)
            #train_writer.add_summary(summary, current_step)
            sum_train_correct = sum_train_correct + train_batch_correct
            train_epoch_loss = train_epoch_loss + train_batch_loss* len(y_batch)
            train_batch_loss_arr.append(train_batch_loss*len(y_batch))
        print("train_loss ={}".format(train_epoch_loss/len(train_set[2])))
        print("total_train_correct={}/total_train={}".format(sum_train_correct, len(train_set[2])))
        train_accuracy.append(sum_train_correct*1.0/len(train_set[2]))
        train_loss.append(train_epoch_loss/len(train_set[2]))

        # Update stored model
        """if current_step % (FLAGS.checkpoint_every) == 0:
            if sum_val_correct >= max_validation_correct:
                max_validation_correct = sum_val_correct
                #saver.save(sess, checkpoint_prefix, global_step=current_step)
                tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph"+str(nn)+".pb", as_text=False)
                print("Saved model {} with checkpoint to {}".format(nn, checkpoint_prefix))"""

        epoch_end_time = time.time()
        print("Total time for {} th-epoch is {}\n".format(nn, epoch_end_time-epoch_start_time))
        save_plot(train_accuracy, val_accuracy, 'epochs', 'accuracy', 'Accuracy vs epochs', [-0.1, nn+0.1, 0, 1.01],  ['train','val' ],'./accuracy_'+str(FLAGS.name))
        save_plot(train_loss, val_loss, 'epochs', 'loss', 'Loss vs epochs', [-0.1, nn+0.1, 0, np.max(train_loss)+0.2],  ['train','val' ],'./loss_'+str(FLAGS.name))
        save_plot(train_batch_loss_arr, val_batch_loss_arr, 'steps', 'loss', 'Loss vs steps', [-0.1, (nn+1)*sum_no_of_batches+0.1, 0, np.max(train_batch_loss_arr)+0.2],  ['train','val' ],'./loss_batch_'+str(FLAGS.name))

    end_time = time.time()
    print("Total time for {} epochs is {}".format(FLAGS.num_epochs, end_time-start_time))

#"""
