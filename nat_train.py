"""Trains a model, saving checkpoints and tensorboard summaries along
   the way.

    [NOTE:] Does not use Adversarial training
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from model import Model
from pgd_attack import LinfPGDAttack

with open('config.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

batch_size = config['training_batch_size']
c_eps = config['c_eps']
nu = config['nu']

# Setting up the data and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model()

# Setting up the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(model.xent,
                                                   global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy train:', model.accuracy)
tf.summary.scalar('xent :', model.xent / batch_size)
merged_summaries = tf.summary.merge_all()

shutil.copy('config.json', model_dir)

compress_op = model.compressWeights(eps = c_eps, nu = nu)

with tf.Session() as sess:
  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  # Main training loop
  for ii in range(max_num_training_steps + 1):
    x_batch, y_batch = mnist.train.next_batch(batch_size)
    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}
    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=nat_dict)
    end = timer()
    training_time += end - start

    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0

    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=nat_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))


    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
        print(" [CC] Saving a Checkpoint for iteration ",ii)
        saver.save(sess,
                   os.path.join(model_dir, 'checkpoint'),
                   global_step=global_step)


  print('=======================================================')
  print('     Training Complete.')
  print('         Compressing the last Fully Connected layer')
  test_dict = {model.x_input: mnist.test.images, model.y_input: mnist.test.labels}
  before_test_acc =  sess.run(model.accuracy, feed_dict=test_dict)

  sess.run(compress_op)

  after_test_acc = sess.run(model.accuracy, feed_dict=test_dict)
  summary = sess.run(merged_summaries, feed_dict=test_dict)
  summary_writer.add_summary(summary, global_step.eval(sess))

  print('    Before Test accuracy {:.4}%'.format(before_test_acc * 100))
  print('    After  Test accuracy {:.4}%'.format(after_test_acc * 100))
  print('=======================================================')


  with open('job_result.json', 'w') as result_file:
    final_result = {'Before_Test_ ccuracy': before_test_acc,
                    'After_Test_Accuracy': after_test_acc,
                    'Compression_eps':c_eps, 'Compression_nu': nu,
                    'Training_Steps': max_num_training_steps}
    json.dump(final_result, result_file, sort_keys=True, indent=4)
