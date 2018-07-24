"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class Model(object):
  def __init__(self):
    self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
    self.y_input = tf.placeholder(tf.int64, shape = [None])

    self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])

    # first convolutional layer
    W_conv1 = self._weight_variable([5,5,1,32])
    b_conv1 = self._bias_variable([32])

    h_conv1 = tf.nn.relu(self._conv2d(self.x_image, W_conv1) + b_conv1)
    h_pool1 = self._max_pool_2x2(h_conv1)

    # second convolutional layer
    W_conv2 = self._weight_variable([5,5,32,64])
    b_conv2 = self._bias_variable([64])

    h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = self._max_pool_2x2(h_conv2)

    # first fully connected layer
    self.W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
    b_fc1 = self._bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + b_fc1)

    # output layer
    self.W_fc2 = self._weight_variable([1024,10])
    b_fc2 = self._bias_variable([10])

    self.pre_softmax = tf.matmul(h_fc1, self.W_fc2) + b_fc2

    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)

    self.xent = tf.reduce_sum(y_xent)

    self.y_pred = tf.argmax(self.pre_softmax, 1)

    correct_prediction = tf.equal(self.y_pred, self.y_input)

    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



  # Projects a 2D matrix according to Sanjeev Arora Paper
  def _matrix_project(self, A, eps, nu):
      k = np.log(1.0 / nu) / (np.square(eps))
      k = k.astype(int)
      print("K:", k)
      A_hat = tf.zeros(A.shape, tf.float32)
      for i in range(k):
          m = self._random_matrix(A.shape)
          z = tf.reduce_sum(tf.multiply(m, A))
          A_hat = tf.add(A_hat, tf.multiply(z, m))
      A_hat = A_hat / k.astype(float)
      return A_hat

  # Returns a Random Matrix of shape with only iid +1 and -1, where +1 appears with probability frac
  def _random_matrix(self, shape, frac=0.5):
      m = tf.convert_to_tensor(np.random.binomial(1, frac, size=shape), dtype=tf.float32)
      m = 2 * m - 1
      return m

  def compressWeights(self, eps=0.05, nu = 0.1):
    print(" ......................... Entering Compress")
    W_fc1_compress = self._matrix_project(self.W_fc1, eps, nu)
    compress_op = self.W_fc1.assign(W_fc1_compress)
    W_fc2_compress = self._matrix_project(self.W_fc2, eps, nu)
    compress_op_2 = self.W_fc2.assign(W_fc2_compress)
    print(" ......................... Exiting Compress")
    return [compress_op, compress_op_2]

  @staticmethod
  def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

  @staticmethod
  def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  @staticmethod
  def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')
