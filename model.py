"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class MLP(object):
  """
        A multilayer percepetron for image classification.
        This has only one hidden layer with configurable number of hidden units. 

  """
  def __init__(self, hidden_units = 1024):
   
    print(" ---- Using a one-layer MLP with ", hidden_units, " Hidden units ----")
    self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
    self.y_input = tf.placeholder(tf.int64, shape = [None])
    self.hidden_units = hidden_units

    self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])

    # first fully connected layer
    self.W_fc1 = self._weight_variable([28 * 28, self.hidden_units])
    b_fc1 = self._bias_variable([self.hidden_units])

    x_image_flat = tf.reshape(self.x_image, [-1, 28 * 28])
    h_fc1 = tf.nn.relu(tf.matmul(x_image_flat, self.W_fc1) + b_fc1)

    # output layer
    self.W_fc2 = self._weight_variable([self.hidden_units, 10])
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
  def _matrix_project(self, sess, A_tf, eps, nu):
      A = sess.run(A_tf)
      k = np.log(1.0 / nu) / (np.square(eps))
      k = k.astype(int)
      A_hat = np.zeros(A.shape)
      total_params = A.shape[0] * A.shape[1]
      compression_ratio = total_params / k
      print('k : {} .. Matrix Params: {}.. Compression Ratio {:.3f}'.format(
	k, total_params, compression_ratio))
      for i in range(k):
          m = self._random_matrix(A.shape)
          z = np.sum(np.multiply(m, A))
          A_hat = A_hat + np.multiply(z, m)
      A_hat = A_hat / k.astype(float)
      return A_hat

  # Returns a Random Matrix of shape with only iid +1 and -1, where +1 appears with probability frac
  def _random_matrix(self, shape, frac=0.5):
      m = np.random.binomial(1, frac, size=shape)
      m = 2 * m - 1
      return m


  def compressWeights(self, sess, eps=0.05, nu = 0.1):
    print(" ......................... Entering Compress")
    print(" ********** Simple MLP with just one hidden layer  **************")
    W_fc1_compress = self._matrix_project(sess, self.W_fc1, eps, nu)
    compress_op = self.W_fc1.assign(W_fc1_compress)
    W_fc2_compress = self._matrix_project(sess, self.W_fc2, eps, nu)
    compress_op_2 = self.W_fc2.assign(W_fc2_compress)
    sess.run([compress_op, compress_op_2])
    print(" ......................... Exiting Compress")

    return

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


class DeepMLP(object):
  """
        A multilayer percepetron for image classification.

  """
  def __init__(self, hidden_units = 128, num_layers = 10):

    print(" ********** Using a Deep MLP **************")
    print("     Number of Layers: ", num_layers)
    print("     Number of Hidden Units: ", hidden_units)
    print(" ******************************************")

    self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
    self.y_input = tf.placeholder(tf.int64, shape = [None])
    self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])
    self.H = hidden_units
    self.weights = []
    self.hidden_FCs = []
    self.biases = []
    self.num_layers = num_layers

    self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])

    # first fully connected layer
    self.weights.append(self._weight_variable([28 * 28, self.H]))
    self.biases.append(self._bias_variable([self.H]))
    x_image_flat = tf.reshape(self.x_image, [-1, 28 * 28])
    self.hidden_FCs.append(tf.nn.relu(tf.matmul(x_image_flat, self.weights[0]) + self.biases[0]))

    for i in range(self.num_layers - 2):
        print("Initializing Layer :", i)
        self.weights.append(self._weight_variable([self.H, self.H]))
        self.biases.append(self._bias_variable([self.H]))
        self.hidden_FCs.append(
            tf.nn.relu(tf.matmul(self.hidden_FCs[i], self.weights[i+1]) + self.biases[i+1]))

    # output layer
    # This doesn't maintain much specifications ?
    self.weights.append( self._weight_variable([self.H, 10]))
    self.biases.append(self._bias_variable([10]))

    self.pre_softmax = tf.matmul(self.hidden_FCs[-1], self.weights[-1]) + self.biases[-1]
    self.hidden_FCs.append(self.pre_softmax)

    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)

    self.xent = tf.reduce_sum(y_xent)

    self.y_pred = tf.argmax(self.pre_softmax, 1)

    correct_prediction = tf.equal(self.y_pred, self.y_input)

    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


  # Projects a 2D matrix according to Sanjeev Arora Paper
  def _matrix_project(self, sess, A_tf, eps, nu):
      A = sess.run(A_tf)
      k = np.log(1.0 / nu) / (np.square(eps))
      k = k.astype(int)
      total_params = A.shape[0] * A.shape[1]
      compression_ratio = total_params / k
      print(" Matrix Shape", A.shape)
      print('k : {} .. Matrix Params: {}.. Compression Ratio {:.3f}'.format(
          k, total_params, compression_ratio))
      A_hat = np.zeros(A.shape)
      for i in range(k):
          m = self._random_matrix(A.shape)
          z = np.sum(np.multiply(m, A))
          A_hat = A_hat + np.multiply(z, m)
      A_hat = A_hat / k.astype(float)
      return A_hat


  def compressWeights(self, sess, eps=0.05, nu=0.1, layers = None):

    # By Default we compress all the layers.
    if not layers:
        layers = [ i for i in range(self.num_layers)]

    for i in layers:
        print("\n============Start ... Compressing Layer :", i)
        W_compress = self._matrix_project(sess, self.weights[i], eps, nu)
        compress_op = self.weights[i].assign(W_compress)
        sess.run(compress_op)




  # Returns a Random Matrix of shape with only iid +1 and -1, where +1 appears with probability frac
  def _random_matrix(self, shape, frac=0.5):
      m = np.random.binomial(1, frac, size=shape)
      m = 2 * m - 1
      return m

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


class Model(object):
  def __init__(self):
    print (" ========== Using a Model with Convolutional Layers =============== ")
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
  def _matrix_project(self, sess, A_tf, eps, nu):
      A = sess.run(A_tf)
      k = np.log(1.0 / nu) / (np.square(eps))
      k = k.astype(int)
     
      total_params = A.shape[0] * A.shape[1]
      compression_ratio = total_params / k
      print(" Matrix Shape", A.shape)
      print('k : {} .. Matrix Params: {}.. Compression Ratio {:.3f}'.format(
          k, total_params, compression_ratio))
      
      A_hat = np.zeros(A.shape)
      for i in range(k):
          m = self._random_matrix(A.shape)
          z = np.sum(np.multiply(m, A))
          A_hat = A_hat + np.multiply(z, m)
      A_hat = A_hat / k.astype(float)
      return A_hat

  # Returns a Random Matrix of shape with only iid +1 and -1, where +1 appears with probability frac
  def _random_matrix(self, shape, frac=0.5):
      m = np.random.binomial(1, frac, size=shape)
      m = 2 * m - 1
      return m

  def compressFirstFC(self, sess, eps=0.05, nu=0.1):
    print(" ......................... Compressing First FC layer ....... ")
    W_fc1_compress = self._matrix_project(sess, self.W_fc1, eps, nu)
    compress_op = self.W_fc1.assign(W_fc1_compress)
    sess.run(compress_op)
    print("......................... Done ................................")

  def compressSecondFC(self, sess, eps=0.05, nu = 0.1):
    print(" ......................... Compressing Second FC layer ....... ")
    W_fc2_compress = self._matrix_project(sess, self.W_fc2, eps, nu)
    compress_op_2 = self.W_fc2.assign(W_fc2_compress)
    sess.run(compress_op_2)
    print("......................... Done ................................")
    return

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
