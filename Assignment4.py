# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:07:37 2016

Deep learning Assignment 4:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/4_convolutions.ipynb

@author: chuang
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'

#%%
with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
  

#%%
image_size = 28
num_labels = 10
num_channels = 1 # grayscale

import numpy as np

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

#%%
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

#%%
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  keep_prob = tf.placeholder(tf.float32) #prob 2

  # Model.
  def model(data):
#    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
#    hidden = tf.nn.relu(conv + layer1_biases)
#    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
#    hidden = tf.nn.relu(conv + layer2_biases)
    
    #probm1:
    conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    hidden1 = tf.nn.relu(conv + layer1_biases)
    h_pool1=tf.nn.max_pool(hidden1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
    keep_prob=0.8 #prob2
    h_pool1=tf.nn.dropout(h_pool1,keep_prob) #prob2
  
    conv2=tf.nn.conv2d(h_pool1,layer2_weights,[1,1,1,1],padding='SAME')
    hidden2=tf.nn.relu(conv2+layer2_biases)
    h_pool2=tf.nn.max_pool(hidden2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
    shape = h_pool2.get_shape().as_list()
    h_pool2_flat = tf.reshape(h_pool2, [shape[0], shape[1] * shape[2] * shape[3]])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, layer3_weights) + layer3_biases)

    return tf.matmul(h_fc1, layer4_weights) + layer4_biases
  
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
  # Optimizer.
#  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  global_step=tf.Variable(0)
  learning_rate=tf.train.exponential_decay(.1,global_step,100,.95,staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
  
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset)) 

#%%
num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
#    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:.8} #prob2
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 300 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))       
  
#%%:
#Prob1: test accuracy 91.5%
#prob2: dropout=.8, test accuracy: 90.1%
#with steps = 3001, test accuracy = 92.4% (note batch size = 16 here)
#with decaying LR, .1, 100, .95, accuracy = 93.1%.
  