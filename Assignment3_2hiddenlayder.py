# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:57:52 2016

Deep learning Assignment 3:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/3_regularization.ipynb

@author: chuang
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle


#%%
pickle_file = 'notMNIST.pickle'

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

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
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

#%% Problem 1 & 2
prob=4

if prob==1:
    batch_size = 128
elif prob ==4:
    batch_size = 128
else:
    batch_size = 10

beta = 0#0.0001 #Prob1 (.001)

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(.1,shape=shape)
    return tf.Variable(initial)

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Number of hidden variables:
  n_hlayer = 2
  
  n0_hidden=[1024, 512, 256] #hidden notes at each layer
  
  n_hidden=n0_hidden[0:n_hlayer]
  n_hidden.append(num_labels)
  
  w1 = weight_variable([image_size * image_size, n_hidden[0]])
  b1 = bias_variable([n_hidden[0]])
  
  h1=tf.nn.relu(tf.matmul(tf_train_dataset,w1)+b1)
  
  keep_prob = tf.placeholder(tf.float32) #prob 3
  h_drop=tf.nn.dropout(h1,keep_prob)
  
  if n_hlayer==1: #only 1 hidden layer
      wo=weight_variable([n_hidden[0],num_labels])
      bo=bias_variable([num_labels])
      logits=tf.matmul(h1,wo)+bo       
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))            
      l2reg=tf.nn.l2_loss(w1)+tf.nn.l2_loss(b1)+tf.nn.l2_loss(wo)+tf.nn.l2_loss(bo)
      
  else:     
      #Prob4: add hidden layers
#      w2=weight_variable([n_hidden[0],n_hidden[1]])
#      b2=bias_variable([n_hidden[1]])
#      h2=tf.nn.relu(tf.matmul(h1,w2)+b2)
#
#      if n_hlayer==2:
#         wo=weight_variable([n_hidden[1],num_labels])
#         bo=bias_variable([num_labels])
#         logits = tf.matmul(h2,wo)+bo
#      
#         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,tf_train_labels))
#         l2reg = tf.nn.l2_loss(w1)+tf.nn.l2_loss(b1)+tf.nn.l2_loss(w2)+tf.nn.l2_loss(b2) +tf.nn.l2_loss(wo)+tf.nn.l2_loss(bo)
#      elif n_hlayer==3:
#     
#          w3=weight_variable([n_hidden[1],n_hidden[2]])
#          b3=bias_variable([n_hidden[2]])
#          h3=tf.nn.relu(tf.matmul(h2,w3)+b3)
#          
#          wo=weight_variable([n_hidden[2],num_labels])
#          bo=bias_variable([num_labels])
#         
#          logits = tf.matmul(h3,wo)+bo
#          
#          loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,tf_train_labels))
#          l2reg = tf.nn.l2_loss(w1)+tf.nn.l2_loss(b1)+tf.nn.l2_loss(w2)+tf.nn.l2_loss(b2) +tf.nn.l2_loss(wo)+tf.nn.l2_loss(bo) + tf.nn.l2_loss(w3)+tf.nn.l2_loss(b3)
      l2reg=tf.nn.l2_loss(w1)+tf.nn.l2_loss(b1)
      
      for layer in range(n_hlayer-1):
           w=weight_variable(n_hidden[layer],n_hidden[layer+1])
           b=bias_variable([n_hidden[layer+1]])
           l2reg=l2reg+tf.nn.l2_loss(w)+tf.nn.l2_loss(b)
           
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,tf_train_labels))

     
  loss=loss+beta*l2reg
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  
  for layer in range(n_hlayer):
      if layer==0:
          relu_out=tf.matmul(tf_valid_dataset,w1)+b1
      elif layer==1:
          relu_out=tf.matmul(tf.nn.relu(relu_out),w2)+b2
      elif layer ==2:
          relu_out=tf.matmul(tf.nn.relu(relu_out),w3)+b3
          
  
  relu_out=tf.matmul(tf.nn.relu(relu_out),wo)+bo
  valid_prediction = tf.nn.softmax(relu_out)
  
  for layer in range(n_hlayer):
      if layer==0:
          relu_out=tf.matmul(tf_test_dataset,w1)+b1
      elif layer==1:
          relu_out=tf.matmul(tf.nn.relu(relu_out),w2)+b2
      elif layer ==2:
          relu_out=tf.matmul(tf.nn.relu(relu_out),w3)+b3
  
  relu_out=tf.matmul(tf.nn.relu(relu_out),wo)+bo
  test_prediction = tf.nn.softmax(relu_out)
           

#%%
num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
#    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels,keep_prob:0.5}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

# Prob1: test accuray improved to 93.2% with L2 regularization with beta = .002
# Prob2: test accuracy dropped to 55% with batch = 10
# Prob3: test accuracy dropped to 55% with batch = 10, keep_prob = 1/.5 (?)
# Prob4: a. test accuracy with 2 hidden layers (1024 and 512): 94.6% with beta = 0, dropout =.5 
# Prob4: b. add learning rate decay (WIP)
