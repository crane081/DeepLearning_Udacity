# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 22:43:52 2016

Deep learning Assignment 1:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_notmnist.ipynb

@author: cranehuang
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

#%%
url = 'http://yaroslavvb.com/upload/notMNIST/'

def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify' + filename + '. Can you get to it with a browser?')
  return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

#%%
num_classes = 10
np.random.seed(133)


def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

#%% Problem 1

from PIL import Image
image=Image.open('MDEtMDEtMDAudHRm.png')
image.show()


#%%
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  image_index = 0
  print(folder)
  for image in os.listdir(folder):
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[image_index, :, :] = image_data
      image_index += 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  num_images = image_index
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

#%% Problem 2
#cd notMNIST_small/

f=open('A.pickle','rb')
data=pickle.load(f)
f.close()

img_test=data[2,:,:]
imgplot=plt.imshow(img_test)

#%% Problem 3

letters=['A','B','C','D','E','F','G','H','I','J']
n_letters=np.zeros(len(letters))
t=0
for i in letters:
    filename = i + '.pickle'
    f=open(filename,'rb')
    data=pickle.load(f)
    f.close()
    n_letters[t]=data.shape[0]
    t=t+1
    
print(n_letters)

#%%
def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

#%%
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

#%% Problem 4
test1=train_dataset[10,:,:]
plt.imshow(test1)

#%%
pickle_file = 'notMNIST.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

#%%  
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)  


#%% Problem 5
f=open('notMNIST.pickle','rb')
data=pickle.load(f)
f.close()

for key in data:
    print(key)
 
train_dataset=data['train_dataset']
test_dataset=data['test_dataset']
valid_dataset=data['valid_dataset']

#%% method 1
#test_ol_train=0
#for i in range(len(test_dataset)):
#    for j in range(len(train_dataset)):
#        if np.array_equal(test_dataset[i,10,:],train_dataset[j,10,:]):
#            if np.array_equal(test_dataset[i,:,:],train_dataset[j,:,:]):
#                test_ol_train=test_ol_train+1
#                print(i)
#                break
#%% method 2
test_ol_train=0
for i in range(len(test_dataset)):
    id_c=12
    id0=np.where(test_dataset[i,id_c,:]!=-.5)[0]
    if not id0.any():
        id_c=27
        id0=np.where(test_dataset[i,id_c,:]!=-.5)[0]
    idx=[]
    idx=np.where(train_dataset[:,id_c,id0[0]]==test_dataset[i,id_c,id0[0]])[0]
    for j in range(len(idx)):
        if np.array_equal(test_dataset[i,:,:],train_dataset[idx[j],:,:]):
            test_ol_train=test_ol_train+1
            print(i)
            break

print('%d out of %d test images are in the training set' % (test_ol_train,len(test_dataset)))     

#: test_ol_train=1324
#%%
test_ol_valid=0
for i in range(len(test_dataset)):
    id_c=12
    id0=np.where(test_dataset[i,id_c,:]!=-.5)[0]
    if not id0.any():
        id_c=27
        id0=np.where(test_dataset[i,id_c,:]!=-.5)[0]
    idx=[]
    idx=np.where(valid_dataset[:,id_c,id0[0]]==test_dataset[i,id_c,id0[0]])[0]
    for j in range(len(idx)):
        if np.array_equal(test_dataset[i,:,:],valid_dataset[idx[j],:,:]):
            test_ol_valid=test_ol_valid+1
            print(i)
            break
        
print('%d out of %d test images are in the validation set' % (test_ol_valid,len(test_dataset)))     
    
#: test_ol_valid=200

#%% problem 6
N_train = 5000
N_test=50

x_train=np.reshape(train_dataset[0:N_train,:,:],(N_train,28*28))
y_train=train_labels[0:N_train]

x_test=np.reshape(test_dataset[0:N_test,:,:],(N_test,28*28))
y_test=test_labels[0:N_test]

#%% method 1: just logistic regression (slow)
lr=LogisticRegression()

lr.fit(x_train,y_train)

print('LogisticRegression score: %f' % lr.score(x_test,y_test))

#accuracy: .46, .62, .72 and .74 for sample size: 50, 100, 1000 and 5000

#%% method2: SGD logistic regression (much faster)
from sklearn.linear_model import SGDClassifier
lr_SGD=SGDClassifier(loss='log',n_iter=10)
lr_SGD.fit(x_train,y_train)

print('LogisticRegression using SGD score: %f' % lr_SGD.score(x_test,y_test))

