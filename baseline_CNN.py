#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install Pillow')
get_ipython().system('pip install gdown')


# In[3]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import gdown


# In[4]:


tf.debugging.set_log_device_placement(True)


# In[5]:


tf.__version__


# In[6]:


tf.test.gpu_device_name()


# In[7]:


strategy = tf.distribute.MirroredStrategy()


# In[8]:


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[9]:


import pathlib

data_dir = get_ipython().run_line_magic('pwd', '')
data_dir = data_dir + '/combined_data'
data_dir = pathlib.Path(data_dir)
print(data_dir)
test_dir = get_ipython().run_line_magic('pwd', '')
print(test_dir)


# In[10]:


image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


# In[11]:


CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "retrained_labels.txt"])
CLASS_NAMES


# In[12]:


# The 1./255 is to convert from uint8 to float32 in range [0,1].
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

IMG_HEIGHT = 256
IMG_WIDTH = 256


# In[13]:


def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')


# import PIL.Image
# image_batch, label_batch = next(train_data_gen)
# show_batch(image_batch, label_batch)

# In[14]:


list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))


# In[15]:


#transform the images broek out by the text files are in converted to a list of images similar to those read in from list_ds
with open('train_list.txt', 'r') as f:
        train_list = []
        for line in f:
            line = line.rstrip('\n')
            line= line.replace('data', 'tf', 1)
            train_list.append(line)        
        #train_list = [tf.strings.split(x, '/')[:-1] for x in train_list]
        
        

with open('test_list.txt', 'r') as f:
        test_list = []
        for line in f:
            line = line.rstrip('\n')
            line= line.replace('data', 'tf', 1)
            test_list.append(line)        
        #test_list = [tf.strings.split(x, '/')[-1] for x in test_list]
        
with open('val_list.txt', 'r') as f:
        val_list = []
        for line in f:
            line = line.rstrip('\n')
            line= line.replace('data', 'tf', 1)
            val_list.append(line)        
        #val_list = [tf.strings.split(x, '/')[-1] for x in val_list]  
print(train_list[0])
print(type(train_list[0]))


# In[16]:


#simple verification that the images taken from the text files are actually in the folder we read the files in from the list_ds
#input
count = 0
for x in list_ds:
    
    #print(train_list[0])
    if (str(x.numpy())[2:-1]) in train_list:
        continue
    elif (str(x.numpy())[2:-1]) in val_list:
        continue
    elif (str(x.numpy())[2:-1]) in test_list:
        continue
    else:
        count += 1
        print(str(x.numpy())[2:-1])
        print(count)


# In[17]:


train_dataset = tf.data.Dataset.from_tensor_slices(train_list)
valid_dataset = tf.data.Dataset.from_tensor_slices(val_list)
test_dataset  = tf.data.Dataset.from_tensor_slices(test_list)


# In[18]:


# set the shuflle and steps per epochs based on size
train_size = len(list(train_dataset))
print('Train Picture Size = ' + str(train_size))

valid_size = len(list(valid_dataset))
print('Valid Picture Size = ' + str(valid_size))

test_size = len(list(test_dataset))
print('Test Picture Size =  ' + str(test_size))

print(test_size + train_size + valid_size)

train_shuffle_size = train_size
valid_shuffle_size = valid_size
test_shuffle_size = test_size
STEPS_PER_EPOCH = np.ceil(train_size/BATCH_SIZE)
VALID_STEPS_PER_EPOCH = np.ceil(valid_size/BATCH_SIZE)
TEST_STEPS_PER_EPOCH = np.ceil(test_size/BATCH_SIZE)


# In[19]:


for f in train_dataset.take(5):
  print(f.numpy())

for f in valid_dataset.take(5):
  print(f.numpy())

for f in test_dataset.take(5):
  print(f.numpy())


# In[20]:


def get_label(file_path):
  # convert the path to a list of path components
    # use '/' for linux
  parts = tf.strings.split(file_path, '/')
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_labeled_ds = train_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
valid_labeled_ds = valid_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
test_labeled_ds = test_dataset.map(process_path, num_parallel_calls=AUTOTUNE)

#see https://www.tensorflow.org/tutorials/load_data/images for above and cells above taken from there.


# In[21]:


#verify images and labels are joined.

for image, label in train_labeled_ds.take(5):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())
  
    
  inputshape = image.numpy().shape

for image, label in test_labeled_ds.take(5):
  
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())
  inputshape = image.numpy().shape
  
print(inputshape)


# In[22]:


print(train_labeled_ds.take(1))
print(test_labeled_ds.take(1))
test2 = test_labeled_ds.take(1)
print(test2)


# In[41]:


def image_transformation(image, label):
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_jpeg_quality(image, min_jpeg_quality=75, max_jpeg_quality=100)
    #image = tf.image.random_brightness(image, max_delta=.2)
    return image, label

def prepare_for_training(ds, shuffle_buffer_size, cache=False):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()
  ds = ds.map(image_transformation, num_parallel_calls=AUTOTUNE)

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds

#the above mostly constructed from TensorFlow tutorial:
#https://www.tensorflow.org/tutorials/load_data/images


# In[24]:


test1 = prepare_for_training(train_labeled_ds, train_shuffle_size)


# image_batch, label_batch = next(iter(test1))
# #image_batch, label_batch = train_ds.take(5)
# show_batch(image_batch.numpy(), label_batch.numpy())

# In[25]:


import keras
from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras import optimizers 
from keras.layers.advanced_activations import LeakyReLU


# In[45]:


#Building a CNN with 1 convulutional layer, an actication layer, a pooling layer then flatten to go into the fully connected
# layer
with strategy.scope():
    model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(100, kernel_size = (3,3), input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.MaxPooling2D((2,2), padding='same'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(50, kernel_size = (5,5)),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.MaxPooling2D((2,2), padding='same'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(25, kernel_size = (5,5)),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.MaxPooling2D((2,2), padding='same'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(15, kernel_size = (3,3)),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.MaxPooling2D((2,2), padding='same'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(5, kernel_size = (3,3)),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.MaxPooling2D((2,2), padding='same'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dense(2, activation = 'softmax')])
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])


# In[27]:


from time import time
print(time())


# In[46]:


start = time()
model.fit(
        prepare_for_training(train_labeled_ds, train_shuffle_size),
        steps_per_epoch = STEPS_PER_EPOCH,
        epochs = 200,
        validation_data=prepare_for_training(valid_labeled_ds, valid_size),
        validation_steps= VALID_STEPS_PER_EPOCH)

end = time()

print((end - start)/60) 
    


# In[47]:


score = model.evaluate(prepare_for_training(test_labeled_ds, test_shuffle_size), steps = TEST_STEPS_PER_EPOCH, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


#potential to save model if needed.
model.save(
    '/saved_model2/my6',
    save_format='tf'    
)


# References:
# https://stackoverflow.com/questions/51125266/how-do-i-split-tensorflow-datasets
# see paper for more references:
# Baseline CNN:  utlized the basis of many CNN from https://towardsdatascience.com/the-4-convolutional-neural-network-models-that-can-classify-your-fashion-images-9fe7f3e5399d, https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5, tutorials in tensorflow.org such as https://www.tensorflow.org/guide/data, https://www.tensorflow.org/tutorials/load_data/images, https://www.tensorflow.org/guide/distributed_training, and https://www.tensorflow.org/guide/gpu.  Also, used articles from medium.com and towardsdatascience.com..  Also, used articles from medium.com and towardsdatascience.com.
# Also, used stackoverflow and other internet searches for understanding, trouble shooting, and general use.  Further context can be provided if needed.

# In[ ]:




