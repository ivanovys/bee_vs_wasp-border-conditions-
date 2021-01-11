# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 01:20:33 2021

@author: ivanov
"""


import pandas as pd, numpy as np
import tensorflow as tf


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

import cv2
import os
import math
from math import ceil, floor, log

import glob
import shutil

from PIL import Image , ImageOps

# to be used to get better performance
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

import datetime

AUTOTUNE = tf.data.experimental.AUTOTUNE


padding = "SAME"  
num_output_classes = 2  
batch_size = 8  
learning_rate = 0.001  


print("TF version:", tf.__version__)
print(tf.config.list_physical_devices('GPU'))

source_dir="C:\\data\\beeVSwasp\\"


path_bees = glob.glob(source_dir+"bee/*")
path_wasps = glob.glob(source_dir+"wasp/*")

print(f'bees: {len(path_bees)}')
print(f'wasps: {len(path_wasps)}')

pd_bees = pd.DataFrame(path_bees,columns=["filename"]) 
pd_wasps = pd.DataFrame(path_wasps,columns=["filename"]) 
pd_bees["label"]=0#[[0,1]]*len(pd_bees)
pd_wasps["label"]=1#[[1,0]]*len(pd_wasps)


label_name={'bee': 0, 'wasp': 1}

df=pd.concat([pd_bees,pd_wasps], ignore_index=True)

#df["Image"]=[np.asarray(Image.open(filename))/255 for filename in df.filename]


def parse_function(filename, label):
    image_string = tf.io.read_file(filename)

    #Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    #This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize(image, [300, 300])  #[224, 224]  [256, 256]
    return image, label


def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    #Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    label=tf.one_hot( label , depth=2 )

    return image, label
    
 
  


pd_train, pd_test= train_test_split(df, test_size=0.1)
# pd_train, pd_val= train_test_split(pd_train, test_size=0.23)  #0.9*0.23=0.2


#train_ds = tf.data.Dataset.from_tensor_slices((get_images(pd_train.filename), pd_train.label))



train_ds = tf.data.Dataset.from_tensor_slices((pd_train.filename, pd_train.label))
train_ds = train_ds.shuffle(len(pd_train.label))
train_ds = train_ds.map(parse_function, num_parallel_calls=4)
train_ds = train_ds.map(train_preprocess, num_parallel_calls=4)
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.prefetch(1)


test_ds = tf.data.Dataset.from_tensor_slices((pd_test.filename, pd_test.label))
test_ds = test_ds.shuffle(len(pd_test.label))
test_ds = test_ds.map(parse_function, num_parallel_calls=4)
test_ds = test_ds.map(train_preprocess, num_parallel_calls=4)
test_ds = test_ds.batch(batch_size)
test_ds = test_ds.prefetch(1)


# val_ds = tf.data.Dataset.from_tensor_slices((pd_val.filename, pd_val.label))
# val_ds = val_ds.shuffle(len(pd_val.label))
# val_ds = val_ds.map(parse_function, num_parallel_calls=4)
# val_ds = val_ds.map(train_preprocess, num_parallel_calls=4)
# val_ds = val_ds.batch(batch_size)
# val_ds = val_ds.prefetch(1)






def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


sample_training_images, _ = next(train_ds.__iter__())

plotImages(sample_training_images[:5])






for feat, targ in train_ds.take(1):
  print ('Путь: {}, Метка: {}'.format(feat, targ))
  print(type(feat.numpy()))
  c=feat.numpy()
  print(c.shape)

# plt.imshow(c[10,:,:,:])


leaky_relu_alpha = 0.2
dropout_rate = 0.5
padding = "SAME"  

def conv2d( inputs , filters , stride_size, b ):
    out = tf.nn.conv2d( inputs , filters , strides=[ 1 , stride_size , stride_size , 1 ] , padding=padding ) + b
    return tf.nn.leaky_relu( out , alpha=leaky_relu_alpha ) 

def maxpool( inputs , pool_size , stride_size ):
    return tf.nn.max_pool2d( inputs , ksize=[ 1 , pool_size , pool_size , 1 ] , padding='VALID' , strides=[ 1 , stride_size , stride_size , 1 ] )

def dense( inputs , weights,b ):
    x = tf.nn.leaky_relu( tf.matmul( inputs , weights ) , alpha=leaky_relu_alpha ) +b
    return tf.nn.dropout( x , rate=dropout_rate )




output_classes = 2


initializer = tf.initializers.glorot_uniform()
def get_weight( shape , name ):
    
    return tf.Variable( initializer( shape ) , name=name , trainable=True , dtype=tf.float32 )

def get_zeros( shape , name ):
    return tf.zeros(  shape=shape  , name=name ,  dtype=tf.float32 )

shapes = [
    [ 3 , 3 , 3 , 64 ] ,   #conv0
    [ 3 , 3 , 64 , 64 ] ,   #conv1
                            # pool
    [ 3 , 3 , 64 , 128 ] ,  #conv2
    [ 3 , 3 , 128 , 128 ] , #conv3
                            # pool
    [ 3 , 3 , 128 , 256 ] , #conv4
    [ 3 , 3 , 256 , 256 ] , #conv5
    [ 3 , 3 , 256 , 256 ] , #conv6
                            # pool
    [ 3 , 3 , 256 , 512 ] , #7
    [ 3 , 3 , 512 , 512 ] , #8
    [ 3 , 3 , 512 , 512 ] , #9
                            #pool
    [ 3 , 3 , 512 , 512 ] , #10
    [ 3 , 3 , 512 , 512 ] , #11
    [ 3 , 3 , 512 , 512 ] , #12
                            # pool
    [ 41472, 4096] ,            #13
    [ 4096, 4096 ] ,              #14
    [ 4096 , 1024 ] ,            #15
    [ 1024 , output_classes ]  ,#16
]

weights = []
for i in range( len( shapes ) ):
    weights.append( get_weight( shapes[ i ] , 'weight{}'.format( i ) ) )



bias = []
for i in range(len(shapes)):
  bias.append( get_zeros([1,shapes[i][-1]] , 'bias{}'.format( i ) ) )



def model( x ) :
    x = tf.cast( x , dtype=tf.float32 )
    c1 = conv2d( x , weights[ 0 ] , stride_size=1, b=bias[0] ) 
    c1 = conv2d( c1 , weights[ 1 ] , stride_size=1, b=bias[1]  ) 
    p1 = maxpool( c1 , pool_size=2 , stride_size=2 )
    
    c2 = conv2d( p1 , weights[ 2 ] , stride_size=1, b=bias[2]  )
    c2 = conv2d( c2 , weights[ 3 ] , stride_size=1, b=bias[3] ) 
    p2 = maxpool( c2 , pool_size=2 , stride_size=2 )
    
    c3 = conv2d( p2 , weights[ 4 ] , stride_size=1, b=bias[4] ) 
    c3 = conv2d( c3 , weights[ 5 ] , stride_size=1, b=bias[5] ) 
    c3 = conv2d( c3 , weights[ 6 ] , stride_size=1, b=bias[6] ) 
    p3 = maxpool( c3 , pool_size=2 , stride_size=2 )
    
    c4 = conv2d( p3 , weights[ 7 ] , stride_size=1, b=bias[7]  )
    c4 = conv2d( c4 , weights[ 8 ] , stride_size=1, b=bias[8]  )
    c4 = conv2d( c4 , weights[ 9 ] , stride_size=1, b=bias[9]  )
    p4 = maxpool( c4 , pool_size=2 , stride_size=2 )

    c5 = conv2d( p4 , weights[ 10 ] , stride_size=1, b=bias[10] )
    c5 = conv2d( c5 , weights[ 11 ] , stride_size=1, b=bias[11] )
    c5 = conv2d( c5 , weights[ 12 ] , stride_size=1, b=bias[12] )
    p5 = maxpool( c5 , pool_size=2 , stride_size=2 )

    flatten = tf.reshape( p5 , shape=( tf.shape( p5 )[0] , -1 ))

    d1 = dense( flatten , weights[ 13 ], b=bias[13] )
    d2 = dense( d1 , weights[ 14 ] , b=bias[14])
    d3 = dense( d2 , weights[ 15 ] , b=bias[15])
    
    
    d4 = dense( d3 , weights[ 16 ], b=bias[16] )
    
    
    logits = tf.nn.softmax( d4 )
    
    return  logits 
    
    
    # logits = tf.matmul( d3 , weights[ 16 ] )

    # return tf.nn.softmax( logits )




def loss( pred , target ):
    return tf.losses.binary_crossentropy( target , pred )

optimizer = tf.optimizers.Adam( learning_rate )





logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer( logdir + "/metrics")
file_writer.set_as_default()

import os
os.system('tensorboard --logdir=')

num_epochs = 5000 #@param {type: "number"}

def test_accuracy(model,dataset):
  """Computes accuracy on the test set."""
  correct, total = 0, 0
 # i=0
  for images, labels in dataset:
    labels=np.argmax(labels, axis=1)
    preds = tf.argmax(model(images), axis=1)
    preds=tf.dtypes.cast(preds, tf.int32)
    correct += tf.math.count_nonzero(tf.equal(preds, labels), dtype=tf.int32)
    total += tf.shape(labels)[0]
    #print(i)
   # i=i+1
  accuracy = (correct / tf.cast(total, tf.int32)) * 100.
  
  return {"accuracy": accuracy, "incorrect": total - correct}


def train_step( model, optimizer, inputs ,  outputs ):
    with tf.GradientTape() as tape:
        current_loss = loss( model( inputs ), outputs)
        grads = tape.gradient( current_loss ,weights )
    
        optimizer.apply_gradients( zip( grads ,weights) )
    #print( tf.reduce_mean( current_loss ) )
    return tf.reduce_mean( current_loss )


def train_epoch( model, optimizer, dataset):
  loss = 0.
  for images, labels in dataset:
      loss = train_step(model, optimizer, images,  labels )
  return loss



train_loss_results = []
train_accuracy_results = []



for e in range( num_epochs ):
    print( 'Epoch {} out of {} {}'.format( e + 1 , num_epochs , '--' * 50 ) ) 
    
    #train_loss = train_epoch(model, optimizer, train_ds)
    test_metrics = test_accuracy(model, test_ds)
 
    summ_loss=0
    
    for image , label in train_ds:
        #label=np.argmax(label, axis=1)
        summ_loss = train_step( model , optimizer, image ,  label  )
        
    if e%1 == 0:
        print(f"[Epoch {e}] train loss: {summ_loss}, test acc: {test_metrics['accuracy']} ({test_metrics['incorrect']} wrong)")
    
    if(summ_loss is not None):
        train_loss_results.append(summ_loss)
        train_accuracy_results.append(test_metrics)



plt.plot(train_loss_results)
plt.title('Loss')
plt.show()
plt.title('Accuracy')
plt.plot(train_accuracy_results)
plt.show()




for image , label in test_ds:
    
    preds = tf.argmax(model(image), axis=1)
     
    
   # print (label.numpy())
    prediction = model(image)
    print (prediction.numpy())
    


image , label  =next( test_ds.__iter__())


print(f"pred \n{model(image)} \n lab \n{label}" )


D=test_accuracy(model,train_ds)

        
        
