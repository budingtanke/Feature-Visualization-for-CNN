import os,sys,os.path
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.spatial import distance
from sklearn.manifold import TSNE
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import math
import pickle


def conv2d(x, W):
    return(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'))
    

def max_pool22(x):
    return(tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'))

    
def max_pool33(x):
    return(tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME'))
    



def compute_cross_entropy(logits, y):
    cross_entropy_terms = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits, name='cross_entropy_terms')
    cross_entropy = tf.reduce_mean(cross_entropy_terms, name='cross_entropy')
    return(cross_entropy)


def compute_accuracy(logits, y):
    pred_labels = tf.argmax(logits, 1, name='pred_labels')
    true_labels = tf.argmax(y, 1, name='true_labels')
    acu = tf.reduce_mean(tf.cast(tf.equal(pred_labels, true_labels), tf.float32))
    return(acu)
    
def train_MNIST(mnist,train_step,x,y,pkeep,accuracy,batch_size, num_step):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for i in range(num_step):
			# Generate training set
			batch = mnist.train.next_batch(batch_size)
			X_batch = batch[0]
			y_batch = batch[1]
		
	
			# Train CNN on training set
			sess.run(train_step, feed_dict={x: X_batch, y: y_batch,pkeep:1.0})

			# For each 100 step calculate the accuracy for validation set and print it out
			if i%100 == 0: 
				valid_acu = sess.run(accuracy, {x:mnist.test.images, y:mnist.test.labels, pkeep:1.0})
				print("\rAfter step {0:3d}, validation accuracy {1:0.4f}".format(i, valid_acu))

			# For each 1000 step save the model and corresponding data
			if i%1000 == 0:
				saver = tf.train.Saver()
				saver.save(sess, "./output/mnist/", global_step=i)

def VisualizeCNNLayer1(image):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('./output/mnist/-1000.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./output/mnist/'))
        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name('x:0')
        y = graph.get_tensor_by_name('y:0')
        pkeep = graph.get_tensor_by_name('pkeep:0')

        h_conv1 = graph.get_tensor_by_name('Relu:0')
        
        layer = sess.run(h_conv1,feed_dict={x:np.reshape(image,[1,784],order='F'),pkeep:1.0})
    
    filters = layer.shape[3]

    plt.figure(1, figsize=(20,15))
    n_columns = 8
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Activation map ' + str(i))
        plt.imshow(layer[0,:,:,i], interpolation="nearest", cmap="gray")

def VisualizeCNNLayer2(image):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('./output/mnist/-1000.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./output/mnist/'))
        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name('x:0')
        y = graph.get_tensor_by_name('y:0')
        pkeep = graph.get_tensor_by_name('pkeep:0')

        h_conv2 = graph.get_tensor_by_name('Relu_1:0')
        
        layer = sess.run(h_conv2,feed_dict={x:np.reshape(image,[1,784],order='F'),pkeep:1.0})
    
    filters = layer.shape[3]

    plt.figure(1, figsize=(20,30))
    n_columns = 8
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Activation map ' + str(i))
        plt.imshow(layer[0,:,:,i], interpolation="nearest", cmap="gray")

def VisualizeWeightWconv1(image):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('./output/mnist/-1000.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./output/mnist/'))
        graph = tf.get_default_graph()

        #x = graph.get_tensor_by_name('x:0')
        #y = graph.get_tensor_by_name('y:0')
        #pkeep = graph.get_tensor_by_name('pkeep:0')

        W_conv1 = graph.get_tensor_by_name('W_conv1:0')
        layer = sess.run(W_conv1)
    
    filters = layer.shape[3]

    plt.figure(1, figsize=(20,20))
    n_columns = 8
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Number ' + str(i))
        plt.imshow(layer[:,:,0,i], interpolation="nearest", cmap="gray")

