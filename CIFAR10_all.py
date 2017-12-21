import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
import os.path
import math
import pickle
import os,sys

import cifar10
import dataset






##############################################
############ Train model #####################
##############################################

##############################################
############ Train model #####################
##############################################

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


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding ='bytes')
    X = dict[b'data']
    fo.close()
    return dict


def random_batch(data_images, data_labels, batch_size):
    """
    Randomly select a batch of data from training set.
    
    :param: 
          data_images: training images
          data_labels: training labels
          batch_size:  number of data in a batch
    :return:
          x_batch: a batch of image data
          y_batch: a batch of label data

    """
    
    num_img = len(data_images)

    # Create random indices
    idx = np.random.choice(num_img, size=batch_size, replace=False)

    # Use the random indices to select images and labels
    x_batch = data_images[idx, :, :, :]
    y_batch = data_labels[idx, :]

    return(x_batch, y_batch)

def train_CNN_cifar10(input_images, input_labels, train_step, batch_size, num_step):
   """
    Train CIFAR10 dataset on CNN model, which is similar to AlexNet.

    :param:
       input_images: all images available
       input_labels: one-hot coded labels
       batch_size: number of data used to train at one time
       num_step: number of steps for training 
    """
   with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(num_step):
            # Training set
            batch_all = random_batch(data_images=input_images, data_labels=input_labels_labels, batch_size=batch_size)
            train_img = batch_all[0]
            train_lab = batch_all[1]
            
            sess.run(train_step, feed_dict={x: train_img, y: train_lab, pkeep: 0.5})
            
            if i%1000 == 0: 
                valid_acu = sess.run(accuracy, {x: valid_img, y: valid_lab, pkeep:1.0})
                print("\rAfter step {0:3d}, validation accuracy {1:0.4f}".format(i, valid_acu))
            if i%10000 == 0:
                saver = tf.train.Saver()
                saver.save(sess, "./alex_on_cifar10/", global_step=i)



 #############################################
 ######## Visualize maximal activations ######
 #############################################
 
def unPool(value):
    """ 
    A n-dimensional version of the unpooling operation 
    
    :param:
        value: A Tensor of shape [batch_size,  d0,   d1,  ...,  dn,  channel]
    :return: 
        output: A Tensor of shape [batch_size, 2*d0, 2*d1, ..., 2*dn, channel]
    """
        
    # Input_shape : [-1, d0, d1, ..., dn, channel]   
    input_shape = value.get_shape().as_list()
    # Number of dimensions
    num_dim = len(input_shape[1 : -1])       
    # Fix other dimensions except the first two
    output = tf.reshape(value, [-1] + input_shape[-num_dim:]) 
        
    for i in range(num_dim, 0, -1):
        output = tf.concat([output, output], axis=1)
            
    # Output_shape : [-1, 2*d0, 2*d1, ..., 2*dn, channel]   
    output_shape = [-1] + [s * 2 for s in input_shape[1: -1]] + [input_shape[-1]]
    output = tf.reshape(output, output_shape)
    return output                



def display_max_activations_CONV1_cifar10(input_images, input_labels, batch_size, n1):
    """
    Display maximal activation maps in the first convolutional layer for a batch of images.
    
    :paras:
        input_images: all images available
        input_labels: one-hot coded labels
        batch_size: number of images, which is equal to the batch_size for training CNN before
        n1: number of activation maps in the first convolutional layer
    :return:
        maximal activation maps in the first concolutional layer for each input image following with its raw image
    """
    # Randomly select input images with a batch size
    batch_all = random_batch(data_images=input_images, 
                             data_labels=input_labels, 
                             batch_size=batch_size)
    input_img = batch_all[0]
    input_lab = batch_all[1]

     
    ###### Conv_Layer1 ######
    
    featu_Conv1 = tf.placeholder(tf.float32, [batch_size, 32, 32, n1])
    # Un-relu for the relu layer
    un_ReLu = tf.nn.relu(featu_Conv1)
    
    with tf.Session() as sess:
        # Load the saved model
        new_saver = tf.train.import_meta_graph("./output/cifar10/-30000.meta")
        new_saver.restore(sess, tf.train.latest_checkpoint('./output/cifar10/'))
        
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        h_conv1 = graph.get_tensor_by_name("Relu:0")
        
        images = sess.run(x, feed_dict={x: input_img})
        act_Conv1 = sess.run(h_conv1, feed_dict={x: images})
        print("")
        print("For the fist convolutional layer:")
        print("")
        
        W_conv1 = graph.get_tensor_by_name("W_conv1:0")
        # Un-conv for the convolutional layer using the weights already trained and the output from un-relu step
        un_Conv1 = tf.nn.conv2d_transpose(un_ReLu, W_conv1, output_shape=[batch_size, 32, 32, 3], strides=[1, 1, 1, 1], padding='SAME')
        
        fig = plt.figure(0, figsize=(20, 80))
        n_cols = 8
        n_rows = math.ceil(n1*2 / n_cols) + 1
        
        for i in range(batch_size):
            iso = act_Conv1.copy()
            # Choose one image
            pic = iso[i]
            # Find the maximal activation map for this image
            best_one = np.argmax(np.sum(pic, axis=(0, 1)), axis=0)
            
            # Set other images and dimensions equal to 0 except this image with the maximal activation map
            iso[:i, :, :, :]=0
            iso[i+1:, :, :, : ]=0
            iso[i, :, :, :best_one]=0
            iso[i, :, :, best_one+1:]=0
            
            # Show the maximal activation map with pixel values by un-relu and un-conv
            pixel_act = sess.run(un_Conv1, feed_dict={featu_Conv1: iso})
            
            fig.add_subplot(n_rows, n_cols, i*2+1)
            plt.title("Max_act {0} for Image{1}".format(best_one+1, i+1))
            plt.imshow(pixel_act[i])
            
            fig.add_subplot(n_rows, n_cols, i*2+2)
            plt.title("Raw Image {0}".format(i+1))
            plt.imshow(input_img[i])
            
        plt.show()



def display_max_activations_CONV2_cifar10(input_images, input_labels, batch_size, n1, n2):
    """
    Display maximal activation maps in the second convolutional layer for a batch of images.
    
    :paras:
        input_images: all images available
        input_labels: one-hot coded labels
        batch_size: number of images, which is equal to the batch_size for training CNN before
        n1: number of activation maps in the first convolutional layer
        n2: number of activation maps in the second convolutional layer
    :return:
        maximal activation maps in the second convolutional layer for each input image following with its raw image
    """
    
    # Randomly select input images with a batch size
    batch_all = random_batch(data_images=input_images, 
                             data_labels=input_labels, 
                             batch_size=batch_size)
    input_img = batch_all[0]
    input_lab = batch_all[1]

    
    ###### Conv_Layer2 ######
    
    featu_Conv2 = tf.placeholder(tf.float32, [batch_size, 16, 16, n2])
    # Un-relu for the relu layer
    un_ReLu2 = tf.nn.relu(featu_Conv2)
         
    with tf.Session() as sess:
        # Load saved model
        new_saver = tf.train.import_meta_graph("./output/cifar10/-30000.meta")
        new_saver.restore(sess, tf.train.latest_checkpoint('./output/cifar10/'))
        
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        h_conv2 = graph.get_tensor_by_name("Relu_1:0")
        
        images = sess.run(x, feed_dict={x: input_img})
        act_Conv2 = sess.run(h_conv2, feed_dict={x: images})
        print("")
        print("For the second convolutional layer:")
        print("")
        
        W_conv1 = graph.get_tensor_by_name("W_conv1:0")
        W_conv2 = graph.get_tensor_by_name("W_conv2:0")
        
        # Un-conv for the convolutional layer using the corresponding weights already trained and the output from the first un-relu step
        un_Conv2 = tf.nn.conv2d_transpose(un_ReLu2, W_conv2, output_shape=[batch_size, 16, 16, n1], strides=[1, 1, 1, 1], padding="SAME")
        # Un-pool for the maxpool layer
        un_Pool = unPool(un_Conv2)
        # Un-relu for the relu layer
        un_ReLu = tf.nn.relu(un_Pool)
        # Un-conv for the convolutional layer using the corresponding weights already trained and the output from the second un-relu step
        un_Conv = tf.nn.conv2d_transpose(un_ReLu, W_conv1, output_shape=[batch_size, 32, 32, 3] , strides=[1, 1, 1, 1], padding="SAME")
        
    
        fig = plt.figure(0, figsize=(20, 80))
        n_cols = 8
        n_rows = math.ceil(n2*2 / n_cols) + 1
        
        for i in range(batch_size):
            iso = act_Conv2.copy()
            # Choose an image
            pic = iso[i]
            # Find the maximal activation map for this image
            best_one = np.argmax(np.sum(pic, axis=(0, 1)), axis=0)
            
            # Set other images and dimensions equal to 0 except this image with the maximal activation map
            iso[:i, :, :, :]=0
            iso[i+1:, :, :, : ]=0
            iso[i, :, :, :best_one]=0
            iso[i, :, :, best_one+1:]=0
            
            # Show the maximal activation map with pixel values by un-relu, un-pool and un-conv
            pixel_act2 = sess.run(un_Conv, feed_dict={featu_Conv2: iso})
            
            fig.add_subplot(n_rows, n_cols, i*2+1)
            plt.title("Max_act {0} for Image{1}".format(best_one+1, i+1))
            plt.imshow(pixel_act2[i])
            
            fig.add_subplot(n_rows, n_cols, i*2+2)
            plt.title("Raw Image {0}".format(i+1))
            plt.imshow(input_img[i])
            
        plt.show()       


##############################################
######## Saliency map and back propagation ####
##############################################
 
def Grad_to_images_cifar10(layer_name, input_images, input_labels, batch_size):
    """
    Take the gradients of a layer with respect to the input images.
    
    :paras:
        layer_name: the layer wanted to take gradients of
        input_images: all images available
        input_labels: one-hot coded labels
        batch_size: number of input images, which is equal to the batch_size for training CNN before
    :returns:
        Saliency maps or backpropagation ones for some input images following with their raw images
    """
    
    batch_all = random_batch(data_images=input_images, data_labels=input_labels, batch_size=batch_size)
    input_img = batch_all[0]
    
    with tf.Session() as sess:
        # Load the saved model
        new_saver = tf.train.import_meta_graph("./output/cifar10/-30000.meta")
        new_saver.restore(sess, tf.train.latest_checkpoint('./output/cifar10/'))
    
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        layer_vals = graph.get_tensor_by_name(layer_name)
    
        # Calculate the gradients of the layer values with respect to input images
        gradients = tf.stack([tf.gradients(yi, x)[0] for yi in tf.unstack(layer_vals, axis=1)], axis=4)
    
        g_vals = sess.run(gradients, feed_dict={x: input_img})
        print("")
        print("The shape after taking gradients with respect to input images: {0}".format(g_vals.shape))
        
        
        Weights = np.zeros((batch_size, 32, 32))
        for p in range(batch_size):
            # Take absolute values 
            pic = np.abs(g_vals[p])
            for i in range(32):
                for j in range(32):
                    # Choose the one maximize over RGB channels and 10 different classes
                    Weights[p, i, j] = np.max(pic[i, j, :, :])
                    
                    
        print("")
        print("Taking gradients with respect to input images:")
        print("")
        # Randomly choose 4 image indices
        image_idx = np.random.choice(100, size=4, replace=False)
        fig = plt.figure(0, figsize=(20, 10))
        for i in range(4):
            idx = image_idx[i]
            fig.add_subplot(2, 4, 2*i+1)
            plt.imshow(Weights[idx], cmap='gray')
            if layer_name=="Add_4:0":
                plt.title("Saliency map")
                plt.xticks([])
                plt.yticks([])
            else:
                plt.title("Backpropagation")
                plt.xticks([])
                plt.yticks([])
            fig.add_subplot(2, 4, 2*i+2)
            plt.imshow(input_img[idx])
            plt.title("Raw image")
            plt.xticks([])
            plt.yticks([])
        plt.show()


##############################################
####### Visualize activation maps ############
##############################################

def plotCNN_actmaps_cifar10(layer_name, input_images, image_idx):
    """
    Plot the activation maps in specific layer for a specific image
    
    :paras:
        layer_name: the layer wanted to show activation maps
        input_images:  raw iamges with the size equals to the one in training step (batch_size)
        image_idx: the index for an image
    :returns:
        All activation maps for a specific layer for a specific image
    """

    with tf.Session() as sess:
        # Load saved model
        new_saver = tf.train.import_meta_graph("./output/cifar10/-30000.meta")
        new_saver.restore(sess, tf.train.latest_checkpoint('./output/cifar10/'))
        
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        layer = graph.get_tensor_by_name(layer_name)
        # Get the activation maps for the whole input images
        filter_units = sess.run(layer, feed_dict={x: input_images})
        
    num_filters = filter_units.shape[3] 
    
    # Original Image
    plt.imshow(input_images[image_idx])
    plt.title('Original Image')
    plt.xticks([])
    plt.yticks([])
    plt.show()    
    
    fig = plt.figure(0, figsize=(20, 60))
    n_rows = math.ceil(num_filters / 6) + 1
    for i in range(num_filters):
        fig.add_subplot(n_rows, 6, i+1)
        plt.imshow(filter_units[image_idx, :, :, i], interpolation='nearest', cmap='gray')
        plt.title('Activation map {0}'.format(i+1))
        plt.xticks([])
        plt.yticks([])
    plt.show()









