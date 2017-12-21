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


#############################################
################# Read data #################
#############################################
def read_all_images(path_to_data):
    """
    This function read SFL-10 dataset 
    :param:
        path_to_data: the file containing the binary images from the STL-10 dataset
    :return:
        images: with shape (13000, 96, 96, 3)
        labels: one hot labels with shape (13000, 10)
    """
    # path to the binary train file with image data
    DATA_PATH_train = os.path.join(path_to_data,'train_X.bin')

    # path to the binary train file with labels
    LABEL_PATH_train = os.path.join(path_to_data,'train_y.bin')

    # path to the binary test file with image data
    DATA_PATH_test = os.path.join(path_to_data,'test_X.bin')

    # path to the binary test file with labels
    LABEL_PATH_test = os.path.join(path_to_data,'test_y.bin')
                          
    with open(LABEL_PATH_train, 'rb') as f:
        rawlabels_train = np.fromfile(f, dtype=np.uint8)
        rawlabels_train = rawlabels_train -1
        labels_train = np.zeros((len(rawlabels_train),10))
        labels_train[np.arange(len(rawlabels_train)),rawlabels_train] = 1
    
    with open(LABEL_PATH_test, 'rb') as f:
        rawlabels_test = np.fromfile(f, dtype=np.uint8)
        rawlabels_test = rawlabels_test -1
        labels_test = np.zeros((len(rawlabels_test),10))
        labels_test[np.arange(len(rawlabels_test)),rawlabels_test] = 1
                               
    labels = np.concatenate((labels_train,labels_test))
    raw_labels = np.argmax(labels,axis = 1)
                                                 
    with open(DATA_PATH_train, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)
        images_train = np.reshape(everything, (-1, 3, 96, 96))
        images_train = np.transpose(images_train, (0, 3, 2, 1))
    with open(DATA_PATH_test, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)
        images_test = np.reshape(everything, (-1, 3, 96, 96))
        images_test = np.transpose(images_test, (0, 3, 2, 1))
    images = np.concatenate((images_train,images_test))
    images = images/255.0
    
    print("shape of images: {}".format(images.shape))
    print("shape of labels: {}".format(labels.shape))
    print("shape of raw labels: {}".format(raw_labels.shape))
    return(images, labels, raw_labels)


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


def train_CNN_stl10(input_images, input_labels, batch_size, num_step):
    """
    Train STL10 dataset on CNN model, which is similar to AlexNet.

    :param:
       input_images: all images available
       input_labels: one-hot coded labels
       batch_size: number of data used to train at one time
       num_step: number of steps for training 
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(num_step):
            # Generate training set
            batch_all = random_batch(data_images=input_images[:10000, :, :, :], data_labels=input_labels[:10000, :], batch_size=batch_size)
            train_img = batch_all[0]
            train_lab = batch_all[1]
            
            # Train CNN on training set
            sess.run(train_step, feed_dict={x: train_img, y: train_lab})
            
            # For each 100 step calculate the accuracy for validation set and print it out
            if i%100 == 0: 
                valid_acu = sess.run(accuracy, {x: valid_img, y: valid_lab})
                print("\rAfter step {0:3d}, validation accuracy {1:0.4f}".format(i, valid_acu))
                
            # For each 1000 step save the model and corresponding data
            if i%1000 == 0:
                saver = tf.train.Saver()
                saver.save(sess, "./alex_on_stl10/", global_step=i)
 
 
 
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


                
def display_max_activations_CONV1_stl10(input_images, input_labels, batch_size, n1):
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
    batch_all = random_batch(data_images=input_images[:10000, :, :, :], data_labels=input_labels[:10000, :], batch_size=batch_size)
    input_img = batch_all[0]
    input_lab = batch_all[1]

     

    ###### Conv_Layer1 ######

    featu_Conv1 = tf.placeholder(tf.float32, [batch_size, 96, 96, n1])
    # Un-relu for the relu layer
    un_ReLu = tf.nn.relu(featu_Conv1)
    
    with tf.Session() as sess:
        # Load the saved model and its checkpoint
        new_saver = tf.train.import_meta_graph("./output/stl10/alex_on_stl10-2000.meta")
        new_saver.restore(sess, tf.train.latest_checkpoint('./output/stl10/'))
        
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
        un_Conv1 = tf.nn.conv2d_transpose(un_ReLu, W_conv1, output_shape=[batch_size, 96, 96, 3], strides=[1, 1, 1, 1], padding='SAME')
        
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
            plt.xticks([])
            plt.yticks([])
            
            fig.add_subplot(n_rows, n_cols, i*2+2)
            plt.title("Raw Image {0}".format(i+1))
            plt.imshow(input_img[i])
            plt.xticks([])
            plt.yticks([])
            
        plt.show()
 
 
        
def display_max_activations_CONV2_stl10(input_images, input_labels, batch_size, n1, n2):
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
    batch_all = random_batch(data_images=input_images[:10000, :, :, :], data_labels=input_labels[:10000, :],batch_size=batch_size)
    input_img = batch_all[0]
    input_lab = batch_all[1]


    

    ###### Conv_Layer2 ######

    featu_Conv2 = tf.placeholder(tf.float32, [batch_size, 48, 48, n2])
    # Un-relu for the relu layer
    un_ReLu2 = tf.nn.relu(featu_Conv2)
         
    with tf.Session() as sess:
        # Load the saved model 
        new_saver = tf.train.import_meta_graph("./output/stl10/alex_on_stl10-2000.meta")
        new_saver.restore(sess, tf.train.latest_checkpoint('./output/stl10/'))
        
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
        un_Conv2 = tf.nn.conv2d_transpose(un_ReLu2, W_conv2, output_shape=[batch_size, 48, 48, n1], strides=[1, 1, 1, 1], padding="SAME")
        # Un-pool for the maxpool layer
        un_Pool = unPool(un_Conv2)
        # Un-relu for the relu layer
        un_ReLu = tf.nn.relu(un_Pool)
        # Un-conv for the convolutional layer using the corresponding weights already trained and the output from the second un-relu step
        un_Conv = tf.nn.conv2d_transpose(un_ReLu, W_conv1, output_shape=[batch_size, 96, 96, 3] , strides=[1, 1, 1, 1], padding="SAME")
        
    
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
            plt.xticks([])
            plt.yticks([])
            
            fig.add_subplot(n_rows, n_cols, i*2+2)
            plt.title("Raw Image {0}".format(i+1))
            plt.imshow(input_img[i])
            plt.xticks([])
            plt.yticks([])
            
        plt.show()
 
 
##############################################
######## Saliency map and back propagation ###
##############################################
 
def Grad_to_images_stl10(layer_name, input_images, input_labels, batch_size):
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
    
    batch_all = random_batch(data_images=input_images[:10000, :, :, :], data_labels=input_labels[:10000, :], batch_size=batch_size)
    input_img = batch_all[0]
    
    with tf.Session() as sess:
        # Load the saved model
        new_saver = tf.train.import_meta_graph("./output/stl10/alex_on_stl10-2000.meta")
        new_saver.restore(sess, tf.train.latest_checkpoint('./output/stl10/'))
    
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        layer_vals = graph.get_tensor_by_name(layer_name)
    
        # Calculate the gradients of the layer values with respect to input images
        gradients = tf.stack([tf.gradients(yi, x)[0] for yi in tf.unstack(layer_vals, axis=1)], axis=4)
    
        g_vals = sess.run(gradients, feed_dict={x: input_img})
        print("")
        print("The shape after taking gradients with respect to input images: {0}".format(g_vals.shape))
        
        
        Weights = np.zeros((batch_size, 96, 96))
        for p in range(batch_size):
            # Take absolute values 
            pic = np.abs(g_vals[p])
            for i in range(96):
                for j in range(96):
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

def plotCNN_actmaps_stl10(layer_name, input_images, image_idx):
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
        # Load the saved model
        new_saver = tf.train.import_meta_graph("./output/stl10/alex_on_stl10-2000.meta")
        new_saver.restore(sess, tf.train.latest_checkpoint('./output/stl10/'))
        
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
       

##############################################
############ Visualize last layer ############
##############################################


def RandomImagesNeeded(images, labels,num_img,lab_needed = list(range(10))):
    """
    This function selects some random images from images of STL-10 with specified labels
    For example, we want 10 images of plane, monkey and ship, then the function will randomly select
    10 images of plane, monkey and ship
    
    :param:
        images: original images array with dimension (130000, 96, 96, 3)
        labels: one hot labels of all iamges with dimension (130000, 10)
        num_img: number of images we want
        lab_needed: specified labels we want
    :return:
        x_batch: selected images with dimension (num_img, 96, 96, 3)
        y_batch: one hot labels of selected images with dimension (num_img, 10)
        idx: index of images selected
    """

    raw_labels = np.argmax(labels,axis = 1)
    needed_index = ([i for i,x in enumerate(raw_labels) if x in lab_needed])
    
    # Create a random index.
    idx = np.random.choice(needed_index, size=num_img, replace=False)

    # Use the random index to select random images and labels.
    x_batch = images[idx, :, :, :]
    y_batch = labels[idx, :]
    return(x_batch,y_batch,idx)


def Extract192Features(random_images,random_labels):
    """
    This function extracts last layer features with dimension 192
    :param:
        random_images: all images we need to extract features with shape (n, 96, 96, 3)
        random_labels: corresponding labels of images (one hot)
    :return:
        features: 192 features of all images, each row is an image, and each column is a feature
    """

    features = np.array([]).reshape((0,192))
    num_images = random_images.shape[0]


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('./output/stl10/alex_on_stl10-2000.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./output/stl10/'))
        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name('x:0')
        y = graph.get_tensor_by_name('y:0')
        pkeep = graph.get_tensor_by_name('pkeep:0')

        h_fc2 = graph.get_tensor_by_name('Relu_3:0')
        
        for i in range(int(num_images/100)):
            this_feature = sess.run(h_fc2, feed_dict={x: random_images[i*100:(i+1)*100], y: random_labels[i*100:(i+1)*100], pkeep:1.0})
            features = np.concatenate((features,this_feature),axis=0)

    return features




def tSNE_show(all_images,features,labels,idx):
    """
    This function calculates tSNE reduced two dimension features,  
    shows tSNE clusters and reorder images according to reduced features
    :param:
        all_images: all images array with dimension (13000, 96, 96, 3)
        features: 192 features of selected images with dimension (n, 192)
        labels: one hot labels of selected iamges with dimension (n, 10)
        idx: index of selected images
    :return:
        plot tSNE clusters: x and y axis are tSNE two dimension reduced features, 
                            different colors mean different classes
        reorder images: reorder all images according to reduced features and show in one big merged image
    
    """
    
    ###### dimension reduction ######
    tsne = TSNE()
    reduced = tsne.fit_transform(features)
    reduced_transformed = reduced - np.min(reduced, axis=0)
    reduced_transformed /= np.max(reduced_transformed, axis=0)
    
    image_xindex_sorted = np.argsort(np.sum(reduced_transformed, axis=1))
    image_index_sorted = idx[image_xindex_sorted]
    
    ##### plot cluster ####
    df = pd.DataFrame({'x':reduced_transformed[:,0],'y':reduced_transformed[:,1],'color':np.argmax(labels,axis=1)})
    ax = sns.lmplot( x="x", y="y", data=df, fit_reg=False, hue='color', legend=False)
    plt.legend(loc='lower right')
    ax.set(xlabel='Dimension 1', ylabel='Dimension 2')
    plt.title('Cluster of different labels')
    
    
    image_width = all_images.shape[1]
    no_of_images = features.shape[0]
    
    merged_width = int(np.ceil(np.sqrt(no_of_images))*image_width)
    merged_image = np.zeros((merged_width, merged_width, 3), dtype='float32')

    for counter, index in enumerate(image_index_sorted):
        b = int(np.mod(counter, np.sqrt(no_of_images)))
        a = int(np.mod(counter//np.sqrt(no_of_images), np.sqrt(no_of_images)))
        
        img = all_images[index]
        merged_image[a*image_width:(a+1)*image_width, b*image_width:(b+1)*image_width,:] = img[:,:,:3]
    
    plt.figure(figsize=(30,30))
    plt.imshow(merged_image)


def FindNN(idx_test_img,images,all_features,raw_labels, num_NN):

    """
    This function finds the neareast neighbors in all dataset of test images according to l2 distance of 
    last layer 192-dim features
    For example, the test image is a dog, this function will find images most similar to the test image
    
    param:
        idx_test_img: a list of index of test image 
        images: original images array with dimension (130000, 96, 96, 3)
        all_features: features of all images in the dataset, with dimension (13000, 192)
        raw_labels: a list raw labels of all images in the datasest (not one hot)
        num_NN: number of nearest neighbors return
    return:
        plot: each row shows a test image and its nearest neighbor images
    
    """
    class_names = {0: 'plane', 1: 'bird',2: 'car', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'horse', 7: 'monkey', 8: 'ship', 9: 'truck'}
    if type(idx_test_img) == int:
        fig = plt.figure(figsize=(20,30))
        l2 = list(np.apply_along_axis(lambda x: distance.euclidean(x,all_features[idx_test_img]),1,all_features))
        l2_best = np.sort(l2)[:num_NN]
        best = [l2.index(i) for i in l2_best]


        for i in range(num_NN):
            if i == 0:
                fig.add_subplot(1, num_NN, i+1)
                plt.imshow(images[best[i]])
                plt.title('test image: {}'.format(class_names[raw_labels[best[i]]]),fontsize = 14)
                plt.xticks([])
                plt.yticks([])
            else:
                fig.add_subplot(1, num_NN, i+1)
                plt.imshow(images[best[i]])
                plt.title('nearest {}'.format(i),fontsize = 14)
                plt.xticks([])
                plt.yticks([])
                
    else:
        num = len(idx_test_img)
        
        for j,index in enumerate(idx_test_img):
            fig = plt.figure(figsize=(20, 30))
            
            l2 = list(np.apply_along_axis(lambda x: distance.euclidean(x,all_features[index]),1,all_features))
            l2_best = np.sort(l2)[:num_NN]
            best = [l2.index(i) for i in l2_best]


            for i in range(num_NN):
                if i == 0:
                    fig.add_subplot(num, num_NN, j*num_NN+i+1)
                    plt.imshow(images[best[i]])
                    plt.title('test image: {}'.format(class_names[raw_labels[best[i]]]), fontsize=14)
                    plt.xticks([])
                    plt.yticks([])
                else:
                    fig.add_subplot(num, num_NN, j*num_NN+i+1)
                    plt.imshow(images[best[i]])
                    plt.title('nearest {}'.format(i), fontsize=14)
                    plt.xticks([])
                    plt.yticks([])
        
