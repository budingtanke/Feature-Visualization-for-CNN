from keras import activations
import numpy as np
import cv2
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation, Input
from keras.models import Sequential


IMG_WEIGHT, IMG_HEIGHT = 96, 96

def get_model(weights_path):

    if K.image_data_format() == 'channels_first':
        input_shape = (3, IMG_WEIGHT, IMG_HEIGHT)
    elif K.image_data_format() == 'channels_last':
        input_shape = (IMG_WEIGHT, IMG_HEIGHT, 3)

    n1 = 96
    n2 = 96
    n3 = 384
    n4 = 192

    model = Sequential()
    model.add(Conv2D(n1, (5, 5), input_shape=input_shape, activation='relu', padding='same', name='conv1'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1'))
    model.add(Conv2D(n2, (3, 3), activation='relu', padding='same', name='conv2'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool2'))
    model.add(Flatten())
    model.add(Dense(n3, activation='relu', name='fc1'))
    model.add(Dense(n4, activation='relu', name='fc2'))
    model.add(Dense(10, activation='softmax', name='preds'))

    model.load_weights(weights_path)

    return model


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x = x.astype('float64')
    
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))

    x = np.clip(x, 0, 255).astype('uint8')
    return x




def gradient_ascent_iter_basic(loss_fn, img, step=1.0, verbose=False):
    ''' To update image based on calculating loss function 
    '''
    loss_value, grads_value = loss_fn([img])
    
    if verbose: print("Loss: {}".format(loss_value))
    
    gradient_ascent_step = img + grads_value * step
    
    grads_row_major = np.transpose(grads_value[0, :], (1, 2, 0))
    img_row_major = np.transpose(gradient_ascent_step[0, :], (1, 2, 0))
    
    img = np.float32([np.transpose(img_row_major, (2, 0, 1))])
    return img




def visualize_activation_basic(model, class_idx, init_img, n_iteration=100, verbose=False,
                        tv_weight=0, lp_weight=0.1):
    '''this function is to conduct Preliminary Trial for Activation Maximization
    '''
    model.layers[-1].activation = activations.linear
    
    layer_output = model.layers[-1].output
    input_tensor = model.input
    
    total_loss = layer_output[:, class_idx]  
    grads = K.gradients(total_loss, input_tensor)[0]

    iterate = K.function([input_tensor], [total_loss, grads])
    
    img = init_img 
    for i in range(n_iteration):
        if verbose: print("iteration: {0}".format(i))
        img = gradient_ascent_iter_basic(iterate, img, step=1000.0, verbose=verbose)
    
    return img






def decay_regularization(img, decay=0.8):       
    '''regularization method: Decay
    '''
    return decay * img



def blur_regularization(img, size=(3,3)):
    '''regularization method: blur
    '''
    return cv2.blur(img, size)



def clip_weak_pixel_regularization(img, grads, percentile=1):
    '''regularization method: Clipping pixels with small norm
    '''
    clipped = img
    threshold = np.percentile(np.abs(img), percentile)
    clipped[np.where(np.abs(img) < threshold)] = 0
    return clipped




class slicerClass(object):
    """Utility class to make image slicing uniform across various `image_data_format`.
       Reference: https://github.com/raghakot/keras-vis
    """

    def __getitem__(self, item_slice):
        """Assuming a slice for shape `(samples, channels, image_dims...)`
        """
        if K.image_data_format() == 'channels_first':
            return item_slice
        else:
            # Move channel index to last position.
            item_slice = list(item_slice)
            item_slice.append(item_slice.pop(1))
            return tuple(item_slice)
        
slicer = slicerClass()





def total_variation_regularizer(img, beta=2.0):
    '''regularization method: Bounded variation
    '''

    image_dims = K.ndim(img) - 2
    
    # Constructing slice [1:] + [:-1] * (image_dims - 1) and [:-1] * (image_dims)
    start_slice = [slice(1, None, None)] + [slice(None, -1, None) for _ in range(image_dims - 1)]
    end_slice = [slice(None, -1, None) for _ in range(image_dims)]
    samples_channels_slice = [slice(None, None, None), slice(None, None, None)]
    
    tv = None
    for i in range(image_dims):
        ss = tuple(samples_channels_slice + start_slice)
        es = tuple(samples_channels_slice + end_slice)
        diff_square = K.square(img[slicer[ss]] - img[slicer[es]])
        tv = diff_square if tv is None else tv + diff_square

        # Roll over to next image dim
        start_slice = np.roll(start_slice, 1).tolist()
        end_slice = np.roll(end_slice, 1).tolist()

    tv = K.sum(K.pow(tv, beta / 2.))
    return tv / np.prod(image_dims)




def lp_regularizer(img, p = 2):
    '''regularization method: Bounded Range
    '''
    if p < 1: raise ValueError("p value should be in [i, inf)")
    
    image_dims = K.ndim(img) - 2
    
    if np.isinf(p):
        value = K.max(img)
    else:
        value = K.pow(K.sum(K.pow(K.abs(img), p)), 1. / p)
    
    return value / np.prod(image_dims)
















