import tensorflow as tf
import numpy as np
import scipy.io as sio
def Dataloader(name, data_path = None):
    if name == 'cifar10':
        return Cifar10()
    if name == 'cifar100':
        return Cifar100()
        
def Cifar10():
    from tensorflow.keras.datasets.cifar10 import load_data
    (train_images, train_labels), (test_images, test_labels) = load_data()
    
    def pre_processing(is_training = False):
        def training(image, *argv):
            image = tf.cast(image, tf.float32)
            image = (image-np.array([113.9,123.0,125.3]))/np.array([66.7,62.1,63.0])
            image = tf.image.random_flip_left_right(image)
            sz = tf.shape(image)
            image = tf.pad(image, [[4,4],[4,4],[0,0]], 'REFLECT')
            image = tf.image.random_crop(image,sz)
            
            return [image] + [arg for arg in argv]
        
        def inference(image, label):
            image = tf.cast(image, tf.float32)
            image = (image-np.array([113.9,123.0,125.3]))/np.array([66.7,62.1,63.0])
            return image, label
        
        return training if is_training else inference
    return train_images, train_labels, test_images, test_labels, pre_processing

def Cifar100():
    from tensorflow.keras.datasets.cifar100 import load_data
    (train_images, train_labels), (test_images, test_labels) = load_data()

    def pre_processing(is_training = False):
        @tf.function
        def training(image, *argv):
            image = tf.cast(image, tf.float32)
            image = (image-np.array([112,124,129]))/np.array([70,65,68])
            image = tf.image.random_flip_left_right(image)
            sz = tf.shape(image)
            image = tf.pad(image, [[4,4],[4,4],[0,0]], 'REFLECT')
            image = tf.image.random_crop(image,sz)
            return [image] + [arg for arg in argv]
        @tf.function
        def inference(image, label):
            image = tf.cast(image, tf.float32)
            image = (image-np.array([112,124,129]))/np.array([70,65,68])
            return image, label
        
        return training if is_training else inference
    return train_images, train_labels, test_images, test_labels, pre_processing

