import tensorflow as tf
import numpy as np
def Dataloader(name, home_path):
    if name == 'cifar10':
        return Cifar10(home_path)
    if name == 'cifar100':
        return Cifar100(home_path)
        
def Cifar10(home_path):
    from tensorflow.keras.datasets.cifar10 import load_data
    (train_images, train_labels), (val_images, val_labels) = load_data()
    
    def pre_processing(is_training = False):
        def training(image, label):
            image = tf.cast(image, tf.float32)
            image = (image-np.array([112,124,129]))/np.array([70,65,68])
            image = tf.image.random_flip_left_right(image)
            sz = tf.shape(image)
            image = tf.pad(image, [[0,0],[4,4],[4,4],[0,0]], 'REFLECT')
            image = tf.image.random_crop(image,sz)
            return image, label
        
        def inference(image, label):
            image = tf.cast(image, tf.float32)
            image = (image-np.array([112,124,129]))/np.array([70,65,68])
            return image, label
        
        return training if is_training else inference
    return train_images, train_labels, val_images, val_labels, pre_processing

def Cifar100(home_path):
    from tensorflow.keras.datasets.cifar100 import load_data
    (train_images, train_labels), (val_images, val_labels) = load_data()
    
    def pre_processing(is_training = False):
        def training(image, label):
            image = tf.cast(image, tf.float32)
            image = (image-np.array([112,124,129]))/np.array([70,65,68])
            image = tf.image.random_flip_left_right(image)
            sz = tf.shape(image)
            image = tf.pad(image, [[0,0],[4,4],[4,4],[0,0]], 'REFLECT')
            image = tf.image.random_crop(image,sz)
            return image, label
        
        def inference(image, label):
            image = tf.cast(image, tf.float32)
            image = (image-np.array([112,124,129]))/np.array([70,65,68])
            return image, label
        
        return training if is_training else inference
    return train_images, train_labels, val_images, val_labels, pre_processing

