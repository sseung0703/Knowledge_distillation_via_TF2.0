import tensorflow as tf
from nets import tcl

class Model(tf.keras.Model):
    def __init__(self, architecture, weight_decay, num_class, name = 'WResNet', trainable = True, **kwargs):
        super(Model, self).__init__(name = name, **kwargs)
        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(kernel_regularizer = tf.keras.regularizers.l2(weight_decay),
                                                  use_biases = False, activation_fn = None, trainable = trainable))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(param_regularizers = {'gamma':tf.keras.regularizers.l2(weight_decay),
                                                                            'beta':tf.keras.regularizers.l2(weight_decay)},
                                                     trainable = trainable))
        setattr(tcl.FC, 'pre_defined', kwargs(kernel_regularizer = tf.keras.regularizers.l2(weight_decay),
                                              biases_regularizer = tf.keras.regularizers.l2(weight_decay),
                                              trainable = trainable))
        
        self.wresnet_layers = {}
        depth, widen_factor = architecture
        self.nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        self.stride = [1,2,2]
        self.n = (depth-4)//6
        
        self.wresnet_layers['conv0'] = tcl.Conv2d([3,3], self.nChannels[0], name = 'conv0')
        in_planes = self.nChannels[0]
        
        for i, (c, s) in enumerate(zip(self.nChannels[1:], self.stride)):
            for j in range(self.n):
                block_name = 'BasicBlock%d.%d'%(i,j)
                with tf.name_scope(block_name):
                    equalInOut = in_planes == c
                    
                    self.wresnet_layers[block_name + '/bn0']   = tcl.BatchNorm(name = 'bn0')
                    self.wresnet_layers[block_name + '/conv1'] = tcl.Conv2d([3,3], c, strides = s if j == 0 else 1, name = 'conv1')
                    self.wresnet_layers[block_name + '/bn1']   = tcl.BatchNorm(activation_fn = tf.nn.relu, name = 'bn1')
                    self.wresnet_layers[block_name + '/conv2'] = tcl.Conv2d([3,3], c, strides = 1, name = 'conv2')
                            
                    if not(equalInOut):
                        self.wresnet_layers[block_name + '/conv3'] = tcl.Conv2d([1,1], c, strides = s if j == 0 else 1, name = 'conv3')
                    in_planes = c
        self.wresnet_layers['bn1']= tcl.BatchNorm(name = 'bn1')
        self.wresnet_layers['fc'] = tcl.FC(num_class, name = 'fc')
    
    def call(self, x, training=None):
        x = self.wresnet_layers['conv0'](x)
        in_planes = self.nChannels[0]
        
        self.feature = []
        self.feature_noact = []
        
        for i, (c, s) in enumerate(zip(self.nChannels[1:], self.stride)):
            for j in range(self.n):
                block_name = 'BasicBlock%d.%d'%(i,j)
                with tf.name_scope(block_name):
                    equalInOut = in_planes == c
                    if not equalInOut:
                        x_ = self.wresnet_layers[block_name + '/bn0'](x, training = training)
                        x = tf.nn.relu(x_)
                        if j == 0 and i > 0:
                            self.feature.append(x)
                            self.feature_noact.append(x_)
                    else:
                        out_ = self.wresnet_layers[block_name + '/bn0'](x, training = training)
                        out = tf.nn.relu(out_)
                        if j == 0 and i > 0:
                            self.feature.append(out)
                            self.feature_noact.append(out_)
                            
                    out = self.wresnet_layers[block_name + '/conv1'](out if equalInOut else x)
                    out = self.wresnet_layers[block_name +   '/bn1'](out, training = training)
                    out = self.wresnet_layers[block_name + '/conv2'](out)
                    if not(equalInOut):
                        x = self.wresnet_layers[block_name + '/conv3'](x)
                    x = x+out
                    in_planes = c
                        
        x_ = self.wresnet_layers['bn1'](x, training = training)
        x = tf.nn.relu(x_)
        self.feature.append(x)
        self.feature_noact.append(x_)
                            
        x = tf.reduce_mean(x,[1,2])
        self.last_embedded = x
        x = self.wresnet_layers['fc'](x)
        self.logits = x
        return x
    
    def get_feat(self, x, feat, training = None):
        y = self.call(x, training)
        return y, getattr(self, feat)
        
