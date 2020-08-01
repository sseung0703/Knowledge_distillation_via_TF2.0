import tensorflow as tf
from nets import tcl

class Model(tf.keras.Model):
    def __init__(self, architecture, num_class, name = 'WResNet', trainable = True, **kwargs):
        super(Model, self).__init__(name = name, **kwargs)
        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(use_biases = False, activation_fn = None, trainable = trainable))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(trainable = trainable))
        setattr(tcl.FC, 'pre_defined', kwargs(trainable = trainable))
        
        self.Layers = {}
        depth, widen_factor = architecture
        self.nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        self.stride = [1,2,2]
        self.n = (depth-4)//6
        
        self.Layers['conv'] = tcl.Conv2d([3,3], self.nChannels[0], name = 'conv')
        in_planes = self.nChannels[0]
        prev_conv_name = 'conv'
        
        for i, (c, s) in enumerate(zip(self.nChannels[1:], self.stride)):
            for j in range(self.n):
                block_name = 'BasicBlock%d.%d/'%(i,j)
                equalInOut = in_planes == c
                in_planes = c
                    
                self.Layers[block_name + 'bn']   = tcl.BatchNorm(name = block_name + 'bn')
                self.Layers[prev_conv_name.replace('conv','bn')] = self.Layers[block_name + 'bn']

                if not(equalInOut):
                    self.Layers[block_name + 'conv2'] = tcl.Conv2d([1,1], c, strides = s if j == 0 else 1, name = block_name + 'conv2')

                self.Layers[block_name + 'conv0'] = tcl.Conv2d([3,3], c, strides = s if j == 0 else 1, name = block_name + 'conv0')
                self.Layers[block_name + 'bn0']   = tcl.BatchNorm(activation_fn = tf.nn.relu, name = block_name + 'bn0')
                self.Layers[block_name + 'conv1'] = tcl.Conv2d([3,3], c, strides = 1, name = block_name + 'conv1')
                prev_conv_name = block_name + 'conv1'

                if not(equalInOut):
                    prev_conv_name = block_name + 'conv2'

        self.Layers['bn_last']= tcl.BatchNorm(name = 'bn_last')
        self.Layers[prev_conv_name.replace('conv','bn')] = self.Layers['bn_last']

        self.Layers['fc'] = tcl.FC(num_class, name = 'fc')
    
    def call(self, x, training=None):
        x = self.Layers['conv'](x)

        in_planes = self.nChannels[0]
        for i, (c, s) in enumerate(zip(self.nChannels[1:], self.stride)):
            for j in range(self.n):
                equalInOut = in_planes == c
                block_name = 'BasicBlock%d.%d'%(i,j)
                out = self.Layers[block_name + '/bn'](x)
                out = tf.nn.relu(out)
                if not(equalInOut):
                    residual = self.Layers[block_name + '/conv2'](x)
                else:
                    residual = x
                out = self.Layers[block_name + '/conv0'](out)
                out = self.Layers[block_name + '/bn0'](out)

                out = self.Layers[block_name + '/conv1'](out)
                x = residual+out

                in_planes = c
        x = self.Layers['bn_last'](x)
        x = tf.nn.relu(x)
                            
        x = tf.reduce_mean(x,[1,2])
        x = self.Layers['fc'](x)
        return x
