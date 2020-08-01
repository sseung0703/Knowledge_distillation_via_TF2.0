import tensorflow as tf
from tensorflow.python.keras.utils.conv_utils import deconv_output_length

def arg_scope(func):
    def func_with_args(self, *args, **kwargs):
        if hasattr(self, 'pre_defined'):
            for k in self.pre_defined.keys():
                kwargs[k] = self.pre_defined[k]
        return func(self, *args, **kwargs)
    return func_with_args

class Conv2d(tf.keras.layers.Layer):
    @arg_scope
    def __init__(self, kernel_size, num_outputs, strides = 1, dilations = 1, padding = 'SAME',
                 kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 2., mode='fan_out'),
                 use_biases = True,
                 biases_initializer  = tf.keras.initializers.Zeros(),
                 activation_fn = tf.nn.relu,
                 name = 'conv',
                 trainable = True,
                 **kwargs):
        super(Conv2d, self).__init__(name = name, **kwargs)
        
        self.kernel_size = kernel_size
        self.num_outputs = num_outputs
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.kernel_initializer = kernel_initializer
        
        self.use_biases = use_biases
        self.biases_initializer = biases_initializer
        
        self.activation_fn = activation_fn

        self.trainable = trainable
        self.keep_feat = False
        
        
    def build(self, input_shape):
        super(Conv2d, self).build(input_shape)
        self.kernel = self.add_weight(name  = 'kernel', 
                                      shape = self.kernel_size + [input_shape[-1], self.num_outputs],
                                      initializer=self.kernel_initializer,
                                      trainable = self.trainable)
        if self.use_biases:
            self.biases  = self.add_weight(name = "biases",
                                           shape=[1,1,1,self.num_outputs],
                                           initializer = self.biases_initializer,
                                           trainable = self.trainable)

    def call(self, input, training = False):
        if self.keep_feat is 'input':
            self.feat = input

        conv = tf.nn.conv2d(input, self.kernel, self.strides, self.padding,
                            dilations=self.dilations, name=None)
        if self.use_biases:
            conv += self.biases

        if self.keep_feat is 'pre_act':
            self.feat = conv
        
        if self.activation_fn:
            conv = self.activation_fn(conv)

        if self.keep_feat is 'output':
            self.feat = conv

        return conv

class Conv2d_transpose(tf.keras.layers.Layer):
    @arg_scope
    def __init__(self, kernel_size, num_outputs, strides = 1, dilations = 1, padding = 'SAME',
                 kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 2., mode='fan_out'),
                 use_biases = True,
                 biases_initializer  = tf.keras.initializers.Zeros(),
                 activation_fn = tf.nn.relu,
                 name = 'conv',
                 trainable = True,
                 **kwargs):
        super(Conv2d_transpose, self).__init__(name = name, **kwargs)

        self.kernel_size = kernel_size
        self.num_outputs = num_outputs
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.kernel_initializer = kernel_initializer
        
        self.use_biases = use_biases
        self.biases_initializer = biases_initializer
        
        self.activation_fn = activation_fn

        self.trainable = trainable
        self.keep_feat = False
        
        
    def build(self, input_shape):
        super(Conv2d_transpose, self).build(input_shape)
        self.kernel = self.add_weight(name  = 'kernel', 
                                      shape = self.kernel_size + [self.num_outputs, input_shape[-1]],
                                      initializer=self.kernel_initializer,
                                      trainable = self.trainable)
        if self.use_biases:
            self.biases  = self.add_weight(name = "biases",
                                           shape=[1,1,1,self.num_outputs],
                                           initializer = self.biases_initializer,
                                           trainable = self.trainable)
        self.spatial_shape = input_shape[1:3]
        self.output_shape_ = [deconv_output_length(input_shape[1], self.kernel_size[0],padding=self.padding.lower(),
                                                        stride=self.strides, dilation=self.dilations),
                              deconv_output_length(input_shape[2], self.kernel_size[1],padding=self.padding.lower(),
                                                        stride=self.strides, dilation=self.dilations),
                              self.num_outputs]


    def call(self, input, training = False):
        if self.keep_feat is 'input':
            self.feat = input

        conv = tf.nn.conv2d_transpose(input, self.kernel, [input.shape[0]] + self.output_shape_, self.strides, self.padding, dilations = self.dilations, name=None)
        if self.use_biases:
            conv += self.biases

        if self.keep_feat is 'pre_act':
            self.feat = conv
        
        if self.activation_fn:
            conv = self.activation_fn(conv)

        if self.keep_feat is 'output':
            self.feat = conv

        return conv

class DepthwiseConv2d(tf.keras.layers.Layer):
    @arg_scope
    def __init__(self, kernel_size, multiplier = 1, strides = [1,1,1,1], dilations = [1,1], padding = 'SAME',
                 kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 2., mode='fan_out'),         
                 use_biases = True,
                 biases_initializer  = tf.keras.initializers.Zeros(),
                 activation_fn = tf.nn.relu,
                 name = 'conv',
                 trainable = True,
                 **kwargs):
        super(DepthwiseConv2d, self).__init__(name = name, **kwargs)
        
        self.kernel_size = kernel_size
        self.strides = strides if isinstance(strides, list) else [1, strides, strides, 1]
        self.padding = padding
        self.dilations = dilations if isinstance(dilations, list) else [dilations, dilations]
        self.multiplier = multiplier
        self.kernel_initializer = kernel_initializer
        
        self.use_biases = use_biases
        self.biases_initializer = biases_initializer
        
        self.activation_fn = activation_fn

        self.trainable = trainable
        
    def build(self, input_shape):
        super(DepthwiseConv2d, self).build(input_shape)
        self.kernel = self.add_weight(name  = 'kernel', 
                                      shape = self.kernel_size + [input_shape[-1], self.multiplier],
                                      initializer=self.kernel_initializer,
                                      trainable = self.trainable)
        if self.use_biases:
            self.biases  = self.add_weight(name = "biases",
                                           shape=[1,1,1, input_shape[-1]*self.multiplier],
                                           initializer = self.biases_initializer,
                                           trainable = self.trainable)

    def call(self, input):
        kernel = self.kernel

        conv = tf.nn.depthwise_conv2d(input, kernel, strides = self.strides, padding = self.padding, dilations=self.dilations)
        if self.use_biases:
            conv += self.biases
        if self.activation_fn:
            conv = self.activation_fn(conv)

        return conv

class FC(tf.keras.layers.Layer):
    @arg_scope
    def __init__(self, num_outputs, 
                 kernel_initializer = tf.keras.initializers.he_normal(),
                 use_biases = True,
                 biases_initializer  = tf.keras.initializers.Zeros(),
                 activation_fn = None,
                 name = 'fc',
                 trainable = True, **kwargs):
        super(FC, self).__init__(name = name, **kwargs)
        self.num_outputs = num_outputs
        self.kernel_initializer = kernel_initializer
        
        self.use_biases = use_biases
        self.biases_initializer = biases_initializer
        
        self.activation_fn = activation_fn
        
        self.trainable = trainable
        self.keep_feat = False
        
    def build(self, input_shape):
        super(FC, self).build(input_shape)
        self.kernel = self.add_weight(name  = 'kernel', 
                                      shape = [int(input_shape[-1]), self.num_outputs],
                                      initializer=self.kernel_initializer,
                                      trainable = self.trainable)
        if self.use_biases:
            self.biases  = self.add_weight(name = "biases",
                                           shape=[1,self.num_outputs],
                                           initializer = self.biases_initializer,
                                           trainable = self.trainable)

    def call(self, input):
        kernel = self.kernel

        if self.keep_feat is 'input':
            self.feat = input

        fc = tf.matmul(input, kernel)
        if self.use_biases:
            fc += self.biases

        if self.keep_feat is 'pre_act':
            self.feat = fc

        if self.activation_fn:
            fc = self.activation_fn(fc)

        if self.keep_feat is 'output':
            self.feat = fc

        return fc

class BatchNorm(tf.keras.layers.Layer):
    @arg_scope
    def __init__(self, param_initializers = None,
                       scale = True,
                       center = True,
                       alpha = 0.9,
                       epsilon = 1e-5,
                       activation_fn = None,
                       name = 'bn',
                       trainable = True,
                       **kwargs):
        super(BatchNorm, self).__init__(name = name, **kwargs)
        if param_initializers == None:
            param_initializers = {}
        if not(param_initializers.get('moving_mean')):
            param_initializers['moving_mean'] = tf.keras.initializers.Zeros()
        if not(param_initializers.get('moving_var')):
            param_initializers['moving_var'] = tf.keras.initializers.Ones()
        if not(param_initializers.get('gamma')) and scale:
            param_initializers['gamma'] = tf.keras.initializers.Ones()
        if not(param_initializers.get('beta')) and center:
            param_initializers['beta'] = tf.keras.initializers.Zeros()
        
        self.param_initializers = param_initializers
        self.scale = scale
        self.center = center
        self.alpha = alpha
        self.epsilon = epsilon
        self.activation_fn = activation_fn
        
        self.trainable = trainable
        self.keep_feat = False

    def build(self, input_shape):
        super(BatchNorm, self).build(input_shape)
        self.moving_mean = self.add_weight(name  = 'moving_mean', trainable = False,
                                      shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                      initializer=self.param_initializers['moving_mean'])
        self.moving_var = self.add_weight(name  = 'moving_var', trainable = False,
                                      shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                      initializer=self.param_initializers['moving_var'])

        if self.scale:
            self.gamma = self.add_weight(name  = 'gamma', 
                                         shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                         initializer=self.param_initializers['gamma'],
                                         trainable = self.trainable)
        else:
            self.gamma = 1.
        if self.center:
            self.beta = self.add_weight(name  = 'beta', 
                                        shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                        initializer=self.param_initializers['beta'],
                                        trainable = self.trainable)
        else:
            self.beta = 0.
        self.spatial_shape = input_shape[1:3]

    def EMA(self, variable, value):
        update_delta = (variable - value) * (1-self.alpha)
        variable.assign(variable-update_delta)
        
    def call(self, input, training=None):
        if self.keep_feat is 'input':
            self.feat = input

        if training:
            mean, var = tf.nn.moments(input, list(range(len(input.shape)-1)), keepdims=True)
            self.EMA(self.moving_mean, mean)
            self.EMA(self.moving_var, var)
        else:
            mean = self.moving_mean
            var = self.moving_var
        gamma, beta = self.gamma, self.beta

        bn = tf.nn.batch_normalization(input, mean, var, offset = beta, scale = gamma, variance_epsilon = self.epsilon)

        if self.keep_feat is 'pre_act':
            self.feat = bn

        if self.activation_fn:
            bn = self.activation_fn(bn)

        if self.keep_feat is 'output':
            self.feat = bn
        return bn


