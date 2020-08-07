import tensorflow as tf
import numpy as np

from nets import tcl
from scipy.stats import norm

class distill:
    '''
     Zagoruyko, Sergey and Komodakis, Nikos.
     Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer.
     arXiv preprint arXiv:1612.03928, 2016.
    '''
    def __init__(self, args, model, teacher):
        self.args = args
        self.student = model
        self.teacher = teacher
        self.student_layers = self.sampled_layer(args.arch, self.student)
        self.teacher_layers = self.sampled_layer(args.teacher_arch, self.teacher)

        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(kernel_initializer = tf.keras.initializers.he_normal(),
                                                  use_biases = False, activation_fn = None, trainable = True))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(trainable = True))
        self.student.aux_layers = [tf.keras.Sequential([tcl.Conv2d([1,1], tl.gamma.shape[-1]), tcl.BatchNorm()] ) 
                           for sl, tl in zip(self.student_layers, self.teacher_layers)]
        self.beta = 1e3
        
        self.build()

    def build(self):
        input = np.zeros([1] + self.args.input_shape, dtype = np.float32)
        self.student(input, training = False)
        for sl, aux in zip(self.student_layers, self.student.aux_layers):
            aux.build(sl.feat.shape)

    def sampled_layer(self, arch, model):
        if 'WResNet' in arch:
            for i in range(1,3):
                model.Layers['BasicBlock%d.0/bn'%i].keep_feat = 'pre_act'
            model.Layers['bn_last'].keep_feat = 'pre_act'
            return [model.Layers['BasicBlock%d.0/bn'%i] for i in range(1,3)] + [model.Layers['bn_last']]

    def loss(self, sl, tl, aux):
        s = aux(sl.feat, training = True)
        t = tf.stop_gradient(tl.feat)
        return tf.reduce_mean(tf.square(tf.nn.l2_normalize(s, [1,2]) - tf.nn.l2_normalize(t, [1,2])))

    def forward(self, input, labels, target_loss):
        self.teacher(input, training = False)
        return target_loss + tf.add_n([self.loss(*data)/2**(len(self.student_layers)-i-1)
                                       for i, data in enumerate(zip(self.student_layers, self.teacher_layers, self.student.aux_layers))])*self.beta
