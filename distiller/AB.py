import tensorflow as tf
import numpy as np

from nets import tcl

class distill:
    '''
    Byeongho Heo,  Minsik Lee,  Sangdoo Yun,  and Jin Young Choi.   
    Knowledge transfer via distillation of activation boundaries formed by hidden neurons.
    AAAI Conference on Artificial Intelligence (AAAI), 2019.
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
        self.aux_layers = [tf.keras.Sequential([tcl.Conv2d([1,1], tl.gamma.shape[-1]), tcl.BatchNorm()] ) 
                           for sl, tl in zip(self.student_layers, self.teacher_layers)]
        self.margin = 1.
        self.weight = 3e-4

    def sampled_layer(self, arch, model):
        if 'WResNet' in arch:
            for i in range(1,3):
                model.Layers['BasicBlock%d.0/bn'%i].keep_feat = 'pre_act'
            model.Layers['bn_last'].keep_feat = 'pre_act'
            return [model.Layers['BasicBlock%d.0/bn'%i] for i in range(1,3)] + [model.Layers['bn_last']]

    def loss(self, sl, tl, aux):
        s = aux(sl.feat)
        t = tf.stop_gradient(tl.feat)
        return tf.reduce_sum(tf.square(s + self.margin) * tf.cast(tf.logical_and(s > -self.margin, t <= 0.), tf.float32)\
                            +tf.square(s - self.margin) * tf.cast(tf.logical_and(s <= self.margin, t > 0.), tf.float32))

    def forward(self, input, labels, target_loss):
        self.teacher(input, training = False)
        return target_loss + tf.add_n([self.loss(*data)/2**(len(self.student_layers)-i-1)
                                       for i, data in enumerate(zip(self.student_layers, self.teacher_layers, self.aux_layers))])/input.shape[0] * self.weight
