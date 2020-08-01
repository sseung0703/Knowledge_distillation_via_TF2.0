import tensorflow as tf
import numpy as np

from nets import tcl

class distill:
    '''
    Ahn, Sungsoo, et al. 
    Variational information distillation for knowledge transfer.
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
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

        self.aux_layers = []
        for s, t in zip(self.student_layers, self.teacher_layers):
            layers = []
            Ds = s.gamma.shape[-1]
            for i in range(3):
                layers.append(tcl.Conv2d([1,1], Ds if i == 2 else Ds*2))
                layers.append(tcl.BatchNorm(activation_fn = None if i == 2 else tf.nn.relu))
            self.aux_layers.append(tf.keras.Sequential(layers))
            self.aux_layers[-1].alpha = self.aux_layers[-1].add_weight(name  = 'alpha', shape = [1,1,1,Ds], initializer=tf.constant_initializer(5.), trainable = True)

    def sampled_layer(self, arch, model):
        if 'WResNet' in arch:
            for i in range(1,3):
                model.Layers['BasicBlock%d.0/bn'%i].keep_feat = 'pre_act'
            model.Layers['bn_last'].keep_feat = 'pre_act'
            return [model.Layers['BasicBlock%d.0/bn'%i] for i in range(1,3)] + [model.Layers['bn_last']]

    def loss(self, sl, tl, aux):
        s = aux(sl.feat)
        t = tf.stop_gradient(tl.feat)
        var = tf.math.softplus(aux.alpha)+1
        return tf.reduce_mean(tf.math.log(var) + tf.square(t - s)/var)/2

    def forward(self, input, labels, target_loss):
        self.teacher(input, training = False)
        return target_loss + tf.add_n([self.loss(*data) for i, data in enumerate(zip(self.student_layers, self.teacher_layers, self.aux_layers))])



