import tensorflow as tf
import numpy as np

from nets import tcl

class distill:
    '''
    Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta,  and  Yoshua  Bengio.
    Fitnets:   Hints  for  thin  deep  nets.
    arXiv preprint arXiv:1412.6550, 2014.
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

    def sampled_layer(self, arch, model):
        if 'WResNet' in arch:
            for i in range(1,3):
                model.Layers['BasicBlock%d.0/bn'%i].keep_feat = 'pre_act'
            model.Layers['bn_last'].keep_feat = 'pre_act'
            return [model.Layers['BasicBlock%d.0/bn'%i] for i in range(1,3)] + [model.Layers['bn_last']]

    def loss(self, sl, tl, aux):
        s = aux(sl.feat)
        t = tf.stop_gradient(tl.feat)
        return tf.reduce_mean(tf.square(s - t))

    def forward(self, input, labels, target_loss):
        self.teacher(input, training = False)
        return target_loss + tf.add_n([self.loss(*data) for i, data in enumerate(zip(self.student_layers, self.teacher_layers, self.aux_layers))])
