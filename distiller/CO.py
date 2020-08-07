import tensorflow as tf
import numpy as np

from nets import tcl
from scipy.stats import norm

class distill:
    '''
    Heo, Byeongho, et al.
    A comprehensive overhaul of feature distillation.
    Proceedings of the IEEE International Conference on Computer Vision. 2019.
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

        def get_margin(s, m):
            s, m = np.abs(s.numpy()), m.numpy()
            margin = np.where(norm.cdf(-m/s) > 1e-3, - s * np.exp(- (m/s) ** 2 / 2) / np.sqrt(2 * np.pi) / norm.cdf(-m/s) + m, -3 * s).astype(np.float32)
            return tf.constant(margin)

        self.margins = [get_margin(tl.gamma, tl.beta) for tl in self.teacher_layers]

    def sampled_layer(self, arch, model):
        if 'WResNet' in arch:
            for i in range(1,3):
                model.Layers['BasicBlock%d.0/bn'%i].keep_feat = 'pre_act'
            model.Layers['bn_last'].keep_feat = 'pre_act'
            return [model.Layers['BasicBlock%d.0/bn'%i] for i in range(1,3)] + [model.Layers['bn_last']]

    def loss(self, sl, tl, aux, m):
        s = aux(sl.feat, training = True)
        t = tf.stop_gradient(tf.maximum(tl.feat, m))
        return tf.reduce_sum(tf.square(s - t) * tf.cast((s > t) | (t > 0), tf.float32))

    def forward(self, input, labels, target_loss):
        self.teacher(input, training = False)
        return target_loss + tf.add_n([self.loss(*data)/2**(len(self.student_layers)-i-1)
                                       for i, data in enumerate(zip(self.student_layers, self.teacher_layers, self.student.aux_layers, self.margins))])/input.shape[0] * 1e-3
