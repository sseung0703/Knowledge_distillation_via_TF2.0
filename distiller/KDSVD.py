import tensorflow as tf
import numpy as np

from nets import tcl
from distiller import SVD

class distill:
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

        self.distilled_SV = 4

    def sampled_layer(self, arch, model):
        if 'WResNet' in arch:
            for i in range(1,3):
                model.Layers['BasicBlock%d.0/bn'%i].keep_feat = 'pre_act'
            model.Layers['bn_last'].keep_feat = 'pre_act'
            return [model.Layers['BasicBlock%d.0/bn'%i] for i in range(1,3)] + [model.Layers['bn_last']]

    def forward(self, input, labels, target_loss):
        self.teacher(input, training = False)

        distill_loss = []
        for i, (aux, s, t) in enumerate(zip(self.aux_layers, self.student_layers, self.teacher_layers)):
            B,H,W,D = s.feat.shape
            with tf.device('cpu'):
                ts, tU, tV = SVD.SVD(t.feat, self.distilled_SV)

            feat = tf.reshape(aux(s.feat), [B,H*W,D])
            sVs = tf.matmul(feat, tU, transpose_a = True)
            sV = tf.nn.l2_normalize(sVs, 1)
            
            ts = tf.expand_dims(ts,1)
            sV *= ts
            tV *= ts
                
            if i > 0:                
                S_rbf = tf.exp(-tf.square(tf.expand_dims(sV,2)-tf.expand_dims(sV_,1))/8)
                T_rbf = tf.exp(-tf.square(tf.expand_dims(tV,2)-tf.expand_dims(tV_,1))/8)
                l2loss = (S_rbf-tf.stop_gradient(T_rbf))**2
                l2loss = tf.where(tf.math.is_finite(l2loss), l2loss, tf.zeros_like(l2loss))
                distill_loss.append(tf.reduce_sum(l2loss))
            sV_ = sV
            tV_ = tV

        self.distill_loss = tf.add_n(distill_loss)
        self.target_loss =  target_loss
        return target_loss + self.distill_loss
    
    def backward(self, tape, total_loss, vars):
        target_grad = tape.gradient(self.target_loss, vars, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        distill_grad = tape.gradient(self.distill_loss, vars, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        return [tg + tf.clip_by_norm(dg, tf.norm(tg)) for tg, dg in zip(target_grad, distill_grad)]




