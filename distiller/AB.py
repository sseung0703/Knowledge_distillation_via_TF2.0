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
        self.student.aux_layers = [tf.keras.Sequential([tcl.Conv2d([1,1], tl.gamma.shape[-1]), tcl.BatchNorm()] ) 
                                   for sl, tl in zip(self.student_layers, self.teacher_layers)]
        self.margin = 1.

    def sampled_layer(self, arch, model):
        if 'WResNet' in arch:
            for i in range(1,3):
                model.Layers['BasicBlock%d.0/bn'%i].keep_feat = 'pre_act'
            model.Layers['bn_last'].keep_feat = 'pre_act'
            return [model.Layers['BasicBlock%d.0/bn'%i] for i in range(1,3)] + [model.Layers['bn_last']]

    def loss(self, sl, tl, aux):
        s = aux(sl.feat, training = True)
        t = tf.stop_gradient(tl.feat)
        B,H,W,D = s.shape

        return tf.reduce_sum(tf.abs(tf.square(s + self.margin) * tf.cast(tf.logical_and(s > -self.margin, t <= 0.), tf.float32)
                                    +tf.square(s - self.margin) * tf.cast(tf.logical_and(s <= self.margin, t > 0.), tf.float32)))/B/H/W

    def initialize_student(self, dataset):
        optimizer = tf.keras.optimizers.SGD(self.args.learning_rate, .9, nesterov=True)
        train_loss = tf.keras.metrics.Mean(name='train_loss')

        @tf.function(jit_compile=True)
        def init_forward(input):
            self.teacher(input, training = False)
            with tf.GradientTape(persistent = True) as tape:
                self.student(input, training = True)
                distill_loss = []
                for i, data in enumerate(zip(self.student_layers, self.teacher_layers, self.student.aux_layers)):
                    distill_loss.append(self.loss(*data)*2**(-len(self.student_layers)+i+1))
                distill_loss = tf.add_n(distill_loss)

            gradients = tape.gradient(distill_loss, self.student.trainable_variables)
            if self.args.weight_decay > 0.:
                gradients = [g+v*self.args.weight_decay if g is not None else g
                             for g,v in zip(gradients, self.student.trainable_variables)]
            optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
            train_loss.update_state(distill_loss)

        for e in range(int(self.args.train_epoch*.3)):
            for imgs, _ in dataset:
                init_forward(imgs)
            print('Aux Epoch: %d: loss: %.4f'%(e,train_loss.result()))
            train_loss.reset_states()
