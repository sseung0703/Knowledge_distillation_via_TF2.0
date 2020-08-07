import tensorflow as tf
import numpy as np

from nets import tcl
    
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
        self.student.aux_layers = [tf.keras.Sequential([tcl.Conv2d([1,1], tl.gamma.shape[-1]), tcl.BatchNorm()] ) 
                                   for sl, tl in zip(self.student_layers, self.teacher_layers)]

    def sampled_layer(self, arch, model):
        if 'WResNet' in arch:
            for i in range(1,3):
                model.Layers['BasicBlock%d.0/bn'%i].keep_feat = 'pre_act'
            model.Layers['bn_last'].keep_feat = 'pre_act'
            return [model.Layers['BasicBlock%d.0/bn'%i] for i in range(1,3)] + [model.Layers['bn_last']]

    def initialize_student(self, dataset):
        optimizer = tf.keras.optimizers.SGD(self.args.learning_rate, .9, nesterov=True)
        train_loss = tf.keras.metrics.Mean(name='train_loss')

        @tf.function(experimental_compile=True)
        def init_forward(input):
            self.teacher(input, training = False)
            with tf.GradientTape(persistent = True) as tape:
                self.student(input, training = True)
                B = input.shape[0]

                distill_loss = []
                for i, (aux, sl, tl) in enumerate(zip(self.student.aux_layers, self.student_layers, self.teacher_layers)):
                    s = aux(sl.feat, training = True)
                    t = tl.feat
                    
                    if i > 0:
                        distill_loss.append(tf.reduce_sum(tf.square(Gram(s, s_) - tf.stop_gradient(Gram(t,t_)))))
                    s_ = s
                    t_ = t
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

def Gram(x, x_):
    B, H, W, D = x.shape
    _, H_,W_,D_= x_.shape
    if H != H_:
        x_ = tf.nn.max_pool2d(x_, [2,2], 2, padding = 'SAME')

    x  = tf.reshape(x,  [B,-1,D])
    x_ = tf.reshape(x_, [B,-1,D_])
    return tf.matmul(x, x_, transpose_a = True)/H/W/B



