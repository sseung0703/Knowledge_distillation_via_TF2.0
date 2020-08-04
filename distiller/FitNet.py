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
        self.student.aux_layers = [tf.keras.Sequential([tcl.Conv2d([1,1], tl.gamma.shape[-1]),
                                                        tcl.BatchNorm(param_initializers = {'gamma' : tf.constant_initializer(1/tl.gamma.shape[-1])})] ) 
                                   for sl, tl in zip(self.student_layers, self.teacher_layers)]
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

    def initialize_student(self, dataset):
        optimizer = tf.keras.optimizers.SGD(1e-2, .9, nesterov=True)
        train_loss = tf.keras.metrics.Mean(name='train_loss')

        @tf.function(experimental_compile=True)
        def init_forward(input):
            self.teacher(input, training = False)
            with tf.GradientTape() as tape:
                self.student(input, training = True)
                B = input.shape[0]

                distill_loss = []
                for i, (aux, sl, tl) in enumerate(zip(self.student.aux_layers, self.student_layers, self.teacher_layers)):
                    s = aux(sl.feat)
                    t = tf.stop_gradient(tl.feat)
                    distill_loss.append(tf.reduce_mean(tf.square(s - t)))
                distill_loss = tf.add_n(distill_loss)

            gradients = tape.gradient(distill_loss, self.student.trainable_variables)
            if self.args.weight_decay > 0.:
                gradients = [g+v*self.args.weight_decay if g is not None else g
                             for g,v in zip(gradients, self.student.trainable_variables)]
            optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
            train_loss.update_state(distill_loss)

        for e in range(int(self.args.train_epoch)):
            for imgs, _ in dataset:
                init_forward(imgs)
            print('Aux Epoch: %d: loss: %.4f'%(e,train_loss.result()))
            train_loss.reset_states()
