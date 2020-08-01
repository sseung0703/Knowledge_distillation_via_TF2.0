import tensorflow as tf
import numpy as np

from nets import tcl

class distill:
    '''
    Jangho Kim, SeoungUK Park, Nojun Kwak.
    Paraphrasing Complex Network: Network Compression via Factor Transfer.
    Advances in Neural Information Processing Systems (NeurIPS). 2018.
    '''
    def __init__(self, args, model, teacher):
        self.args = args
        self.student = model
        self.teacher = teacher
        self.student_layer = self.sampled_layer(args.arch, self.student)
        self.teacher_layer = self.sampled_layer(args.teacher_arch, self.teacher)

        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(activation_fn = tf.nn.leaky_relu, trainable = False))
        setattr(tcl.Conv2d_transpose, 'pre_defined', kwargs(activation_fn = tf.nn.leaky_relu, trainable = False))

        rate = .5
        D = self.teacher_layer.gamma.shape[-1]
        self.aux_layers =  []
        self.aux_layers.append( tf.keras.Sequential([tcl.Conv2d([3,3], int(D*rate), 1),
                                                     tcl.Conv2d([3,3], int(D*rate**2), int(1/rate)),
                                                     tcl.Conv2d([3,3], int(D*rate**3), 1, activation_fn = None)]))
        self.aux_layers.append( tf.keras.Sequential([tcl.Conv2d_transpose([3,3], int(D*rate**2), 1),
                                                     tcl.Conv2d_transpose([3,3], int(D*rate), int(1/rate)),
                                                     tcl.Conv2d_transpose([3,3], D, 1, activation_fn = None)]))
        self.aux_layers.append( tf.keras.Sequential([tcl.Conv2d([3,3], int(D*rate), 1, trainable = True),
                                                     tcl.Conv2d([3,3], int(D*rate**2), int(1/rate), trainable = True),
                                                     tcl.Conv2d([3,3], int(D*rate**3), 1, activation_fn = None, trainable = True)]))

        # build
        self.teacher(np.zeros([1] + args.input_size, np.float32))
        self.student(np.zeros([1] + args.input_size, np.float32))
        self.aux_layers[1](self.aux_layers[0](self.teacher_layer.feat))
        self.aux_layers[2](self.student_layer.feat)
        self.beta = 1e2
        
    def sampled_layer(self, arch, model):
        if 'WResNet' in arch:
            model.Layers['bn_last'].keep_feat = 'pre_act'
            return model.Layers['bn_last']

    def forward(self, input, labels, target_loss):
        self.teacher(input, training = False)
        t = tf.nn.l2_normalize(self.aux_layers[0](self.teacher_layer.feat), -1)
        s = tf.nn.l2_normalize(self.aux_layers[2](self.student_layer.feat), -1)
        return target_loss + tf.reduce_mean(tf.abs(t-s))*self.beta

    def auxiliary_training(self, dataset):
        optimizer = tf.keras.optimizers.SGD(1e-3, .9, nesterov=True)
        train_loss = tf.keras.metrics.Mean(name='train_loss')

        @tf.function(experimental_compile=True)
        def training(images):
            teacher_aux = self.aux_layers[0].variables + self.aux_layers[1].variables
            self.teacher(images)
            with tf.GradientTape() as tape:
                tape.watch(teacher_aux)
                feat = self.teacher_layer.feat
                enc = tf.nn.leaky_relu(self.aux_layers[0](feat))
                dec = self.aux_layers[1](enc)
                loss = tf.reduce_mean(tf.abs(feat - dec))
            gradients = tape.gradient(loss, teacher_aux)
            gradients = [g+v*1e-4 for g,v in zip(gradients, teacher_aux)]
            optimizer.apply_gradients(zip(gradients, teacher_aux))
            train_loss.update_state(loss)

        for e in range(int(self.args.train_epoch*.3)):
            for images,_ in dataset:
                training(images)    
            print('Aux Epoch: %d: loss: %.4f'%(e,train_loss.result()))
            train_loss.reset_states()
