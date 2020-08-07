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
        setattr(tcl.Conv2d, 'pre_defined', kwargs(use_biases = False, trainable = True))
        setattr(tcl.Conv2d_transpose, 'pre_defined', kwargs(use_biases = False, trainable = True))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(activation_fn = tf.nn.leaky_relu, trainable = True))

        rate = .5
        D = self.teacher_layer.gamma.shape[-1]
        self.aux_layers =  []
        self.aux_layers.append( tf.keras.Sequential([tcl.Conv2d([3,3], int(D*rate), 1, name = 'conv0'),
                                                     tcl.BatchNorm(name = 'bn0'),
                                                     tcl.Conv2d([3,3], int(D*rate**2), int(1/rate), name = 'conv1'),
                                                     tcl.BatchNorm(name = 'bn1'),
                                                     tcl.Conv2d([3,3], int(D*rate**3), 1, name = 'conv2'),
                                                     tcl.BatchNorm(activation_fn = None, name = 'bn2'),]))
        self.aux_layers.append( tf.keras.Sequential([tcl.Conv2d_transpose([3,3], int(D*rate**2), 1, name = 'convt0'),
                                                     tcl.BatchNorm(name = 'bnt0'),
                                                     tcl.Conv2d_transpose([3,3], int(D*rate), int(1/rate), name = 'convt1'),
                                                     tcl.BatchNorm(name = 'bnt1'),
                                                     tcl.Conv2d_transpose([3,3], D, 1, use_biases = True, name = 'convt2')]))

        self.student.aux_layers = tf.keras.Sequential([tcl.Conv2d([3,3], int(D*rate), 1, name = 'conv0'),
                                                       tcl.BatchNorm(name = 'bn0'),
                                                       tcl.Conv2d([3,3], int(D*rate**2), int(1/rate), name = 'conv1'),
                                                       tcl.BatchNorm(name = 'bn1'),
                                                       tcl.Conv2d([3,3], int(D*rate**3), 1, activation_fn = None, name = 'conv2'),
                                                       tcl.BatchNorm(activation_fn = None, name = 'bn2'),])

        # build
        self.teacher(np.zeros([1] + args.input_shape, np.float32))
        self.student(np.zeros([1] + args.input_shape, np.float32))
        self.aux_layers[1](self.aux_layers[0](self.teacher_layer.feat))
        self.student.aux_layers(self.student_layer.feat)
        self.beta = 1e2
        
    def sampled_layer(self, arch, model):
        if 'WResNet' in arch:
            model.Layers['bn_last'].keep_feat = 'output'
            return model.Layers['bn_last']

    def forward(self, input, labels, target_loss):
        self.teacher(input, training = False)
        t = tf.nn.l2_normalize(self.aux_layers[0](self.teacher_layer.feat, training = False), [1,2,3])
        s = tf.nn.l2_normalize(self.student.aux_layers(self.student_layer.feat, training = True), [1,2,3])
        B,H,W,D = s.shape
        return target_loss + tf.reduce_mean(tf.abs(t-s))*self.beta

    def auxiliary_training(self, dataset):
        optimizer = tf.keras.optimizers.SGD(1e-3, .9, nesterov=True)
        train_loss = tf.keras.metrics.Mean(name='train_loss')

        teacher_aux = self.aux_layers[0].trainable_variables + self.aux_layers[1].trainable_variables
        @tf.function(experimental_compile=True)
        def training(images):
            self.teacher(images)
            with tf.GradientTape() as tape:
                tape.watch(teacher_aux)
                feat = self.teacher_layer.feat
                enc = tf.nn.leaky_relu(self.aux_layers[0](feat))
                dec = self.aux_layers[1](enc)
                dec = self.teacher_layer.activation_fn(dec)
                B,H,W,D = dec.shape
                loss = tf.reduce_sum(tf.abs(feat - dec))/B/H/W + tf.reduce_sum(tf.abs(enc))/B/H/W*1e-6
            gradients = tape.gradient(loss, teacher_aux)
            gradients = [g+v*self.args.weight_decay for g,v in zip(gradients, teacher_aux)]
            optimizer.apply_gradients(zip(gradients, teacher_aux))
            train_loss.update_state(loss)

        for e in range(int(self.args.train_epoch*.3)):
            for images,_ in dataset:
                training(images)    
            print('Aux Epoch: %d: loss: %.4f'%(e,train_loss.result()))
            train_loss.reset_states()
