import tensorflow as tf
import numpy as np

class distill:
    '''
    Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.  
    Distilling the knowledge in a neural network.
    arXiv preprint arXiv:1503.02531, 2015.
    '''
    def __init__(self, args, model, teacher):
        self.args = args
        self.student = model
        self.teacher = teacher

        self.student_layer = self.sampled_layer(args.arch, self.student)
        self.teacher_layer = self.sampled_layer(args.teacher_arch, self.teacher)

        self.T = 5

    def sampled_layer(self, arch, model):
        if 'WResNet' in arch:
            model.Layers['fc'].keep_feat = 'output'
            return model.Layers['fc']

    def KLD(self,x, y):
        x /= self.T
        y /= self.T
        B = x.shape[0]
        return tf.reduce_sum(tf.nn.softmax(x, 1) * (tf.nn.log_softmax(x, 1) - tf.nn.log_softmax(y, 1) ))/B

    def forward(self, input, labels, target_loss):
        B = input.shape[0]
        self.teacher(input, training = False)
        return (target_loss + self.KLD(self.teacher_layer.feat, self.student_layer.feat))/2