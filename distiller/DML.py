import tensorflow as tf
import numpy as np

class distill:
    '''
    Ying Zhang, Tao Xiang, Timothy M. Hospedales, Huchuan Lu. 
    Deep mutual learning.
    IEEE Conference on Computer Vision and Pattern Recognition. 2018.
    '''
    def __init__(self, args, model, teacher):
        self.args = args
        self.student = model
        self.teacher = teacher

        self.student_layer = self.sampled_layer(args.arch, self.student)
        self.teacher_layer = self.sampled_layer(args.teacher_arch, self.teacher)

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.T = 5

    def sampled_layer(self, arch, model):
        if 'WResNet' in arch:
            model.Layers['fc'].keep_feat = 'output'
            return model.Layers['fc']

    def KLD(self, x, y):
        B = x.shape[0]
        x /= self.T
        y /= self.T
        return tf.reduce_sum(tf.nn.softmax(x, 1) * (tf.nn.log_softmax(x, 1) - tf.nn.log_softmax(y, 1) ))/B

    def forward(self, input, labels, target_loss):
        with tf.GradientTape(persistent = True) as tape:
            pred = self.teacher(input, training = True)
            teacher_loss = self.loss_object(labels, pred)
            total_loss = (teacher_loss + self.KLD(self.student_layer.feat, pred))/2

        trainable_variables = self.teacher.trainable_variables
        gradients = tape.gradient(total_loss, trainable_variables)
        if self.args.weight_decay > 0.:
            gradients = [g+v*self.args.weight_decay for g,v in zip(gradients, trainable_variables)]
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        pred = self.teacher(input, training = False)
        return (target_loss + self.KLD(pred, self.student_layer.feat))/2
            

        
        
        