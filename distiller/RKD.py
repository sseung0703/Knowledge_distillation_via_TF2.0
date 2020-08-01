import tensorflow as tf
import numpy as np

def Huber_loss(x,y):
    return tf.reduce_mean(tf.where(tf.less_equal(tf.abs(x-y), 1.), tf.square(x-y)/2, tf.abs(x-y)-1/2))
            
def Distance_wise_potential(x):
    x_square = tf.reduce_sum(tf.square(x),-1)
    prod = tf.matmul(x,x,transpose_b=True)
    distance = tf.sqrt(tf.maximum(tf.expand_dims(x_square,1)+tf.expand_dims(x_square,0) -2*prod, 1e-12))
    mu = tf.reduce_sum(distance)/tf.reduce_sum(tf.where(distance > 0., tf.ones_like(distance), tf.zeros_like(distance)))
    return distance/(mu+1e-8)
            
def Angle_wise_potential(x):
    e = tf.expand_dims(x,0)-tf.expand_dims(x,1)
    e_norm = tf.nn.l2_normalize(e,2)
    return tf.matmul(e_norm, e_norm,transpose_b=True)

class distill:
    '''
    Park, Wonpyo, et al.
    Relational knowledge distillation.
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
    '''
    def __init__(self, args, model, teacher):
        self.args = args
        self.student = model
        self.teacher = teacher

        self.student_layer = self.sampled_layer(args.arch, self.student)
        self.teacher_layer = self.sampled_layer(args.teacher_arch, self.teacher)
        self.l = [1e2,2e2]

    def sampled_layer(self, arch, model):
        if 'WResNet' in arch:
            model.Layers['fc'].keep_feat = 'input'
            return model.Layers['fc']

    def loss(self, sl, tl):
        s = sl.feat
        t = tl.feat
        distance_loss = Huber_loss(Distance_wise_potential(s),Distance_wise_potential(t))
        angle_loss    = Huber_loss(   Angle_wise_potential(s),   Angle_wise_potential(t))    
        return distance_loss*self.l[0]+angle_loss*self.l[1]

    def forward(self, input, labels, target_loss):
        self.teacher(input, training = False)
        return target_loss + self.loss(self.student_layer, self.teacher_layer)
