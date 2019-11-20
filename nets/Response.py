import tensorflow as tf
from nets import tcl
class FitNet(tf.keras.layers.Layer):
    '''
     Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta,  and  Yoshua  Bengio.
     Fitnets:   Hints  for  thin  deep  nets.
     arXiv preprint arXiv:1412.6550, 2014.
    '''
    def __init__(self, target, source):
        super(FitNet, self).__init__(target, source)
        self.source = source
        self.target = target
        self.linear_map = []
        self.feat_name = 'feature'
        for t, s in zip(target, source):
            Ds = source.shape[-1]
            Dt = target.shape[-1]
            if Ds != Dt:
                self.linearmap.append(tcl.Conv2d([3,3], Dt, activation_fn = None))
            else:
                self.linear_map.append(None)
                
    def call(self, target_feat, source_feat):
        def l2_loss(t,s,lm):
            if lm:
                t = lm(t)
            return tf.resuce_mean(tf.square(t-s))
            
        return tf.add_n([l2_loss(t, s, lm) for t,s,lm in zip(target_feat, source_feat, self.linear_map)])
    
def Soft_logits(student, teacher, T = 2):
    '''
    Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.  
    Distilling the knowledge in a neural network.
    arXiv preprint arXiv:1503.02531, 2015.
    '''
    with tf.variable_scope('KD'):
        return tf.reduce_mean(tf.reduce_sum( tf.nn.softmax(teacher/T)*(tf.nn.log_softmax(teacher/T)-tf.nn.log_softmax(student/T)),1 ))

def DML(student, teacher):
    '''
    Ying Zhang, Tao Xiang, Timothy M. Hospedales, Huchuan Lu. 
    Deep mutual learning.
    IEEE Conference on Computer Vision and Pattern Recognition. 2018.
    '''
    with tf.variable_scope('KD'):
        return (tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(teacher)*(tf.nn.log_softmax(teacher)-tf.nn.log_softmax(student)),1)) +
                tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(student)*(tf.nn.log_softmax(student)-tf.nn.log_softmax(teacher)),1)))/2

def Factor_Transfer(sfm, tfm, beta = 1e2):
    '''
    Jangho Kim, SeoungUK Park, Nojun Kwak.
    Paraphrasing Complex Network: Network Compression via Factor Transfer.
    Advances in Neural Information Processing Systems (NeurIPS). 2018.
    '''
    def Factor_transfer(X, rate, scope, reuse = False, ):
        with tf.variable_scope(scope):
            with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.conv2d_transpose], weights_regularizer=None,
                                                variables_collections = [tf.GraphKeys.GLOBAL_VARIABLES, 'Para']):
                D = tfm.get_shape().as_list()[-1]
                conv = tf.contrib.layers.conv2d(X,    int(D*rate**1), [3,3], 1,          scope='conv0', reuse = reuse)
                conv = tf.contrib.layers.conv2d(conv, int(D*rate**2), [3,3], int(1/rate),scope='conv1', reuse = reuse)
                conv = tf.contrib.layers.conv2d(conv, int(D*rate**3), [3,3], 1, activation_fn = None, scope='conv2', reuse = reuse)
                
                if reuse:
                    return tf.nn.l2_normalize(conv, -1)
                conv = tf.nn.leaky_relu(conv)
                deconv = tf.contrib.layers.conv2d_transpose(conv,   int(D*rate**2), [3,3], 1,          scope='convt0', reuse = reuse)
                deconv = tf.contrib.layers.conv2d_transpose(deconv, int(D*rate**1), [3,3], int(1/rate),scope='convt1', reuse = reuse)
                deconv = tf.contrib.layers.conv2d_transpose(deconv, D, [3,3], 1,  activation_fn = None, scope='convt2', reuse = reuse)
                return deconv

    with tf.variable_scope('Factor_Transfer'):
        with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.conv2d_transpose], trainable = True,
                                            biases_initializer = tf.zeros_initializer(), activation_fn = tf.nn.leaky_relu):
            rate = 0.5
            tfm_ = Factor_transfer(tfm, rate, 'Factor_transfer')
            if tfm.get_shape().as_list()[1] %2 == 1:
                _,H,W,_ = tfm.get_shape().as_list()
                tfm_ = tf.slice(tfm_, [0,0,0,0],[-1,H,W,-1])
            tf.add_to_collection('Para_loss', tf.reduce_mean(tf.reduce_mean(tf.abs(tfm-tfm_),[1,2,3])))
                    
            F_T = Factor_transfer(tfm, rate, 'Factor_transfer', True)

            with tf.variable_scope('Translator'):
                D = tfm.get_shape().as_list()[-1]
                conv = tf.contrib.layers.conv2d(sfm,  int(D*rate**1), [3,3], 1,          scope='conv0')
                conv = tf.contrib.layers.conv2d(conv, int(D*rate**2), [3,3], int(1/rate),scope='conv1')
                conv = tf.contrib.layers.conv2d(conv, int(D*rate**3), [3,3], 1, activation_fn = None, scope='conv2')
                F_S = tf.nn.l2_normalize(conv, -1)
            return tf.reduce_mean(tf.reduce_mean(tf.abs(F_T-F_S),[1,2,3]))*beta
