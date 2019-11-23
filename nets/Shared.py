import tensorflow as tf
from nets import SVP
from nets import tcl

class FSP(tf.keras.layers.Layer):
    '''
    Junho Yim, Donggyu Joo, Jihoon Bae, and Junmo Kim.
    A gift from knowledge distillation: Fast optimization, network minimization and transfer learning. 
    In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4133–4141, 2017.
    '''
    def __init__(self, target, source, weight_decay, weight = 1e-3, **kwargs):
        super(FSP, self).__init__(**kwargs)
        self.linear_map = []
        self.weight = weight
        self.feat_name = 'feature_noact'
        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(kernel_regularizer = tf.keras.regularizers.l2(weight_decay),
                                                  use_biases = False, activation_fn = None))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(param_regularizers = {'gamma':tf.keras.regularizers.l2(weight_decay),
                                                                            'beta':tf.keras.regularizers.l2(weight_decay)}))
        for t, s in zip(target.feature, source.feature):
            Dt = t.shape[-1]
            Ds = s.shape[-1]
            if Ds != Dt:
                self.linear_map.append(tcl.Conv2d([3,3], Ds))
            else:
                self.linear_map.append(None)

    def Grammian(self, top, bot):
        t_sz = top.shape
        b_sz = bot.shape
    
        if t_sz[1] > b_sz[1]:
            top = tf.nn.max_pool2d(top, 2, 2, 'SAME')
                            
        top = tf.reshape(top,[-1, b_sz[1]*b_sz[2], t_sz[-1]])
        bot = tf.reshape(bot,[-1, b_sz[1]*b_sz[2], b_sz[-1]])
    
        Gram = tf.matmul(top, bot, transpose_a = True)/(b_sz[1]*b_sz[2])
        return Gram

    def call(self, student_feat, teacher_feat, training = True):
        KD_losses = []
        for i, (s,t,lm) in enumerate(zip(student_feat, teacher_feat, self.linear_map)):
            if lm:
                s = lm(s)
            if i > 0:
                gs0 = self.Grammian(sb, s)
                gt0 = self.Grammian(tb, t)
     
                KD_losses.append(tf.reduce_mean(tf.reduce_sum(tf.square(gt0-gs0),[1,2])/2 ))
            sb, tb = s, t

        return tf.add_n(KD_losses)*self.weight

class KD_SVD(tf.keras.layers.Layer):
    '''
    Seung Hyun Lee, Dae Ha Kim, and Byung Cheol Song.
    Self-supervised knowledge distillation using singular value decomposition. In
    European Conference on ComputerVision, pages 339–354. Springer, 2018.
    '''
    def __init__(self, target, source, weight_decay, K = 4, beta = 8, dist_type = 'SVD', **kwargs):
        super(KD_SVD, self).__init__(**kwargs)
        self.linear_map = []
        self.K = K
        self.beta = beta
        self.dist_type = dist_type
        self.feat_name = 'feature'
        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(kernel_regularizer = tf.keras.regularizers.l2(weight_decay),
                                                  use_biases = False, activation_fn = None))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(param_regularizers = {'gamma':tf.keras.regularizers.l2(weight_decay),
                                                                            'beta':tf.keras.regularizers.l2(weight_decay)}))
        for t, s in zip(target.feature, source.feature):
            Dt = t.shape[-1]
            Ds = s.shape[-1]
            if Ds != Dt:
                self.linear_map.append(tcl.Conv2d([3,3], Ds))
            else:
                self.linear_map.append(None)
                
    def call(self, student_feat, teacher_feat, training = True):
        KD_losses = []
        for i, (s,t,lm) in enumerate(zip(student_feat, teacher_feat, self.linear_map)):
            if lm:
                s = lm(s)
            Sigma_T, U_T, V_T = SVP.SVD(t, self.K,   self.dist_type)
            Sigma_S, U_S, V_S = SVP.SVD(s, self.K+3, self.dist_type)
            B, D,_ = V_S.shape
            V_S, V_T = SVP.Align_rsv(V_S, V_T)
                    
            Sigma_T = tf.expand_dims(Sigma_T,1)
            V_T *= Sigma_T
            V_S *= Sigma_T
            if i > 0:
                with tf.name_scope('RBF%d'%i):    
                    S_rbf = tf.exp(-tf.square(tf.expand_dims(V_S,2)-tf.expand_dims(V_Sb,1))/self.beta)
                    T_rbf = tf.exp(-tf.square(tf.expand_dims(V_T,2)-tf.expand_dims(V_Tb,1))/self.beta)

                    l2loss = tf.square(S_rbf-T_rbf)
                    KD_losses.append(tf.reduce_sum(l2loss))
            V_Tb = V_T
            V_Sb = V_S

        return tf.add_n(KD_losses)
        
