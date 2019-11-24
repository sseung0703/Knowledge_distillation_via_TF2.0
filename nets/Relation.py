import tensorflow as tf
from nets import SVP
from nets import tcl

class RKD(tf.keras.layers.Layer):
    '''
    Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho.
    relational knowledge distillation.
    arXiv preprint arXiv:1904.05068, 2019.
    '''
    def __init__(self, target, source, l = [25,50], **kwargs):
        super(RKD, self).__init__(**kwargs)
        self.l = l
        self.feat_name = 'last_embedded'

    def call(self, target_feat, source_feat):
        def Huber_loss(x,y):
            with tf.name_scope('Huber_loss'):
                return tf.reduce_mean(tf.where(tf.less_equal(tf.abs(x-y), 1.),
                                               tf.square(x-y)/2, tf.abs(x-y)-1/2))

        def Distance_wise_potential(x):
            with tf.name_scope('DwP'):
                x_square = tf.reduce_sum(tf.square(x),-1)
                prod = tf.matmul(x,x,transpose_b=True)
                distance = tf.sqrt(tf.maximum(tf.expand_dims(x_square,1)+tf.expand_dims(x_square,0) -2*prod, 1e-12))
                mu = tf.reduce_sum(distance)/tf.reduce_sum(tf.cast(distance > 0., tf.float32))
                return distance/(mu+1e-8)

        def Angle_wise_potential(x):
            with tf.name_scope('AwP'):
                e = tf.expand_dims(x,0)-tf.expand_dims(x,1)
                e_norm = tf.nn.l2_normalize(e,2)
            return tf.matmul(e_norm, e_norm,transpose_b=True)

        t = tf.nn.l2_normalize(target_feat,1)
        s = tf.nn.l2_normalize(source_feat,1)

        distance_loss = Huber_loss(Distance_wise_potential(t),Distance_wise_potential(s))
        angle_loss    = Huber_loss(   Angle_wise_potential(t),   Angle_wise_potential(s))

        return distance_loss*self.l[0]+angle_loss*self.l[1]

class MHGD(tf.keras.layers.Layer):
    '''
    Seunghyun Lee, Byung Cheol Song.
    Graph-based Knowledge Distillation by Multi-head Attention Network.
    British Machine Vision Conference (BMVC) 2019
    '''
    def __init__(self, target, source, weight_decay, num_head = 8,  **kwargs):
        super(MHGD, self).__init__(**kwargs)
        self.num_head = num_head
        self.linear_map = []
        self.Q = [None]
        self.K = [None]
        self.E = [None]
        self.feat_name = 'feature'

        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.FC, 'pre_defined', kwargs(kernel_regularizer = tf.keras.regularizers.l2(weight_decay),
                                              use_biases = False, activation_fn = None))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(param_regularizers = {'gamma' : tf.keras.regularizers.l2(weight_decay),
                                                                            'beta' : tf.keras.regularizers.l2(weight_decay)}))

        for i, (t,s) in enumerate(zip(target.feature, source.feature)):
            if t.shape[-1] != s.shape[-1]:
                self.linear_map.append(tf.keras.Sequential([tcl.FC(s.shape[-1]), tcl.BatchNorm()]))
            else:
                self.linear_map.append(None)

            if i > 0:
                with tf.name_scope('Aux'): 
                    D = (t.shape[-1]+t_.shape[-1])//2
                    self.Q.append(tf.keras.Sequential([tcl.FC(D*num_head), tcl.BatchNorm()]))
                    self.K.append(tf.keras.Sequential([tcl.FC(D*num_head), tcl.BatchNorm()]))
                    self.E.append([tcl.FC(D), tcl.BatchNorm(activation_fn = tf.nn.relu), 
                                   tcl.FC(t.shape[-1], use_biases = True, biases_regularizer = tf.keras.regularizers.l2(weight_decay))])
            t_ = t

    def Attention_head(self, f, b, K, Q, D, num_head, training = False):
        B = f.shape[0]

        k = K(f, training = training)
        q = Q(b, training = training)

        k = tf.reshape(k, [B, D, num_head])
        q = tf.reshape(q, [B, D, num_head])
                
        k = tf.transpose(k,[2,0,1])
        q = tf.transpose(q,[2,1,0])
        return tf.matmul(k, q)

    def Estimator(self, f, G, E, Db, num_head, training):
        B = G.shape[1]
        G = tf.nn.softmax(G)

        noise = tf.keras.backend.random_normal([num_head, B, 1])
        G *= tf.where(noise - tf.reduce_mean(noise, 0, keepdims=True) > 0, tf.ones_like(noise), tf.zeros_like(noise))
        G = tf.reshape(G, [num_head*B, B])
                
        Df = f.shape[-1]
        D = (Df+Db)//2

        f = E[1](E[0](f), training = training)
        f = tf.reshape(tf.matmul(G, f), [num_head, B, D])
        f = tf.reshape(tf.transpose(f,[1,0,2]),[B,D*num_head])

        f = E[2](f)
        b_ = tf.nn.l2_normalize(f, -1)
        return b_
    
    def kld_loss(self, X, Y):
        return tf.reduce_sum( tf.nn.softmax(X)*(tf.nn.log_softmax(X)-tf.nn.log_softmax(Y)) )

    def call(self, student_feat, teacher_feat, training = False):
        KD_losses = []
        for i, (s,t,lm,q,k,e) in enumerate(zip(student_feat, teacher_feat, self.linear_map, self.Q, self.K, self.E)):
            if lm:
                s = lm(s)
            _,_,t = SVP.SVD(t, 1, 'EID')
            _,_,s = SVP.SVD(s, 4, 'EID')
            s, t = SVP.Align_rsv(s, t)
            D = s.shape[1]
            
            t = tf.reshape(t,[-1,D])
            s = tf.reshape(s,[-1,D])

            if i > 0:
                D_ = sb.shape[1]
                D2 = (D+D_)//2
                G_T = self.Attention_head(t, tb, k, q, D2, self.num_head, training = False)
                G_S = self.Attention_head(s, sb, k, q, D2, self.num_head, training = False)

                G_T = tf.tanh(G_T)
                G_S = tf.tanh(G_S)
               
                KD_losses.append(self.kld_loss(G_S, G_T))
                    
            tb, sb = t, s
        return tf.add_n(KD_losses)

    def aux_call(self, teacher_feat, training = False):
        MHA_losses = []
        with tf.name_scope('Aux'): 
            for i, (t,q,k,e) in enumerate(zip(teacher_feat, self.Q, self.K, self.E)):
                _,_,t = SVP.SVD(t, 1, 'EID')
                D = t.shape[1]
            
                t = tf.reshape(t,[-1,D])

                if i > 0:
                    D_ = tb.shape[1]
                    D2 = (D+D_)//2
                    G_T = self.Attention_head(t, tb, k, q, D2, self.num_head, training = training)
                    t_  = self.Estimator(tb, G_T, e, D, self.num_head, training = training)
                    MHA_losses.append(tf.reduce_mean(1-tf.reduce_sum(t_*t,1)))
                tb = t
            return tf.add_n(MHA_losses)
