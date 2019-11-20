import tensorflow as tf
from nets import tcl
class FitNet(tf.keras.layers.Layer):
    '''
     Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta,  and  Yoshua  Bengio.
     Fitnets:   Hints  for  thin  deep  nets.
     arXiv preprint arXiv:1412.6550, 2014.
    '''
    def __init__(self, target, source, weight_decay, **kwargs):
        super(FitNet, self).__init__(**kwargs)
        self.linear_map = []
        self.feat_name = 'feature'
        for t, s in zip(target.feature, source.feature):
            Ds = s.shape[-1]
            Dt = t.shape[-1]
            if Ds != Dt:
                self.linearmap.append(tcl.Conv2d([3,3], Ds, activation_fn = None))
            else:
                self.linear_map.append(None)
                
    def call(self, target_feat, source_feat):
        def l2_loss(t,s,lm):
            if lm:
                t = lm(t)
            return tf.reduce_mean(tf.square(t-s))
            
        return tf.add_n([l2_loss(t, s, lm) for t,s,lm in zip(target_feat, source_feat, self.linear_map)])

class AT(tf.keras.layers.Layer):
    '''
     Zagoruyko, Sergey and Komodakis, Nikos.
     Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer.
     arXiv preprint arXiv:1612.03928, 2016.
    '''
    def __init__(self, target, source, beta = 1e3):
        super(AT, self).__init__(target, source, beta)
        self.beta = beta
        self.feat_name = 'feature_noact'
        
    def call(self, target_feat, source_feat):
        def at(t,s):
            s = tf.reduce_mean(tf.square(s),-1)
            s = tf.nn.l2_normalize(s, [1,2])
            
            t = tf.reduce_mean(tf.square(t),-1)
            t = tf.nn.l2_normalize(t, [1,2])
            return tf.resuce_mean(tf.square(t-s))
            
        return tf.add_n([at(t, s) for t,s,lm in zip(target_feat, source_feat)])*self.beta
    
class AB(tf.keras.layers.Layer):
    '''
    Byeongho Heo,  Minsik Lee,  Sangdoo Yun,  and Jin Young Choi.   
    Knowledge transfer via distillation of activation boundaries formed by hidden neurons.
    AAAI Conference on Artificial Intelligence (AAAI), 2019.
    '''
    def __init__(self, target, source, weight_decay, margin=1., weight = 3e-3, **kwargs):
        super(AB, self).__init__(**kwargs)
        self.margin = margin
        self.weight = weight
        self.linear_map = []
        self.feat_name = 'feature_noact'
        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(kernel_regularizer = tf.keras.regularizers.l2(weight_decay),
                                                  use_biases = False, activation_fn = None))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(param_regularizers = {'gamma':tf.keras.regularizers.l2(weight_decay),
                                                                           'beta':tf.keras.regularizers.l2(weight_decay)}))
    
        for t, s in zip(target.feature_noact, source.feature_noact):
            Ds = s.shape[-1]
            Dt = t.shape[-1]
            if Ds != Dt:
                self.linear_map.append(tf.keras.Sequential([tcl.Conv2d([3,3], Ds),tcl.BatchNorm()]))
            else:
                self.linear_map.append(None)
                
    def call(self, target_feat, source_feat):
        def criterion_alternative_L2(t, s, lm, margin):
            if lm is not None:
                s = lm(s)
            loss = tf.square(t + margin) * tf.cast(tf.logical_and(t > -margin, s <= 0.), tf.float32)\
                  +tf.square(t - margin) * tf.cast(tf.logical_and(t <= margin, s > 0.), tf.float32)
            return tf.reduce_mean(tf.reduce_sum(tf.abs(loss),[1,2,3]))
            
        return tf.add_n([criterion_alternative_L2(t, s, lm, self.margin)*2**(-len(target_feat)+i+1 )
                         for i, (t,s,lm) in enumerate(zip(target_feat, source_feat, self.linear_map))])*self.weight
    
class VID(tf.keras.layers.Layer):
    def __init__(self, target, source, weight_decay, **kwargs):
        super(VID, self).__init__(**kwargs)
        self.source = source
        self.target = target
        self.linear_map = []
        self.alpha = []
        self.feat_name = 'feature'
        
        for t, s in zip(target.feature, source.feature):
            Ds = s.shape[-1]
            layers = []
            for i in range(3):
                tcl.Conv2d([1,1], Ds if i == 2 else Ds*2)
                tcl.BatchNorm(activation_fn = None if i == 2 else tf.nn.relu)
            self.linear_map.append(tf.keras.Sequential(layers))
            self.alpha.append(self.add_weight(name  = 'alpha', shape = [1,1,1,Ds], initializer=tf.constant_initializer(5.)))
            
    def call(self, target_feat, source_feat):
        def VID_loss(t, s, lm, alpha):
            s = lm(s)
            var   = tf.math.softplus(alpha)+1
            return tf.reduce_mean(tf.log(var) + tf.square(t - s)/var)/2
        
        return tf.add_n([VID_loss(*args) for args in zip(target_feat, source_feat, self.linear_map, self.alpha)])
