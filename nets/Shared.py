import tensorflow as tf
from nets import SVP
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
    
def FSP(students, teachers, weight = 1e-3):
    '''
    Junho Yim, Donggyu Joo, Jihoon Bae, and Junmo Kim.
    A gift from knowledge distillation: Fast optimization, network minimization and transfer learning. 
    In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4133–4141, 2017.
    '''
    def Grammian(top, bot):
        with tf.variable_scope('Grammian'):
            t_sz = top.get_shape().as_list()
            b_sz = bot.get_shape().as_list()
    
            if t_sz[1] > b_sz[1]:
                top = tf.contrib.layers.max_pool2d(top, [2, 2], 2)
                            
            top = tf.reshape(top,[-1, b_sz[1]*b_sz[2], t_sz[-1]])
            bot = tf.reshape(bot,[-1, b_sz[1]*b_sz[2], b_sz[-1]])
    
            Gram = tf.matmul(top, bot, transpose_a = True)/(b_sz[1]*b_sz[2])
            return Gram
    with tf.variable_scope('FSP'):
        Dist_loss = []
        for i in range(len(students)-1):
            gs0 = Grammian(students[i], students[i+1])
            gt0 = Grammian(teachers[i], teachers[i+1])
     
            Dist_loss.append(tf.reduce_mean(tf.reduce_sum(tf.square(tf.stop_gradient(gt0)-gs0),[1,2])/2 ))

        return tf.add_n(Dist_loss)*weight
    
def KD_SVD(student_feature_maps, teacher_feature_maps, dist_type = 'SVD'):
    '''
    Seung Hyun Lee, Dae Ha Kim, and Byung Cheol Song.
    Self-supervised knowledge distillation using singular value decomposition. In
    European Conference on ComputerVision, pages 339–354. Springer, 2018.
    '''
    with tf.variable_scope('Distillation'):
        GNN_losses = []
        K = 1
        V_Tb = V_Sb = None
        for i, sfm, tfm in zip(range(len(student_feature_maps)), student_feature_maps, teacher_feature_maps):
            with tf.variable_scope('Compress_feature_map%d'%i):
                if dist_type == 'SVD':
                    Sigma_T, U_T, V_T = SVP.SVD(tfm, K, name = 'TSVD%d'%i)
                    Sigma_S, U_S, V_S = SVP.SVD(sfm, K+3, name = 'SSVD%d'%i)
                    B, D,_ = V_S.get_shape().as_list()
                    V_S, V_T = SVP.Align_rsv(V_S, V_T)
                    
                elif dist_type == 'EID':
                    Sigma_T, U_T, V_T = SVP.SVD_eid(tfm, K, name = 'TSVD%d'%i)
                    Sigma_S, U_S, V_S = SVP.SVD_eid(sfm, K+3, name = 'SSVD%d'%i)
                    B, D,_ = V_S.get_shape().as_list()
                    V_S, V_T = SVP.Align_rsv(V_S, V_T)
                
                Sigma_T = tf.expand_dims(Sigma_T,1)
                V_T *= Sigma_T
                V_S *= Sigma_T
                
            if i > 0:
                with tf.variable_scope('RBF%d'%i):    
                    S_rbf = tf.exp(-tf.square(tf.expand_dims(V_S,2)-tf.expand_dims(V_Sb,1))/8)
                    T_rbf = tf.exp(-tf.square(tf.expand_dims(V_T,2)-tf.expand_dims(V_Tb,1))/8)

                    l2loss = (S_rbf-tf.stop_gradient(T_rbf))**2
                    l2loss = tf.where(tf.is_finite(l2loss), l2loss, tf.zeros_like(l2loss))
                    GNN_losses.append(tf.reduce_sum(l2loss))
            V_Tb = V_T
            V_Sb = V_S

        transfer_loss =  tf.add_n(GNN_losses)

        return transfer_loss
