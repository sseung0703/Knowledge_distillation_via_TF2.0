import tensorflow as tf
from tensorflow.python.framework import function

def removenan(x):
    return tf.where(tf.math.is_finite(x), x,tf.zeros_like(x))

def SVD(X, n, method = 'SVD'):
    sz = X.get_shape().as_list()
    if len(sz)==4:
        x = tf.reshape(X,[-1,sz[1]*sz[2],sz[3]])
    elif len(sz)==3:
        x = X
    else:
        x = tf.expand_dims(X, 1)
        n = 1
    _, HW, D = x.get_shape().as_list()

    if method == 'SVD':
        s,u,v = SVD_custom(x)
            
    if method == 'EID':
        s,u,v = SVD_lite_custom(x)

    s = removenan(s)
    v = removenan(v)
    u = removenan(u)
       
    if n > 0:
        s = tf.nn.l2_normalize(tf.slice(s,[0,0],[-1,n]),1)
        u = tf.nn.l2_normalize(tf.slice(u,[0,0,0],[-1,-1,n]),1)
        v = tf.nn.l2_normalize(tf.slice(v,[0,0,0],[-1,-1,n]),1)
    return s, u, v

def SVD_eid(x):
    _, HW, D = x.get_shape().as_list()
    if HW/D < 3/2  and 2/3 < HW/D:
        with tf.device('CPU'):
            s,u,v = tf.linalg.svd(x,full_matrices=False)

    else:
        if HW < D:
            xxt = tf.matmul(x,x,transpose_b = True)
            with tf.device('CPU'):
                _,u,_ = tf.linalg.svd(xxt,full_matrices=False)
            v = tf.matmul(x, u, transpose_a = True)
            s = tf.linalg.norm(v, axis = 1)
            v = removenan(v/tf.expand_dims(s,1))
                
        else:
            xtx = tf.matmul(x,x,transpose_a = True)
            with tf.device('CPU'):
                _, v = tf.linalg.eigh(xtx)
            v = tf.reshape(tf.image.flip_left_right(tf.reshape(v,[-1,D,D,1])),[-1,D,D])

            u = tf.matmul(x, v)
            s = tf.linalg.norm(u, axis = 1)
            u = removenan(u/tf.expand_dims(s,1))
    
    return s, u, v
    
def SVP(X, n, is_training, reuse = False, name = None, whitening = False):
    with tf.variable_scope(name):
        s, U, V = SVD_eid(X, n, name = 'EID')
        N = U.get_shape().as_list()[1]
            
        ## rotation to coordinate center
        U, _ = Rotation(U, is_training, reuse, 'U')
        sign = tf.stop_gradient(tf.sign( tf.slice(U,[0,N-2,0],[-1,1,-1]) ))
        U *= sign

        V, b = Rotation(V, is_training, reuse, 'V')
        V *= sign
        
        V = Coordinate_conversion(V)
        V = removenan(V)
        
        if is_training:
            U = Coordinate_conversion(U)
            estimate_center(U)
            estimate_center(V)

        return s, U, V
    
def Align_rsv(x, y):
    with tf.name_scope('Align'):
        cosine = tf.matmul(x, y, transpose_a=True)
        mask = tf.where(tf.equal(tf.reduce_max(tf.abs(cosine), 1,keepdims=True), tf.abs(cosine)),
                        tf.sign(cosine), tf.zeros_like(cosine))
        x = tf.matmul(x, mask)
        return x, y

def Rotation(v, is_training, reuse, scope):
    with tf.variable_scope('Rotation_'+ scope):
        R, b = Rodrigues(v, is_training, reuse, scope)
        v_rot = tf.reduce_sum(R*tf.expand_dims(v,1),2)
        b_rot = tf.reduce_sum(R*tf.expand_dims(b,1),2)
        return v_rot, b_rot

def Coordinate_conversion(x):
    with tf.variable_scope('Coordinate_conversion'):
        B,D0,k = x.get_shape().as_list()
        n = tf.transpose(tf.matrix_band_part(tf.ones([1,1,D0-1,D0]),0,-1),[0,2,3,1])
        S = tf.sqrt(tf.reduce_sum(n*tf.expand_dims(x**2,1),2))+1e-3
        x_sphere = -tf.slice(x,[0,0,0],[-1,D0-1,-1])/S
        x_sphere = tf.concat([tf.slice(x_sphere,[0,0,0],[-1,D0-2,-1]),
                             (tf.slice(x_sphere,[0,D0-2,0],[-1,-1,-1])+1)*tf.stop_gradient(tf.sign(tf.slice(x,[0,D0-1,0],[-1,-1,-1]))) ],1)
        return x_sphere

def Rodrigues(X, is_training, reuse, scope):
    B, length, k = X.get_shape().as_list()
    with tf.variable_scope(scope, reuse = reuse):
        b = tf.get_variable('basis', [k,length], tf.float32, trainable = False,
                            collections = [tf.GraphKeys.GLOBAL_VARIABLES, 'basises'],
                            initializer=tf.ones_initializer())
        b = tf.nn.l2_normalize(b,1)
        b = tf.expand_dims(b,-1)

    with tf.variable_scope('Rodrigues'):
        import numpy as np
        y = np.zeros([k,length])
        y[:,-2] = 1
        y = tf.cast(np.expand_dims(y,-1),tf.float32)

        cos = tf.reduce_sum(y*b,[1,2],keepdims=True)
        I = tf.expand_dims(tf.eye(length),0)
    
        K = y*tf.matrix_transpose(b)-b*tf.matrix_transpose(y)
        R = I + K + tf.matmul(K,K)/(1+cos)
        R = tf.expand_dims(tf.transpose(R,[1,2,0]),0)

    return R, tf.transpose(b,[2,1,0])

def estimate_center(x):
    with tf.variable_scope('learn_basis'):
        tf.add_to_collection('basis_loss', kld_reg(tf.contrib.layers.flatten(x)))

def kld_reg(pred):
    with tf.variable_scope('kld'):
        pred = removenan(pred)

        mean, var = tf.nn.moments(pred,0,keep_dims=True)
        noise = tf.random_normal(tf.shape(pred))*tf.stop_gradient(tf.sqrt(var))

        pred  = tf.nn.softmax(tf.transpose(pred))
        noise = tf.nn.softmax(tf.transpose(noise))

        loss = tf.reduce_sum(tf.reduce_sum(noise * tf.log(noise/pred),1))
        return loss
    

@tf.custom_gradient
def SVD_custom(x):
    with tf.device('CPU'):
        s, U, V =  tf.linalg.svd(x)
    def gradient_svd(ds, dU, dV):
        u_sz = tf.squeeze(tf.slice(tf.shape(dU),[1],[1]))
        v_sz = tf.squeeze(tf.slice(tf.shape(dV),[1],[1]))
        s_sz = tf.squeeze(tf.slice(tf.shape(ds),[1],[1]))

        S = tf.linalg.diag(s)
        s_2 = tf.square(s)

        eye = tf.expand_dims(tf.eye(s_sz),0) 
        k = (1 - eye)/(tf.expand_dims(s_2,2)-tf.expand_dims(s_2,1) + eye)
        KT = tf.transpose(k,[0,2,1])
        KT = removenan(KT)
    
        def msym(X):
            return (X+tf.transpose(X,[0,2,1]))
    
        def left_grad(U,S,V,dU,dV):
            U, V = (V, U); dU, dV = (dV, dU)
            D = tf.matmul(dU,tf.linalg.diag(1/(s+1e-8)))
    
            grad = tf.matmul(D - tf.matmul(U, tf.linalg.diag(tf.linalg.diag_part(tf.matmul(U,D,transpose_a=True)))
                               + 2*tf.matmul(S, msym(KT*(tf.matmul(D,tf.matmul(U,S),transpose_a=True))))), V,transpose_b=True)
        
            grad = tf.transpose(grad, [0,2,1])
            return grad

        def right_grad(U,S,V,dU,dV):
            grad = tf.matmul(2*tf.matmul(U, tf.matmul(S, msym(KT*(tf.matmul(V,dV,transpose_a=True)))) ),V,transpose_b=True)
            return grad
    
        grad = tf.cond(tf.greater(v_sz, u_sz), lambda :  left_grad(U,S,V,dU,dV), 
                                               lambda : right_grad(U,S,V,dU,dV))
        return [grad, None]
    return [s,U,V], gradient_svd

@tf.custom_gradient
def SVD_lite_custom(x):
    s, U, V = SVD_eid(x)
    def gradient_svd(ds, dU, dV):
        u_sz = tf.squeeze(tf.slice(tf.shape(dU),[1],[1]))
        v_sz = tf.squeeze(tf.slice(tf.shape(dV),[1],[1]))
        s_sz = tf.squeeze(tf.slice(tf.shape(ds),[1],[1]))

        S = tf.linalg.diag(s)
        s_2 = tf.square(s)

        eye = tf.expand_dims(tf.eye(s_sz),0) 
        k = (1 - eye)/(tf.expand_dims(s_2,2)-tf.expand_dims(s_2,1) + eye)
        KT = tf.transpose(k,[0,2,1])
        KT = removenan(KT)
    
        def msym(X):
            return (X+tf.transpose(X,[0,2,1]))
    
        def left_grad(U,S,V,dU,dV):
            U, V = (V, U); dU, dV = (dV, dU)
            D = tf.matmul(dU,tf.linalg.diag(1/(s+1e-8)))
    
            grad = tf.matmul(D - tf.matmul(U, tf.linalg.diag(tf.linalg.diag_part(tf.matmul(U,D,transpose_a=True)))
                               + 2*tf.matmul(S, msym(KT*(tf.matmul(D,tf.matmul(U,S),transpose_a=True))))), V,transpose_b=True)
        
            grad = tf.transpose(grad, [0,2,1])
            return grad

        def right_grad(U,S,V,dU,dV):
            grad = tf.matmul(2*tf.matmul(U, tf.matmul(S, msym(KT*(tf.matmul(V,dV,transpose_a=True)))) ),V,transpose_b=True)
            return grad
    
        grad = tf.cond(tf.greater(v_sz, u_sz), lambda :  left_grad(U,S,V,dU,dV), 
                                               lambda : right_grad(U,S,V,dU,dV))
        return [grad, None]
    return [s,U,V], gradient_svd

