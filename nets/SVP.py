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

