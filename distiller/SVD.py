import tensorflow as tf

def removenan(x):
    return tf.where(tf.math.is_finite(x), x,tf.zeros_like(x))

def SVD(x, n):
    B,H,W,D = x.shape
    x = tf.reshape(x, [B,H*W, D])
    s,u,v = SVD_custom(x)
    s = removenan(s)
    v = removenan(v)
    u = removenan(u)
       
    s = tf.nn.l2_normalize(tf.slice(s,[0,0],[-1,n]),1)
    return s, u, v

def Align_rsv(x, y):
    cosine = tf.matmul(x, y, transpose_a=True)
    mask = tf.where(tf.equal(tf.reduce_max(tf.abs(cosine), 1,keepdims=True), tf.abs(cosine)),
                    tf.sign(cosine), tf.zeros_like(cosine))
    x = tf.matmul(x, mask)
    return x, y

@tf.custom_gradient
def SVD_custom(x):
    with tf.device('CPU'):
        s, U, V =  tf.linalg.svd(x)
        B,M,N = x.shape
        
        xx = tf.matmul(x, x, transpose_a = M > N, transpose_b = M <= N)
        _, V = tf.linalg.eigh(xx)
        V = tf.gather(V, list(range(min(M,N)))[::-1], axis = 2)
        
        if M > N:
            Us = tf.matmul(x, V)
            s = tf.norm(Us, axis = 1, keepdims=True)
            U = Us/tf.maximum(s, 1e-12)
        else:
            U = V
            Vs = tf.matmul(x, U, transpose_a = True)
            s = tf.norm(Vs, axis = 1, keepdims=True)
            V = Vs/tf.maximum(s, 1e-12)
        s = tf.reshape(s, [B,-1])

    def gradient_svd(ds, dU, dV):
        _,M,N = x.shape
        K = min(M,N)
        
        S = tf.linalg.diag(s)
        s_2 = tf.square(s)

        eye = tf.expand_dims(tf.eye(K),0) 
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
    
        grad = tf.cond(tf.greater(M, N), lambda :  left_grad(U,S,V,dU,dV), 
                                         lambda : right_grad(U,S,V,dU,dV))
        return [grad]
    return [s,U,V], gradient_svd
