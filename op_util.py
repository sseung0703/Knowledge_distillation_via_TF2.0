import tensorflow as tf
import numpy as np

COMPILE_MODE = True

def Optimizer(model, weight_decay, LR):
    with tf.name_scope('Optimizer_w_Distillation'):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.SGD(LR, .9, nesterov=True)
        if hasattr(model, 'distiller'):
            model.distiller.optimizer = optimizer
            model.distiller.weight_decay = weight_decay
        
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        
    @tf.function(experimental_compile = COMPILE_MODE)
    def training(images, labels):
        with tf.GradientTape(persistent = True) as tape:
            pred = model(images, training = True)
            target_loss = loss_object(labels, pred)

            try:
                total_loss = model.distiller.forward(images, labels, target_loss)
            except:
                total_loss = target_loss

        trainable_variables = model.trainable_variables
        try:
            gradients = model.distiller.backward(tape, total_loss, trainable_variables)
        except:
            gradients = tape.gradient(total_loss, trainable_variables)                

        if weight_decay > 0.:
            gradients = [g+v*weight_decay for g,v in zip(gradients, trainable_variables)]

        optimizer.apply_gradients(zip(gradients, trainable_variables))
        
        train_loss.update_state(total_loss)
        train_accuracy.update_state(labels, pred)

        return optimizer._decayed_lr(var_dtype = tf.float32)
        
    @tf.function(experimental_compile = COMPILE_MODE)
    def validation(images, labels):
        pred = model(images, training = False)
        loss = loss_object(labels, pred)
        
        test_loss.update_state(loss)
        test_accuracy.update_state(labels, pred)
        return pred
    return training, train_loss, train_accuracy, validation, test_loss, test_accuracy, optimizer
