import tensorflow as tf
import numpy as np
import scipy.io as sio
import os, time, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dataloader import Dataloader
import op_util
from nets import WResNet
from nets import Multiple


home_path = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description='Variational_Information_Distillation Implementation via TF2.0 low-level coding')

parser.add_argument("--train_dir", default="test", type=str)
parser.add_argument("--architecture", default=[16,4], nargs='+', type=int)
parser.add_argument("--Distillation", default="None", type=str,
                    help = 'Distillation method : Soft_logits, FitNet, AT, FSP, DML, KD-SVD, FT, AB, RKD')
parser.add_argument("--trained_param", default="None", type=str)
parser.add_argument("--dataset", default="cifar100", type=str)
args = parser.parse_args()

if __name__ == '__main__':
    ### define path and hyper-parameter
    Learning_rate =1e-1
    batch_size = 128
    val_batch_size = 200
    train_epoch = 100
    weight_decay = 5e-4
    
    should_log = 100
    gpu_num = 0
    
    tf.debugging.set_log_device_placement(False)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_visible_devices(gpus[gpu_num], 'GPU')
    
    summary_writer = tf.summary.create_file_writer(args.train_dir)
    
    train_images, train_labels, val_images, val_labels, pre_processing = Dataloader(args.dataset, '')
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(100*batch_size).batch(batch_size)
    train_ds = train_ds.map(pre_processing(is_training = True))
    test_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(val_batch_size)
    test_ds = test_ds.map(pre_processing(is_training = False))
    
    student_model = WResNet.Model(architecture=args.architecture, weight_decay = weight_decay, num_class = np.max(train_labels)+1,
                                  name = 'Student', trainable = True)
    student_model(np.zeros([1]+list(train_images.shape[1:]), dtype=np.float32), training = False)
    
    if args.Distillation != 'None':
        teacher_model = WResNet.Model(architecture=[40,4], weight_decay = 0., num_class = np.max(train_labels)+1,
                                      name = 'Teacher', trainable = False)
        teacher_model(np.zeros([1]+list(train_images.shape[1:]), dtype=np.float32), training = False)
    
    if args.Distillation in {'AB','FitNet'}: ## Initialize via KD
        init_epoch = 20
        if args.Distillation == 'AB':
            distill_model = Multiple.AB(student_model, teacher_model, weight_decay = weight_decay)
        elif args.Distillation == 'FitNet':
            distill_model = Multiple.FitNet(student_model, teacher_model, weight_decay = weight_decay)
            
        dist_step, dist_loss = op_util.Initializer_Optimizer(student_model, teacher_model, distill_model, Learning_rate)
        train_step, train_loss, train_accuracy,\
        test_step,  test_loss,  test_accuracy = op_util.Optimizer(student_model, Learning_rate)
    elif args.Distillation in {'VID', 'AT'}: ## take constraint via KD
        init_epoch = 0
        if args.Distillation == 'AT':
            distill_model = Multiple.AT(student_model, teacher_model)
        elif args.Distillation == 'VID':
            distill_model = Multiple.VID(student_model, teacher_model, weight_decay = weight_decay)
        
        train_step, train_loss, train_accuracy,\
        test_step,  test_loss,  test_accuracy = op_util.Multitask_Optimizer(student_model, teacher_model, distill_model, Learning_rate)
    else:
        init_epoch = 0
        train_step, train_loss, train_accuracy,\
        test_step,  test_loss,  test_accuracy = op_util.Optimizer(student_model, Learning_rate)
    
    LR_scheduler = op_util.learning_rate_scheduler(Learning_rate, train_epoch, decay_points = [0.3, 0.6, 0.8], decay_rate = 2e-1)
   
    with summary_writer.as_default():
        step = 0
        logs = {'training_acc' : [], 'validation_acc' : []}
        student_name = student_model.variables[0].name.split('/')[0]
        
        if args.Distillation != 'None':
            teacher_name = teacher_model.variables[0].name.split('/')[0]
            trained = sio.loadmat(args.trained_param)
            n = 0
            for v in teacher_model.variables + teacher_model.non_trainable_variables:
                v.assign(trained[v.name[len(teacher_name)+1:]])
                n += 1
            print (n, ' : teacher params loaded')
        
        for epoch in range(init_epoch):
            train_time = time.time()
            for images, labels in train_ds:
                dist_step(images)
                step += 1
                if step % should_log == 0:
                    template = 'Global step {0:5d}: loss = {1:0.4f} ({2:1.3f} sec/step)'
                    print (template.format(step, dist_loss.result(), (time.time()-train_time)/should_log))
                    train_time = time.time()
            tf.summary.scalar('Initializer_loss/train', dist_loss.result(), step=epoch+1)
            dist_loss.reset_states()
            
        for epoch in range(train_epoch):
            lr = LR_scheduler(epoch)
            train_time = time.time()
            for images, labels in train_ds:
                train_step(images, labels, lr)
                step += 1
                if step % should_log == 0:
                    template = 'Global step {0:5d}: loss = {1:0.4f} ({2:1.3f} sec/step)'
                    print (template.format(step, train_loss.result(), (time.time()-train_time)/should_log))
                    train_time = time.time()
            tf.summary.scalar('Categorical_loss/train', train_loss.result(), step=epoch+1)
            tf.summary.scalar('Accuracy/train', train_accuracy.result()*100, step=epoch+1)
            tf.summary.scalar('learning_rate', lr, step=epoch)
                
            for test_images, test_labels in test_ds:
                test_step(test_images, test_labels)
            tf.summary.scalar('Categorical_loss/test', test_loss.result(), step=epoch+1)
            tf.summary.scalar('Accuracy/test', test_accuracy.result()*100, step=epoch+1)
                
            template = 'Epoch: {0:3d}, train_loss: {1:0.4f}, train_Acc.: {2:2.2f}, val loss: {3:0.4f}, val_Acc.: {4:2.2f}'
            print (template.format(epoch+1, train_loss.result(), train_accuracy.result()*100,
                                             test_loss.result(),  test_accuracy.result()*100))
            
            logs['training_acc'].append(train_accuracy.result()*100)
            logs['validation_acc'].append(test_accuracy.result()*100)
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()
            
        params = {}
        for v in student_model.variables:
            params[v.name[len(student_name)+1:]] = v.numpy()
        sio.savemat(args.train_dir+'/trained_params.mat', params)
        sio.savemat(args.train_dir + '/log.mat',logs)

