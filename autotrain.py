import subprocess
import tensorflow as tf
import glob, os
import scipy.io as sio
import numpy as np

def get_avg_plot(base_path):
    summary_writer = tf.summary.create_file_writer(base_path)
    with summary_writer.as_default():
        pathes = glob.glob(base_path[:-len(base_path.split('/')[-1])] + '*')
        training_acc   = []
        validation_acc = []
        for path in pathes:
            logs = sio.loadmat(path + '/log.mat')
            training_acc.append(logs['training_acc'])
            validation_acc.append(logs['validation_acc'])
        training_acc   = np.mean(np.vstack(training_acc),0)
        validation_acc = np.mean(np.vstack(validation_acc),0)
    
        for i, (train_acc, val_acc) in enumerate(zip(training_acc,validation_acc)):
            tf.summary.scalar('Accuracy/train', train_acc, step=i+1)
            tf.summary.scalar('Accuracy/test', val_acc, step=i+1)
            

conf = 0
home_path = os.path.dirname(os.path.abspath(__file__))
#Soft_logits, FitNet, AT, FSP, DML, KD-SVD, FT, AB, RKD
if conf == 0:
    for d in ['FitNet']:
        base_path = '/home/cvip/Documents/VID2.0/%s/%s'%(d,d)
        for i in range(3):
            subprocess.call('python %s/train_w_distill.py '%home_path
                           +' --train_dir %s%d'%(base_path,i)
#                           +' --architecture 40 4'
                           +' --dataset cifar100'
                           +' --Distillation %s'%d
                           +' --trained_param %s'%(home_path+'/pre_trained/WResNet40-4.mat'), 
                            shell=True)
            print ('Training Done')
        get_avg_plot(base_path)
        
