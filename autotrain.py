import subprocess
import tensorflow as tf
import glob, os, argparse
import scipy.io as sio
import numpy as np

def get_avg_plot(base_path):
    pathes = glob.glob(base_path[:-len(base_path.split('/')[-1])] + '*')
    summary_writer = tf.summary.create_file_writer(base_path[:-len(base_path.split('/')[-1])]+'/average')
    with summary_writer.as_default():
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

parser = argparse.ArgumentParser()
parser.add_argument("--Distillation", default="Student", nargs = '+', type=str,
                    help = 'Distillation method : Soft_logits, FitNet, AT, FSP, DML, KD-SVD, FT, AB, RKD')
args = parser.parse_args()

if __name__ == '__main__':            
    conf = 0
    home_path = os.path.dirname(os.path.abspath(__file__))

    if conf == 0:
    #    for d in ['MHGD']:
        for d in args.Distillation:
            arc = [16, 4] if d != 'Teacher' else [40,4]
            base_path = '/home/cvip/Documents/TF_bench/%s/%s'%(d,d)
            for i in range(3):
                subprocess.call('python %s/train_w_distill.py '%home_path
                               +' --train_dir %s%d'%(base_path,i)
                               +' --architecture %d %d'%(arc[0], arc[1])
                               +' --dataset cifar100'
                               +' --Distillation %s'%d
                               +' --trained_param %s'%(home_path+'/pre_trained/WResNet40-4.mat'), 
                                shell=True)
                print ('Training Done')
            get_avg_plot(base_path)
        
