import tensorflow as tf
import numpy as np
import scipy.io as sio
import glob, os, time, argparse, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dataloader import Dataloader
import op_util
from nets import WResNet
from math import ceil
import shutil

import importlib
import distiller

home_path = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description='')

parser.add_argument("--train_path", default="test", type=str)
parser.add_argument("--arch", default='WResNet-16-4', type=str)
parser.add_argument("--dataset", default="cifar100", type=str)
parser.add_argument("--data_path", type=str)

parser.add_argument("--learning_rate", default = 0.1, type=float)
parser.add_argument("--decay_points", default = [.3, .6, .8], type=float, nargs = '+')
parser.add_argument("--decay_rate", default=.2, type=float)
parser.add_argument("--weight_decay", default=5e-4, type=float)

parser.add_argument("--batch_size", default = 128, type=int)
parser.add_argument("--val_batch_size", default=200, type=int)
parser.add_argument("--train_epoch", default=100, type=int)

parser.add_argument("--Knowledge", type=str)
parser.add_argument("--teacher_arch", default='ResNet-56', type=str)
parser.add_argument("--trained_param", type=str)

parser.add_argument("--gpu_id", default=0, type=int)
parser.add_argument("--do_log", default=200, type=int)
args = parser.parse_args()

def validation(test_step, test_ds, test_loss, test_accuracy,
               train_loss = None, train_accuracy = None, epoch = None, lr = None, logs = None, bn_statistics_update = False):
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    tf.summary.scalar('Categorical_loss/train', train_loss.result(), step=epoch+1)
    tf.summary.scalar('Categorical_loss/test', test_loss.result(), step=epoch+1)
    tf.summary.scalar('Accuracy/train', train_accuracy.result()*100, step=epoch+1)
    tf.summary.scalar('Accuracy/test', test_accuracy.result()*100, step=epoch+1)
    tf.summary.scalar('learning_rate', lr, step=epoch)

    template = 'Epoch: {0:3d}, train_loss: {1:0.4f}, train_Acc.: {2:2.2f}, val_loss: {3:0.4f}, val_Acc.: {4:2.2f}'
    print (template.format(epoch+1, train_loss.result(), train_accuracy.result()*100,
                                     test_loss.result(),  test_accuracy.result()*100))

    logs['training_acc'].append(train_accuracy.result()*100)
    logs['validation_acc'].append(test_accuracy.result()*100)

    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()

def build_dataset_proviers(train_images, train_labels, test_images, test_labels, pre_processing):
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_ds = test_ds.map(pre_processing(is_training = False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(args.val_batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).cache()
    train_ds = train_ds.map(pre_processing(is_training = True),  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.shuffle(100*args.batch_size).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return {'train': train_ds, 'test': test_ds}

def load_model(arch, num_class):
    if 'WResNet' in arch:
        arch = [int(a) for a in arch.split('-')[1:]]
        model = WResNet.Model(architecture=arch, num_class = num_class, name = 'WResNet', trainable = True)
    return model

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[args.gpu_id], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu_id], True)
    
    train_images, train_labels, test_images, test_labels, pre_processing = Dataloader(args.dataset, args.data_path)
    datasets = build_dataset_proviers(train_images, train_labels, test_images, test_labels, pre_processing)
    args.input_shape = list(train_images.shape[1:])
    
    model = load_model(args.arch, np.max(test_labels) + 1)
    model(np.zeros([1]+args.input_shape, dtype=np.float32), training = False)

    if args.Knowledge is not None:
        teacher = load_model(args.teacher_arch, np.max(test_labels) + 1)
        teacher(np.zeros([1]+args.input_shape, dtype=np.float32), training = False)

        model_name = teacher.variables[0].name.split('/')[0]
        trained = sio.loadmat(args.trained_param)
        n = 0
        for v in teacher.variables:
            if model_name in v.name:
                v.assign(trained[v.name[len(model_name)+1:]])
                n += 1
        print (n, 'params loaded')

        Knowledge = importlib.import_module('distiller.' + args.Knowledge)
        model.distiller = Knowledge.distill(args, model, teacher)
        model(np.zeros([1]+args.input_shape, dtype=np.float32), training = False)
    
    train_step, train_loss, train_accuracy,\
    test_step,  test_loss,  test_accuracy, optimizer = op_util.Optimizer(model, args.weight_decay, args.learning_rate)

    args.decay_points = [int(dp*args.train_epoch) if dp < 1 else int(dp) for dp in args.decay_points]
    def scheduler(epoch):
        lr = args.learning_rate
        for dp in args.decay_points:
            if epoch >= dp:
                lr *= args.decay_rate
        return lr

    summary_writer = tf.summary.create_file_writer(args.train_path, flush_millis = 30000)
    with summary_writer.as_default():
        step = 0
        logs = {'training_acc' : [], 'validation_acc' : []}

        model_name = model.name.split('/')[0]
        train_time = time.time()
        init_epoch = 0

        try: 
            os.mkdir(os.path.join(args.train_path,'codes'))
        except:
            pass
            
        with open(os.path.join(args.train_path, 'arguments.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        if hasattr(model.distiller, 'auxiliary_training'):
            model.distiller.auxiliary_training(datasets['train'])

        if hasattr(model.distiller, 'initialize_student'):
            model.distiller.initialize_student(datasets['train'])
            del model.aux_layers

        ## Conventional training routine
        train_time = time.time()
        for epoch in range(init_epoch, init_epoch + args.train_epoch):
            optimizer.learning_rate = scheduler(epoch)

            for images, labels in datasets['train']:
                lr = train_step(images, labels)
                step += 1
                if step % args.do_log == 0:
                    template = 'Global step {0:5d}: loss = {1:0.4f} ({2:1.3f} sec/step)'
                    print (template.format(step, train_loss.result(), (time.time()-train_time)/args.do_log))
                    train_time = time.time()

            val_time = time.time()
            validation(test_step, datasets['test'], test_loss, test_accuracy,
                       train_loss, train_accuracy, epoch = epoch, lr = lr, logs = logs, bn_statistics_update = False)
            train_time += time.time() - val_time
            summary_writer.flush()

        params = {}
        for v in model.variables:
            if model_name in v.name:
                params[v.name[len(model_name)+1:]] = v.numpy()

        sio.savemat(args.train_path+'/trained_params.mat', params)
        sio.savemat(args.train_path + '/log.mat',logs)
