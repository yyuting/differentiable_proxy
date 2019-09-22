from __future__ import division
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import numpy
import numpy.random
import random
import argparse_util
import pickle
from tensorflow.python.client import timeline
import copy
from unet import unet
import importlib
import importlib.util
import subprocess
import shutil
from local_laplacian_tf import local_laplacian_tf
from local_laplacian import local_laplacian
import bayesian_optimization
import skimage.io
import scipy
import scipy.optimize

no_L1_reg_other_layers = True

dtype = tf.float32

batch_norm_is_training = True

allow_nonzero = False

identity_output_layer = True

less_aggresive_ini = False

allowed_img_ext = ['.jpg', '.png']

def lrelu(x):
    return tf.maximum(x*0.2,x)

def identity_initializer(in_channels=[], allow_map_to_less=False):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if not allow_nonzero:
            print('initializing all zero')
            array = np.zeros(shape, dtype=float)
        else:
            x = np.sqrt(6.0 / (shape[2] + shape[3])) / 1.5
            array = numpy.random.uniform(-x, x, size=shape)
            print('initializing xavier')
            #return tf.constant(array, dtype=dtype)
        cx, cy = shape[0]//2, shape[1]//2
        if len(in_channels) > 0:
            input_inds = in_channels
            output_inds = range(len(in_channels))
            #for k in range(len(in_channels)):
            #    array[cx, cy, in_channels[k], k] = 1
        elif allow_map_to_less:
            input_inds = range(min(shape[2], shape[3]))
            output_inds = input_inds
            #for i in range(min(shape[2], shape[3])):
            #    array[cx, cy, i, i] = 1
        else:
            input_inds = range(shape[2])
            output_inds = input_inds
            #for i in range(shape[2]):
            #    array[cx, cy, i, i] = 1
        for i in range(len(input_inds)):
            if less_aggresive_ini:
                array[cx, cy, input_inds[i], output_inds[i]] *= 10.0
            else:
                array[cx, cy, input_inds[i], output_inds[i]] = 1.0
        return tf.constant(array, dtype=dtype)
    return _initializer

batch_norm_only = False

def adaptive_nm(x):
    if not batch_norm_only:
        w0=tf.Variable(1.0,name='w0')
        w1=tf.Variable(0.0,name='w1')
        return w0*x+w1*slim.batch_norm(x, is_training=batch_norm_is_training) # the parameter "is_training" in slim.batch_norm does not seem to help so I do not use it
    else:
        return slim.batch_norm(x, is_training=batch_norm_is_training)

nm = adaptive_nm

conv_channel = 24
actual_conv_channel = conv_channel

dilation_remove_large = False
dilation_clamp_large = False
dilation_remove_layer = False
dilation_threshold = 8

def build(input, ini_id=True, regularizer_scale=0.0, final_layer_channels=-1, identity_initialize=False, grayscale=False):
    regularizer = None
    if not no_L1_reg_other_layers and regularizer_scale > 0.0:
        regularizer = slim.l1_regularizer(regularizer_scale)
    if ini_id or identity_initialize:
        net=slim.conv2d(input,actual_conv_channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(allow_map_to_less=True),scope='g_conv1',weights_regularizer=regularizer)
    else:
        net=slim.conv2d(input,actual_conv_channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,scope='g_conv1',weights_regularizer=regularizer)

    dilation_schedule = [2, 4, 8, 16, 32, 64]
    for ind in range(len(dilation_schedule)):
        dilation_rate = dilation_schedule[ind]
        conv_ind = ind + 2
        if dilation_rate > dilation_threshold:
            if dilation_remove_large:
                dilation_rate = 1
            elif dilation_clamp_large:
                dilation_rate = dilation_threshold
            elif dilation_remove_layer:
                continue
        print('rate is', dilation_rate)
        net=slim.conv2d(net,actual_conv_channel,[3,3],rate=dilation_rate,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv'+str(conv_ind),weights_regularizer=regularizer)
#    net=slim.conv2d(net,24,[3,3],rate=128,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv8')

    net=slim.conv2d(net,actual_conv_channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv9',weights_regularizer=regularizer)
    if final_layer_channels > 0:
        if actual_conv_channel > final_layer_channels and (not identity_initialize):
            net = slim.conv2d(net, final_layer_channels, [1, 1], rate=1, activation_fn=lrelu, normalizer_fn=nm, scope='final_0', weights_regularizer=regularizer)
            nlayers = [1, 2]
        else:
            nlayers = [0, 1, 2]
        for nlayer in nlayers:
            net = slim.conv2d(net, final_layer_channels, [1, 1], rate=1, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(allow_map_to_less=True), scope='final_'+str(nlayer),weights_regularizer=regularizer)

    print('identity last layer?', identity_initialize and identity_output_layer)
    if not grayscale:
        out_ch = 3
    else:
        out_ch = 1
    net=slim.conv2d(net,out_ch,[1,1],rate=1,activation_fn=None,scope='g_conv_last',weights_regularizer=regularizer, weights_initializer=identity_initializer(allow_map_to_less=True) if (identity_initialize and identity_output_layer) else tf.contrib.layers.xavier_initializer())
    return net

def prepare_data_root(dataroot, gradient_loss=False):
    input_names=[]
    output_names=[]
    val_names=[]
    val_img_names=[]
    map_names = []
    val_map_names = []
    grad_names = []
    val_grad_names = []
    validate_names = []
    validate_img_names = []

    train_input_dir = os.path.join(dataroot, 'train_label')
    test_input_dir = os.path.join(dataroot, 'test_label')
    train_output_dir = os.path.join(dataroot, 'train_img')
    test_output_dir = os.path.join(dataroot, 'test_img')
    validate_input_dir = os.path.join(dataroot, 'validate_label')
    validate_output_dir = os.path.join(dataroot, 'validate_img')

    for (dir, names) in [(train_input_dir, input_names),
                         (train_output_dir, output_names),
                         (test_input_dir, val_names),
                         (test_output_dir, val_img_names),
                         (validate_input_dir, validate_names),
                         (validate_output_dir, validate_img_names)]:
        for file in sorted(os.listdir(dir)):
            _, ext = os.path.splitext(file)
            if ext in allowed_img_ext:
                names.append(os.path.join(dir, file))

    if gradient_loss:
        train_grad_dir = os.path.join(dataroot, 'train_grad')
        test_grad_dir = os.path.join(dataroot, 'test_grad')
        for file in sorted(os.listdir(train_grad_dir)):
            grad_names.append(os.path.join(train_grad_dir, file))
        for file in sorted(os.listdir(test_grad_dir)):
            val_grad_names.append(os.path.join(test_grad_dir, file))

    return input_names, output_names, val_names, val_img_names, validate_names, validate_img_names, map_names, val_map_names, grad_names, val_grad_names


os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
os.system('rm tmp')

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse_util.ArgumentParser(description='FastImageProcessing')
    parser.add_argument('--name', dest='name', default='', help='name of task')
    parser.add_argument('--dataroot', dest='dataroot', default='../data', help='directory to store training and testing data')
    parser.add_argument('--is_npy', dest='is_npy', action='store_true', help='whether input is npy format')
    parser.add_argument('--is_train', dest='is_train', action='store_true', help='state whether this is training or testing')
    parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='number of channels for input')
    parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='number of epochs to train, seperated by comma')
    parser.add_argument('--no_preload', dest='preload', action='store_false', help='whether to preload data')
    parser.add_argument('--is_bin', dest='is_bin', action='store_true', help='whether input is stored in bin files')
    parser.add_argument('--L1_regularizer_scale', dest='regularizer_scale', type=float, default=0.0, help='scale for L1 regularizer')
    parser.add_argument('--L2_regularizer_scale', dest='L2_regularizer_scale', type=float, default=0.0, help='scale for L2 regularizer')
    parser.add_argument('--test_training', dest='test_training', action='store_true', help='use training data for testing purpose')
    parser.add_argument('--which_epoch', dest='which_epoch', type=int, default=0, help='decide which epoch to read the checkpoint')
    parser.add_argument('--generate_timeline', dest='generate_timeline', action='store_true', help='generate timeline files')
    parser.add_argument('--feature_reduction', dest='encourage_sparse_features', action='store_true', help='if true, encourage selecting sparse number of features')
    parser.add_argument('--collect_validate_loss', dest='collect_validate_loss', action='store_true', help='if true, collect validation loss (and training score) and write to tensorboard')
    parser.add_argument('--collect_validate_while_training', dest='collect_validate_while_training', action='store_true', help='if true, collect validation loss while training')
    parser.add_argument('--no_normalize', dest='normalize_weights', action='store_false', help='if specified, does not normalize weight on feature selection layer')
    parser.add_argument('--abs_normalize', dest='abs_normalize', action='store_true', help='when specified, use sum of abs values as normalization')
    parser.add_argument('--rowwise_L2_normalize', dest='rowwise_L2_normalize', action='store_true', help='when specified, normalize feature selection matrix by divide row-wise L2 norm sum, then regularize the resulting matrix with L1')
    parser.add_argument('--Frobenius_normalize', dest='Frobenius_normalize', action='store_true', help='when specified, use Frobenius norm to normalize feature selecton matrix, followed by L1 regularization')
    parser.add_argument('--add_initial_layers', dest='add_initial_layers', action='store_true', help='add initial conv layers without dilation')
    parser.add_argument('--initial_layer_channels', dest='initial_layer_channels', type=int, default=-1, help='number of channels in initial layers')
    parser.add_argument('--add_final_layers', dest='add_final_layers', action='store_true', help='add final conv layers without dilation')
    parser.add_argument('--final_layer_channels', dest='final_layer_channels', type=int, default=-1, help='number of channels in final layers')
    parser.add_argument('--dilation_remove_large', dest='dilation_remove_large', action='store_true', help='when specified, use ordinary conv layer instead of dilated conv layer with large dilation rate')
    parser.add_argument('--dilation_clamp_large', dest='dilation_clamp_large', action='store_true', help='when specified, clamp large dilation rate to a give threshold')
    parser.add_argument('--dilation_threshold', dest='dilation_threshold', type=int, default=8, help='threshold used to remove or clamp dilation')
    parser.add_argument('--dilation_remove_layer', dest='dilation_remove_layer', action='store_true', help='when specified, use less dilated conv layers')
    parser.add_argument('--update_bn', dest='update_bn', action='store_true', help='accurately update batch normalization')
    parser.add_argument('--conv_channel_no', dest='conv_channel_no', type=int, default=-1, help='directly specify number of channels for dilated conv layers')
    parser.add_argument('--unet', dest='unet', action='store_true', help='if specified, use unet instead of dilated conv network')
    parser.add_argument('--unet_base_channel', dest='unet_base_channel', type=int, default=32, help='base channel (1st conv layer channel) for unet')
    parser.add_argument('--batch_norm_only', dest='batch_norm_only', action='store_true', help='if specified, use batch norm only (no adaptive normalization)')
    parser.add_argument('--no_batch_norm', dest='batch_norm', action='store_false', help='if specified, do not apply batch norm')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.0001, help='learning rate for adam optimizer')
    parser.add_argument('--identity_initialize', dest='identity_initialize', action='store_true', help='if specified, initialize weights such that output is 1 sample RGB')
    parser.add_argument('--nonzero_ini', dest='allow_nonzero', action='store_true', help='if specified, use xavier for all those supposed to be 0 entries in identity_initializer')
    parser.add_argument('--no_identity_output_layer', dest='identity_output_layer', action='store_false', help='if specified, do not use identity mapping for output layer')
    parser.add_argument('--less_aggresive_ini', dest='less_aggresive_ini', action='store_true', help='if specified, use a less aggresive way to initialize RGB weights (multiples of the original xavier weights)')
    parser.add_argument('--perceptual_loss', dest='perceptual_loss', action='store_true',help='if specified, use perceptual loss as well as L2 loss')
    parser.add_argument('--perceptual_loss_term', dest='perceptual_loss_term', default='conv1_1', help='specify to use which layer in vgg16 as perceptual loss')
    parser.add_argument('--perceptual_loss_scale', dest='perceptual_loss_scale', type=float, default=0.0001, help='used to scale perceptual loss')
    parser.add_argument('--gradient_loss', dest='gradient_loss', action='store_true', help='if specified, also use gradient at canny edge regions as a loss term')
    parser.add_argument('--normalize_grad', dest='normalize_grad', action='store_true', help='if specified, use normalized gradient as loss')
    parser.add_argument('--grayscale_grad', dest='grayscale_grad', action='store_true', help='if specified, use grayscale gradient as loss')
    parser.add_argument('--cos_sim', dest='cos_sim', action='store_true', help='use cosine similarity to compute gradient loss')
    parser.add_argument('--gradient_loss_scale', dest='gradient_loss_scale', type=float, default=1.0, help='scale multiplied to gradient loss')
    parser.add_argument('--gradient_loss_all_pix', dest='gradient_loss_all_pix', action='store_true', help='if specified, use all pixels to calculate gradient loss')
    parser.add_argument('--train_res', dest='train_res', action='store_true', help='if specified, out_img = in_noisy_img + out_network')
    parser.add_argument('--RGB_norm', dest='RGB_norm', type=int, default=2, help='specify which p-norm to use for RGB loss')
    parser.add_argument('--force_training_mode_inference', dest='force_training_mode', action='store_true', help='if specified, force use training mode for batch normalization inference')
    parser.add_argument('--optimize_hyperparameter', dest='optimize_hyperparameter', action='store_true', help='if specified, instead of training the network, use trained network to optimize input parameters')
    parser.add_argument('--input_img', dest='input_img', default='', help='when optimize hyperparameters, name of input image')
    parser.add_argument('--output_img', dest='output_img', default='', help='when optimize hyperparameters, name of output image')
    parser.add_argument('--optimizer', dest='optimizer', default='adam', help='specify which optimizer to use during hyperparameter optimization')
    parser.add_argument('--use_orig_program', dest='use_orig_program', action='store_true', help='if specified, use the original program instead of builidng a proxy')
    parser.add_argument('--optimize_prefix', dest='optimize_prefix', default='', help='a unique prefix use to store optimized result')
    parser.add_argument('--specific_par', dest='specific_par', default='', help='use specified parameter and input_img for inference')
    parser.add_argument('--inference_loss', dest='inference_loss', default='', help='for hyperparameter optimization, stores the proxy inference loss when using the ground truth parameters')
    parser.add_argument('--test_dirname', dest='test_dirname', default='', help='if specified, use this name for testing directory')
    parser.add_argument('--no_input_img', dest='use_input_img', action='store_false', help='if specified, input only parameters')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='if specified, output grayscale image instead of color image')
    parser.add_argument('--multi_level_loss', dest='multi_level_loss', action='store_true', help='if specified, use a multiple level loss')
    parser.add_argument('--orig_program_name', dest='orig_program_name', default='local_laplacian_tf', help='specify the original program name for optimization')
    parser.add_argument('--fc', dest='fc', action='store_true', help='if specified, use fully connected network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='batch size used for training/testing.')
    parser.add_argument('--direct_scipy_optimize', dest='direct_scipy_optimize', action='store_true', help='if specified, use scipy optimization directly, (not using scipy wrapper in tensorflow) for comparable timing')

    parser.set_defaults(is_npy=False)
    parser.set_defaults(is_train=False)
    parser.set_defaults(preload=True)
    parser.set_defaults(is_bin=False)
    parser.set_defaults(test_training=False)
    parser.set_defaults(generate_timeline=False)
    parser.set_defaults(encourage_sparse_features=False)
    parser.set_defaults(collect_validate_loss=False)
    parser.set_defaults(collect_validate_while_training=False)
    parser.set_defaults(normalize_weights=True)
    parser.set_defaults(abs_normalize=False)
    parser.set_defaults(rowwise_L2_normalize=False)
    parser.set_defaults(Frobenius_normalize=False)
    parser.set_defaults(add_initial_layers=False)
    parser.set_defaults(add_final_layers=False)
    parser.set_defaults(dilation_remove_large=False)
    parser.set_defaults(dilation_clamp_large=False)
    parser.set_defaults(dilation_remove_layer=False)
    parser.set_defaults(update_bn=False)
    parser.set_defaults(unet=False)
    parser.set_defaults(batch_norm_only=False)
    parser.set_defaults(batch_norm=True)
    parser.set_defaults(identity_initialize=False)
    parser.set_defaults(allow_nonzero=False)
    parser.set_defaults(identity_output_layer=True)
    parser.set_defaults(less_aggresive_ini=False)
    parser.set_defaults(perceptual_loss=False)
    parser.set_defaults(gradient_loss=False)
    parser.set_defaults(normalize_grad=False)
    parser.set_defaults(grayscale_grad=False)
    parser.set_defaults(cos_sim=False)
    parser.set_defaults(gradient_loss_all_pix=False)
    parser.set_defaults(train_res=False)
    parser.set_defaults(intersection=True)
    parser.set_defaults(force_training_mode=False)
    parser.set_defaults(optimize_hyperparameter=False)
    parser.set_defaults(use_orig_program=False)
    parser.set_defaults(fc=False)
    parser.set_defaults(direct_scipy_optimize=False)

    args = parser.parse_args()

    main_network(args)

def copy_option(args):
    new_args = copy.copy(args)
    delattr(new_args, 'is_train')
    delattr(new_args, 'dataroot')
    delattr(new_args, 'test_training')
    delattr(new_args, 'which_epoch')
    delattr(new_args, 'generate_timeline')
    delattr(new_args, 'collect_validate_loss')
    delattr(new_args, 'collect_validate_while_training')
    delattr(new_args, 'preload')
    delattr(new_args, 'force_training_mode')
    delattr(new_args, 'optimize_hyperparameter')
    delattr(new_args, 'input_img')
    delattr(new_args, 'output_img')
    delattr(new_args, 'optimizer')
    delattr(new_args, 'use_orig_program')
    delattr(new_args, 'optimize_prefix')
    delattr(new_args, 'specific_par')
    delattr(new_args, 'inference_loss')
    delattr(new_args, 'test_dirname')
    delattr(new_args, 'orig_program_name')
    delattr(new_args, 'batch_size')
    delattr(new_args, 'direct_scipy_optimize')
    return new_args

def main_network(args):

    if args.is_bin:
        assert not args.preload

    if args.name == '':
        args.name = ''.join(random.choice(string.digits) for _ in range(5))

    if not os.path.isdir(args.name):
        os.makedirs(args.name)

    option_file = os.path.join(args.name, 'option.txt')
    option_copy = copy_option(args)
    if os.path.exists(option_file):
        option_str = open(option_file).read()
        print(str(option_copy))
        assert option_str == str(option_copy)
    else:
        open(option_file, 'w').write(str(option_copy))

    nfeatures = args.input_nc

    global batch_norm_is_training
    batch_norm_is_training = args.force_training_mode or args.is_train

    global batch_norm_only
    batch_norm_only = args.batch_norm_only

    global actual_conv_channel
    actual_conv_channel = args.conv_channel_no
    if args.initial_layer_channels < 0:
        args.initial_layer_channels = actual_conv_channel
    if args.final_layer_channels < 0:
        args.final_layer_channels = actual_conv_channel

    global dilation_threshold
    dilation_threshold = args.dilation_threshold
    assert (not args.dilation_clamp_large) or (not args.dilation_remove_large) or (not args.dilation_remove_layer)
    global dilation_clamp_large
    dilation_clamp_large = args.dilation_clamp_large
    global dilation_remove_large
    dilation_remove_large = args.dilation_remove_large
    global dilation_remove_layer
    dilation_remove_layer = args.dilation_remove_layer

    if not args.add_initial_layers:
        args.initial_layer_channels = -1
    if not args.add_final_layers:
        args.final_layer_channels = -1

    if not args.batch_norm:
        global nm
        nm = None
        args.update_bn = False

    global allow_nonzero
    allow_nonzero = args.allow_nonzero

    global identity_output_layer
    identity_output_layer = args.identity_output_layer

    global less_aggresive_ini
    less_aggresive_ini = args.less_aggresive_ini

    if not args.fc:
        input_names, output_names, val_names, val_img_names, validate_names, validate_img_names, map_names, val_map_names, grad_names, val_grad_names = prepare_data_root(args.dataroot, gradient_loss=args.gradient_loss)
        if args.test_training:
            val_names = input_names
            val_img_names = output_names
            val_map_names = map_names
            val_grad_names = grad_names
        assert args.batch_size == 1
    else:
        train_label = numpy.load(os.path.join(args.dataroot, 'train_label.npy'))
        train_val = numpy.load(os.path.join(args.dataroot, 'train_val.npy'))
        test_label = numpy.load(os.path.join(args.dataroot, 'test_label.npy'))
        test_val = numpy.load(os.path.join(args.dataroot, 'test_val.npy'))

    def read_ind(img_arr, name_arr, id, is_npy):
        img_arr[id] = read_name(name_arr[id], is_npy)
        if img_arr[id] is None:
            return False
        elif img_arr[id].shape[0] * img_arr[id].shape[1] > 2200000:
            img_arr[id] = None
            return False
        return True

    def read_name(name, is_npy, is_bin=False):
        if not os.path.exists(name):
            return None
        if not is_npy and not is_bin:
            arr = np.float32(cv2.imread(name, -1)) / 255.0
            if len(arr.shape) < 3:
                arr = numpy.expand_dims(arr, axis=2)
            return arr
        elif is_npy:
            ans = np.load(name)
            return ans
        else:
            return np.fromfile(name, dtype=np.float32).reshape([640, 960, args.input_nc])

    read_data_from_file = True

    train_from_queue = False

    niters = 150

    if not args.fc:
        if args.grayscale:
            img_ch = 1
        else:
            img_ch = 3
        input=tf.placeholder(tf.float32,shape=[None,None,None,img_ch])
        output=tf.placeholder(tf.float32,shape=[None,None,None,img_ch])
        #if args.optimize_hyperparameter and (args.optimizer not in ['nelder-mead', 'powell']):
        #if args.optimize_hyperparameter and (args.optimizer in ['adam', 'gradient', 'adagrad', 'proximal']):
        if args.optimize_hyperparameter and (not args.direct_scipy_optimize):
            ini_parameters = numpy.random.rand(args.input_nc)
            # a hack because eps is not correctly scaled during training
            if args.orig_program_name == 'local_laplacian_tf':
                ini_parameters[2] *= 0.1
            input_parameters = tf.Variable(ini_parameters, dtype=tf.float32, name='input_parameters')
        else:
            input_parameters = tf.placeholder(tf.float32, shape=args.input_nc)
        filled = []
        for i in range(args.input_nc):
            filled.append(tf.expand_dims(tf.fill(tf.shape(output)[:3], input_parameters[i]), axis=3))
        if args.use_input_img:
            input_to_network = tf.concat([input]+filled, axis=3)
        else:
            input_to_network = tf.concat(filled, axis=3)
    else:
        if not args.optimize_hyperparameter:
            input = tf.placeholder(tf.float32, [None, args.input_nc])
        else:
            #if args.optimizer in ['adam', 'gradient', 'adagrad', 'proximal']:
            if not args.direct_scipy_optimize:
                ini_parameters = numpy.random.rand(args.input_nc)
                input_parameters = tf.Variable(ini_parameters, dtype=tf.float32, name='input_parameters')
            else:
                input_parameters = tf.placeholder(tf.float32, shape=args.input_nc)
            input = tf.expand_dims(input_parameters, axis=0)

        output = tf.placeholder(tf.float32, [None, 1])
        input_to_network = input

    with tf.control_dependencies([input_to_network]):
        if args.use_orig_program:
            if args.orig_program_name == 'local_laplacian_tf':
                network = local_laplacian_tf(input, input_parameters[0], input_parameters[1], input_parameters[2])
            elif args.orig_program_name == 'local_laplacian_categorical':
                from local_laplacian_categorical import local_laplacian_categorical
                network = local_laplacian_categorical(input, input_parameters[0], input_parameters[1], input_parameters[2], input_parameters[3], input_parameters[4])
            elif args.orig_program_name == 'ackley':
                network = bayesian_optimization.ackley(input, args.input_nc)
            elif args.orig_program_name == 'laplacian_nlmeans':
                network = 0
            regularizer_loss = 0
            manual_regularize = 0
        elif args.unet:
            with tf.variable_scope("unet"):
                network, intermediate_layers = unet(input_to_network, args.unet_base_channel, args.update_bn, batch_norm_is_training)
                regularizer_loss = 0
                manual_regularize = 0
        elif args.fc:
            with tf.variable_scope("fc"):
                network = input_to_network
                #slim.stack(network, slim.fully_connected, [32, 64, 128, 1], activation_fn=lrelu)
                for neuron in [32, 64, 128, 256, 128, 64, 32, 1]:
                    network = slim.fully_connected(network, neuron, activation_fn=lrelu)
                regularizer_loss = 0.0
                manual_regularize = 0
        else:
            if 3 + args.input_nc <= actual_conv_channel:
                ini_id = True
            else:
                ini_id = False

            regularizer_loss = 0
            manual_regularize = args.rowwise_L2_normalize or args.Frobenius_normalize
            if args.encourage_sparse_features:
                regularizer = None
                if (args.regularizer_scale > 0 or args.L2_regularizer_scale > 0) and not manual_regularize:
                    regularizer = slim.l1_l2_regularizer(scale_l1=args.regularizer_scale, scale_l2=args.L2_regularizer_scale)
                actual_initial_layer_channels = args.initial_layer_channels
                actual_nfeatures = nfeatures
                with tf.variable_scope("feature_reduction"):
                    weights = tf.get_variable('w0', [1, 1, actual_nfeatures, actual_initial_layer_channels], initializer=tf.contrib.layers.xavier_initializer() if not args.identity_initialize else identity_initializer(color_inds), regularizer=regularizer)
                    if args.normalize_weights:
                        if args.abs_normalize:
                            column_sum = tf.reduce_sum(tf.abs(weights), [0, 1, 2])
                        elif args.rowwise_L2_normalize:
                            column_sum = tf.reduce_sum(tf.abs(tf.square(weights)), [0, 1, 2])
                        elif args.Frobenius_normalize:
                            column_sum = tf.reduce_sum(tf.abs(tf.square(weights)))
                        else:
                            column_sum = tf.reduce_sum(weights, [0, 1, 2])
                        weights_to_input = weights / column_sum
                    else:
                        weights_to_input = weights

                    input_to_network = tf.nn.conv2d(input_to_network, weights_to_input, [1, 1, 1, 1], "SAME")
                    if manual_regularize:
                        regularizer_loss = args.regularizer_scale * tf.reduce_mean(tf.abs(weights_to_input))
                    if actual_initial_layer_channels <= actual_conv_channel:
                        ini_id = True
                    else:
                        ini_id = False
                if args.add_initial_layers:
                    for nlayer in range(3):
                        input_to_network = slim.conv2d(input_to_network, actual_initial_layer_channels, [1, 1], rate=1, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='initial_'+str(nlayer), weights_regularizer=regularizer)

            network=build(input_to_network, ini_id, regularizer_scale=args.regularizer_scale, final_layer_channels=args.final_layer_channels, identity_initialize=args.identity_initialize, grayscale=args.grayscale)

    if not args.train_res:
        diff = network - output
    else:
        input_color = tf.stack([input[:, :, :, ind] for ind in range(3)], axis=3)
        diff = network + input_color - output
        network += input_color

    if args.RGB_norm % 2 != 0:
        diff = tf.abs(diff)
    powered_diff = diff ** args.RGB_norm

    loss=tf.reduce_mean(powered_diff)

    if args.multi_level_loss:
        nlevels = 9
        for n in range(nlevels):
            diff = tf.nn.avg_pool(diff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            if args.RGB_norm % 2 != 0:
                diff = tf.abs(diff)
            powered_diff = diff ** args.RGB_norm
            loss += tf.reduce_mean(powered_diff)

    loss_l2 = loss
    loss_add_term = loss

    if args.gradient_loss:
        canny_edge = tf.placeholder(tf.float32, shape=[None, None, None])
        if not args.grayscale_grad:
            dx_ground = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            dy_ground = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            dx_network, dy_network = image_gradients(network)
        else:
            dx_ground = tf.placeholder(tf.float32, shape=[None, None, None, 1])
            dy_ground = tf.placeholder(tf.float32, shape=[None, None, None, 1])
            bgr_weights = [0.0721, 0.7154, 0.2125]
            network_gray = tf.expand_dims(tf.tensordot(network, bgr_weights, [[-1], [-1]]), axis=3)
            dx_network, dy_network = image_gradients(network_gray)
        if args.normalize_grad:
            grad_norm_network = tf.sqrt(tf.square(dx_network) + tf.square(dy_network) + 1e-8)
            grad_norm_ground = tf.sqrt(tf.square(dx_ground) + tf.square(dy_ground) + 1e-8)
            dx_ground /= grad_norm_ground
            dy_ground /= grad_norm_ground
            dx_network /= grad_norm_network
            dy_network /= grad_norm_network
        if not args.cos_sim:
            gradient_loss_term = tf.reduce_mean(tf.square(dx_network - dx_ground) + tf.square(dy_network - dy_ground), axis=3)
        else:
            gradient_loss_term = -tf.reduce_mean(dx_network * dx_ground + dy_network * dy_ground, axis=3)

        if args.gradient_loss_all_pix:
            loss_add_term = tf.reduce_mean(gradient_loss_term)
        else:
            loss_add_term = tf.reduce_sum(gradient_loss_term * canny_edge) / tf.reduce_sum(canny_edge)

        loss += args.gradient_loss_scale * loss_add_term

    if args.perceptual_loss:
        vgg_in = vgg16.Vgg16()
        vgg_in.build(network)
        vgg_out = vgg16.Vgg16()
        vgg_out.build(output)
        loss_vgg = tf.reduce_mean(tf.square(getattr(vgg_in, args.perceptual_loss_term) - getattr(vgg_out, args.perceptual_loss_term)))
        loss += args.perceptual_loss_scale * loss_vgg

    avg_loss = 0
    tf.summary.scalar('avg_loss', avg_loss)
    avg_test = 0
    tf.summary.scalar('avg_test', avg_test)
    avg_validate = 0
    tf.summary.scalar('avg_validate', avg_validate)
    gradient_loss = 0
    tf.summary.scalar('gradient_loss', gradient_loss)
    l2_loss = 0
    tf.summary.scalar('l2_loss', l2_loss)

    loss_to_opt = loss + regularizer_loss

    if args.optimize_hyperparameter:
        var_list = [input_parameters]
        if args.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        elif args.optimizer == 'gradient':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
        elif args.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=args.learning_rate)
        elif args.optimizer == 'proximal':
            optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=args.learning_rate)
        elif not args.direct_scipy_optimize:
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss_to_opt, method=args.optimizer, var_list=var_list, options={'maxiter': niters})
        else:
            optimizer = None
        #elif args.optimizer in ['nelder-mead', 'powell']:
        #    optimizer = None
        #else:
        #    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss_to_opt, method=args.optimizer, var_list=var_list, options={'maxiter': niters})
        if args.optimizer in ['adam', 'gradient', 'adagrad', 'proximal']:
            opt = optimizer.minimize(loss_to_opt,var_list=var_list)
            regular_optimize = True
        else:
            regular_optimize = False
    else:
        var_list = tf.trainable_variables()
        adam_optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        if args.update_bn:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                opt=adam_optimizer.minimize(loss_to_opt,var_list=var_list)
        else:
            opt=adam_optimizer.minimize(loss_to_opt,var_list=var_list)

    if not args.use_orig_program:
        saver=tf.train.Saver(tf.trainable_variables(), max_to_keep=1000)
        var_only = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if ('Adam' not in var.name) and (not var.name.startswith('input_parameters'))]
        saver_vars_only = tf.train.Saver(var_list=var_only, max_to_keep=1000)

    print("start sess")
    #sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=10, intra_op_parallelism_threads=3))
    sess = tf.Session()
    print("after start sess")
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(args.name, sess.graph)
    print("initialize local vars")
    sess.run(tf.local_variables_initializer())
    print("initialize global vars")
    sess.run(tf.global_variables_initializer())

    read_from_epoch = True
    if not args.use_orig_program:
        if args.optimize_hyperparameter:
            ckpt = tf.train.get_checkpoint_state(os.path.join(args.name, 'var_only'))
        else:
            ckpt = tf.train.get_checkpoint_state(os.path.join(args.name, "%04d"%int(args.which_epoch)))
            if not ckpt:
                ckpt=tf.train.get_checkpoint_state(args.name)
                read_from_epoch = False

        if ckpt:
            print('loaded '+ckpt.model_checkpoint_path)
            if args.optimize_hyperparameter:
                saver_vars_only.restore(sess, ckpt.model_checkpoint_path)
            else:
                saver.restore(sess,ckpt.model_checkpoint_path)
            print('finished loading')

    if not args.optimize_hyperparameter:
        save_frequency = 1
        num_epoch = args.epoch
        #assert num_epoch % save_frequency == 0

        if not args.fc:
            if args.is_train or args.test_training:
                parameters = np.load(os.path.join(args.dataroot, 'train.npy'))
            else:
                parameters = np.load(os.path.join(args.dataroot, 'test.npy'))

        print("arriving before train branch")

        if args.is_train:
            if not args.fc:
                all=np.zeros(len(input_names), dtype=float)
            else:
                all = numpy.zeros(train_label.shape[0], dtype=float)

            if read_data_from_file and args.preload and not args.fc:
                input_images=[None]*len(input_names)
                output_images=[None]*len(input_names)

                for id in range(len(input_names)):
                    if read_ind(input_images, input_names, id, args.is_npy):
                        read_ind(output_images, output_names, id, False)
                    input_images[id] = np.expand_dims(input_images[id], axis=0)
                    output_images[id] = np.expand_dims(output_images[id], axis=0)

            min_avg_loss = 1e20

            for epoch in range(1, num_epoch+1):

                if read_from_epoch:
                    if epoch <= args.which_epoch:
                        continue
                else:
                    next_save_point = int(np.ceil(float(epoch) / save_frequency)) * save_frequency
                    if os.path.isdir("%s/%04d"%(args.name,next_save_point)):
                        continue

                cnt=0

                if args.fc:
                    dataset_len = train_label.shape[0]
                else:
                    dataset_len = len(input_names)
                permutation = np.random.permutation(dataset_len)
                nupdates = int(numpy.ceil(dataset_len / args.batch_size))

                for i in range(nupdates):
                    st=time.time()

                    feed_dict={}
                    if args.gradient_loss:
                        grad_arr = read_name(grad_names[permutation[i]], True)
                        feed_dict[canny_edge] = grad_arr[:, :, :, 0]
                        if args.grayscale_grad:
                            feed_dict[dx_ground] = grad_arr[:, :, :, 1:2]
                            feed_dict[dy_ground] = grad_arr[:, :, :, 2:3]
                        else:
                            feed_dict[dx_ground] = grad_arr[:, :, :, 1:4]
                            feed_dict[dy_ground] = grad_arr[:, :, :, 4:]

                    if not args.fc:
                        if args.preload:
                            input_image = input_images[permutation[i]]
                            output_image = output_images[permutation[i]]
                            if input_image is None:
                                continue
                        else:
                            input_image = np.expand_dims(read_name(input_names[permutation[i]], args.is_npy, args.is_bin), axis=0)
                            output_image = np.expand_dims(read_name(output_names[permutation[i]], False), axis=0)

                        feed_dict[input] = input_image
                        feed_dict[input_parameters] = parameters[:, permutation[i]]
                        feed_dict[output] = output_image
                    else:
                        start_ind = i * args.batch_size
                        end_ind = min((i + 1) * args.batch_size, permutation.shape[0])
                        feed_dict[input] = train_label[permutation[start_ind:end_ind], :]
                        feed_dict[output] = numpy.expand_dims(train_val[permutation[start_ind:end_ind]], axis=1)

                    _,current=sess.run([opt,loss],feed_dict=feed_dict)

                    if args.batch_size == 1:
                        current *= 255.0 * 255.0
                        all[permutation[i]]=current
                    else:
                        all[permutation[start_ind:end_ind]] = current
                    cnt += 1
                    print("%d %d %.2f %.2f %.2f %s"%(epoch,cnt,current,np.mean(all[np.where(all)]),time.time()-st,os.getcwd().split('/')[-2]))

                avg_loss = np.mean(all[np.where(all)])

                if min_avg_loss > avg_loss:
                    min_avg_loss = avg_loss

                summary = tf.Summary()
                summary.value.add(tag='avg_loss', simple_value=avg_loss)
                if manual_regularize:
                    summary.value.add(tag='reg_loss', simple_value=sess.run(regularizer_loss) / args.regularizer_scale)


                if args.collect_validate_while_training:
                    validate_parameters = np.load(os.path.join(args.dataroot, 'validate.npy'))
                    if args.preload:
                        validate_images = [None] * len(validate_names)
                        validate_out_images = [None] * len(validate_names)
                        for id in range(len(validate_names)):
                            read_ind(validate_images, validate_names, id, args.is_npy)
                            validate_images[id] = np.expand_dims(validate_images[id], axis=0)
                            read_ind(validate_out_images, validate_img_names, id, False)
                            validate_out_images[id] = np.expand_dims(validate_out_images[id], axis=0)

                    all_test=np.zeros(len(validate_names), dtype=float)
                    for ind in range(len(validate_names)):
                        if args.preload:
                            input_image = validate_images[ind]
                            output_image = validate_out_images[ind]
                        else:
                            input_image = np.expand_dims(read_name(validate_names[ind], args.is_npy, args.is_bin), axis=0)
                            output_image = np.expand_dims(read_name(validate_img_names[ind], False, False), axis=0)
                        if input_image is None:
                            continue
                        st=time.time()
                        feed_dict = {}
                        feed_dict[input] = input_image
                        feed_dict[output] = output_image
                        feed_dict[input_parameters] = validate_parameters[:, ind]
                        current=sess.run(loss,feed_dict=feed_dict)
                        print("%.3f"%(time.time()-st))
                        all_test[ind] = current * 255.0 * 255.0

                    avg_validate = np.mean(all_test[np.where(all_test)])
                    summary.value.add(tag='avg_validate', simple_value=avg_validate)

                #train_writer.add_run_metadata(run_metadata, 'epoch%d' % epoch)
                train_writer.add_summary(summary, epoch)

                if epoch % save_frequency == 0:
                    os.makedirs("%s/%04d"%(args.name,epoch))
                    target=open("%s/%04d/score.txt"%(args.name,epoch),'w')
                    target.write("%f"%np.mean(all[np.where(all)]))
                    target.close()

                    #target = open("%s/%04d/score_breakdown.txt"%(args.name,epoch),'w')
                    #target.write("%f, %f, %f, %f"%(avg_test_close, avg_test_far, avg_test_middle, avg_test_all))
                    #target.close()

                    if min_avg_loss == avg_loss:
                        saver.save(sess,"%s/model.ckpt"%args.name)
                    saver.save(sess,"%s/%04d/model.ckpt"%(args.name,epoch))

            #var_list_gconv1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='g_conv1')
            #g_conv1_dict = {}
            #for var_gconv1 in var_list_gconv1:
            #    g_conv1_dict[var_gconv1.name] = sess.run(var_gconv1)
            #save_obj(g_conv1_dict, "%s/g_conv1.pkl"%(args.name))

        if not args.is_train:
            if not os.path.exists('%s/var_only'%args.name):
                os.makedirs('%s/var_only'%args.name)
            saver_vars_only.save(sess, "%s/var_only/model.ckpt"%args.name)
            if args.preload and not args.fc:
                eval_images = [None] * len(val_names)
                eval_out_images = [None] * len(val_names)
                for id in range(len(val_names)):
                    read_ind(eval_images, val_names, id, args.is_npy)
                    eval_images[id] = np.expand_dims(eval_images[id], axis=0)
                    read_ind(eval_out_images, val_img_names, id, False)
                    eval_out_images[id] = np.expand_dims(eval_out_images[id], axis=0)

            if args.test_dirname == '':
                test_dirbase = 'train' if args.test_training else 'test'
                test_dirname = "%s/%s"%(args.name, test_dirbase)

                if read_from_epoch:
                    test_dirname += "_epoch_%04d"%args.which_epoch
            else:
                test_dirname = "%s/%s"%(args.name, args.test_dirname)

            if not os.path.isdir(test_dirname):
                os.makedirs(test_dirname)

            if args.fc:
                test_len = test_label.shape[0]
            else:
                test_len = len(val_names)

            if not args.fc:
                assert test_len <= parameters.shape[1]
            all_test=np.zeros(test_len, dtype=float)
            for ind in range(test_len):
                if args.generate_timeline:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                else:
                    run_options = None
                    run_metadata = None
                feed_dict = {}
                if not args.fc:
                    if args.preload:
                        input_image = eval_images[ind]
                        output_image = eval_out_images[ind]
                    else:
                        input_image = np.expand_dims(read_name(val_names[ind], args.is_npy, args.is_bin), axis=0)
                        output_image = np.expand_dims(read_name(val_img_names[ind], False, False), axis=0)
                    if input_image is None:
                        continue
                    if args.specific_par != '':
                        specific_parameters = args.specific_par.split(',')
                        specific_parameters = numpy.array([float(item) for item in specific_parameters])
                        feed_dict[input] = np.expand_dims(read_name(args.input_img, False), axis=0)
                        feed_dict[output] = np.expand_dims(read_name(args.output_img, False), axis=0)
                        feed_dict[input_parameters] = specific_parameters
                    else:
                        feed_dict[input] = input_image
                        feed_dict[output] = output_image
                        feed_dict[input_parameters] = parameters[:, ind]

                    if args.gradient_loss:
                        grad_arr = read_name(val_grad_names[ind], True)
                        feed_dict[canny_edge] = grad_arr[:, :, :, 0]
                        if args.grayscale_grad:
                            feed_dict[dx_ground] = grad_arr[:, :, :, 1:2]
                            feed_dict[dy_ground] = grad_arr[:, :, :, 2:3]
                        else:
                            feed_dict[dx_ground] = grad_arr[:, :, :, 1:4]
                            feed_dict[dy_ground] = grad_arr[:, :, :, 4:]
                else:
                    feed_dict[input] = test_label[ind:ind+1, :]
                    feed_dict[output] = numpy.expand_dims(test_val[ind:ind+1], axis=1)
                st=time.time()
                output_image, current=sess.run([network, loss_l2],feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                print("%.3f"%(time.time()-st))
                if not args.fc:
                    current *= 255.0 * 255.0
                    print(current)
                    output_image=np.minimum(np.maximum(output_image,0.0),1.0)*255.0
                    if args.specific_par != '':
                        cv2.imwrite('%s/%s.png'%(args.name, args.optimize_prefix), np.uint8(output_image[0,:,:,:]))
                        print(current * 255.0 * 255.0)
                        sess.close()
                        return
                    cv2.imwrite("%s/%06d.png"%(test_dirname, ind),np.uint8(output_image[0,:,:,:]))
                all_test[ind] = current
                if args.generate_timeline:
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open("%s/nn_%d.json"%(test_dirname, ind+1), 'w') as f:
                        f.write(chrome_trace)

            target=open(os.path.join(test_dirname, 'score.txt'),'w')
            target.write("%f"%np.mean(all_test[np.where(all_test)]))
            target.close()
            numpy.savetxt(os.path.join(test_dirname, 'l2_all.txt'), all_test[np.where(all_test)], fmt="%f, ")
            numpy.save(os.path.join(test_dirname, 'l2_all.npy'), all_test[np.where(all_test)])

            if not args.fc:
                if args.test_training:
                    grounddir = os.path.join(args.dataroot, 'train_img')
                else:
                    grounddir = os.path.join(args.dataroot, 'test_img')

                check_command = 'source activate pytorch36 && CUDA_VISIBLE_DEVICES=2, python plot_clip_weights.py ' + test_dirname + ' ' + grounddir + ' && source activate tensorflow36'
                subprocess.check_output(check_command, shell=True)
    else:
        idx = 0
        def loss_callback_functor(convergence_vec, is_scipy=False, func=None):
            def loss_callback(eval_loss):
                nonlocal idx
                if idx < convergence_vec.shape[0]:
                    convergence_vec[idx] = eval_loss
                print(idx, eval_loss * 255 * 255)
                idx += 1
            if not is_scipy:
                return loss_callback
            else:
                assert func is not None
                def callback(x):
                    loss_callback(func(x))
                return callback

        neval = 0

        if optimizer is None:
            def scipy_objective_functor(feed_dict):
                def img_func(x):
                    if args.orig_program_name == 'laplacian_nlmeans':
                        from laplacian_nlmeans import laplacian_nlmeans
                        out = laplacian_nlmeans(numpy.squeeze(feed_dict[input]), x)
                    else:
                        feed_dict[input_parameters] = x
                        out = sess.run(network, feed_dict=feed_dict)
                    return out

                def objective(x):
                    if args.orig_program_name == 'laplacian_nlmeans':
                        nonlocal neval
                        print(x)
                        out = img_func(x)
                        ans = numpy.mean((out - numpy.squeeze(feed_dict[output])) ** 2.0)
                        print(neval, ans)
                        neval += 1
                    else:
                        feed_dict[input_parameters] = x
                        ans = sess.run(loss, feed_dict=feed_dict)
                    return ans
                return objective, img_func

        if (args.input_img != '' or not args.use_input_img) and args.output_img != '':
            feed_dict = {}
            if args.use_input_img:
                input_img = read_name(args.input_img, False)
                feed_dict[input] = numpy.expand_dims(input_img, axis=0)
            output_img = read_name(args.output_img, False)
            feed_dict[output] = numpy.expand_dims(output_img, axis=0)
            convergence = -numpy.ones(niters)
            if optimizer is None:
                # do not use scipy interface in tensorflow
                # to make sure that gradient information is not used during optimization
                x0 = numpy.random.rand(args.input_nc)
                if args.orig_program_name == 'local_laplacian_tf':
                    x0[2] *= 0.1
                objective = scipy_objective_functor(feed_dict)
                res = scipy.optimize.minimize(objective, x0, method=args.optimizer, callback=loss_callback_functor(convergence, True, objective), options={'disp': True, 'maxiter': niters})
                print(res)
            elif regular_optimize:
                for i in range(niters):
                    _, current = sess.run([opt, loss], feed_dict=feed_dict)
                    print(i, current * 255 * 255)
                    convergence[i] = current * 255 * 255
            else:
                optimizer.minimize(sess, feed_dict=feed_dict, fetches=[loss], loss_callback=loss_callback_functor(convergence))
            if optimizer is None:
                optimized_parameters = res.x
                feed_dict[input_parameters] = optimized_parameters
            else:
                optimized_parameters = sess.run(input_parameters, feed_dict=feed_dict)
            img, current = sess.run([network, loss], feed_dict=feed_dict)
            print(current * 255.0 * 255.0)
            numpy.save('%s/%s_%s_convergence%s.npy'%(args.name, args.optimize_prefix, args.optimizer, '_orig' if args.use_orig_program else ''), convergence)
            numpy.save('%s/%s_%s_optimized_parameters%s.npy'%(args.name, args.optimize_prefix, args.optimizer, '_orig' if args.use_orig_program else ''), optimized_parameters)
            cv2.imwrite('%s/%s_%s_%s.png'%(args.name, args.optimize_prefix, args.optimizer, '_orig' if args.use_orig_program else ''), np.uint8(np.clip(img[0, :, :, :], 0.0, 1.0) * 255.0))
        else:
            if args.test_dirname == '':
                dirbase = 'train' if args.test_training else 'test'
            else:
                dirbase = args.test_dirname
            optimize_dirname = '%s/optimize_%s%s'%(args.name, dirbase, '_orig' if args.use_orig_program else '')
            if args.optimizer in ['nelder-mead', 'powell']:
                optimize_dirname += '_%s'%args.optimizer

            if not os.path.isdir(optimize_dirname):
                os.makedirs(optimize_dirname)

            if args.fc:
                data_len = test_val.shape[0]
            else:
                data_len = len(val_names)

            # only use the first 100 images
            if not args.fc and data_len > 100:
                val_names = val_names[:100]
                data_len = 100
            nrestarts = 3
            print(data_len, niters, nrestarts)
            convergence = -numpy.ones([data_len, niters, nrestarts])
            loss_record = numpy.zeros([data_len, nrestarts])
            iters_used = numpy.zeros([data_len, nrestarts])
            time_used = numpy.zeros([data_len, nrestarts])
            parameters_stored = numpy.zeros([data_len, args.input_nc])
            convergence_single = numpy.empty(niters)
            for ind in range(data_len):
                feed_dict = {}
                if not args.fc:
                    if args.preload:
                        input_image = eval_images[ind]
                        output_image = eval_out_images[ind]
                    else:
                        input_image = np.expand_dims(read_name(val_names[ind], args.is_npy, args.is_bin), axis=0)
                        output_image = np.expand_dims(read_name(val_img_names[ind], False, False), axis=0)
                    if input_image is None:
                        continue
                    feed_dict[input] = input_image
                    feed_dict[output] = output_image
                else:
                    feed_dict[output] = numpy.expand_dims(test_val[ind:ind+1], axis=1)

                min_current = 1e8
                min_current_iters = -1

                if optimizer is None:
                    objective, get_img = scipy_objective_functor(feed_dict)

                for k in range(nrestarts):
                    time_start = time.time()
                    idx = 0
                    convergence_single[:] = -1
                    x0 = numpy.random.rand(args.input_nc)
                    if args.orig_program_name == 'local_laplacian_tf':
                        x0[2] *= 0.1
                    if optimizer is not None:
                        sess.run(tf.assign(input_parameters, x0))
                    if optimizer is None:
                        res = scipy.optimize.minimize(objective, x0, method=args.optimizer, callback=loss_callback_functor(convergence_single, True, objective), options={'disp': True, 'maxiter': niters})
                        #print(res)
                        iters_used[ind, k] = idx
                    elif regular_optimize:
                        for i in range(niters):
                            _, current = sess.run([opt, loss], feed_dict=feed_dict)
                            #print(i, current * 255 * 255)
                            convergence_single[i] = current * 255 * 255
                            #convergence[ind, i] = current * 255 * 255
                        iters_used[ind, k] = i
                    else:
                        optimizer.minimize(sess, feed_dict=feed_dict, fetches=[loss], loss_callback=loss_callback_functor(convergence_single))
                        iters_used[ind, k] = idx
                    if optimizer is None:
                        optimized_parameters = res.x
                        feed_dict[input_parameters] = optimized_parameters
                    else:
                        optimized_parameters = sess.run(input_parameters, feed_dict=feed_dict)
                    current = sess.run(loss, feed_dict=feed_dict)
                    time_end = time.time()
                    time_used[ind, k] = time_end - time_start
                    convergence[ind, :, k] = convergence_single[:]
                    loss_record[ind, k] = current * 255 * 255
                    print(ind, k, iters_used[ind], current * 255 * 255)
                    if current < min_current:
                        min_current = current
                        min_current_iters = iters_used[ind]
                        parameters_stored[ind, :] = optimized_parameters[:]

                    numpy.save('%s/convergence.npy'%(optimize_dirname), convergence)
                    numpy.save('%s/loss_record.npy'%(optimize_dirname), loss_record)
                    numpy.save('%s/iters_used.npy'%(optimize_dirname), iters_used)
                    numpy.save('%s/parameters_stored.npy'%(optimize_dirname), parameters_stored)
                    numpy.save('%s/time.npy'%(optimize_dirname), time_used)

                if optimizer is None:
                    feed_dict[input_parameters] = parameters_stored[ind, :]
                else:
                    sess.run(tf.assign(input_parameters, parameters_stored[ind, :]))
                if not args.fc:
                    #img = sess.run(network, feed_dict=feed_dict)
                    if optimizer is None:
                        img = get_img(parameters_stored[ind, :])
                    else:
                        img = sess.run(network, feed_dict=feed_dict)
                    cv2.imwrite('%s/%06d.png'%(optimize_dirname, ind), np.uint8(np.clip(img[0, :, :, :], 0.0, 1.0) * 255.0))
                #loss_record[ind] = min_current * 255.0 * 255.0
            numpy.save('%s/convergence.npy'%(optimize_dirname), convergence)
            numpy.save('%s/loss_record.npy'%(optimize_dirname), loss_record)
            numpy.save('%s/iters_used.npy'%(optimize_dirname), iters_used)
            numpy.save('%s/parameters_stored.npy'%(optimize_dirname), parameters_stored)
            numpy.save('%s/time.npy'%(optimize_dirname), time_used)

    sess.close()

if __name__ == '__main__':
    main()
