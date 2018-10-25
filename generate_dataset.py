import numpy
import os
import skimage.io
import skimage
import sys
from local_laplacian_categorical import local_laplacian_categorical
import shutil
import tensorflow as tf

dest_dir = '/home/yy2bb/test_images/Images_cropped_256_result'

def main():
    args = sys.argv[1:]
    #filenames = open(args[0]).read().split('\n')
    #parameters = numpy.load(args[1])

    #assert len(filenames) == parameters.shape[1]

    def local_laplacian_single(input_args):
        file = input_args[0]
        alpha = input_args[1]
        beta = input_args[2]
        eps = input_args[3]
        print("generating data for %s", file)
        img = skimage.img_as_float(skimage.io.imread(file))
        output = local_laplacian.local_laplacian(img, alpha, beta, eps)
        skimage.io.imsave(os.path.join(dest_dir, os.path.split(file)[1]), output)
        print("finished generating data for %s"%file)
        return

    #pool = ProcessingPool(nproc)
    #pool.map(local_laplacian_single, zip(filenames, list(parameters[0, :]), list(parameters[1, :]), list(parameters[2, :])))

    input=tf.placeholder(tf.float32,shape=[None,None,None,3])
    input_parameters = tf.placeholder(tf.float32, shape=5)
    network = local_laplacian_categorical(input, input_parameters[0], input_parameters[1], input_parameters[2], input_parameters[3], input_parameters[4])
    feed_dict = {}
    sess = tf.Session()

    source_dir_base = args[0]
    target_dir = args[1]
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    source_dir = os.path.join(source_dir_base, 'train_label')
    files = sorted(os.listdir(source_dir))
    ndatas = 100
    source_parameter_file = os.path.join(source_dir_base, 'test.npy')
    target_parameter_file = os.path.join(target_dir, 'test.npy')
    sampled_parameters = numpy.load(source_parameter_file)
    shutil.copyfile(source_parameter_file, target_parameter_file)

    target_label_dir = os.path.join(target_dir, 'test_label')
    target_img_dir = os.path.join(target_dir, 'test_img')
    if not os.path.isdir(target_label_dir):
        os.makedirs(target_label_dir)
    if not os.path.isdir(target_img_dir):
        os.makedirs(target_img_dir)

    for i in range(ndatas):
        file = files[i]
        img = skimage.img_as_float(skimage.io.imread(os.path.join(source_dir, file)))
        feed_dict[input] = numpy.expand_dims(img, axis=0)
        feed_dict[input_parameters] = sampled_parameters[:, i]
        out_img = sess.run(network, feed_dict=feed_dict)
        label_filename = os.path.join(target_label_dir, file)
        shutil.copyfile(os.path.join(source_dir, file), label_filename)
        img_filename = os.path.join(target_img_dir, file)
        skimage.io.imsave(img_filename, numpy.clip(numpy.squeeze(out_img), 0.0, 1.0))

    if False:
    #for mode in ['train', 'test', 'validate']:
        source_dir = os.path.join(source_dir_base, '%s_label'%mode)
        files = sorted(os.listdir(source_dir))
        ndatas = len(files)
        sampled_parameters = numpy.random.rand(5, ndatas)
        # normalize parameters
        # alpha: 0 - 1
        # beta: 0 - 1
        # eps: 0 - 0.1
        # levels: 2 - 8.99, but since numpy.random.rand takes value from [0, 1), can actually make it at range [2, 9)
        # J: 2 - 8.99, similar argument with levels, can scale to [2, 9)
        numpy.save(os.path.join(target_dir, '%s.npy'%mode), sampled_parameters)

        target_label_dir = os.path.join(target_dir, '%s_label'%mode)
        target_img_dir = os.path.join(target_dir, '%s_img'%mode)
        if not os.path.isdir(target_label_dir):
            os.makedirs(target_label_dir)
        if not os.path.isdir(target_img_dir):
            os.makedirs(target_img_dir)

        for i in range(len(files)):
            file = files[i]
            img = skimage.img_as_float(skimage.io.imread(os.path.join(source_dir, file)))
            feed_dict[input] = numpy.expand_dims(img, axis=0)
            feed_dict[input_parameters] = sampled_parameters[:, i]
            out_img = sess.run(network, feed_dict=feed_dict)
            label_filename = os.path.join(target_label_dir, file)
            shutil.copyfile(os.path.join(source_dir, file), label_filename)
            img_filename = os.path.join(target_img_dir, file)
            skimage.io.imsave(img_filename, numpy.clip(numpy.squeeze(out_img), 0.0, 1.0))

if __name__ == '__main__':
    main()
