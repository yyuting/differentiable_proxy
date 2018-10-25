import tensorflow as tf
import numpy
import numpy as np
import skimage
import skimage.io
import sys
import numpy.random
import os
from local_laplacian import local_laplacian

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
os.system('rm tmp')

def local_laplacian_categorical(input_img, alpha=1.0 / 7.0, beta=1.0, eps=0.01, levels=8, J=8, feed_dict=None):
    # assume input_img = tf.placeholder([1, None, None, 3])
    # assume all input parameters are at range [0, 1)
    # generally, they're generated from numpy.random.rand
    eps *= 0.1
    levels *= 7.0
    levels += 2.0
    J *= 7.0
    J += 2.0
    levels = tf.cast(levels, tf.int32)
    J = tf.cast(J, tf.int32)

    max_levels = 8
    max_J = 8

    # gray dim: [1, None, None]
    gray = 0.299 * input_img[:, :, :, 0] + 0.587 * input_img[:, :, :, 1] + 0.114 * input_img[:, :, :, 2]

    gPyramid = [None] * max_J
    lPyramid = [None] * max_J
    inGPyramid = [None] * max_J
    outLPyramid = [None] * max_J
    outGPyramid = [None] * max_J
    gPyramid0 = [None] * max_levels

    for k in range(max_levels):
        level = k * (1.0 / tf.cast(levels - 1, tf.float32))
        # idx shape: [1, None, None, 1]
        idx = tf.cast(gray * (tf.cast(levels, tf.float32) - 1.0) * 256.0, tf.int32)
        idx = tf.clip_by_value(idx, 0, 256 * (levels - 1))
        fx = (tf.cast(idx, tf.float32) - 256.0 * k) / 256.0
        gPyramid0[k] = beta * (gray - level) + level + alpha * fx * tf.exp(-fx * fx / 2.0)
    gPyramid[0] = tf.stack(gPyramid0, axis=3)
    inGPyramid[0] = tf.expand_dims(gray, 3)

    filter_base = numpy.array([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype=numpy.float32)
    filter_gPyramid = numpy.zeros([4, 4, max_levels, max_levels])
    for i in range(max_levels):
        filter_gPyramid[:, :, i, i] = filter_base
    filter_inGPyramid = numpy.expand_dims(numpy.expand_dims(filter_base, 2), 3)

    for j in range(1, max_J):
        def update_gPyramid():
            gPyramid_old = gPyramid[j-1]
            gPyramid_old_pad = tf.pad(gPyramid_old, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
            return tf.nn.conv2d(gPyramid_old_pad, filter_gPyramid, [1, 2, 2, 1], "VALID") / 64.0
        gPyramid[j] = tf.cond(j < J,
                              update_gPyramid,
                              lambda: gPyramid[j-1])

        def update_inGPyramid():
            inGPyramid_old = inGPyramid[j-1]
            inGPyramid_old_pad = tf.pad(inGPyramid_old, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
            return tf.nn.conv2d(inGPyramid_old_pad, filter_inGPyramid, [1, 2, 2, 1], "VALID") / 64.0
        inGPyramid[j] = tf.cond(j < J,
                                update_inGPyramid,
                                lambda: inGPyramid[j-1])

    lPyramid[max_J-1] = gPyramid[max_J-1]

    level = inGPyramid[max_J-1] * tf.cast(levels - 1, tf.float32)
    li = tf.clip_by_value(tf.cast(level, tf.int32), 0, levels - 2)
    lf = level - tf.cast(li, tf.float32)
    #outLPyramid[J-1] = (1.0 - lf) * tf.gather(lPyramid[J-1], li, axis=3) + lf * tf.gather(lPyramid[J-1], li+1, axis=3)
    meshx, meshy = tf.meshgrid(tf.range(tf.shape(li)[1]), tf.range(tf.shape(li)[2]), indexing='ij')
    #indices = tf.stack([tf.zeros_like(meshx), meshx, meshy, tf.squeeze(li)], axis=2)
    indices = tf.stack([meshx, meshy, tf.squeeze(tf.squeeze(li, axis=3), axis=0)], axis=2)
    indices2 = tf.stack([meshx, meshy, tf.squeeze(tf.squeeze(li+1, axis=3), axis=0)], axis=2)
    outLPyramid[max_J-1] = (1.0 - lf) * tf.expand_dims(tf.expand_dims(tf.gather_nd(tf.squeeze(lPyramid[max_J-1], axis=0), indices), axis=0), axis=3) + lf * tf.expand_dims(tf.expand_dims(tf.gather_nd(tf.squeeze(lPyramid[max_J-1], axis=0), indices2), axis=0), axis=3)

    outGPyramid[max_J-1] = outLPyramid[max_J-1]

    filter_base2 = numpy.array([[1, 3], [3, 9]])
    filter_lPyramid = numpy.zeros([2, 2, max_levels, max_levels])
    for i in range(max_levels):
        filter_lPyramid[:, :, i, i] = filter_base2
    filter_outGPyramid = numpy.expand_dims(numpy.expand_dims(filter_base2, axis=2), axis=3)

    for j in range(max_J - 2, -1, -1):
        #gPyramid_pad = tf.pad(gPyramid[j+1], [[0, 0], [1, 0], [1, 0], [0, 0]], 'SYMMETRIC')
        def update_lPyramid():
            gPyramid_old = gPyramid[j+1]
            gPyramid_resize = tf.image.resize_images(gPyramid_old, tf.stack([tf.shape(gPyramid_old)[1]*2, tf.shape(gPyramid_old)[2]*2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            gPyramid_pad = tf.pad(gPyramid_resize, [[0, 0], [1, 0], [1, 0], [0, 0]], 'SYMMETRIC')
            return gPyramid[j] - tf.nn.conv2d(gPyramid_pad, filter_lPyramid, [1, 1, 1, 1], "VALID") / 16.0
        lPyramid[j] = tf.cond(j < J - 1,
                              update_lPyramid,
                              lambda: lPyramid[j+1])

        def update_outLPyramid():
            level = inGPyramid[j] * tf.cast(levels - 1, tf.float32)
            li = tf.clip_by_value(tf.cast(level, tf.int32), 0, levels - 2)
            lf = level - tf.cast(li, tf.float32)
            meshx, meshy = tf.meshgrid(tf.range(tf.shape(li)[1]), tf.range(tf.shape(li)[2]), indexing='ij')
            indices = tf.stack([meshx, meshy, tf.squeeze(tf.squeeze(li, axis=3), axis=0)], axis=2)
            indices2 = tf.stack([meshx, meshy, tf.squeeze(tf.squeeze(li+1, axis=3), axis=0)], axis=2)
            return (1.0 - lf) * tf.expand_dims(tf.expand_dims(tf.gather_nd(tf.squeeze(lPyramid[j], axis=0), indices), axis=0), axis=3) + lf * tf.expand_dims(tf.expand_dims(tf.gather_nd(tf.squeeze(lPyramid[j], axis=0), indices2), axis=0), axis=3)
        outLPyramid[j] = tf.cond(j < J - 1,
                                 update_outLPyramid,
                                 lambda: outLPyramid[j+1])

        def update_outGPyramid():
            outGPyramid_old = outGPyramid[j+1]
            outGPyramid_resize = tf.image.resize_images(outGPyramid_old, tf.stack([tf.shape(outGPyramid_old)[1]*2, tf.shape(outGPyramid_old)[2]*2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            outGPyramid_pad = tf.pad(outGPyramid_resize, [[0, 0], [1, 0], [1, 0], [0, 0]], 'SYMMETRIC')
            return outLPyramid[j] + tf.nn.conv2d(outGPyramid_pad, filter_outGPyramid, [1, 1, 1, 1], "VALID") / 16.0
        outGPyramid[j] = tf.cond(j < J - 1,
                                 update_outGPyramid,
                                 lambda: outGPyramid[j+1])

    output = outGPyramid[0] * (input_img + eps) / (tf.expand_dims(gray, axis=3) + eps)
    output = tf.clip_by_value(output, 0.0, 1.0)
    return output

if False:
    input_img = tf.placeholder(tf.float32, [1, None, None, 3])
    output_img = tf.placeholder(tf.float32, [1, None, None, 3])
    ini_parameters = numpy.random.rand(3)
    ini_parameters[2] *= 0.1
    input_parameters = tf.Variable(ini_parameters, dtype=tf.float32, name='input_parameters')

    #img_raw = skimage.img_as_float(skimage.io.imread('/home/yy2bb/test_images/Images_cropped_256/000000.png'))
    img_raw = skimage.img_as_float(skimage.io.imread('test_label/000000.png'))
    img = numpy.expand_dims(img_raw, axis=0)
    output_raw = skimage.img_as_float(skimage.io.imread('test_img/000000.png'))
    feed_dict = {input_img: img}
    ans = local_laplacian_tf(input_img, input_parameters[0], input_parameters[1], input_parameters[2], feed_dict=feed_dict)
    loss = tf.reduce_mean((ans - output_img) ** 2.0)

    opt = tf.train.AdamOptimizer().minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    arr, parameters = sess.run([ans, input_parameters], feed_dict={input_img: img})
    print("tf finished")
    arr2 = local_laplacian(img_raw, parameters[0], parameters[1], parameters[2])
    print("python finished")
    print(numpy.allclose(numpy.squeeze(arr), arr2))

    sess.run(opt, feed_dict={input_img: img})

if __name__ == '__main__':
    input_img = tf.placeholder(tf.float32, [1, None, None, 3])
    input_parameters = tf.placeholder(tf.float32, [5])

    #os.chdir('/localtmp/yuting/datas_local_laplacian_test_optimization')
    #parameters = numpy.load('test.npy')
    #parameters = numpy.random.rand(5)
    parameters = numpy.load('test.npy')
    sess = tf.Session()

    img_raw = skimage.img_as_float(skimage.io.imread('train_label/000000.png'))
    img = numpy.expand_dims(img_raw, axis=0)
    feed_dict = {input_img: img, input_parameters: parameters}

    ans = local_laplacian_categorical(input_img, input_parameters[0], input_parameters[1], input_parameters[2], input_parameters[3], input_parameters[4], feed_dict=feed_dict)

    arr = sess.run(ans, feed_dict=feed_dict)
    skimage.io.imsave('test.png', numpy.squeeze(arr))

    arr_true = local_laplacian(img_raw, parameters[0], parameters[1], parameters[2]*0.1, parameters[3]*7+2, parameters[4]*7+2)
    skimage.io.imsave('test_true.png', arr_true)
    assert numpy.allclose(arr_true, numpy.squeeze(arr))

    if False:
    #for i in range(parameters.shape[1]):
        img_raw = skimage.img_as_float(skimage.io.imread('test_label/%06d.png'%i))
        img = numpy.expand_dims(img_raw, axis=0)
        arr = sess.run(ans, feed_dict={input_img: img, input_parameters:parameters[:, i]})
        skimage.io.imsave('test_img/%06d.png'%i, numpy.squeeze(arr))
