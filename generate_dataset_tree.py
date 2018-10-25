import numpy
import tensorflow as tf
import skimage.io
import os
import sys; sys.path += ['../procedural_test']
import tensorflow_test as tree
import numpy.random
import numpy as np

dest_dir = 'tree'
ndatas = 1200

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
os.system('rm tmp')

def main():
    tree_var_len = len(tree.tree) + tree.no_branch
    tree_var = numpy.maximum(1.0 + numpy.random.randn(ndatas, tree_var_len) / 2.0, 0.0)
    #tree_var = numpy.ones((ndatas, tree_var_len))
    numpy.save(os.path.join(dest_dir, 'train.npy'), tree_var)

    tree_var_pl = tf.placeholder(dtype=tf.float32, shape=tree_var_len)

    blank_img = tf.zeros(tree.size+(3,), dtype=tf.float32)

    rendered = tree.render(blank_img, tree.tree, tree.startpos, from_data=tree_var_pl, var_len=True)[0]

    sess = tf.Session()
    for i in range(ndatas):
        img = sess.run(rendered, feed_dict={tree_var_pl: tree_var[i, :]})
        skimage.io.imsave(os.path.join(dest_dir, '%05d.png'%i), numpy.clip(img[:, :, 0], 0.0, 1.0))

if __name__ == '__main__':
    main()
