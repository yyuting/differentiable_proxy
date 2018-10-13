import numpy
import os
import skimage.io
import skimage
import sys
import local_laplacian
from pathos.multiprocessing import ProcessingPool, cpu_count
nproc = cpu_count() // 2

dest_dir = '/home/yy2bb/test_images/Images_cropped_256_result'

def main():
    args = sys.argv[1:]
    filenames = open(args[0]).read().split('\n')
    parameters = numpy.load(args[1])

    assert len(filenames) == parameters.shape[1]

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

    pool = ProcessingPool(nproc)
    pool.map(local_laplacian_single, zip(filenames, list(parameters[0, :]), list(parameters[1, :]), list(parameters[2, :])))

if __name__ == '__main__':
    main()
