import os
import skimage
import skimage.io
import numpy
import numpy.random
import glob

dest_dir = '/home/yy2bb/test_images/Images_cropped_256'
nprocs = 3

def main():
    files = numpy.array(glob.glob(os.path.join(dest_dir, '*.png')))
    nfiles = len(files)
    alpha = 1.0 - numpy.random.rand(nfiles)
    beta = numpy.random.rand(nfiles)
    eps = 0.1 * (1.0 - numpy.random.rand(nfiles))
    parameters = numpy.array([alpha, beta, eps])
    numpy.save(os.path.join(dest_dir, 'local_laplacian_parameter.npy'), parameters)

    rand_idx = numpy.random.permutation(nfiles)
    for i in range(nprocs):
        start_ind = (nfiles // nprocs) * i
        end_ind = (nfiles // nprocs) * (i + 1)
        if i == nprocs - 1:
            end_ind = nfiles
        current_ind = rand_idx[start_ind:end_ind]
        current_files = files[current_ind]
        open('filename%05d.txt'%i, 'wt').write('\n'.join(current_files))
        numpy.save('parameter%05d.npy'%i, parameters[:, current_ind])

    with open('cmd.txt', 'wt') as f:
        for i in range(nprocs):
            f.write('python generate_dataset.py %s %s\n'%('filename%05d.txt'%i, 'parameter%05d.npy'%i))

if __name__ == '__main__':
    main()
