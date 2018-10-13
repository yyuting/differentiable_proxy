import os
import numpy
import skimage.io
import skimage
import sys
max_len = 1024

def main():
    args = sys.argv[1:]
    dir = args[0]
    for mode in ['train', 'test', 'validate']:
        files = sorted(os.listdir(os.path.join(dir, '%s_img'%mode)))
        ind = []
        for i in range(len(files)):
            file = files[i]
            img = skimage.img_as_float(skimage.io.imread(os.path.join(dir, '%s_img/%s'%(mode, file))))
            if img.shape[0] > max_len or img.shape[1] > max_len:
                ind.append(i)
                os.remove(os.path.join(dir, '%s_img/%s'%(mode, file)))
                os.remove(os.path.join(dir, '%s_label/%s'%(mode, file)))
        ind_valid = set(range(len(files))) - set(ind)

        parameters = numpy.load(os.path.join(dir, '%s.npy'%mode))
        valid_parameters = parameters[:, list(ind_valid)]
        numpy.save(os.path.join(dir, '%s.npy'%mode), valid_parameters)

        print(valid_parameters.shape[1])

if __name__ == '__main__':
    main()
