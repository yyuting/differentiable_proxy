import numpy
import numpy.random
import os
import sys
import skimage
import skimage.util
import skimage.io
import cv2

def laplacian_nlmeans(img, parameters):
    """
    parameters length: 3 * (levels + 1)
    """
    levels = 4
    laplacian = []
    old_gaussian = img.astype('f')
    for i in range(levels):
        gaussian = cv2.pyrDown(old_gaussian)
        temp = cv2.pyrUp(gaussian)
        laplacian.append(((old_gaussian - temp) * 0.5 + 128.0).astype('uint8'))
        old_gaussian = gaussian
    laplacian.append(gaussian.astype('uint8'))

    filtered_laplacian = []
    for i in range(len(laplacian)):
        # h: float [0, 40]
        # templatesize: odd int [3, 15]
        # searchsize: odd int [15, 45]
        h = parameters[i*3] * 40.0
        templatesize = int(parameters[i*3+1] * 7) * 2 + 3
        searchsize = int(parameters[i*3+2] * 16) * 2 + 15
        filtered_laplacian.append(cv2.fastNlMeansDenoising(laplacian[i], None, h, templatesize, searchsize))

    new_gaussian = filtered_laplacian[-1].astype('f')
    for i in range(len(filtered_laplacian)-2, -1, -1):
        temp = cv2.pyrUp(new_gaussian)
        new_gaussian = temp + (filtered_laplacian[i].astype('f') * 2.0 - 256.0)

    return new_gaussian.astype('uint8')

ndims = 15
nparameter_sets = 8

def main():
    args = sys.argv[1:]
    base_dir = args[0]

    label_dirs = []
    img_dirs = []
    for mode in ['train', 'test', 'validate']:
        current_dir = os.path.join(base_dir, mode+'_label')
        save_dir = os.path.join(base_dir, mode+'_img')

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        files = sorted(os.listdir(current_dir))
        print(current_dir)
        print(len(files))
        parameters = numpy.empty([ndims, len(files) * nparameter_sets])
        for i in range(len(files)):
            file = files[i]
            orig_img = cv2.imread(os.path.join(current_dir, file))
            # 8 different parameters for each image
            for j in range(nparameter_sets):
                current_parameters = numpy.random.rand(ndims)
                denoised_img = laplacian_nlmeans(orig_img, current_parameters)
                print('finished', i, j)
                orig_name = os.path.splitext(file)[0]
                new_name = orig_name + str(j) + '.png'
                new_name = os.path.join(save_dir, new_name)
                cv2.imwrite(new_name, denoised_img)
                ind = i * nparameter_sets + j
                parameters[:, ind] = current_parameters[:]
        parameter_name = mode + '.npy'
        parameter_path = os.path.join(base_dir, parameter_name)
        numpy.save(parameter_path, parameters)
        print('finished', mode)

if __name__ == '__main__':
    main()
