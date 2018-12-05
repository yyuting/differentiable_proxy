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
        templatesize = max(int(parameters[i*3+1] * 7) * 2 + 3, 1)
        searchsize = max(int(parameters[i*3+2] * 16) * 2 + 15, 1)
        filtered_laplacian.append(cv2.fastNlMeansDenoising(laplacian[i], None, h, templatesize, searchsize))

    new_gaussian = filtered_laplacian[-1].astype('f')
    for i in range(len(filtered_laplacian)-2, -1, -1):
        temp = cv2.pyrUp(new_gaussian)
        new_gaussian = temp + (filtered_laplacian[i].astype('f') * 2.0 - 256.0)

    return new_gaussian

ndims = 15
nparameter_sets = 8

def main():
    args = sys.argv[1:]
    base_dir = args[0]

    label_dirs = []
    img_dirs = []
    for mode in ['train', 'test', 'validate']:
        current_dir = os.path.join(base_dir, mode+'_label')
        files = sorted(os.listdir(current_dir))

        if False:
        #for file in files:
            filename, ext = os.path.splitext(file)
            assert ext.endswith('.png')
            print(filename)
            new_filename = os.path.join(current_dir, filename+'0.png')
            os.rename(os.path.join(current_dir, file), new_filename)
            for i in range(1, 8):
                os.symlink(new_filename, os.path.join(current_dir, filename+str(i)+'.png'))

        if True:
            save_dir = os.path.join(base_dir, mode+'_img')

            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            files = sorted(os.listdir(current_dir))
            print(current_dir)
            print(len(files))
            #parameters = numpy.empty([ndims, len(files) * nparameter_sets])
            parameter_name = mode + '.npy'
            parameter_path = os.path.join(base_dir, parameter_name)
            parameters = numpy.load(parameter_path)
            for i in range(len(files)):
                file = files[i]
                orig_img = cv2.imread(os.path.join(current_dir, file))
                # 8 different parameters for each image
                #for j in range(nparameter_sets):
                if True:
                    #current_parameters = numpy.random.rand(ndims)
                    current_parameters = parameters[:, i]
                    denoised_img = laplacian_nlmeans(orig_img, current_parameters)
                    print('finished', i)
                    orig_name = os.path.splitext(file)[0]
                    new_name = orig_name + '.png'
                    new_name = os.path.join(save_dir, new_name)
                    cv2.imwrite(new_name, denoised_img)
                    #ind = i * nparameter_sets + j
                    parameters[:, i] = current_parameters[:]
            numpy.save(parameter_path, parameters)
            print('finished', mode)

if __name__ == '__main__':
    main()
