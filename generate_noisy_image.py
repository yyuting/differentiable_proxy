import numpy
import numpy.random
import os
import sys
import skimage
import skimage.util
import skimage.io

valid_noise_modes = ['gaussian', 'poisson', 'speckle']

def main():
    args = sys.argv[1:]
    source_dir = args[0]
    target_dir = args[1]

    source_sub_dirs = []
    target_sub_label_dirs = []
    target_sub_img_dirs = []
    for mode in ['train', 'test', 'validate']:
        source_sub_dirs.append(os.path.join(source_dir, mode+'_label'))
        target_sub_label_dirs.append(os.path.join(target_dir, mode+'_label'))
        target_sub_img_dirs.append(os.path.join(target_dir, mode+'_img'))

    for dir in [target_dir] + target_sub_label_dirs + target_sub_img_dirs:
        if not os.path.isdir(dir):
            os.makedirs(dir)

    for i in range(len(source_sub_dirs)):
        current_dir = source_sub_dirs[i]
        save_dir = target_sub_label_dirs[i]
        files = sorted(os.listdir(current_dir))
        print(current_dir)
        print(len(files))
        mode_count = {}
        for mode in valid_noise_modes:
            mode_count[mode] = 0
        for file in files:
            img = skimage.img_as_float(skimage.io.imread(os.path.join(current_dir, file)))
            if img.size >= 512 * 1024 * 3:
                noise_mode = numpy.random.choice(valid_noise_modes)
                if noise_mode == 'poisson':
                    kwargs = {}
                else:
                    var = numpy.random.uniform(0, 0.1)
                    kwargs = {'var': var}
                noisy_img = skimage.util.random_noise(img, mode=noise_mode, **kwargs)
                new_img_name = '%s%05d.png'%(noise_mode, mode_count[noise_mode])
                mode_count[noise_mode] += 1
                skimage.io.imsave(os.path.join(save_dir, new_img_name), numpy.clip(noisy_img, 0.0, 1.0))
                print(new_img_name)

if __name__ == '__main__':
    main()
