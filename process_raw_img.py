import skimage
import skimage.io
import glob
import numpy
import numpy.random
import os

"""
read raw image and select those whose size are larger than required,
crop them to the shape desired, and stored
"""

min_width = 256
min_height = 256
raw_dir = '/home/yy2bb/test_images/Images'
dest_dir = '/home/yy2bb/test_images/Images_cropped_256'

def main():
    files = glob.glob(os.path.join(raw_dir, '*/*.jpg')) + glob.glob(os.path.join(raw_dir, '*/*/*.jpg'))
    print(len(files))
    valid_file_count = 0
    for i in range(len(files)):
        file = files[i]
        print('processing %d out of %d'%(i, len(files)))
        img = skimage.img_as_float(skimage.io.imread(file))
        if len(img.shape) == 3 and img.shape[2] == 3:
            if img.shape[0] > min_height and img.shape[1] > min_width:
                crop_height = (img.shape[0] // min_height) * min_height
                crop_width = (img.shape[1] // min_width) * min_width
                offset_x = numpy.random.randint(img.shape[0] - crop_height) if crop_height < img.shape[0] else 0
                offset_y = numpy.random.randint(img.shape[1] - crop_width) if crop_width < img.shape[1] else 0
                crop_img = img[offset_x:offset_x+crop_height, offset_y:offset_y+crop_width, :]
                skimage.io.imsave(os.path.join(dest_dir, '%06d.png'%i), crop_img)
                valid_file_count += 1
    print('valid files stored: %d', valid_file_count)

if __name__ == '__main__':
    main()
