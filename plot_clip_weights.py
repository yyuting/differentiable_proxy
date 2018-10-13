import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import sys
import skimage.measure
import numpy
import skimage.io

MSE_ONLY = False

#TEST_GROUND = '/localtmp/yuting/out_1x_1_sample/train/zigzag_plane_normal_spheres/test_img'
#TRAIN_GROUND = '/localtmp/yuting/out_1x_1_sample/train/zigzag_plane_normal_spheres/train_img'
#TEST_GROUND = '/bigtemp/yy2bb/out_4_sample_features_random_camera_uniform_all_features/zigzag_plane_normal_spheres/datas/test_img'
#TRAIN_GROUND = '/bigtemp/yy2bb/out_4_sample_features_random_camera_uniform_all_features/zigzag_plane_normal_spheres/datas/train_img'
TEST_GROUND = '/localtmp/yuting/datas_features_only/datas_feature_only/test_img'
TRAIN_GROUND = '/localtmp/yuting/datas_features_only/datas_feature_only/train_img'

if not MSE_ONLY:
    sys.path += ['../PerceptualSimilarity']
    from models import dist_model as dm
    import torch
    from util import util

    cwd = os.getcwd()
    os.chdir('../PerceptualSimilarity')
    model = dm.DistModel()
    model.initialize(model='net-lin',net='alex',use_gpu=True)
    print('Model [%s] initialized'%model.name())
    os.chdir(cwd)

def compute_metric(dir1, dir2, mode):
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)
    img_files1 = sorted([file for file in files1 if file.endswith('.png') or file.endswith('.jpg')])
    img_files2 = sorted([file for file in files2 if file.endswith('.png') or file.endswith('.jpg')])
    assert len(img_files1) == len(img_files2)

    vals = numpy.empty(len(img_files1))
    if mode == 'perceptual':
        global model

    for ind in range(len(img_files1)):
        if mode == 'ssim' or mode == 'l2':
            img1 = skimage.img_as_float(skimage.io.imread(os.path.join(dir1, img_files1[ind])))
            img2 = skimage.img_as_float(skimage.io.imread(os.path.join(dir2, img_files2[ind])))
            if mode == 'ssim':
                vals[ind] = skimage.measure.compare_ssim(img1, img2, datarange=img2.max()-img2.min(), multichannel=True)
            else:
                vals[ind] = numpy.mean((img1 - img2) ** 2) * 255.0 * 255.0
        elif mode == 'perceptual':
            img1 = util.im2tensor(util.load_image(os.path.join(dir1, img_files1[ind])))
            img2 = util.im2tensor(util.load_image(os.path.join(dir2, img_files2[ind])))
            vals[ind] = model.forward(img1, img2)[0]
        else:
            raise

    filename_all = mode + '_all.txt'
    filename_breakdown = mode + '_breakdown.txt'
    filename_single = mode + '.txt'
    numpy.savetxt(os.path.join(dir1, filename_all), vals, fmt="%f, ")
    target=open(os.path.join(dir1, filename_single),'w')
    target.write("%f"%numpy.mean(vals))
    target.close()
    if len(img_files1) == 30:
        target=open(os.path.join(dir1, filename_breakdown),'w')
        target.write("%f, %f, %f"%(numpy.mean(vals[:5]), numpy.mean(vals[5:10]), numpy.mean(vals[10:])))
        target.close()
    return vals

def compute_ssim(dir1, dir2):
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)
    img_files1 = sorted([file for file in files1 if file.endswith('.png') or file.endswith('.jpg')])
    img_files2 = sorted([file for file in files2 if file.endswith('.png') or file.endswith('.jpg')])
    #img_files1 = sorted([file for file in files1 if file.endswith('.png') or file.endswith('synthesized_image.jpg')])
    #img_files2 = sorted([file for file in files1 if file.endswith('.png') or file.endswith('real_image.jpg')])
    assert len(img_files1) == len(img_files2)

    vals = numpy.empty(len(img_files1))

    for ind in range(len(img_files1)):
        img1 = skimage.img_as_float(skimage.io.imread(os.path.join(dir1, img_files1[ind])))
        img2 = skimage.img_as_float(skimage.io.imread(os.path.join(dir2, img_files2[ind])))
        vals[ind] = skimage.measure.compare_ssim(img1, img2, datarange=img2.max()-img2.min(), multichannel=True)

    numpy.savetxt(os.path.join(dir1, 'ssim_all.txt'), vals, fmt="%f, ")
    target=open(os.path.join(dir1, 'ssim.txt'),'w')
    target.write("%f"%numpy.mean(vals))
    target.close()
    if len(img_files1) == 30:
        target=open(os.path.join(dir1, 'ssim_breakdown.txt'),'w')
        target.write("%f, %f, %f"%(numpy.mean(vals[:5]), numpy.mean(vals[5:10]), numpy.mean(vals[10:])))
        target.close()
    return vals

def compute_perceptual(dir1, dir2):
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)
    img_files1 = sorted([file for file in files1 if file.endswith('.png') or file.endswith('.jpg')])
    img_files2 = sorted([file for file in files2 if file.endswith('.png') or file.endswith('.jpg')])
    #img_files1 = sorted([file for file in files1 if file.endswith('.png') or file.endswith('synthesized_image.jpg')])
    #img_files2 = sorted([file for file in files1 if file.endswith('.png') or file.endswith('real_image.jpg')])
    assert len(img_files1) == len(img_files2)

    vals = numpy.empty(len(img_files1))

    global model

    for ind in range(len(img_files1)):
        img1 = util.im2tensor(util.load_image(os.path.join(dir1, img_files1[ind])))
        img2 = util.im2tensor(util.load_image(os.path.join(dir2, img_files2[ind])))
        vals[ind] = model.forward(img1, img2)[0]

    numpy.savetxt(os.path.join(dir1, 'perceptual_all.txt'), vals, fmt="%f, ")
    target=open(os.path.join(dir1, 'perceptual.txt'),'w')
    target.write("%f"%numpy.mean(vals))
    target.close()
    if len(img_files1) == 30:
        target=open(os.path.join(dir1, 'perceptual_breakdown.txt'),'w')
        target.write("%f, %f, %f"%(numpy.mean(vals[:5]), numpy.mean(vals[5:10]), numpy.mean(vals[10:])))
        target.close()
    return vals

def get_score(name):
    train_x = []
    train_y = []
    test_all_y = []
    test_close_y = []
    test_far_y = []
    test_middle_y = []
    test_x = []
    if not MSE_ONLY:
        train_ssim_y = []
        test_all_ssim_y = []
        test_close_ssim_y = []
        test_far_ssim_y = []
        test_middle_ssim_y = []
        train_perceptual_y = []
        test_all_perceptual_y = []
        test_close_perceptual_y = []
        test_far_perceptual_y = []
        test_middle_perceptual_y = []
    dirs = sorted(os.listdir(name))
    for dir in dirs:
        if dir.startswith('test_pct_norm'):
            test_x.append(float(dir.replace('test_pct_norm', '')) / 10)
            score_file = os.path.join(name, dir, 'score.txt')
            test_all_y.append(float(open(score_file).read()))
            score_breakdown_file = os.path.join(name, dir, 'score_breakdown.txt')
            vals = open(score_breakdown_file).read().split(',')
            test_close_y.append(float(vals[0]))
            test_far_y.append(float(vals[1]))
            test_middle_y.append(float(vals[2]))

            if not MSE_ONLY:
                ssim_file = os.path.join(name, dir, 'ssim.txt')
                ssim_breakdown_file = os.path.join(name, dir, 'ssim_breakdown.txt')
                if not (os.path.exists(ssim_file) and os.path.exists(ssim_breakdown_file)):
                    compute_ssim(os.path.join(name, dir), TEST_GROUND)
                test_all_ssim_y.append(float(open(ssim_file).read()))

                perceptual_file = os.path.join(name, dir, 'perceptual.txt')
                perceptual_breakdown_file = os.path.join(name, dir, 'perceptual_breakdown.txt')
                if not os.path.exists(perceptual_file):
                    compute_perceptual(os.path.join(name, dir), TEST_GROUND)
                test_all_perceptual_y.append(float(open(perceptual_file).read()))

                ssim_vals = open(ssim_breakdown_file).read().split(', ')
                test_close_ssim_y.append(float(ssim_vals[0]))
                test_far_ssim_y.append(float(ssim_vals[1]))
                test_middle_ssim_y.append(float(ssim_vals[2]))

                perceptual_vals = open(perceptual_breakdown_file).read().split(', ')
                test_close_perceptual_y.append(float(perceptual_vals[0]))
                test_far_perceptual_y.append(float(perceptual_vals[1]))
                test_middle_perceptual_y.append(float(perceptual_vals[2]))

        elif dir.startswith('train_pct_norm'):
            train_x.append(float(dir.replace('train_pct_norm', '')) / 10)
            score_file = os.path.join(name, dir, 'score.txt')
            train_y.append(float(open(score_file).read()))

            if not MSE_ONLY:
                ssim_file = os.path.join(name, dir, 'ssim.txt')
                if not os.path.exists(ssim_file):
                    compute_ssim(os.path.join(name, dir), TRAIN_GROUND)
                train_ssim_y.append(float(open(ssim_file).read()))

                perceptual_file = os.path.join(name, dir, 'perceptual.txt')
                if not os.path.exists(perceptual_file):
                    compute_perceptual(os.path.join(name, dir), TRAIN_GROUND)
                train_perceptual_y.append(float(open(perceptual_file).read()))

    if len(train_x) == 0:
        #compute_ssim(name, sys.argv[2])
        #compute_perceptual(name, sys.argv[2])
        compute_metric(name, sys.argv[2], 'ssim')
        compute_metric(name, sys.argv[2], 'perceptual')
        compute_metric(name, sys.argv[2], 'l2')
        return

    figure = pyplot.figure()
    pyplot.plot(train_x, train_y, label='train_loss')
    pyplot.plot(test_x, test_all_y, label='test_all_loss')
    pyplot.plot(test_x, test_close_y, label='test_close_loss')
    pyplot.plot(test_x, test_far_y, label='test_far_loss')
    pyplot.plot(test_x, test_middle_y, label='test_middle_loss')
    pyplot.legend()
    #pyplot.ylim((50, 1000))
    figure.savefig(os.path.join(name, 'clip_weights_mse.png'))
    pyplot.ylim((50, 200))
    figure.savefig(os.path.join(name, 'clip_weights_mse_200.png'))
    pyplot.ylim((50, 1000))
    figure.savefig(os.path.join(name, 'clip_weights_mse_1000.png'))
    pyplot.close(figure)

    if not MSE_ONLY:
        figure = pyplot.figure()
        pyplot.plot(train_x, train_ssim_y, label='train_ssim')
        pyplot.plot(test_x, test_all_ssim_y, label='test_all_ssim')
        pyplot.plot(test_x, test_close_ssim_y, label='test_close_ssim')
        pyplot.plot(test_x, test_far_ssim_y, label='test_far_ssim')
        pyplot.plot(test_x, test_middle_ssim_y, label='test_middle_ssim')
        pyplot.legend()
        #pyplot.ylim((0.5, 1.0))
        figure.savefig(os.path.join(name, 'clip_weights_ssim.png'))
        pyplot.close(figure)

        figure = pyplot.figure()
        pyplot.plot(train_x, train_perceptual_y, label='train_perceptual')
        pyplot.plot(test_x, test_all_perceptual_y, label='test_all_perceptual')
        pyplot.plot(test_x, test_close_perceptual_y, label='test_close_perceptual')
        pyplot.plot(test_x, test_far_perceptual_y, label='test_far_perceptual')
        pyplot.plot(test_x, test_middle_perceptual_y, label='test_middle_perceptual')
        pyplot.legend()
        figure.savefig(os.path.join(name, 'clip_weights_perceptual.png'))
        pyplot.close(figure)

if __name__ == '__main__':
    get_score(sys.argv[1])
