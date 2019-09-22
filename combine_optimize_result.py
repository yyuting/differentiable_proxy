import numpy
import os
import sys

def main():
    base_dir = sys.argv[1]
    ndatas = int(sys.argv[2])
    #for mode in ['test', 'train', 'test_for_training_with_different_par']:
    for mode in ['test']:
        #result_dir = 'local_laplacian_unet_no_batch_norm_small/%s'%mode
        #proxy_dir = 'local_laplacian_unet_no_batch_norm_small/optimize_%s'%mode
        #program_dir = 'local_laplacian_unet_no_batch_norm_small/optimize_%s_orig'%mode
        #program_nelder_mead_dir = 'test/optimize_%s_orig_nelder-mead'%mode
        #program_powell_dir = 'test/optimize_%s_orig_powell'%mode

        #result_dir = 'local_laplacian_categorical_unet_no_batch_norm/%s'%mode
        #proxy_dir = 'local_laplacian_categorical_unet_no_batch_norm/optimize_%s'%mode
        #program_dir = 'local_laplacian_categorical_unet_no_batch_norm/optimize_%s_orig'%mode
        #program_nelder_mead_dir = 'local_laplacian_categorical_unet_no_batch_norm/optimize_%s_orig_nelder-mead'%mode
        #program_powell_dir = 'local_laplacian_categorical_unet_no_batch_norm/optimize_%s_orig_powell'%mode

        #result_dir = 'ackley_fc/%s'%mode
        #proxy_dir = 'ackley_fc/optimize_%s'%mode
        #program_dir = 'ackley_fc/optimize_%s_orig'%mode
        #program_nelder_mead_dir = 'local_laplacian_categorical_unet_no_batch_norm/optimize_%s_orig_nelder-mead'%mode
        #program_powell_dir = 'ackley_fc/optimize_%s_orig_powell'%mode

        result_dir = '%s/%s'%(base_dir, mode)
        proxy_dir = '%s/optimize_%s'%(base_dir, mode)
        program_dir = '%s/optimize_%s_orig'%(base_dir, mode)
        program_powell_dir = '%s/optimize_%s_orig_powell'%(base_dir, mode)

        loss_inference = numpy.load(os.path.join(result_dir, 'l2_all.npy'))

        max_iters = 149
        ncolumns = 6

        loss = numpy.empty([ndatas, 2+3*ncolumns])
        loss[:, 0] = range(ndatas)
        loss[:, 1] = loss_inference[:ndatas]

        for i in range(3):
            if i == 0:
                dir = proxy_dir
            elif i == 1:
                dir = program_dir
            elif i == 2:
                dir = program_powell_dir

            loss_filename = os.path.join(dir, 'loss_record.npy')
            iters_filename = os.path.join(dir, 'iters_used.npy')
            time_filename = os.path.join(dir, 'time.npy')

            if not (os.path.exists(loss_filename) and os.path.exists(iters_filename) and os.path.exists(time_filename)):
                for j in range(ncolumns):
                    loss[:, 2+ncolumns*i+j] = numpy.nan
                continue

            read_loss = numpy.load(loss_filename)
            read_iters = numpy.load(iters_filename)
            read_time = numpy.load(time_filename)

            # mean loss
            loss[:, 2+ncolumns*i] = numpy.mean(read_loss, axis=1)
            # mean iter
            loss[:, 2+ncolumns*i+1] = numpy.mean(read_iters, axis=1)
            # mean time
            loss[:, 2+ncolumns*i+2] = numpy.mean(read_time, axis=1)
            # min loss
            loss[:, 2+ncolumns*i+3] = numpy.min(read_loss, axis=1)
            # argmin idx
            argmin_idx = numpy.argmin(read_loss, axis=1)
            # iter for min loss
            loss[:, 2+ncolumns*i+4] = read_iters[[range(ndatas), argmin_idx]]
            # time for min loss
            loss[:, 2+ncolumns*i+5] = read_time[[range(ndatas), argmin_idx]]

        #loss_inference = numpy.load(os.path.join(result_dir, 'l2_all.npy'))
        #loss_proxy = numpy.load(os.path.join(proxy_dir, 'loss_record.npy'))
        #loss_program = numpy.load(os.path.join(program_dir, 'loss_record.npy'))
        #loss_nelder_mead = numpy.load(os.path.join(program_nelder_mead_dir, 'loss_record.npy'))

        #loss = numpy.empty([loss_proxy.shape[0], 5])
        #loss[:, 0] = range(loss_proxy.shape[0])
        #loss[:, 1] = loss_inference[:loss_proxy.shape[0]]
        #loss[:, 2] = loss_proxy[:]
        #loss[:, 3] = loss_program[:]
        #loss[:, 4] = loss_nelder_mead[:]

        numpy.savetxt(os.path.join(proxy_dir, 'combined_loss.txt'), loss, fmt="%10.3f", delimiter=',',)

        proxy_better_pct_mean = numpy.sum(loss[:, 2] < loss[:, 2+ncolumns]) * 100 / ndatas
        #proxy_approx_no_less_pct = numpy.sum((loss[:, 2] - loss[:, 4]) / loss[:, 4] < 0.05) * 100 / ndatas
        proxy_better_pct_min = numpy.sum(loss[:, 5] < loss[:, 5+ncolumns]) * 100 / ndatas

        #proxy_better_than_nelder_pct = numpy.sum(loss[:, 2] < loss[:, 6]) * 100 / ndatas
        #proxy_approx_no_less_nelder_pct = numpy.sum((loss[:, 2] - loss[:, 6]) / loss[:, 6] < 0.05) * 100 / ndatas

        proxy_better_than_powell_pct_mean = numpy.sum(loss[:, 2] < loss[:, 2+2*ncolumns]) * 100 / ndatas
        #proxy_approx_no_less_powell_pct = numpy.sum((loss[:, 2] - loss[:, 8]) / loss[:, 8] < 0.05) * 100 / ndatas
        proxy_better_than_powell_pct_min = numpy.sum(loss[:, 5] < loss[:, 5+2*ncolumns]) * 100 / ndatas

        #program_better_than_nelder_pct = numpy.sum(loss[:, 4] < loss[:, 6]) * 100 / ndatas
        #program_approx_no_less_nelder_pct = numpy.sum((loss[:, 4] - loss[:, 6]) / loss[:, 6] < 0.05) * 100 / ndatas

        #program_better_than_powell_pct = numpy.sum(loss[:, 4] < loss[:, 8]) * 100 / ndatas
        #program_approx_no_less_powell_pct = numpy.sum((loss[:, 4] - loss[:, 8]) / loss[:, 8] < 0.05) * 100 / ndatas

        #nelder_mead_not_converged = numpy.sum(loss[:, 7] == max_iters) * 100 / ndatas
        #powell_not_converged = numpy.sum(loss[:, 9] == max_iters) * 100 / ndatas

        mean_inference_loss = numpy.mean(loss[:, 1])
        mean_proxy_optimize_loss = numpy.mean(loss[:, 2])
        mean_program_optimize_loss = numpy.mean(loss[:, 2+ncolumns])
        #mean_nelder_optimize_loss = numpy.mean(loss[:, 6])
        mean_powell_optimize_loss = numpy.mean(loss[:, 2+2*ncolumns])

        mean_proxy_optimize_loss_min = numpy.mean(loss[:, 5])
        mean_program_optimize_loss_min = numpy.mean(loss[:, 5+ncolumns])
        mean_powell_optimize_loss_min = numpy.mean(loss[:, 5+2*ncolumns])

        mean_proxy_optimize_runtime = numpy.mean(loss[:, 4])
        mean_program_optimize_runtime = numpy.mean(loss[:, 4+ncolumns])
        mean_powell_optimize_runtime = numpy.mean(loss[:, 4+2*ncolumns])

        mean_proxy_optimize_runtime_min = numpy.mean(loss[:, 7])
        mean_program_optimize_runtime_min = numpy.mean(loss[:, 7+ncolumns])
        mean_powell_optimize_runtime_min = numpy.mean(loss[:, 7+2*ncolumns])

        open(os.path.join(proxy_dir, 'pct.txt'), 'w').write(
        """
        {proxy_better_pct_mean}% proxy result better than program (mean on multiple restarts)
        {proxy_better_pct_min}% proxy result better than program (min of multiple restarts)

        {proxy_better_than_powell_pct_mean}% proxy result better than powell (mean on multiple restarts)
        {proxy_better_than_powell_pct_min}% proxy result better than powell (min of multiple restarts)

        mean inference loss: {mean_inference_loss}

        mean proxy optimization loss (mean across all restarts): {mean_proxy_optimize_loss}
        mean program optimization loss (mean across all restarts): {mean_program_optimize_loss}
        mean powell optimization loss (mean across all restarts): {mean_powell_optimize_loss}

        mean proxy optimization loss (min of all restarts): {mean_proxy_optimize_loss_min}
        mean program optimization loss (min of all restarts): {mean_program_optimize_loss_min}
        mean powell optimization loss (min of all restarts): {mean_powell_optimize_loss_min}

        mean proxy optimization runtime (mean across all restarts): {mean_proxy_optimize_runtime}
        mean program optimization runtime (mean across all restarts): {mean_program_optimize_runtime}
        mean powell optimization runtime (mean across all restarts): {mean_powell_optimize_runtime}

        mean proxy optimization runtime (min of all restarts): {mean_proxy_optimize_runtime_min}
        mean program optimization runtime (min of all restarts): {mean_program_optimize_runtime_min}
        mean powell optimization runtime (min of all restarts): {mean_powell_optimize_runtime_min}
        """.format(**locals())
        )

if __name__ == '__main__':
    main()
