import numpy
import os

for mode in ['test', 'train', 'test_for_training_with_different_par']:
    #result_dir = 'local_laplacian_unet_no_batch_norm_small/%s'%mode
    #proxy_dir = 'local_laplacian_unet_no_batch_norm_small/optimize_%s'%mode
    #program_dir = 'local_laplacian_unet_no_batch_norm_small/optimize_%s_orig'%mode
    #program_nelder_mead_dir = 'test/optimize_%s_orig_nelder-mead'%mode
    #program_powell_dir = 'test/optimize_%s_orig_powell'%mode

    result_dir = 'local_laplacian_categorical_unet_no_batch_norm/%s'%mode
    proxy_dir = 'local_laplacian_categorical_unet_no_batch_norm/optimize_%s'%mode
    program_dir = 'local_laplacian_categorical_unet_no_batch_norm/optimize_%s_orig'%mode
    program_nelder_mead_dir = 'local_laplacian_categorical_unet_no_batch_norm/optimize_%s_orig_nelder-mead'%mode
    program_powell_dir = 'local_laplacian_categorical_unet_no_batch_norm/optimize_%s_orig_powell'%mode

    loss_inference = numpy.load(os.path.join(result_dir, 'l2_all.npy'))

    ndatas = 100
    max_iters = 149

    loss = numpy.empty([ndatas, 10])
    loss[:, 0] = range(ndatas)
    loss[:, 1] = loss_inference[:ndatas]

    for i in range(4):
        if i == 0:
            dir = proxy_dir
        elif i == 1:
            dir = program_dir
        elif i == 2:
            dir = program_nelder_mead_dir
        elif i == 3:
            dir = program_powell_dir

        read_loss = numpy.load(os.path.join(dir, 'loss_record.npy'))
        loss[:, 2+2*i] = read_loss[:]

        read_iters = numpy.load(os.path.join(dir, 'iters_used.npy'))
        loss[:, 2+2*i+1] = read_iters[:]

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

    numpy.savetxt(os.path.join(proxy_dir, 'combined_loss.txt'), loss, fmt="%f, ")

    proxy_better_pct = numpy.sum(loss[:, 2] < loss[:, 4]) * 100 / ndatas
    proxy_approx_no_less_pct = numpy.sum((loss[:, 2] - loss[:, 4]) / loss[:, 4] < 0.05) * 100 / ndatas

    proxy_better_than_nelder_pct = numpy.sum(loss[:, 2] < loss[:, 6]) * 100 / ndatas
    proxy_approx_no_less_nelder_pct = numpy.sum((loss[:, 2] - loss[:, 6]) / loss[:, 6] < 0.05) * 100 / ndatas

    proxy_better_than_powell_pct = numpy.sum(loss[:, 2] < loss[:, 8]) * 100 / ndatas
    proxy_approx_no_less_powell_pct = numpy.sum((loss[:, 2] - loss[:, 8]) / loss[:, 8] < 0.05) * 100 / ndatas

    program_better_than_nelder_pct = numpy.sum(loss[:, 4] < loss[:, 6]) * 100 / ndatas
    program_approx_no_less_nelder_pct = numpy.sum((loss[:, 4] - loss[:, 6]) / loss[:, 6] < 0.05) * 100 / ndatas

    program_better_than_powell_pct = numpy.sum(loss[:, 4] < loss[:, 8]) * 100 / ndatas
    program_approx_no_less_powell_pct = numpy.sum((loss[:, 4] - loss[:, 8]) / loss[:, 8] < 0.05) * 100 / ndatas

    nelder_mead_not_converged = numpy.sum(loss[:, 7] == max_iters) * 100 / ndatas
    powell_not_converged = numpy.sum(loss[:, 9] == max_iters) * 100 / ndatas

    mean_inference_loss = numpy.mean(loss[:, 1])
    mean_proxy_optimize_loss = numpy.mean(loss[:, 2])
    mean_program_optimize_loss = numpy.mean(loss[:, 4])
    mean_nelder_optimize_loss = numpy.mean(loss[:, 6])
    mean_powell_optimize_loss = numpy.mean(loss[:, 8])

    open(os.path.join(proxy_dir, 'pct.txt'), 'w').write(
    """
    {proxy_better_pct}% proxy result better than program
    {proxy_approx_no_less_pct}% proxy result approximately better or equal to program (less than 5% error).

    {proxy_better_than_nelder_pct}% proxy result better than nelder-mead
    {proxy_approx_no_less_nelder_pct}% proxy result approximately better or equal to nelder-mead (less than 5% error)

    {proxy_better_than_powell_pct}% proxy result better than powell
    {proxy_approx_no_less_powell_pct}% proxy result approximately better or equal to powell (less than 5% error)

    {program_better_than_nelder_pct}% program result better than nelder-mead
    {program_approx_no_less_nelder_pct}% program result approximately better or equal to nelder-mead (less than 5% error).

    {program_better_than_powell_pct}% program result better than powell
    {program_approx_no_less_powell_pct}% program result approximately better or equal to powell (less than 5% error)

    {nelder_mead_not_converged}% percentage that nelder mead does not converge till maximum niters.
    {powell_not_converged}% percentage that powell does not converge till maximum niters.

    mean inference loss: {mean_inference_loss}
    mean proxy optimization loss: {mean_proxy_optimize_loss}
    mean program optimization loss: {mean_program_optimize_loss}
    mean nelder mead optimization loss: {mean_nelder_optimize_loss}
    mean powell optimization loss: {mean_powell_optimize_loss}
    """.format(**locals())
    )
