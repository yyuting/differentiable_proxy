import numpy
import os

for mode in ['test', 'train', 'test_for_training_with_different_par']:
    result_dir = 'local_laplacian_unet_no_batch_norm_small/%s'%mode
    proxy_dir = 'local_laplacian_unet_no_batch_norm_small/optimize_%s'%mode
    program_dir = 'local_laplacian_unet_no_batch_norm_small/optimize_%s_orig'%mode

    loss_inference = numpy.load(os.path.join(result_dir, 'l2_all.npy'))
    loss_proxy = numpy.load(os.path.join(proxy_dir, 'loss_record.npy'))
    loss_program = numpy.load(os.path.join(program_dir, 'loss_record.npy'))

    loss = numpy.empty([loss_proxy.shape[0], 4])
    loss[:, 0] = range(loss_proxy.shape[0])
    loss[:, 1] = loss_inference[:loss_proxy.shape[0]]
    loss[:, 2] = loss_proxy[:]
    loss[:, 3] = loss_program[:]
    #loss = numpy.concatenate([loss_inference, loss_proxy, loss_program])
    numpy.savetxt(os.path.join(proxy_dir, 'combined_loss.txt'), loss, fmt="%f, ")

    proxy_better_pct = numpy.sum(loss_proxy < loss_program) * 100 / loss_proxy.shape[0]
    proxy_approx_no_less_pct = numpy.sum((loss_proxy - loss_program) / loss_program < 0.05) * 100 / loss_proxy.shape[0]
    mean_inference_loss = numpy.mean(loss_inference)
    mean_proxy_optimize_loss = numpy.mean(loss_proxy)
    mean_program_optimize_loss = numpy.mean(loss_program)

    open(os.path.join(proxy_dir, 'pct.txt'), 'w').write(
    """
    {proxy_better_pct}% proxy result better than program
    {proxy_approx_no_less_pct}% proxy result approximately better or equal to program (less than 5% error).
    mean inference loss: {mean_inference_loss}
    mean proxy optimization loss: {mean_proxy_optimize_loss}
    mean program optimization loss: {mean_program_optimize_loss}
    """.format(**locals())
    )
