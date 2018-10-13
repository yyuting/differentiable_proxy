import tensorflow as tf
import tensorflow.contrib.slim as slim

base_channel = 32

def get_conv_channels(base):

    ini_channels = [base, base]

    downsample_channels = [round(base * 4.0 / 3.0),
                           round(base * 4.0 ** 2 / 3.0 ** 2),
                           round(base * 4.0 ** 3 / 3.0 ** 3),
                           round(base * 4.0 ** 4 / 3.0 ** 4),
                           round(base * 4.0 ** 4 / 3.0 ** 4)]

    upsample_channels = [round(base * 4.0 ** 3 / 3.0 ** 3),
                         round(base * 4.0 ** 2 / 3.0 ** 2),
                         round(base * 4.0 / 3.0),
                         base]

    final_channels = [4 * base, 2 * base]

    return ini_channels, downsample_channels, upsample_channels, final_channels

output_channels = 3

conv_size = [3, 3]

k2 = conv_size[0] * conv_size[1]

pool_size = [1, 2, 2, 1]

alpha = 0.1

# current experiment: same capacity
# ours: with batch norm, 48, 48, 48
# encoder-decoder: no batch norm, 10 initially

def total_weights_unet_no_batch_norm(c):
    p1 = 9 * c * c * \
        (16 + 2 + 4 +
        5 * (4 ** 2 / 3 ** 2 + 4 ** 4 / 3 ** 4 + 4 ** 6 / 3 ** 6 + 4 ** 8 / 3 ** 8))
    p2 = 9 * 153 * c + \
        c + 2 * c + c + \
        3 * c * (4 / 3 + 4 ** 2 / 3 ** 2 + 4 ** 3 / 3 ** 3 + 4 ** 4 / 3 ** 4) - c * 4 ** 4 / 3 ** 4 + \
        6 * c + \
        54 * c + 3
    return p1 + p2

def total_weights_unet_batch_norm(c):
    p1 = 9 * c * c * \
        (16 + 2 + 4 +
        5 * (4 ** 2 / 3 ** 2 + 4 ** 4 / 3 ** 4 + 4 ** 6 / 3 ** 6 + 4 ** 8 / 3 ** 8))
    p2 = 9 * 153 * c + \
        3 * (c + 2 * c + c + \
        3 * c * (4 / 3 + 4 ** 2 / 3 ** 2 + 4 ** 3 / 3 ** 3 + 4 ** 4 / 3 ** 4) - c * 4 ** 4 / 3 ** 4 + \
        6 * c) + \
        54 * c + 3
    return p1 + p2

def total_weights_ours(c):
    return 51 * c * c + 189 * c + 25

def lrelu(x):
    # alpha = 0.1 in paper
    return tf.maximum(x * alpha,x)

def prectifier_initializer(channel):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        nl = channel * k2
        std = (2.0 / ((1.0 + alpha ** 2) * nl)) ** 0.5
        return tf.random_normal(shape, stddev=std, dtype=dtype)
    return _initializer

def unet(input, unet_base_channel=base_channel, batch_norm=False, batch_norm_is_training=True, extra_info=None):

    def nm(x):
        return slim.batch_norm(x, is_training=batch_norm_is_training)
    net = input
    skip_features = []
    intermediate_layers = []
    ini_channels, downsample_channels, upsample_channels, final_channels = get_conv_channels(unet_base_channel)
    # TODO: what normalization used?
    if not batch_norm:
        normalizer_fn = None
    else:
        normalizer_fn = nm
    for i in range(len(ini_channels)):
        channel = ini_channels[i]
        net=slim.conv2d(net, channel, conv_size, activation_fn=lrelu, normalizer_fn=normalizer_fn, weights_initializer=prectifier_initializer(channel),scope='ini_conv'+str(i))
        intermediate_layers.append(net)
    skip_features.append(net)

    for i in range(len(downsample_channels)):
        channel = downsample_channels[i]
        with tf.variable_scope('down'+str(i)):
            net = slim.conv2d(net, channel, conv_size, activation_fn=lrelu, normalizer_fn=normalizer_fn, weights_initializer=prectifier_initializer(channel))
            intermediate_layers.append(net)
            net = tf.nn.max_pool(net, ksize=pool_size, strides=pool_size, padding="SAME")
            intermediate_layers.append(net)
            skip_features.append(net)

    skip_features = skip_features[:-1]

    for i in range(len(upsample_channels) + 1):
        if i < len(upsample_channels):
            channel = upsample_channels[i]
        else:
            channel = final_channels[0]
        with tf.variable_scope('up'+str(i)):
            net = tf.image.resize_images(net, tf.stack([tf.shape(net)[1] * 2, tf.shape(net)[2] * 2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            intermediate_layers.append(net)
            net = tf.concat([net, skip_features[-i-1]], 3)
            if i < len(upsample_channels):
                conv_channels = [channel, channel]
            else:
                conv_channels = [channel, final_channels[1]]
            for k in range(2):
                net = slim.conv2d(net, conv_channels[k], conv_size, activation_fn=lrelu, normalizer_fn=normalizer_fn, weights_initializer=prectifier_initializer(conv_channels[k]))
                intermediate_layers.append(net)

    net = slim.conv2d(net, output_channels, conv_size, activation_fn=None, weights_initializer=prectifier_initializer(output_channels), scope='output')
    intermediate_layers.append(net)
    return net, intermediate_layers
