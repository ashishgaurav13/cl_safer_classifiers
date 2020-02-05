import tensorflow.compat.v1 as tf
import tensorflow.contrib as contrib
import os

# https://github.com/SamKirkiles/vgg-cifar100/blob/master/vgg.py
# https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/vgg.py
def vgg16(X, training):
    layer = X
    global_seed = int(os.environ["PYTHONHASHSEED"])
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    all_layers = []
    for l in cfg:
        if l == 'M':
            max_pool = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding = 'same')
            layer = max_pool(layer)
            dropout = tf.keras.layers.Dropout(rate = 0.5)
            layer = dropout(layer, training = training)
            all_layers += [max_pool, dropout]
            continue
        conv2d = tf.keras.layers.Conv2D(filters=l, kernel_size = (3, 3), padding = 'same', use_bias = True,
            kernel_initializer = tf.keras.initializers.glorot_uniform(seed = global_seed))
        layer = conv2d(layer)
        layer = contrib.layers.batch_norm(layer, activation_fn = tf.nn.relu, is_training = training)
        all_layers += [conv2d, 'bn']
    layer = tf.keras.layers.Flatten()(layer)
    return layer, all_layers

def vgg16_reuse(X, training, layers):
    layer = X
    global_seed = int(os.environ["PYTHONHASHSEED"])
    for l in layers:
        if l == 'bn':
            layer = contrib.layers.batch_norm(layer, activation_fn = tf.nn.relu, is_training = training)
            continue
        layer = l(layer)
    layer = tf.keras.layers.Flatten()(layer)
    return layer