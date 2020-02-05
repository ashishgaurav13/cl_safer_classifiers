import tensorflow.compat.v1 as tf
import tensorflow.contrib as contrib
import numpy as np
import os, sys
import random

def set_seed(seed = 0, dataset = None):
    tempseed = seed
    if dataset in [3, 4]:
        tempseed = 0 # For CIFAR100, Sim-CIFAR100 (but why?)
    np.random.seed(tempseed)
    tf.random.set_random_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(tempseed)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    random.seed(tempseed)
    session_config = tf.ConfigProto(
        intra_op_parallelism_threads = 1,
        inter_op_parallelism_threads = 1
    )
    session_config.gpu_options.allow_growth = True
    return session_config

def dense(x, i, o):
    seed = int(os.environ["PYTHONHASHSEED"])
    init = tf.initializers.glorot_uniform(seed = seed)
    init_bias = tf.zeros_initializer()
    W = tf.Variable(init([i, o]))
    b = tf.Variable(init_bias([1, o]))
    return tf.add(tf.matmul(x, W), b)

# Named
def dense2(x, i, o, name):
    seed = int(os.environ["PYTHONHASHSEED"])
    init = tf.initializers.glorot_uniform(seed = seed)
    init_bias = tf.zeros_initializer()
    with tf.variable_scope(name):
        W = tf.Variable(init([i, o]), name = "w")
        b = tf.Variable(init_bias([1, o]), name = "b")
    return tf.add(tf.matmul(x, W), b)

def average(xs):
    if len(xs) == 1:
        return xs[0]
    else:
        return tf.keras.layers.average(xs)

def waverage(xs, r, n):
    assert(n >= 1)
    ret = tf.constant(0.0, dtype = tf.float32)
    for i in range(n):
        ret += tf.gather(xs, i) * tf.gather(r, i)
    ret /= tf.reduce_sum(r)
    return ret

def get_var(name):
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=""):
        if var.name.startswith(name):
            if var.name.split(":")[0] != name:
                print("Requested: %s, Returning: %s" % (name, var.name))
            return var
    return None

def say(*args):
    return tf.print(*args, output_stream = sys.stdout)

def softplus_inv(x):
    return tf.math.log(tf.math.exp(x)-1)

def get_activation(aname):
    allowed_activations = {
        "relu": tf.nn.relu,
        "softplus": tf.nn.softplus,
        "tanh": tf.nn.tanh
    }
    assert(aname in allowed_activations.keys())
    return allowed_activations[aname]

def sum_up(var_list):
    return tf.reduce_sum([tf.reduce_sum(item) for item in var_list])

# https://zhuanlan.zhihu.com/p/38716569
def conv2d(x, kernel_shape, strides = [1, 1, 1, 1], padding = 'SAME'):
    kernel = tf.Variable(tf.truncated_normal(kernel_shape, dtype = tf.float32,
        stddev = 1e-1))
    b = tf.Variable(tf.zeros_initializer()([kernel_shape[-1]]))
    c = tf.nn.conv2d(x, kernel, strides, padding = padding)
    return tf.nn.bias_add(c, b)

def gradient_clip_minimize_op(opt, loss, var_list = None):
    gradients, variables = zip(*opt.compute_gradients(loss, var_list = var_list))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    return opt.apply_gradients(zip(gradients, variables))
