import tensorflow.compat.v1 as tf
import tensorflow.contrib as contrib
import utils
from tqdm import tqdm
import numpy as np
import argparse
import os

class BaselineNetwork(utils.Network):

    def __init__(self, layer_sizes, feature_extractor_needed = False, use_dropout = False,
        activation = 'relu', dropoutv = 0.5, reshape_dims = None, seed = 0, 
        session_config = None, it = None, lr = 0.0001, embedding = False):

        super(BaselineNetwork, self).__init__(
            layer_sizes = layer_sizes,
            feature_extractor_needed = feature_extractor_needed,
            use_dropout = use_dropout,
            activation = activation,
            dropoutv = dropoutv,
            reshape_dims = reshape_dims,
            seed = seed,
            session_config = session_config,
            it = it,
            embedding = embedding,
        )
        self.lr = lr

    def setup_phs(self):

        dtype = tf.uint8
        if self.embedding: dtype = tf.float32
        self.phs['X'] = tf.placeholder(dtype, [None, *self.reshape_dims])
        self.phs['Y'] = tf.placeholder(tf.uint8, [None])


    def forward(self):

        X = self.phs['X']
        if not self.embedding: X = tf.cast(X, tf.float32) * (1.0 / 255)
        layer = self.apply_feature_extractor(X)

        n_layers = len(self.layer_sizes)-1
        for i in range(n_layers):

            layer = utils.dense(layer, self.layer_sizes[i], self.layer_sizes[i+1])
            print('Applied dense (%d, %d)' % (self.layer_sizes[i], self.layer_sizes[i+1]))

            if i+1 != len(self.layer_sizes)-1:
                if self.use_dropout:
                    layer = self.activation(layer)
                    layer = tf.keras.layers.Dropout(
                        rate = self.dropoutv, 
                        seed = self.seed)(layer, training = self.glob_training_ph)
                    print('Applied activation -> dropout')
                else:
                    layer = self.activation(layer)
                    print('Applied activation')

        self.vars['fX'] = layer

    def backward(self):
    
        assert('X' in self.phs)
        assert('Y' in self.phs)
        assert('fX' in self.vars)
        fX = self.vars['fX']
        Y = self.phs['Y']

        Y_one_hot = tf.one_hot(Y, depth = self.layer_sizes[-1], dtype = tf.float32)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fX, labels = Y_one_hot)
        loss = tf.reduce_mean(loss)
        self.vars['loss'] = loss

        var_list = self.get_trainable_vars()

        opt = tf.train.AdamOptimizer(self.lr)
        self.objs['opt'] = opt

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        op = utils.gradient_clip_minimize_op(opt, loss, var_list = var_list)
        self.vars['train_op'] = op

        predictions = tf.argmax(tf.nn.softmax(fX), axis = 1)
        predictions = tf.cast(predictions, tf.uint8) # cast to uint8, like Y
        self.vars['predictions'] = predictions

        acc = tf.equal(Y, predictions)
        acc = tf.cast(acc, tf.float32) # For averaging, first cast bool to float32
        acc = tf.reduce_mean(acc)
        self.vars['acc'] = acc

