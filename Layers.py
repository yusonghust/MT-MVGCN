# -*- coding: utf-8 -*-
import tensorflow as tf
import os


def _dot(x, y, sparse=False):
    if sparse:
        return tf.sparse_tensor_dense_matmul(x, y)
    return tf.matmul(x, y)

class GraphConvLayer:
    def __init__(self, cfg, input_dim, output_dim, name):
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_devices
        self.input_dim                     = input_dim
        self.output_dim                    = output_dim
        self.act                           = cfg.act
        self.biases                        = cfg.biases

        with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
            with tf.name_scope('weights'):
                self.w           = tf.get_variable(
                    name         = 'w',
                    dtype        = tf.float32,
                    shape        = (self.input_dim, self.output_dim),
                    initializer  = tf.contrib.layers.xavier_initializer())

            if self.biases:
                with tf.name_scope('biases'):
                    self.b       = tf.get_variable(
                    name         = 'b',
                    dtype        = tf.float32,
                    initializer  = tf.constant(0.1, shape=(self.output_dim,)))

    def call(self, adj_norm, x, sparse=False):
        '''
        adj_norm : adj matrix
        x: feature matrix or the last layer output
        '''
        hw                       = _dot(x = x, y = self.w, sparse=sparse)
        ahw                      = _dot(x = adj_norm, y = hw, sparse=True)

        if not self.biases:
            return self.act(ahw)

        return self.act(tf.add(ahw, self.b))

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)