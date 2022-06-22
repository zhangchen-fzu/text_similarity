'''
直接构造矩阵的方法：
如果单词相同就是1，否则为0
'''
import tensorflow as tf


class Indicator(tf.keras.layers.Layer):

    def __init__(self, width, height, **kwargs):
        super(Indicator, self).__init__(kwargs)
        self.width = width
        self.height = height

    def call(self, inputs, **kwargs):
        q, d = inputs
        q_paddings = tf.constant([[0, 0], [0, self.width]])
        q_pad = tf.pad(q, q_paddings, mode='constant', constant_values=-1)
        q_pad = tf.slice(q_pad, [0, 0], [-1, self.width, ])
        q_pad = tf.expand_dims(q_pad, -1)
        q_pad = tf.tile(q_pad, [1, 1, self.height])

        d_paddings = tf.constant([[0, 0], [0, self.height]])
        d_pad = tf.pad(d, d_paddings, mode='constant', constant_values=-2)
        d_pad = tf.slice(d_pad, [0, 0], [-1, self.height])
        d_pad = tf.expand_dims(d_pad, -1)
        d_pad = tf.tile(d_pad, [1, 1, self.width])
        d_pad = tf.transpose(d_pad, perm=[0, 2, 1])

        m = tf.cast(tf.equal(q_pad, d_pad), dtype=tf.float32)
        return m

    def compute_output_shape(self, input_shape):
        """output shape: [batch_size, self.width, self.height]"""
        q_shape, d_shape = input_shape
        assert q_shape[0] <= self.width
        assert d_shape[0] <= self.height
        return (self.width, self.height)