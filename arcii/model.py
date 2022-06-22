'''
arcii模型部分:
模型重点：交叉卷积
交叉卷积可以转化为：先1维卷积，reshape之后再对相同纬度的两个矩阵做处理。
'''
import tensorflow as tf
from textsimilarity.arcii import match_layer


'''
模型结构：
Q-emb-Conv1-reshape\
                    match_matrix-(conv2-maxpool)-flatten-dense1-dense2-dense3
D-emb-Conv1-reshape/
'''

config = {
    'query_max_len':8,
    'doc_max_len':16,
    'vocab_size':10000,
    'embedding_size':256,
    'num_conv_layers':1,
    'filter':[32],
    'kernel_size':[[3, 3]],
    'pool_size':[[2, 2]],
    'num_dense_layers':2,
    'dense_size':[128, 64]
}

def build_model(config):
    q_input = tf.keras.layers.Input(shape=(config['query_max_len'],), name='q_input')
    d_input = tf.keras.layers.Input(shape=(config['doc_max_len'],), name='d_input')

    embedding = tf.keras.layers.Embedding(config['vocab_size'], config['embedding_size'], name='embedding')
    q_embedding = embedding(q_input)
    d_embedding = embedding(d_input)

    q_conv1 = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu',
        name='q_conv1')(q_embedding)
    d_conv1 = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu',
        name='d_conv1')(d_embedding)

    matching_layer = match_layer.MatchingLayer(matching_type='plus')
    matrix = matching_layer([q_conv1, d_conv1])

    for i in range(config['num_conv_layers']):
        matrix = tf.keras.layers.Conv2D(
            filters=config['filter'][i],
            kernel_size=config['kernel_size'][i],
            padding='same',
            activation='relu',
            name='conv_%d' %i)(matrix)
        matrix = tf.keras.layers.MaxPooling2D(pool_size=tuple(config['pool_size'][i]), name='max_pool_%d' %i)(matrix)

    flatten = tf.keras.layers.Flatten()(matrix)
    dense = flatten
    for j in range(config['num_dense_layers']):
        dense = tf.keras.layers.Dense(config['dense_size'][j], activation='relu', name='dense_%d' %j)(dense)

    out = tf.keras.layers.Dense(1, activation='sigmoid', name='out')(dense)

    model = tf.keras.Model(inputs=[q_input, d_input], outputs=out)
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.Accuracy()]
    )
    return model


