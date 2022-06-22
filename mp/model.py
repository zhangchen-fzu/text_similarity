'''
match pyramid模型部分：
包括三种构建矩阵的方法：内积、余弦、直接
'''
import tensorflow as tf
from textsimilarity.mp.indicator import Indicator

model_config = {
    'query_max_len': 1000,
    'doc_max_len': 1000,
    'num_conv_layers': 3,
    'filters': [8, 16, 32], #kernel的层数
    'kernel_size': [[5, 5], [3, 3], [3, 3]], ##kernel的大小
    'pool_size': [[2, 2], [2, 2], [2, 2]],  ##池化的窗口，注意你的Q及D的长度要够pool才行。比如这个，最大长度要大于等于8才行
    'dropout': 0.5,
    'batch_size': 32,
    'vocab_size': 100,
    'embedding_size': 128
}

'''
内积模型:
Q-emb-\
        Dot-Reshape-(conv1-pooling1-BatchNormalization)-(conv2-pooling2-BatchNormalization)-(conv3-pooling3-BatchNormalization)-flatten-dense1-dense2
D-emb-/
'''
def build_dot_model(config):
    q_input = tf.keras.layers.Input(shape=(config['query_max_len'],), name='q_input')
    d_input = tf.keras.layers.Input(shape=(config['doc_max_len'],), name='d_input')

    embedding = tf.keras.layers.Embedding(config['vocab_size'], config['embedding_size'], name='embedding')
    q_embedding = embedding(q_input)
    d_embedding = embedding(d_input)

    dot = tf.keras.layers.Dot(axes=-1, name='dot')([q_embedding, d_embedding])
    matrix = tf.keras.layers.Reshape((config['query_max_len'], config['doc_max_len'], 1), name='matrix')(dot)

    for i in range(config['num_conv_layers']):
        matrix = tf.keras.layers.Conv2D(
            filters=config['filters'][i],
            kernel_size=config['kernel_size'][i],
            padding='same',
            activation='relu',
            name='conv_%d' % i)(matrix)
        matrix = tf.keras.layers.MaxPool2D(pool_size=tuple(config['pool_size'][i]), name='max_pooling_%d' % i)(matrix)
        matrix = tf.keras.layers.BatchNormalization()(matrix)
    flatten = tf.keras.layers.Flatten()(matrix)
    dense = tf.keras.layers.Dense(32, activation='relu')(flatten)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='out')(dense)

    model = tf.keras.Model(inputs=[q_input, d_input], outputs=out)
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )
    return model


'''
余弦模型
Q-emb-\
        cosine-Reshape-(conv1-pooling1-BatchNormalization)-(conv2-pooling2-BatchNormalization)-(conv3-pooling3-BatchNormalization)-flatten-dense1-dense2
D-emb-/
'''
def build_cosine_model(config):
    q_input = tf.keras.layers.Input(shape=(config['query_max_len'],), name='q_input')
    d_input = tf.keras.layers.Input(shape=(config['doc_max_len'],), name='d_input')

    embedding = tf.keras.layers.Embedding(config['vocab_size'], config['embedding_size'], name='embedding')
    q_embedding = embedding(q_input)
    d_embedding = embedding(d_input)

    cosine = tf.keras.layers.Dot(axes=-1, normalize=True, name='cosine')([q_embedding, d_embedding])
    matrix = tf.keras.layers.Reshape((config['query_max_len'], config['doc_max_len'], 1), name='matrix')(cosine)

    for i in range(config['num_conv_layers']):
        matrix = tf.keras.layers.Conv2D(
            filters=config['filters'][i],
            kernel_size=config['kernel_size'][i],
            padding='same',
            activation='relu',
            name='conv_%d' % i)(matrix)
        matrix = tf.keras.layers.MaxPool2D(pool_size=tuple(config['pool_size'][i]), name='max_pooling_%d' % i)(matrix)
        matrix = tf.keras.layers.BatchNormalization()(matrix)

    flatten = tf.keras.layers.Flatten()(matrix)
    dense = tf.keras.layers.Dense(32, activation='relu')(flatten)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='out')(dense)

    model = tf.keras.Model(inputs=[q_input, d_input], outputs=out)
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )
    return model

'''
如果单词相同就是1，否则为0
Q-emb-\
        Indicator-Reshape-(conv1-pooling1-BatchNormalization)-(conv2-pooling2-BatchNormalization)-(conv3-pooling3-BatchNormalization)-flatten-dense1-dense2
D-emb-/
'''

def build_indicator_model(config):
    q_input = tf.keras.layers.Input(shape=(config['query_max_len'],), name='q_input')
    d_input = tf.keras.layers.Input(shape=(config['doc_max_len'],), name='d_input')

    indicator = Indicator(config['query_max_len'], config['doc_max_len'], name='indicator')((q_input, d_input))
    matrix = tf.keras.layers.Reshape((config['query_max_len'], config['doc_max_len'], 1), name='matrix')(indicator)

    for i in range(config['num_conv_layers']):
        matrix = tf.keras.layers.Conv2D(
            filters=config['filters'][i],
            kernel_size=config['kernel_size'][i],
            padding='same',
            activation='relu',
            name='conv_%d' % i)(matrix)
        matrix = tf.keras.layers.MaxPool2D(pool_size=tuple(config['pool_size']), name='max_pooling_%d' % i)(matrix)
        matrix = tf.keras.layers.BatchNormalization()(matrix)

    flatten = tf.keras.layers.Flatten()(matrix)
    dense = tf.keras.layers.Dense(32, activation='relu')(flatten)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='out')(dense)

    model = tf.keras.Model(inputs=[q_input, d_input], outputs=out)
    model.compile(
        loss='binary_crossenrtopy',
        optimizer='sgd',
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )
    return model