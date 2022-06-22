'''
MNRM模型部分：
有kernel的引入，来对相似度矩阵做分层处理
'''

import tensorflow as tf

# config = {
# #     'query_max_len':8,
# #     'doc_max_len':16,
# #     'vocab_size':100000,
# #     'embedding_size':256,
# #     'kernel_num':11,
# #     'sigma':0.1,
# #     'exact_sigma':0.001
# # }

'''
KNRM模型结构：
Q-emb\
      Dot-kernel_pooling-dense
D-emb/
'''
def kernel_layer(mu, sigma):
    def kernel(x):
        return tf.math.exp(-0.5 * (x - mu) * (x - mu) / sigma / sigma)
    return tf.keras.layers.Activation(kernel)


def build_model(config):
    q_input = tf.keras.layers.Input(shape=(config['query_max_len'],), name='q_input')
    d_input = tf.keras.layers.Input(shape=(config['doc_max_len'],), name='d_input')

    embedding = tf.keras.layers.Embedding(config['vocab_size'], config['embedding_size'], name='embedding')
    q_embedding = embedding(q_input)
    d_embedding = embedding(d_input)

    matrix = tf.keras.layers.Dot(axes=-1, normalize=True, name='cosine')([q_embedding, d_embedding])

    KM = []
    for i in range(config['kernel_num']):
        mu = 1. / (config['kernel_num'] - 1) + (2. * i) / (config['kernel_num'] - 1) - 1.0
        sigma = config['sigma']
        if mu > 1.0:
            mu = 1.0
            sigma = config['exact_sigma']
        mm_exp = kernel_layer(mu, sigma)(matrix)
        mm_doc_sum = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, 2))(mm_exp)
        mm_log = tf.keras.layers.Activation(tf.math.log1p)(mm_doc_sum)
        mm_sum = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, 1))(mm_log)
        KM.append(mm_sum)
    phi = tf.keras.layers.Lambda(lambda  x: tf.stack(x, 1))(KM)

    out = tf.keras.layers.Dense(1, activation='sigmoid', name='out')(phi)

    model = tf.keras.Model(inputs=[q_input, d_input], outputs=out)
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )
    return model

# model = build_model(config)
# model.summary()