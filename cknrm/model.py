'''
Conv_KNRM模型部分：
与KNRM相比，在kernel-pooling层之前加入了多种卷积操作，以模拟n-gram过程
'''

import tensorflow as tf

config = {
    'query_max_len':8,
    'doc_max_len':16,
    'vocab_size':100000,
    'embedding_size':256,
    'kernel_num':11,
    'sigma':0.1,
    'exact_sigma':0.001,
    'max_ngram':3,
    'filters':128
}

'''
Conv_KNRM模型结构：
Q-emb-多种CNN\
             多种CNN结果矩阵两两组合形成多组矩阵-每组矩阵cosine-每组矩阵kernel_pooling-所有组的输出concat-dense
D-emb-多种CNN/
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

    q_conv = []
    d_conv = []
    for i in range(config['max_ngram']):
        conv = tf.keras.layers.Conv1D(
            filters=config['filters'],
            kernel_size=i + 1,
            padding='same',
            activation='relu'
        )
        q_conv.append(conv(q_embedding))
        d_conv.append(conv(d_embedding))

    KM = []
    for q in range(len(q_conv)):
        for d in range(len(d_conv)):
            q_matrix = q_conv[q]
            d_matrix = d_conv[d]

            cosine = tf.keras.layers.Dot(axes=-1, normalize=True)([q_matrix, d_matrix])

            for j in range(config['kernel_num']):
                mu = 1. / config['kernel_num'] - 1 + 2. * j / config['kernel_num'] - 1 + 1.0
                sigma = config['sigma']
                if mu > 1.0:
                    mu = 1.0
                    sigma = config['exact_sigma']
                mm = kernel_layer(mu, sigma)(cosine)
                mm_doc_sum = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, 2))(mm)
                mm_log = tf.keras.layers.Activation(tf.math.log1p)(mm_doc_sum)
                mm_sum = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, 1))(mm_log)
                KM.append(mm_sum)
    phi = tf.keras.layers.Lambda(lambda x: tf.stack(x, 1))(KM)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(phi)

    model = tf.keras.Model(inputs=[q_input, d_input], outputs=out)
    model.compile(
        loss='binary_cross_entropy',
        optimizer='sgd',
        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.Accuracy()]
    )
    return model

model = build_model(config)
model.summary()