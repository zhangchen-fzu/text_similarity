'''
TextCNN模型部分:
textcnn本身为文本分类模型，现在只是借用分类之前形成的文本向量部分，来做
文本匹配任务dot之前的部分。
'''

import tensorflow as tf



'''
模型结构：
               
             /conv3-maxpool\
Q-emb-reshape-conv4-maxpool-concate-flatten\
             \conv5-maxpool/
                                             Dot-dense
             /conv3-maxpool\
D-emb-reshape-conv4-maxpool-concate-flatten/
             \conv5-maxpool/
'''
def build_model(config):
    q_input = tf.keras.layers.Input(shape=(config['query_max_len'],), name='q_input')
    d_input = tf.keras.layers.Input(shape=(config['doc_max_len'],), name='d_input')

    embedding = tf.keras.layers.Embedding(config['vocab_size'], config['embedding_size'], name='embedding')
    q_embedding = embedding(q_input)
    d_enbedding = embedding(d_input)

    q_embedding = tf.keras.layers.Reshape((config['query_max_len'], config['embedding_size'], 1))(q_embedding)
    d_enbedding = tf.keras.layers.Reshape((config['doc_max_len'], config['embedding_size'], 1))(d_enbedding)

    q_out, d_out = [], []
    for i in range(config['num_conv_layers']): ##3种卷积，每种两个
        q_conv = tf.keras.layers.Conv2D(
            filters=config['filters'],
            kernel_size=(config['kernel_size'][i], config['embedding_size']),
            activation='relu',
            padding='valid', name='q_conv_%d' % i)(q_embedding)
        pool_size = (config['query_max_len'] - config['kernel_size'][i] + 1, 1)
        q_pool = tf.keras.layers.MaxPool2D(pool_size=pool_size, name='q_maxpool_%d' % i)(q_conv)
        q_out.append(q_pool)

        d_conv = tf.keras.layers.Conv2D(
            filters=config['filters'],
            kernel_size=(config['kernel_size'][i], config['embedding_size']),
            activation='relu',
            padding='valid', name='d_conv_%d' % i)(d_enbedding)
        pool_size = (config['doc_max_len'] - config['kernel_size'][i] + 1, 1)
        d_pool = tf.keras.layers.MaxPool2D(pool_size=pool_size, name='d_maxpool_%d' % i)(d_conv)
        d_out.append(d_pool)

    q_connect = tf.keras.layers.concatenate(q_out, axis=-1)
    q_flatten = tf.keras.layers.Flatten()(q_connect)
    d_connect = tf.keras.layers.concatenate(d_out, axis=-1)
    d_flatten = tf.keras.layers.Flatten()(d_connect)

    dot = tf.keras.layers.Dot(axes=-1, normalize=True, name='cosine')([q_flatten, d_flatten])
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='out')(dot)

    model = tf.keras.Model(inputs=[q_input, d_input], outputs=out)
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )

    return model