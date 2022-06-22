'''
CDSSM模型部分
'''

import tensorflow as tf



'''
模型结构：
Q-emb-LSTM-(conv-maxpooling)-dense\
                                   cosine-Dense
D-emb-LSTM-(conv-maxpooling)-sense/
'''
def build_model(config):
    q_input = tf.keras.layers.Input(shape=(config['query_max_len'],), name='q_input')
    d_input = tf.keras.layers.Input(shape=(config['doc_max_len'],), name='d_input')

    embedding = tf.keras.layers.Embedding(config['vocab_size'], config['embedding_size'], name='embedding')
    q_embedding = embedding(q_input)
    d_embedding = embedding(d_input)

    q_conv = tf.keras.layers.Conv1D(
        filters=config['filters'],
        kernel_size=config['kernel_size'], ##3行卷积一次
        strides=1,
        padding='same',
        activation='relu',
        kernel_initializer=config['w_initializer'],
        bias_initializer=config['b_initializer'])(q_embedding)
    q_pooling = tf.keras.layers.GlobalMaxPool1D()(q_conv)
    d_conv = tf.keras.layers.Conv1D(
        filters=config['filters'],
        kernel_size=config['kernel_size'],
        strides=config['strides'],
        padding=config['padding'],
        activation=config['conv_activation_func'],
        kernel_initializer=config['w_initializer'],
        bias_initializer=config['b_initializer'])(d_embedding)
    d_pooling = tf.keras.layers.GlobalMaxPool1D()(d_conv)

    q_dense = tf.keras.layers.Dense(32, activation='relu')(q_pooling)
    d_dense = tf.keras.layers.Dense(32, activation='relu')(d_pooling)

    cosine = tf.keras.layers.Dot(axes=-1, normalize=True, name='dot')([q_dense, d_dense])
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='out')(cosine)

    model = tf.keras.Model(inputs=[q_input, d_input], outputs=out)
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )
    return model

