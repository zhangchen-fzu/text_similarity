'''
ESIM模型部分：
模型attention部分的解释可参考：https://zhuanlan.zhihu.com/p/86978155
'''

import tensorflow as tf
from textsimilarity.esim.attention_part import SoftAttention

# config = {
#     'query_max_len':8,
#     'doc_max_len':16,
#     'vocab_size':100000,
#     'embedding_size':256,
#     'lstm_unit':64,
#     'num_dense_layer':2,
#     'dense_size':[256, 128]
# }


'''
模型结构：

Q-emb-bilstm --------- D_out(由四种输出concat组成)-bilstm-pool(由两种pool组成)     
            \         /                                                   \
             attention                                                     concat-dense1-dense2-dense3
            /         \                                                   /
D-emb-bilstm --------- Q_out(由四种输出concat组成)-bilstm-pool(由两种pool组成)
          
'''
def build_model(config):
    q_input = tf.keras.layers.Input(shape=(config['query_max_len'],), name='q_input')
    d_input = tf.keras.layers.Input(shape=(config['doc_max_len'],), name='d_input')

    embedding = tf.keras.layers.Embedding(config['vocab_size'], config['embedding_size'], name='embedding')
    q_embedding = embedding(q_input)
    d_embedding = embedding(d_input)

    bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config['lstm_unit'], return_sequences=True))
    q_lstm = bilstm(q_embedding)
    d_lstm = bilstm(d_embedding)

    atten_a, atten_b = SoftAttention()([q_lstm, d_lstm]) #[batch* 8 * 128]-----[batch* 16 * 128]
    sub_a_atten = tf.keras.layers.Lambda(lambda x: x[0] - x[1])([q_lstm, atten_a])
    sub_b_atten = tf.keras.layers.Lambda(lambda x: x[0] - x[1])([d_lstm, atten_b])
    mul_a_atten = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([q_lstm, atten_a])
    mul_b_atten = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([d_lstm, atten_b])
    m_a = tf.keras.layers.concatenate([q_lstm, atten_a, sub_a_atten, mul_a_atten])
    m_b = tf.keras.layers.concatenate([d_lstm, atten_b, sub_b_atten, mul_b_atten])

    q_lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config['lstm_unit'], return_sequences=True))(m_a)
    q_avg_pool = tf.keras.layers.GlobalAveragePooling1D()(q_lstm1)
    q_max_pool = tf.keras.layers.GlobalMaxPooling1D()(q_lstm1)
    d_lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config['lstm_unit'], return_sequences=True))(m_b)
    d_avg_pool = tf.keras.layers.GlobalAveragePooling1D()(d_lstm1)
    d_max_pool = tf.keras.layers.GlobalMaxPooling1D()(d_lstm1)
    concate = tf.keras.layers.concatenate([q_avg_pool, q_max_pool, d_avg_pool, d_max_pool])

    dense = concate
    for i in range(config['num_dense_layer']):
        dense = tf.keras.layers.Dense(config['dense_size'][i], activation='relu', name='dense_%d' %i)(dense)

    out = tf.keras.layers.Dense(1, activation='sigmoid', name='out')(dense)

    model = tf.keras.Model(inputs=[q_input, d_input], outputs=out)
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )
    return model

#
# model = build_model(config)
# model.summary()








