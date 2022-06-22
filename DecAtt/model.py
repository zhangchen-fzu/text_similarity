'''
DecAtt模型部分：
有注意力机制的使用
DecAtt是ESIM的先驱
'''
import tensorflow as tf
from textsimilarity.DecAtt.attention_part import SoftAttention

# config = {
#     'query_max_len':8,
#     'doc_max_len':16,
#     'vocab_size':100000,
#     'embedding_size':256,
#     'lstm_unit':64,
#     'q_d_dense_size':256,
#     'num_dense_layer':2,
#     'dense_size':[256, 128]
# }

'''
模型结构：

Q-emb --------- D_out(由两种输出concat组成)-dense-矩阵第一维度求和   
     \         /                                              \
      attention                                                 concat-dense1-dense2-dense3
     /         \                                              /
D-emb --------- Q_out(由两种输出concat组成)-dense-矩阵第一维度求和

'''

def build_model(config):
    q_input = tf.keras.layers.Input(shape=(config['query_max_len'],), name='q_input')
    d_input = tf.keras.layers.Input(shape=(config['doc_max_len'],), name='d_input')

    embedding = tf.keras.layers.Embedding(config['vocab_size'], config['embedding_size'], name='embedding')
    q_enbedding = embedding(q_input)
    d_embedding = embedding(d_input)

    q_attention_out, d_attention_out = SoftAttention()([q_enbedding, d_embedding])
    q_out = tf.keras.layers.concatenate([q_enbedding, q_attention_out])
    d_out = tf.keras.layers.concatenate([d_embedding, d_attention_out])

    q_dense = tf.keras.layers.Dense(config['q_d_dense_size'])(q_out)
    d_dense = tf.keras.layers.Dense(config['q_d_dense_size'])(d_out)

    q_sum = tf.reduce_sum(q_dense, axis=1)
    d_sum = tf.reduce_sum(d_dense, axis=1)
    concat = tf.keras.layers.concatenate([q_sum, d_sum])

    dense = concat
    for i in range(config['num_dense_layer']):
        dense = tf.keras.layers.Dense(config['dense_size'][i], activation='relu')(dense)

    out = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

    model = tf.keras.Model(inputs=[q_input, d_input], outputs=out)
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )
    return model

# model = build_model(config)
# model.summary()