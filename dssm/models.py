'''
网络结构部分：
1. MLP结构
Q-emb-avg_emb-Dense1-Dense2\
                            Dot-Dense
D-emb-avg_emb-Dense1-Dense2/

2. LSTM结构
Q-emb-LSTM-Dense1\
                  Dot-Dense
D-emb-LSTM-Dense1/
'''
import tensorflow as tf

def build_mlp_model(config):
    query = tf.keras.layers.Input(shape=(config['query_max_len'],), dtype=tf.int64, name='query')
    doc = tf.keras.layers.Input(shape=(config['doc_max_len'],), dtype=tf.int64, name='doc')
    embedding = tf.keras.layers.Embedding(config['vocab_size'], config['embedding_size'], name='embedding')
    avg_embedding = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, 1))
    query_embed = embedding(query)
    query_embed = avg_embedding(query_embed)
    doc_embed = embedding(doc)
    doc_embed = avg_embedding(doc_embed)
    query_dense0 = tf.keras.layers.Dense(1024, activation='tanh')(query_embed)
    query_vec = tf.keras.layers.Dense(config['vec_dim'], activation='tanh', name='query_vec')(query_dense0)
    doc_dense0 = tf.keras.layers.Dense(1024, activation='tanh')(doc_embed)
    doc_vec = tf.keras.layers.Dense(config['vec_dim'], activation='tanh', name='doc_vec')(doc_dense0)
    cos = tf.keras.layers.Dot(axes=-1, normalize=True, name='cosine')([query_vec, doc_vec])
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='out')(cos)
    model = tf.keras.Model(inputs=[query, doc], outputs=[out])
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

def build_lstm_model(config):
    query = tf.keras.Input(shape=(config['query_max_len'],), dtype=tf.int64, name='query_input')
    doc = tf.keras.Input(shape=(config['doc_max_len'],), dtype=tf.int64, name='doc_input')
    embedding = tf.keras.layers.Embedding(config['vocab_size'], config['embedding_size'], name='embedding')
    query_embed = embedding(query)
    doc_embed = embedding(doc)
    query_lstm = tf.keras.layers.LSTM(256)(query_embed)
    doc_lstm = tf.keras.layers.LSTM(256)(doc_embed)
    query_vec = tf.keras.layers.Dense(config['vec_dim'])(query_lstm)
    doc_vec = tf.keras.layers.Dense(config['vec_dim'])(doc_lstm)
    cos = tf.keras.layers.Dot(axes=-1, normalize=True, name='cosine')([query_vec, doc_vec])
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='out')(cos)
    model = tf.keras.Model(inputs=[query, doc], outputs=out)
    model.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model
