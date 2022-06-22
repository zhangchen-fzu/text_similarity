import tensorflow as tf


config={
    'query_max_len':8,
    'doc_max_len':16,
    'vocab_size':100000,
    'embedding_size':256,
    'num_conv_layers':1,
    'filters':[8],
    'kernel_size':[3],
    'pool_size':[2],
    'num_dense_layers':2,
    'dense_size':[128, 64]
}

def build_model(config):
    q_input = tf.keras.layers.Input(shape=(config['query_max_len'],), name='q_input')
    d_input = tf.keras.layers.Input(shape=(config['doc_max_len'],), name='d_input')

    embedding = tf.keras.layers.Embedding(config['vocab_size'], config['embedding_size'], name='embedding')
    q_embedding = embedding(q_input)
    d_embedding = embedding(d_input)

    for i in range(config['num_conv_layers']):
        q_embedding = tf.keras.layers.Conv1D(
            filters=config['filters'][i],
            kernel_size=config['kernel_size'][i],
            padding='valid',
            activation='relu',
            name='qconv_%d' %i)(q_embedding)
        q_embedding = tf.keras.layers.MaxPooling1D(pool_size=config['pool_size'][i], name='qmaxpool_%d' %i)(q_embedding)

        d_embedding = tf.keras.layers.Conv1D(
            filters=config['filters'][i],
            kernel_size=config['kernel_size'][i],
            padding='valid',
            activation='relu',
            name='dconv_%d' % i)(d_embedding)
        d_embedding = tf.keras.layers.MaxPooling1D(pool_size=config['pool_size'][i], name='dmaxpool_%d' % i)(d_embedding)

    q_flatten = tf.keras.layers.Flatten()(q_embedding)
    d_flatten = tf.keras.layers.Flatten()(d_embedding)

    q_d_concate = tf.keras.layers.Concatenate(axis=1)([q_flatten, d_flatten])

    dense = q_d_concate
    for j in range(config['num_dense_layers']):
        dense = tf.keras.layers.Dense(config['dense_size'][j], activation='relu')(dense)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

    model = tf.keras.Model(inputs=[q_input, d_input], outputs=out)
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )
    return model

