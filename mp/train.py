'''
模型训练部分：
1. 构建词表，方便将词语转化为id的形式
2. 将训练数据、验证数据转化为dataset类型
3. 将测试数据转化为dataset类型
4. 训练
5. 预测
6. 测试结果的输出
'''

import tensorflow as tf
import pandas as pd
from nlp_datasets.tokenizers import SpaceTokenizer
from textsimilarity.mp import model
import codecs
from sklearn.model_selection import train_test_split
import csv

config1 = {
    'matrix_tool': 'cosine',
    'query_max_len': 6,  # 调参
    'doc_max_len': 16,  # 调参
    'vocab_size': 0,
    'embedding_size': 256,  # 调参
    'vec_dim': 256,  # 调参
    'model_dir': '/content/gdrive/MyDrive/KDD_DATA/model_parameter',
    'prefetch_size': tf.data.experimental.AUTOTUNE,
    'num_parallel_calls': tf.data.experimental.AUTOTUNE,
    'buffer_size': 10000000,
    'seed': None,
    'reshuffle_each_iteration': True,
    'train_batch_size': 8,  # 调参
    'test_batch_size': 8,  # 调参
    'bucket_width': 5,
    'padding_by_eos': False,
    'model_name': 'dssm',
    'ckpt_only_weights': True,
    'ckpt_period': 1,
    'early_stopping_patience': 10,  # 调参
    'epoch': 1,  # 调参
    'num_conv_layers': 3,
    'filters': [8, 16, 32], #kernel的层数
    'kernel_size': [[5, 5], [3, 3], [3, 3]], ##kernel的大小
    'pool_size': [[2, 2], [2, 2], [2, 2]],  ##池化的窗口，注意你的Q及D的长度要够pool才行。比如这个，最大长度要大于等于8才行
    'dropout': 0.5,
    'batch_size': 32
}


##构建词表，用于转化为id的形式
def word_txt(train_path, test_path, word_path):
    word_lst = set()
    ##拿到训练数据的词表
    # train_data = train_data.reset_index()
    train_data = pd.read_csv(train_path, keep_default_na=False)
    for i in range(len(train_data)):
        query = train_data['query'][i].strip().split()
        for w1 in query:
            word_lst.add(w1)
        doc = train_data['doc'][i].strip().split()
        for w2 in doc:
            word_lst.add(w2)
    ##拿到测试数据的词表
    test_data = pd.read_csv(test_path, keep_default_na=False)
    for j in range(len(test_data)):
        query = test_data['query'][j].strip().split()
        for w1 in query:
            word_lst.add(w1)
        doc = test_data['doc'][j].strip().split()
        for w2 in doc:
            word_lst.add(w2)
    f = codecs.open(word_path, 'w', 'utf-8')
    for w3 in word_lst:
        f.write(w3 + '\n')
    f.close()
    return word_lst


##根据词表构建hash词表，将word转化为id的形式。输入：词表，由build_train_dataset那边传过来的处理好的词语集合
def convert_word2id(word_table):
    tokenizer = SpaceTokenizer()
    tokenizer.build_from_vocab(word_table)
    config1['vocab_size'] = tokenizer.vocab_size
    return tokenizer


# 将字符型的z转化为数值型的z
def norm_z(z):
    if tf.equal(z, '0'):
        return tf.constant(0, dtype=tf.dtypes.int64)
    if tf.equal(z, '1'):
        return tf.constant(1, dtype=tf.dtypes.int64)
    return tf.cast(tf.strings.to_number(z), dtype=tf.dtypes.int64)


# 处理训练数据，转化为模型的输入形式
def build_train_dataset(train_data_path, tokenizer):
    dataset = tf.data.Dataset.from_tensor_slices([train_data_path])
    dataset = dataset.flat_map(lambda x: tf.data.TextLineDataset(x).skip(1))
    ##过滤不合格的样本，即通过,分割之后，不是qdl三段数据
    dataset = dataset.filter(
        lambda x: tf.equal(3, tf.size(tf.strings.split([x], sep=',').values)))
    ##将数据映射为三元组的形式
    dataset = dataset.map(
        lambda x: (tf.strings.split(x, ",")[0],
                   tf.strings.split(x, ",")[1],
                   tf.strings.split(x, ",")[-1]),
        num_parallel_calls=config1['num_parallel_calls']
    ).prefetch(config1['prefetch_size'])
    ##打乱数据
    dataset = dataset.shuffle(
        buffer_size=config1['buffer_size'],
        seed=config1['seed'],
        reshuffle_each_iteration=config1['reshuffle_each_iteration'])
    ##根据空格分离数据，并对label进行数值化的处理
    dataset = dataset.map(
        lambda x, y, z: (tf.strings.split([x], ' ').values,
                         tf.strings.split([y], ' ').values,
                         norm_z(z)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).prefetch(tf.data.experimental.AUTOTUNE)
    ##截取长度，q要多少长度，d要多少长度。
    dataset = dataset.map(
        lambda q, d, l: (q[:config1['query_max_len']], d[:config1['doc_max_len']], l),
        num_parallel_calls=config1['num_parallel_calls']
    ).prefetch(tf.data.experimental.AUTOTUNE)
    # ##删除过于长的样本数据，如果加上收尾的话，输入的长度要加2
    # dataset = filter_sequence(dataset)
    ##将字符根据hash词表，映射为数值的类型
    dataset = dataset.map(
        lambda x, y, z: (tokenizer.encode(x), tokenizer.encode(y), z),
        num_parallel_calls=config1['num_parallel_calls']
    ).prefetch(config1['prefetch_size'])
    ##添加每句话的开始和结束标记
    # dataset = add_special_tokens(dataset, tokenizer)
    ##训练集的batch及pad操作
    dataset = dataset.padded_batch(
        batch_size=config1['train_batch_size'],  # 每批次的样本数量
        padded_shapes=([config1['query_max_len']], [config1['doc_max_len']], []),
        padding_values=(tf.constant(tokenizer.unk_id, dtype=tf.dtypes.int64),
                        tf.constant(tokenizer.unk_id, dtype=tf.dtypes.int64),
                        tf.constant(0, dtype=tf.int64))
    )
    ##输入和输出的控制
    dataset = dataset.map(
        lambda q, d, l: [(q, d), l],
        num_parallel_calls=config1['num_parallel_calls']
    ).prefetch(config1['prefetch_size'])
    return dataset


# 将test的那部分df类型的三列数据变成三元组的形式：类似于（[第一列], [第二列], [第三列]），并转换为id的形式
def build_test_dataset(test_data_path, tokenizer):
    dataset = tf.data.Dataset.from_tensor_slices([test_data_path])
    dataset = dataset.flat_map(lambda x: tf.data.TextLineDataset(x).skip(1))

    ##将数据映射为三元组的形式
    dataset = dataset.map(
        lambda x: (tf.strings.split(x, ",")[0],
                   tf.strings.split(x, ",")[1],
                   tf.strings.split(x, ",")[-1]),
        num_parallel_calls=config1['num_parallel_calls']
    ).prefetch(config1['prefetch_size'])
    # dataset = dataset.map(
    #     lambda x: (tf.strings.split(x, ",")[0],
    #                tf.strings.split(x, ",")[1]),
    #     num_parallel_calls=config1['num_parallel_calls']
    # ).prefetch(config1['prefetch_size'])
    dataset = dataset.map(
        lambda x, y, z: (tf.strings.split([x], '　').values,
                         tf.strings.split([y], '　').values,
                         norm_z(z)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        lambda q, d, l: (q[:config1['query_max_len']], d[:config1['doc_max_len']], l),
        num_parallel_calls=config1['num_parallel_calls']
    ).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        lambda x, y, z: (tokenizer.encode(x), tokenizer.encode(y), z),
        num_parallel_calls=config1['num_parallel_calls']
    ).prefetch(config1['prefetch_size'])
    dataset = dataset.padded_batch(
        batch_size=config1['train_batch_size'],  # 每批次的样本数量
        padded_shapes=([config1['query_max_len']], [config1['doc_max_len']], []),
        padding_values=(tf.constant(tokenizer.unk_id, dtype=tf.dtypes.int64),
                        tf.constant(tokenizer.unk_id, dtype=tf.dtypes.int64),
                        tf.constant(0, dtype=tf.int64))
    )
    ##输入和输出的控制
    dataset = dataset.map(
        lambda q, d, l: [(q, d), l],
        num_parallel_calls=config1['num_parallel_calls']
    ).prefetch(config1['prefetch_size'])

    # dataset = dataset.map(
    #     lambda q, d, l: (q, d),
    #     num_parallel_calls=config1['num_parallel_calls']
    # ).prefetch(tf.data.experimental.AUTOTUNE)
    # dataset = dataset.padded_batch(
    #     batch_size=config1['train_batch_size'],  # 每批次的样本数量
    #     padded_shapes=([config1['query_max_len']], [config1['doc_max_len']]),
    #     padding_values=(tf.constant(tokenizer.unk_id, dtype=tf.dtypes.int64),
    #                 tf.constant(tokenizer.unk_id, dtype=tf.dtypes.int64))
    # )
    ##输入和输出的控制
    # dataset = dataset.map(lambda q, d, l: [(q, d),l])
    return dataset


def out_process(outputs, test):
    test_data = pd.read_csv(test, keep_default_na=False)
    query_id = test_data['query_id']
    product_id = test_data['product_id']
    true_len = len(test_data)
    out_val = [list(v)[0] for v in outputs][:true_len]  ##batch带来的长度误差
    out_val = pd.DataFrame(out_val, columns=['out_val'])
    res_data = pd.concat([product_id, query_id, out_val], axis=1)
    sub_res_date = res_data.groupby("query_id")

    with open(r'D:\KDD相关\任务1数据集\res.csv', 'w', newline='') as f:
        fieldnames = ["product_id", "query_id"]
        writer = csv.writer(f)
        writer.writerow(fieldnames)

    for query_id, group in sub_res_date:
        group = group.sort_values(by=['out_val'], ascending=False)
        p_id = group['product_id']
        q_id = group['query_id']
        tmp = pd.concat([p_id, q_id], axis=1)
        tmp.to_csv(r'D:\KDD相关\任务1数据集\res.csv', index=False, encoding='utf-8', mode='a', header=0)
    return 'end'


if __name__ == '__main__':
    ## --------------------#读取数据
    train_path = r'D:\KDD相关\任务1数据集\new-train-v0.2.csv'
    test_path = r'D:\KDD相关\任务1数据集\new-test-v0.2.csv'
    word_path = r'D:\KDD相关\任务1数据集\word.txt'
    ## --------------------#将分割好的数据写入csv中
    new_train_path = r'D:\KDD相关\任务1数据集\train.csv'
    new_val_path = r'D:\KDD相关\任务1数据集\val.csv'
    new_test_path = r'D:\KDD相关\任务1数据集\test.csv'
    ## --------------------#构建词表，word2id与id2word
    word_txt(train_path, test_path, word_path)  # 创建词表
    tokenizer = convert_word2id(word_path)  # 根据词表，将词语转化为id的形式
    ## --------------------#切分数据，此时结果是df类型的！
    all_train_data = pd.read_csv(train_path, keep_default_na=False)
    train, val = train_test_split(all_train_data, test_size=0.2)  # train-val = 8:2
    train, test = train_test_split(train, test_size=0.2)
    train.to_csv(new_train_path, index=False, encoding='utf-8')
    val.to_csv(new_val_path, index=False, encoding='utf-8')
    test.to_csv(new_test_path, index=False, encoding='utf-8')
    ## --------------------#构建模型需要的输入，训练集、验证集、测试集的转化
    train_ds = build_train_dataset(new_train_path, tokenizer)
    val_ds = build_train_dataset(new_val_path, tokenizer)
    test_ds = build_test_dataset(new_test_path, tokenizer)  ##有l
    ## --------------------#加载模型
    if 'dot' == config1['matrix_tool']:
        model = model.build_dot_model(config1)
    elif 'cosine' == config1['matrix_tool']:
        model = model.build_cosine_model(config1)
    elif 'indicator' == config1['matrix_tool']:
        model = model.build_indicator_model(config1)
    else:
        raise ValueError('Invalid model')
    ## --------------------#训练模型
    cp_callback = [tf.keras.callbacks.EarlyStopping(patience=config1['early_stopping_patience'])]
    model.fit(train_ds, validation_data=val_ds, epochs=config1['epoch'], callbacks=cp_callback)
    ## --------------------#预测
    test_out = model.evaluate(test_ds)
    print(test_out)