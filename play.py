# import MeCab
import itertools
import collections
import pandas as pd
# mecab = MeCab.Tagger ("-Owakati")

# a = [1, 2, 3, 4, 3]
# b = a.count(0)
# print(b)



# strs = ''
# lst = mecab.parse(strs).split()
# print(lst)

#

# res = []
# dics = {'a':['1', '2'], 'b':['3', '2', '4', '2']}
# df = pd.DataFrame.from_dict({k: ' '.join(v) for k, v in dics.items()}, orient='index', columns=['product_id'])
# df = df.reset_index().rename(columns = {'index':'word'})
# df.to_csv(r'D:\KDD相关\任务1数据集\aaa.csv', index=False, encoding='utf-8')
# # from bs4 import BeautifulSoup
# import re
# def clean(desstr, restr=''):
#     soup = BeautifulSoup(desstr, 'html.parser')
#     strs = soup.get_text()
#     try:
#         co = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]+')
#     except re.error:
#         co = re.compile(u'('u'\ud83c[\udf00-\udfff]|'u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'u'[\u2600-\u2B55])+')
#     return co.sub(restr, strs)
# print(clean(''))

##清洗product的文本，并构建词表
# def create_product_word_lst():
#     stop_words = set([line.strip() for line in open(r'D:\KDD相关\任务1数据集\停用词.txt', encoding='utf-8').readlines()])
#     word_dic = {'es':set(), 'us':set(), 'jp':set()}  ##相应语言的词表
#     product_data = pd.read_csv(r'D:\KDD相关\任务1数据集\product_catalogue-v0.1.csv')
#     n_word = 0
#     for i in range(len(product_data)):
#         strs =  str(product_data['product_title'][i])
#         if not strs:
#             strs = str(product_data['product_description'][i])
#         if not strs:
#             strs = str(product_data['product_bullet_point'][i])
#         strs = clean(strs)
#         local = product_data['product_locale'][i]
#         if local != 'jp': ##英西直接根据空格分词
#             lst = strs.strip().split()
#             for v in lst:
#                 if v not in stop_words:
#                     n_word += 1
#                     word_dic[local].add(v)
#         else: ##日调用分词软件分词
#             lst = mecab.parse(strs).split()
#             for v in lst:
#                 if v not in stop_words:
#                     n_word += 1
#                     word_dic[local].add(v)
#     print("词表总词数：", n_word)  ##词表总词数： 116,231,275 构建倒排索引计算量太大  13,910,047
#     return word_dic

##词所对应的商品id列表（倒排索引）
# def create_query_productid_dic():
#     product_data = pd.read_csv(r'D:\KDD相关\任务1数据集\product_catalogue-v0.1.csv')
#     word_dic = create_product_word_lst()
#     word_productid_dic = {} ##词语及词语对应的product_id列表
#     indx = 0
#     for local in word_dic:  ##时间复杂度：3*word的数量*1*某语言的句子（word的数量*所有句子）
#         word_lst = word_dic[local] ##某语言的词表
#         sub_data = product_data.groupby('product_locale')
#         for word in word_lst:
#             indx += 1
#             if indx % 100000 == 0:
#                 print(indx)
#             for l, group in sub_data:
#                 if l == local:
#                     group1 = group.reset_index()
#                     for i in range(len(group1)):
#                         strs = str(product_data['product_title'][i])
#                         if not strs:
#                             strs = str(product_data['product_description'][i])
#                         if not strs:
#                             strs = str(product_data['product_bullet_point'][i])
#                         strs = clean(strs)
#                         lst = strs.split()
#                         if word in lst:
#                             word_productid_dic.setdefault(word, set()).add(group1['product_id'][i])
#     df = pd.DataFrame.from_dict({k: ' '.join(v) for k, v in word_productid_dic.items()}, orient='index', columns=['product_id'])
#     df.to_csv(r'D:\KDD相关\任务1数据集\word_productid_dic.csv', encoding='utf-8', index=False)
#     return word_productid_dic


# for i in range(len(test_data)): ##进入query
#     query = test_data['query'][i]
#
#     id_score_dic = {}
#     for j in range(len(product_data)):  ##进入product
#         if product_data['product_locale'][j] == test_data['query_locale'][i]:
#
#
#             dl_lst, avgdl = avg_len_procuct() ##
#             dl = dl_lst[j] ##
#             score_q_d = 0
#             for word in query_lst:
#                 nqi = 0  ##
#                 if word in word_id_dic:
#                     nqi = len(word_id_dic[word])
#                 fi = doc_lst.count(word)   ##
#                 part1 = log((N - nqi + 0.5) / (nqi + 0.5))
#                 part2 = fi * (k1 + 1) / (fi + k1 * (1 - b + b * (dl / avgdl)))
#                 score_q_d += (part1 * part2)
#             id_score_dic[product_data['product_id'][j]] = score_q_d
#     sort_id_score_dic = sorted(id_score_dic.items(), key = lambda s:s[1], reverse=True)
#     for n in range(10):
#         top_10_product_id.append(sort_id_score_dic[n][0])
#         query_id.append(test_data['query_id'][i])
#         queryid_productid_dic.setdefault(test_data['query_id'][i], []).append(sort_id_score_dic[n][0])
# top_10_product_id = pd.DataFrame(top_10_product_id, columns=['product_id'])
# query_id = pd.DataFrame(query_id, columns=['query_id'])
# res = pd.concat([top_10_product_id, query_id], axis=1)
# res.to_csv(r'D:\KDD相关\任务1数据集\submit.csv', index=False, encoding='utf-8')
# return queryid_productid_dic

#
# from nlp_datasets.seq_match.seq_match_dataset import SeqMatchDataset

# 参数们：
'''
'buffer_size': 10000000,  重复
'seed': None,  重复
'reshuffle_each_iteration': True,   重复
'prefetch_size': tf.data.experimental.AUTOTUNE, 
'num_parallel_calls': tf.data.experimental.AUTOTUNE,
'add_sos': True,
'add_eos': True,
'skip_count': 0,
'padding_by_eos': False,
'drop_remainder': True,
'bucket_width': 10,
'train_batch_size': 32,  重复
'eval_batch_size': 32,  重复
'predict_batch_size': 32,  重复
'repeat': 1,
-----
'sep': '@',
'num_parallel_calls': 1,
'buffer_size': 1000,   重复
'seed': None,  重复
'reshuffle_each_iteration': True,  重复
'train_batch_size': 2,  重复
'eval_predict_size': 2,  重复
'predict_batch_size': 2,   重复
'query_max_len': 5,
'doc_max_len': 5,
'vocab_file': 'data/vocab.txt', 
'train_files': ['data/train.txt'], 
'eval_files': ['data/train.txt'], 
'predict_files': ['data/train.txt'] 
-----
'vocab_size': 10,
'embedding_size': 256,
'vec_dim': 256, 
-----
'ckpt_period': 1, 
'model_dir': '/tmp/dssm' 
-----
'xyz_sep': '@',
'sep': ' ', 
'x_max_len': -1,
'y_max_len': -1,
'''
# files = ['‪C:/Users/admin\Desktop/train1.txt']
# dataset = tf.data.Dataset.from_tensor_slices(files)

# import MeCab
# mecab = MeCab.Tagger ("-Owakati")
# sentence = '無印良品 体にフィットするソファー用綿デニムカバー ネイビー 44105634'
# print(mecab.parse(sentence))
# def len_control(word_lst, max_len, pad_token):
#     m = len(word_lst)
#     if m > max_len:
#         word_lst = word_lst[:max_len]
#     elif m < max_len:
#         cha = max_len - m
#         padding = [pad_token] * cha
#         word_lst += padding
#     return ' '.join(word_lst)
# print(len_control(['dqe', 'rte', 'qee', 'dqe', 'rte', 'qee'], 5, 'unk'))




'''
罗马数字包含以下七种字符： I， V， X， L，C，D 和 M。
字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。


通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。
数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：

I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 
C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
给你一个整数，将其转为罗马数字。
输入: num = 1994
输出: "MCMXCIV"
解释: M = 1000, CM = 900, XC = 90, IV = 4.

'''


# def roma_num(num):
#     lst1 = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
#     lst2 = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
#     res = ''
#     for i in range(len(lst1)):
#         if num // lst1[i] != 0:
#             n, num = divmod(num, lst1[i])
#             res += lst2[i] * n
#     return res
# num = 1001
# print(roma_num(num))

'''
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

网格中的障碍物和空位置分别用 1 和 0 来表示。

说明：m 和 n 的值均不超过 100。

示例 1:

输入:
[
  [0,0,0],
  [0,1,0],
  [0,0,0]
]
输出: 2
解释:
3x3 网格的正中间有一个障碍物。
从左上角到右下角一共有 2 条不同的路径：
1. 向右 -> 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右 -> 向右


'''


# def uniquePathsWithObstacles (obstacleGrid):
#     m = len(obstacleGrid)
#     n = len(obstacleGrid[0])
#     dp = [[1] * n for _ in range(m)]
#     if obstacleGrid[0][0] == 1:
#         return 0
#     for i in range(1, n):
#         if obstacleGrid[0][i] == 1 or obstacleGrid[0][i - 1] == 0:
#             dp[0][i] = 0
#     for j in range(1, m):
#         if obstacleGrid[j][0] == 1 or obstacleGrid[j - 1][0] == 0:
#             dp[j][0] = 0
#
#     for u in range(1, m):
#         for v in range(1, n):
#             if obstacleGrid[u][v] == 1:
#                 dp[u][v] = 0
#             else:
#                 dp[u][v] = dp[u - 1][v] + dp[u][v - 1]
#     return dp[m - 1][n - 1]


# def split_line(line):
#    arr = tf.strings.split(line, "\t")
#    query = tf.expand_dims(arr[0], axis=0)
#    doc = tf.expand_dims(arr[1], axis=0)
#    label = tf.expand_dims(tf.cast(tf.strings.to_number(arr[2]), tf.int32), axis=0)
#    return (query, doc, label)
# def split_line(line):
#    arr = tf.strings.split(line, "\t")
#    query = arr[0]
#    doc = arr[1]
#    label = arr[2]
#    return (query, doc, label)


# from nlp_datasets.tokenizers import SpaceTokenizer
# import MeCab
# mecab = MeCab.Tagger ("-Owakati")
#
#
# config1 = {
#     'feature_tool':'mlp',
#     'query_max_len': 5,
#     'doc_max_len': 5,
#     'vocab_size': 10,
#     'embedding_size': 256,
#     'vec_dim': 256,
#     'model_dir':'D:/KDD相关/DSSM',
#     'prefetch_size': tf.data.experimental.AUTOTUNE,
#     'num_parallel_calls': tf.data.experimental.AUTOTUNE,
#     'buffer_size': 10000000,
#     'seed': None,
#     'reshuffle_each_iteration': True
#
# }
#
# def norm(z):
#     if tf.equal(z, '0'):
#         return tf.constant(0, dtype=tf.dtypes.int64)
#     if tf.equal(z, '1'):
#         return tf.constant(1, dtype=tf.dtypes.int64)
#     return tf.cast(tf.strings.to_number(z), dtype=tf.dtypes.int64)
#
# def convert_word2id(word_table, dataset):
#     ##构建hash词表
#     tokenizer = SpaceTokenizer()
#     tokenizer.build_from_vocab(word_table)  # 读取词表，创建w2i及i2w的哈希表，最终的输出形式是啥？列表吗？
#     ##word2id的映射map
#     dataset = dataset.map(
#         lambda x, y, z: (tokenizer.encode(x), tokenizer.encode(y), z),
#         num_parallel_calls=config1['num_parallel_calls']
#     ).prefetch(config1['prefetch_size'])
#     return dataset
#
# def build_train_dataset(train_data_path, word_table):
#     dataset = tf.data.TextLineDataset(filenames=[train_data_path])
#     ##将数据映射为三元组的形式
#     dataset = dataset.map(
#         lambda x: (tf.strings.split(x, "@")[0],
#                    tf.strings.split(x, "@")[1],
#                    tf.strings.split(x, "@")[2]),
#         num_parallel_calls=config1['num_parallel_calls']
#     ).prefetch(config1['prefetch_size'])
#     ##打乱数据
#     dataset = dataset.shuffle(
#             buffer_size=config1['buffer_size'],
#             seed=config1['seed'],
#             reshuffle_each_iteration=config1['reshuffle_each_iteration'])
#     ##根据空格分离数据，并对label进行数值化的处理
#     dataset = dataset.map(
#         lambda x, y, z: (tf.strings.split([x], ' ').values,
#                          tf.strings.split([y], ' ').values,
#                          norm(z)),
#         num_parallel_calls=tf.data.experimental.AUTOTUNE
#     ).prefetch(tf.data.experimental.AUTOTUNE)
#     ##将字符根据hash词表，映射为数值的类型
#     dataset = convert_word2id(word_table, dataset)
#     return dataset

# train_all_data = r'C:/Users/admin/Desktop/train1.txt'
# word_path = r'C:/Users/admin/Desktop/vocab.txt'
# print(build_train_dataset(train_all_data, word_path))

# dataset =  tf.data.TextLineDataset(filenames = ['D:/KDD相关/任务1数据集/play/xugou.csv'])
# dataset = dataset.map(
#                 lambda x: (tf.strings.split(x, "\t")[0],
#                            tf.strings.split(x, "\t")[1],
#                            tf.strings.split(x, "\t")[2]),
#                 num_parallel_calls=tf.data.experimental.AUTOTUNE
#             ).prefetch(tf.data.experimental.AUTOTUNE)
# dataset = dataset.shuffle(buffer_size = 1000)
# dataset = dataset.batch(32)
# dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
# print(list(dataset.as_numpy_iterator()))
#
# # dataset = tf.data.Dataset.from_tensor_slices(['C:/Users/admin/Desktop/train.txt'])
# # dataset = dataset.flat_map(lambda x: tf.data.TextLineDataset(x).skip(1))
# # print(list(dataset.as_numpy_iterator()))
# # # print(list(dataset.as_numpy_iterator()))
# # dataset = dataset.map(funStringSplit, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# # print(list(dataset.as_numpy_iterator()))
import tensorflow as tf
from nlp_datasets.tokenizers import SpaceTokenizer
from nlp_datasets.seq_match import seq_match_dataset
config1 = {
    'feature_tool':'mlp',
    'query_max_len': 5,
    'doc_max_len': 5,
    'vocab_size': 10,
    'embedding_size': 256,
    'vec_dim': 256,
    'model_dir':'D:/KDD相关/DSSM',
    'prefetch_size': tf.data.experimental.AUTOTUNE,
    'num_parallel_calls': tf.data.experimental.AUTOTUNE,
    'buffer_size': 10000000,
    'seed': None,
    'reshuffle_each_iteration': True,
    'train_batch_size':5,  ##几个样本一组
    'bucket_width':5,
    'padding_by_eos': False,
    'drop_remainder': True,
    'add_sos': True,
    'add_eos': True

}


##根据词表构建hash词表，将word转化为id的形式。输入：词表，由build_train_dataset那边传过来的处理好的词语集合
def convert_word2id(word_table):
    tokenizer = SpaceTokenizer()
    tokenizer.build_from_vocab(word_table)
    config1['vocab_size'] = tokenizer.vocab_size
    return tokenizer

def norm_z(z):
    if tf.equal(z, '0 1'):
        return tf.constant(1, dtype=tf.dtypes.int64)
    if tf.equal(z, '1 0'):
        return tf.constant(0, dtype=tf.dtypes.int64)
    if tf.equal(z, '0'):
        return tf.constant(0, dtype=tf.dtypes.int64)
    if tf.equal(z, '1'):
        return tf.constant(1, dtype=tf.dtypes.int64)
    return tf.cast(tf.strings.to_number(z), dtype=tf.dtypes.int64)

def filter_sequence(dataset):
    dataset = dataset.filter(lambda x, y, z: tf.logical_and(tf.size(x) > 0, tf.size(y) > 0))
    x_max_len = -1
    if x_max_len > 0:
        dataset = dataset.filter(lambda x, y, z: tf.size(x) <= x_max_len)  ##只要这样的数据：长度小于x_max_len的！！
    y_max_len = -1
    if y_max_len > 0:
        dataset = dataset.filter(lambda x, y, z: tf.size(y) <= y_max_len)
    return dataset

def _add_special_tokens(dataset, tokenizer):
    if config1['add_sos']:
        x_sos_id = tf.constant(tokenizer.sos_id, dtype=tf.dtypes.int64)
        y_sos_id = tf.constant(tokenizer.sos_id, dtype=tf.dtypes.int64)
        dataset = dataset.map(
            lambda x, y, z: (tf.concat(([x_sos_id], x), axis=0), tf.concat(([y_sos_id], y), axis=0), z),
            num_parallel_calls=config1['num_parallel_calls']
        ).prefetch(config1['prefetch_size'])
    if config1['add_eos']:
        x_eos_id = tf.constant(tokenizer.eos_id, dtype=tf.dtypes.int64)
        y_eos_id = tf.constant(tokenizer.eos_id, dtype=tf.dtypes.int64)
        dataset = dataset.map(
            lambda x, y, z: (tf.concat((x, [x_eos_id]), axis=0), tf.concat((y, [y_eos_id]), axis=0), z),
            num_parallel_calls=config1['num_parallel_calls']
        ).prefetch(config1['prefetch_size'])
    return dataset

# 将train、val的那部分df类型的三列数据变成三元组的形式：类似于（[第一列], [第二列], [第三列]），并转换为id的形式
def build_train_dataset(train_data_path, tokenizer):
    dataset = tf.data.Dataset.from_tensor_slices([train_data_path])
    dataset = dataset.flat_map(lambda x: tf.data.TextLineDataset(x).skip(1))
    print("读取数据：", list(dataset.as_numpy_iterator()))
    ##过滤数据，消除某些句子中含有,的样本
    dataset = dataset.filter(
        lambda x: tf.equal(3, tf.size(tf.strings.split([x], sep=',').values)))

    dataset = dataset.map(
        lambda x: (tf.strings.split(x, ",")[0],
                   tf.strings.split(x, ",")[1],
                   tf.strings.split(x, ",")[2]),
        num_parallel_calls=config1['num_parallel_calls']
    ).prefetch(config1['prefetch_size'])
    print("分为QDL：", list(dataset.as_numpy_iterator()))

    dataset = dataset.shuffle(
            buffer_size=config1['buffer_size'],
            seed=config1['seed'],
            reshuffle_each_iteration=config1['reshuffle_each_iteration'])
    print("打乱数据", list(dataset.as_numpy_iterator()))

    dataset = dataset.map(
        lambda x, y, z: (tf.strings.split([x], sep=' ').values,
                         tf.strings.split([y], sep=' ').values,
                         norm_z(z)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).prefetch(tf.data.experimental.AUTOTUNE)
    print("分词：", list(dataset.as_numpy_iterator()))

    dataset = filter_sequence(dataset)
    print("删除过于长的数据：", list(dataset.as_numpy_iterator()))

    dataset = dataset.map(
        lambda x, y, z: (tokenizer.encode(x), tokenizer.encode(y), z),
        num_parallel_calls=config1['num_parallel_calls']
    ).prefetch(config1['prefetch_size'])
    print("转化为id：", list(dataset.as_numpy_iterator()))

    dataset = _add_special_tokens(dataset, tokenizer)
    print("添加开始和结尾的符号：", list(dataset.as_numpy_iterator()))

    batch_size = config1['train_batch_size']
    bucket_width = config1['bucket_width']
    tmp = seq_match_dataset.SeqMatchDataset(x_tokenizer=tokenizer, y_tokenizer=tokenizer, config=None)
    dataset = tmp._padding_and_batching(dataset, batch_size, bucket_width)
    print("pad & batch操作：", list(dataset.as_numpy_iterator()))

    dataset = dataset.map(lambda q, d, l: ((q, d), l))
    print("分为数据和标签：", list(dataset.as_numpy_iterator()))
    return dataset


word_path = r'd:\KDD相关\任务1数据集\play\word.txt'
train_path = r'd:\KDD相关\任务1数据集\play\new_train.csv'
tokenizer = convert_word2id(word_path)
train_ds = build_train_dataset(train_path, tokenizer)





