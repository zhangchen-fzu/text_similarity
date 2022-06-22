'''
初步的数据分析
换了数据之后就没啥用了
'''

import pandas as pd
from numpy import *
import csv
import itertools
from bs4 import BeautifulSoup
import re
import collections
import MeCab
mecab = MeCab.Tagger ("-Owakati")




'''
##训练集数据分析，并重新定义训练集的query_id##
训练集的总商品数目：64,3908；总query的数目：3,3777
训练集下每个query对应的商品数目的均值： 23.14426976936969
exact均值：10；substitute均值：7；complement均值：1；irrelevant均值：3
'''
def train_set_analysis():
    train_data = pd.read_csv(r'd:\KDD相关\任务1数据集\train-v0.1.csv')
    total_product_num = set() ##训练集总商品的数量（去重）
    product_num = [] #训练集每个query对应的商品的数量
    query_num = 0 #训练集query的数量（去重）
    esci_dic = {'exact':[], 'substitute':[], 'complement': [], 'irrelevant':[]} #训练集每个query对应的esci数量
    sub_train_date = train_data.groupby("query")
    with open(r'd:\KDD相关\任务1数据集\new_train-v0.1.csv', 'w', newline='') as f:
        fieldnames = ["query_id", "query", "query_locale", "product_id", "esci_label"]
        writer = csv.writer(f)
        writer.writerow(fieldnames)
    for query, group in sub_train_date:
        group['query_id'] = query_num
        query_num += 1
        esci_count = dict(group['esci_label'].value_counts())
        esci_dic['exact'].append(esci_count.get('exact', 0))
        esci_dic['substitute'].append(esci_count.get('substitute', 0))
        esci_dic['complement'].append(esci_count.get('complement', 0))
        esci_dic['irrelevant'].append(esci_count.get('irrelevant', 0))
        product_num.append(len(group))
        group1 = group.reset_index()
        for i in range(len(group1)):
            total_product_num.add(group1['product_id'][i])
        group.to_csv(r'd:\KDD相关\任务1数据集\new_train-v0.1.csv', index=False, encoding='utf-8', mode='a', header=0)
    print("训练集的总商品数目：%d；总query的数目：%d" %(len(total_product_num), query_num))
    print("训练集下每个query对应的商品数目的均值：", mean(product_num))
    print('exact均值：%d；substitute均值：%d；complement均值：%d；irrelevant均值：%d' %(mean(esci_dic['exact']), mean(esci_dic['substitute']), mean(esci_dic['complement']), mean(esci_dic['irrelevant'])))
    return


'''
##商品表数据分析##
商品总数：88,3868
缺失值占比：
product_id               0.000000
product_title            0.016858 √
product_description     49.459648 √
product_bullet_point    16.368055 √
product_brand            8.591894
product_color_name      39.622206
product_locale           0.000000
3列都不存在的商品数： 0
'''
def produce_set_analysis():
    res = 0
    product_data = pd.read_csv(r'D:\KDD相关\任务1数据集\product_catalogue-v0.1.csv')
    product_num = len(product_data) #总的商品数量
    nan_percent = 100 * (product_data.isnull().sum() / len(product_data)) #每列的缺失值占比
    for i in range(len(product_data)):
        if not product_data['product_title'][i] and not product_data['product_description'][i] and not product_data['product_bullet_point'][i]:
            res += 1
    print("3列都不存在的商品数：", res)
    print(product_num)
    print(nan_percent)
    return


#商品表与训练集join，join的列为：product_id及locale
def train_product_process():
    # train_data = pd.read_csv(r'd:\KDD相关\任务1数据集\new_train-v0.1.csv')
    #
    # join_table = pd.merge(train_data, product_date, left_on=['product_id', 'query_locale'], right_on=['product_id', 'product_locale'], how='left')
    # join_table.to_csv(r'D:\KDD相关\任务1数据集\train_product_join-v0.1.csv', index = False, encoding='utf-8')
    test_date = pd.read_csv(r'D:\KDD相关\任务1数据集\test_public-v0.2.csv')
    product_date = pd.read_csv(r'D:\KDD相关\任务1数据集\product_catalogue-v0.2.csv')
    join_table = pd.merge(test_date, product_date, left_on=['product_id', 'query_locale'], right_on=['product_id', 'product_locale'], how='left')
    join_table.to_csv(r'D:\KDD相关\任务1数据集\test_product_join-v0.1.csv', index=False, encoding='utf-8')
    return
print(train_product_process())
'''
##看看query是日语对应的item一定是日语吗，有的商品由多种语言组成，因此在连接表时要选择语言。（有883868-879290=4578个商品存在一个商品用多种语言书写的情况，所以在join的时候要选择语言）##
es {'es': 152920} {'exact': 63144, 'substitute': 50809, 'irrelevant': 26478, 'complement': 12489}
jp {'jp': 209094} {'exact': 96170, 'substitute': 69500, 'irrelevant': 34454, 'complement': 8970}
us {'us': 419730} {'exact': 181856, 'substitute': 147654, 'irrelevant': 71125, 'complement': 19095}
'''
def check_language():
    join_data = pd.read_csv(r'D:\KDD相关\任务1数据集\train_product_join-v0.1.csv')
    sub_data = join_data.groupby('query_locale')
    for local, group in sub_data:
        product_locale_count = dict(group['product_locale'].value_counts())
        esci_count = dict(group['esci_label'].value_counts())
        print(local, product_locale_count, esci_count)
    return

'''
文本召回策略：（倒排索引）
1. 先建立prodect表中的product_title、product_description、product_bullet_point的倒排索引(分词，去停用词)
2. 获取训练集的query，对每个query获取出现频次高的文档id
3. 召回频次在：出现的词数/总词数>0.5的商品们，对训练集做一波测试，看看是否把训练集中的高相关的商品都召回了
4. 按需修改召回数量

拿到词表（遍历一遍）
每个词出现的文档id列表（又遍历一遍）
'''
