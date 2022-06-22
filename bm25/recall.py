'''
换了数据之后就不用召回了
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


##过滤表情符号及html标签
def clean(desstr, restr=''):
    soup = BeautifulSoup(desstr, 'html.parser')
    strs = soup.get_text()
    try:
        co = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]+')
    except re.error:
        co = re.compile(u'('u'\ud83c[\udf00-\udfff]|'u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'u'[\u2600-\u2B55])+')
    return co.sub(restr, strs)

##获取每个product的中词语对应的id(倒排索引，只与product表有关)
def create_word_productid_dic():
    stop_words = set([line.strip() for line in open(r'D:\KDD相关\任务1数据集\停用词.txt', encoding='utf-8').readlines()])
    word_id_dic = {}
    product_data = pd.read_csv(r'D:\KDD相关\任务1数据集\product_catalogue-v0.1.csv')
    for i in range(len(product_data)):
        strs =  str(product_data['product_title'][i])
        if not strs:
            strs = str(product_data['product_description'][i])
        if not strs:
            strs = str(product_data['product_bullet_point'][i])
        strs = clean(strs)
        local = product_data['product_locale'][i]
        if local != 'jp':  ##英西直接根据空格分词
            lst = strs.strip().split()
            for v in lst:
                if v not in stop_words:
                    word_id_dic.setdefault(v, set()).add(product_data['product_id'][i])
        else:  ##日调用分词软件分词
            lst = mecab.parse(strs).split()
            for v in lst:
                if v not in stop_words:
                    word_id_dic.setdefault(v, set()).add(product_data['product_id'][i])
    df = pd.DataFrame.from_dict({k: ' '.join(v) for k, v in word_id_dic.items()}, orient='index', columns=['product_id'])
    df = df.reset_index().rename(columns={'index': 'word'})
    df.to_csv(r'D:\KDD相关\任务1数据集\word_productid_dic.csv', encoding='utf-8', index=False)
    print("product的总词数：", len(word_id_dic))
    return word_id_dic


#为每个query召回商品id
def recall_product():
    word_productid_dic = create_word_productid_dic()
    # dics = pd.read_csv(r'D:\KDD相关\任务1数据集\word_productid_dic.csv')
    # word_productid_dic = {dics[''][i] for i in range(len(dics))}
    train_date = pd.read_csv(r'D:\KDD相关\任务1数据集\new_train-v0.1.csv')
    query_candidate_product = {}  ##每个query的候选productid
    default_recall_num = 200 ##获取：出现频次/总次数>0.5的id们，当没有这种情况时获取前100个id，当没有100个时，全都获取##
    for i in range(len(train_date)):
        if i > 0 and train_date['query_id'][i] == train_date['query_id'][i - 1]:  ##在train中query重复
            continue
        local = train_date['query_locale'][i]
        query = clean(train_date['query'][i])
        if local != 'jp':
            lst = query.strip().split()
        else:
            lst = mecab.parse(query).split()
        query_lens = len(lst)
        queryword_productid_dic = {} ##词对应的商品id
        for j in range(len(lst)):
            if lst[j] in word_productid_dic:
                queryword_productid_dic.setdefault(lst[j], []).extend(word_productid_dic[lst[j]])
        a = list(itertools.chain.from_iterable(queryword_productid_dic.values()))
        id_fre_dic = dict(collections.Counter(a))
        tmp = sorted(id_fre_dic.items(), key=lambda s:s[1], reverse=True)
        #=========================召回策略，按需修改=========================#
        id_lst = []
        for f in range(len(tmp)):
            if tmp[f][1] / query_lens > 0.3:
                id_lst.append(tmp[f][0])
        if len(id_lst) == 0:
            if len(tmp) < default_recall_num:
                id_lst.extend(id_fre_dic.keys())
            else:
                id_lst.extend([tmp[k][0] for k in range(default_recall_num)])
        query_candidate_product[train_date['query_id'][i]] = id_lst
    return query_candidate_product ##最多100个，个数不固定


##判断召回策略是否可以把训练集中的exact对应的商品全部召回
'''
100;0.5
product的总词数： 862097
召回情况判断： 988 33777 2.9250673535245877
200;0.3
product的总词数： 862097
召回情况判断： 2071 33777 6.131391183349617
'''
def recall_accuracy():
    train_date = pd.read_csv(r'D:\KDD相关\任务1数据集\new_train-v0.1.csv')
    query_candidate_product = recall_product()
    sub_data = train_date.groupby('query_id')
    all_query = 0
    success_recall = 0 ##如果训练集中的exact对应的商品全部在该query召回的商品中，则认为召回成功
    for id, group in sub_data:
        all_query += 1
        exact_num = 0
        recall_num = 0
        group1 = group.reset_index()
        for i in range(len(group1)):
            if group1['esci_label'][i] == 'exact':
                exact_num += 1
                if group1['product_id'][i] in query_candidate_product[id]:
                    recall_num += 1
        if exact_num == recall_num:
            success_recall += 1
    print("召回情况判断：", success_recall, all_query, success_recall / all_query * 100)
    return



if __name__ == '__main__':
    recall_accuracy()

