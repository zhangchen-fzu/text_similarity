'''
使用BM25计算相似度
BM25资料：https://www.jianshu.com/p/1e498888f505
NDCG：0.751
'''
import pandas as pd
from numpy import *
from textsimilarity import recall
import MeCab
mecab = MeCab.Tagger ("-Owakati")
import math
from bs4 import BeautifulSoup
import re

test_data = pd.read_csv(r'D:\KDD相关\任务1数据集\test_product_join-v0.1.csv', encoding='utf-8')
product_data = pd.read_csv(r'D:\KDD相关\任务1数据集\product_catalogue-v0.2.csv', encoding='utf-8')
# indx_table = pd.read_csv(r'D:\KDD相关\任务1数据集\word_productid_dic.csv', encoding='utf-8')
# word_id_dic = {indx_table['word'][i]:indx_table['product_id'][i].strip().split(' ') for i in range(len(indx_table))}
# train_date = pd.read_csv(r'D:\KDD相关\任务1数据集\train-v0.2.csv', encoding='utf-8')

##针对不同语言的分词
def split_word(query, local):
    if local != 'jp':
        lst = query.strip().split()
    else:
        lst = mecab.parse(query).split()
    return lst


'''
def avg_len_procuct():
    len_lst = []
    for i in range(len(product_data)):
        doc = str(product_data['product_title'][i])
        if not doc:
            doc = str(product_data['product_description'][i])
        if not doc:
            doc = str(product_data['product_bullet_point'][i])
        new_doc = recall.clean(doc)
        doc_lst = split_word(new_doc, product_data['product_locale'][i])
        dl = len(doc_lst)
        len_lst.append(dl)
    return len_lst, mean(len_lst)
'''

##过滤表情符号及html标签
def clean(desstr, restr=''):
    soup = BeautifulSoup(desstr, 'html.parser')
    strs = soup.get_text()
    try:
        co = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]+')
    except re.error:
        co = re.compile(u'('u'\ud83c[\udf00-\udfff]|'u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'u'[\u2600-\u2B55])+')
    return co.sub(restr, strs)

def bm25():
    k1 = 2
    b = 0.75
    sub_test_date =  test_data.groupby('query_id')
    product_id = []
    query_id = []
    # score_id = []
    for id, group in sub_test_date:
        N = len(group)
        group1 = group.reset_index()
        query = group1['query'][0]
        new_query = recall.clean(query)
        query_lst = split_word(new_query, group1['query_locale'][0])

        word_id_dic = {}
        lens_lst = []
        fre_lst = []
        for i in range(len(group1)):
            doc = str(group1['product_title'][i])
            if not doc:
                doc = str(group1['product_description'][i])
            if not doc:
                doc = str(group1['product_bullet_point'][i])
            new_doc = recall.clean(doc)
            doc_lst = split_word(new_doc, group1['product_locale'][i])
            for w in doc_lst:
                word_id_dic.setdefault(w, set()).add(group1['product_id'][i])
            lens_lst.append(len(doc_lst))
            fre = []
            for word in query_lst:
                fre.append(doc_lst.count(word))
            fre_lst.append(fre)
        avgdl = mean(lens_lst)
        productid_score = {}
        for j in range(len(group1)):
            score_q_d = 0
            for k in range(len(query_lst)):
                nqi = 0
                if query_lst[k] in word_id_dic:
                    nqi = len(word_id_dic[query_lst[k]])
                fi = fre_lst[j][k]
                dl = lens_lst[j]
                part1 = log((N - nqi + 0.5) / (nqi + 0.5))
                part2 = fi * (k1 + 1) / (fi + k1 * (1 - b + b * (dl / avgdl)))
                score_q_d += (part1 * part2)
            productid_score[group1['product_id'][j]] = score_q_d
        sort_productid_score = sorted(productid_score.items(), key=lambda s:s[1], reverse=True)
        for indx in range(len(sort_productid_score)):
            product_id.append(sort_productid_score[indx][0])
            query_id.append(id)
            # score_id.append(sort_productid_score[indx][1])
    product_id = pd.DataFrame(product_id, columns=['product_id'])
    query_id = pd.DataFrame(query_id, columns=['query_id'])
    # score_id = pd.DataFrame(score_id, columns=['score_id'])
    res = pd.concat([product_id, query_id], axis=1)
    res.to_csv(r'D:\KDD相关\任务1数据集\submit.csv', index=False, encoding='utf-8')
    return


'''
def recall_accuracy():
    query_candidate_product = bm25()
    sub_data = test_data.groupby('query_id')
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
'''


if __name__ == '__main__':
    bm25()












