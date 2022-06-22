'''
准备训练数据部分：
1. train与product做join,拿到需要的QDL及语言数据
2. 做分词、停用词过滤处理，只要'irrelevant', 'exact'的数据，给予标签0,1
准备测试数据部分：
1. test与product做join,拿到需要的QD数据
2. 做分词、停用词过滤处理
'''

import pandas as pd
from bs4 import BeautifulSoup
import re
import MeCab
mecab = MeCab.Tagger ("-Owakati")
import nltk
from sklearn.model_selection import train_test_split

##过滤表情符号及html标签，主要集中在D中
def clean(desstr, restr=''):
    soup = BeautifulSoup(desstr, 'html.parser')
    strs = soup.get_text()
    try:
        co = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]+')
    except re.error:
        co = re.compile(u'('u'\ud83c[\udf00-\udfff]|'u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'u'[\u2600-\u2B55])+')
    return co.sub(restr, strs)

##准备好q、d、l以及local的数据
def query_doc_label(q_path, p_path):
    left_data = pd.read_csv(q_path, keep_default_na=False)
    right_date = pd.read_csv(p_path, keep_default_na=False)
    join_table = pd.merge(left_data, right_date, left_on=['product_id', 'query_locale'], right_on=['product_id', 'product_locale'], how='left')
    query = join_table['query']
    local = join_table['query_locale']
    esci_label = join_table['esci_label']
    product_info = []
    for i in range(len(join_table)):
        doc = str(join_table['product_title'][i])
        if not doc:
            doc = str(join_table['product_description'][i])
        if not doc:
            doc = str(join_table['product_bullet_point'][i])
        new_doc = clean(doc)
        product_info.append(new_doc)
    product_info = pd.DataFrame(product_info, columns=['doc'])
    data = pd.concat([query, product_info, esci_label, local], axis = 1)
    return data

##分词处理
def split_word(stence, local):
    if local == 'us':
        lst = nltk.word_tokenize(stence, 'english')
    elif local == 'es':
        lst = nltk.word_tokenize(stence, 'Spanish')
    else:
        lst = mecab.parse(stence).split()
    return lst


##停用词删除
def del_stop_word(word_lst, stop_word_path):
    stop_words = set([line.strip() for line in open(stop_word_path, encoding='utf-8').readlines()])
    lst = []
    for w in word_lst:
        if w not in stop_words:
            lst.append(w)
    return lst

##筛选可用的数据（'irrelevant', 'exact'），并进行分词处理，删除停用词
def filter_data(data, stop_word_path):
    query, doc, label = [], [], []
    for i in range(len(data)):
        if data['esci_label'][i] in ['irrelevant', 'exact']:
            local = data['query_locale'][i]
            query_lst = split_word(data['query'][i], local)
            query_lst = del_stop_word(query_lst, stop_word_path)
            doc_lst = split_word(data['doc'][i], local)
            doc_lst = del_stop_word(doc_lst, stop_word_path)
            query.append(' '.join(query_lst))
            doc.append(' '.join(doc_lst))
            if data['esci_label'][i] == 'exact':
                label.append(1)
            else:
                label.append(0)
    query = pd.DataFrame(query, columns=['query'])
    doc = pd.DataFrame(doc, columns=['doc'])
    label = pd.DataFrame(label, columns=['label'])
    res = pd.concat([query, doc, label], axis=1)
    return res ##返回的是dataframe类型的数据


##转化为Q，D的形式，并且已经分好词
def get_test_data(path, product_all_data, stop_word_path):
    test_data = pd.read_csv(path, keep_default_na=False)
    product_data = pd.read_csv(product_all_data, keep_default_na=False)
    join_table = pd.merge(test_data, product_data, left_on=['product_id', 'query_locale'], right_on=['product_id', 'product_locale'], how='left')
    query, doc = [], []
    for i in range(len(join_table)):
        local = join_table['query_locale'][i]
        query_lst = split_word(join_table['query'][i], local)
        query_lst = del_stop_word(query_lst, stop_word_path)
        query.append(' '.join(query_lst))

        doc_str = str(join_table['product_title'][i])
        if not doc_str:
            doc_str = str(join_table['product_description'][i])
        if not doc_str:
            doc_str = str(join_table['product_bullet_point'][i])
        new_doc = clean(doc_str)
        doc_lst = split_word(new_doc, local)
        doc_lst = del_stop_word(doc_lst, stop_word_path)
        doc.append(' '.join(doc_lst))
    query = pd.DataFrame(query, columns=['query'])
    doc = pd.DataFrame(doc, columns=['doc'])
    res = pd.concat([query, doc], axis=1)
    return res ##返回的是dataframe类型的数据





if __name__ == '__main__':
    ## --------------------#原始数据path
    train_all_data = r'd:\KDD相关\任务1数据集\train-v0.2.csv'
    product_all_data = r'd:\KDD相关\任务1数据集\product_catalogue-v0.2.csv'
    test_all_data = r'd:\KDD相关\任务1数据集\test_public-v0.2.csv'
    stop_word_path = r'd:\KDD相关\任务1数据集\停用词.txt'
    ## --------------------#新数据path
    train_path = r'd:\KDD相关\任务1数据集\play\new_train.csv'
    val_path = r'd:\KDD相关\任务1数据集\play\new_val.csv'
    test_path = r'd:\KDD相关\任务1数据集\play\new_test.csv'
    ## --------------------#对总训练集的处理
    train_all_data = query_doc_label(train_all_data, product_all_data) ##准备好可用的列
    all_train = filter_data(train_all_data, stop_word_path) ##过滤数据
    train, val = train_test_split(all_train, test_size=0.2)  # train-val = 8:2
    train.to_csv(train_path, index=False, encoding='utf-8')
    val.to_csv(val_path, index=False, encoding='utf-8')
    ## --------------------#对总测试集的处理
    test = get_test_data(test_all_data, product_all_data, stop_word_path)
    test.to_csv(test_path, index = False, encoding='utf-8')
