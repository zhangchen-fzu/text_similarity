'''
两种模型的训练部分
'''

import argparse
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import evaluation
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path_file", type=str, help="Input training CSV with the pairs of queries and products.")
    parser.add_argument("product_catalogue_path_file", type=str, help="Input product catalogue CSV.")
    parser.add_argument("locale", type=str, choices=['us', 'es', 'jp'], help="Locale of the queries.")
    parser.add_argument("model_save_path", type=str, help="Directory to save the model.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    parser.add_argument("--n_dev_queries", type=int, default=200, help="Number of development examples.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size.")
    args = parser.parse_args()

    """ 0. Init variables """
    col_query_id = "query_id"
    col_query = "query"
    col_query_locale = "query_locale"
    col_product_id = "product_id"
    col_product_title = "product_title"
    col_product_locale = "product_locale"
    col_esci_label = "esci_label"
    esci_label2gain = {
        'exact': 1.0,
        'substitute': 0.1,
        'complement': 0.01,
        'irrelevant': 0.0,}
    col_gain = 'gain'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ 1. Load data """
    df = pd.read_csv(args.train_path_file) #训练数据的地址
    df_product_catalogue = pd.read_csv(args.product_catalogue_path_file) #product数据的地址
    df = df[df[col_query_locale] == args.locale]
    df_product_catalogue = df_product_catalogue[df_product_catalogue[col_product_locale] == args.locale]
    df = pd.merge(
        df,
        df_product_catalogue,
        how='left',
        left_on=[col_product_id, col_query_locale],
        right_on=[col_product_id, col_product_locale],) #数据合并
    df = df[df[col_product_title].notna()] #获取col_product_title不为null的数据
    list_query_id = df[col_query_id].unique()
    dev_size = args.n_dev_queries / len(list_query_id) #200个q/总q来获取dev的比例
    list_query_id_train, list_query_id_dev = train_test_split(list_query_id, test_size=dev_size, random_state=args.random_state)
    df[col_gain] = df[col_esci_label].apply(lambda label: esci_label2gain[label])
    df = df[[col_query_id, col_query, col_product_title, col_gain]] #四个特征：q_id, q, d, gain
    df_train = df[df[col_query_id].isin(list_query_id_train)] #训练数据
    df_dev = df[df[col_query_id].isin(list_query_id_dev)] #验证数据

    """ 2. Prepare data loaders """
    train_samples = [] #只包含两个特征（q, d）和gain
    for (_, row) in df_train.iterrows():
        train_samples.append(InputExample(texts=[row[col_query], row[col_product_title]], label=float(row[col_gain])))
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size, drop_last=True)
    if args.locale == "us": ##英语部分
        dev_samples = {} #{qid:{'query':q内容，'positive'：set(分数>0的product_title), 'negative':set(分数=0的0的product_title)}......}
        query2id = {} #{q内容：id......}
        for (_, row) in df_dev.iterrows():
            try:
                qid = query2id[row[col_query]]
            except KeyError:
                qid = len(query2id)
                query2id[row[col_query]] = qid
            if qid not in dev_samples:
                dev_samples[qid] = {'query': row[col_query], 'positive': set(), 'negative': set()}
            if row[col_gain] > 0:
                dev_samples[qid]['positive'].add(row[col_product_title])
            else:
                dev_samples[qid]['negative'].add(row[col_product_title])
        evaluator = CERerankingEvaluator(dev_samples, name='train-eval')  #评估器

        """ 3. Prepare Cross-enconder model:
            https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_cross-encoder_kd.py
        """
        model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
        num_epochs = 1
        num_labels = 1
        max_length = 512
        #不区分参数的占位符标识运算符。百度翻译，其实意思就是这个网络层的设计是用于占位的，即不干活，只是有这么一个层，放到残差网络里就是在跳过连接的地方用这个层，显得没有那么空虚
        default_activation_function = torch.nn.Identity()
        model = CrossEncoder(
            model_name,
            num_labels=num_labels,
            max_length=max_length,
            default_activation_function=default_activation_function,
            device=device)
        loss_fct = torch.nn.MSELoss()
        evaluation_steps = 5000 #？？
        warmup_steps = 5000 #？？
        lr = 7e-6
        """ 4. Train Cross-encoder model """
        model.fit(
            train_dataloader=train_dataloader,
            loss_fct=loss_fct,
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=evaluation_steps,
            warmup_steps=warmup_steps,
            output_path=f"{args.model_save_path}_tmp", ##模型保存地址
            optimizer_params={'lr': lr},
        )
        model.save(args.model_save_path)
    else: #西语与日语部分
        dev_queries = df_dev[col_query].to_list()
        dev_titles = df_dev[col_product_title].to_list()
        dev_scores = df_dev[col_gain].to_list()
        evaluator = evaluation.EmbeddingSimilarityEvaluator(dev_queries, dev_titles, dev_scores)

        """ 3. Prepare sentence transformers model: 
            https://www.sbert.net/docs/training/overview.html 
        """
        model_name = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
        model = SentenceTransformer(model_name)
        train_loss = losses.CosineSimilarityLoss(model=model)
        num_epochs = 1
        evaluation_steps = 1000
        """ 4. Train Sentence transformer model """
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=evaluation_steps,
            output_path=args.model_save_path,
        )


if __name__ == "__main__":
    main()