'''
测试集的预测部分
'''
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from tqdm import tqdm #进度条


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_path_file", type=str, help="Input test CSV with the pairs of queries and products.")
    parser.add_argument("product_catalogue_path_file", type=str, help="Input product catalogue CSV.")
    parser.add_argument("locale", type=str, choices=['us', 'es', 'jp'], help="Locale of the queries.")
    parser.add_argument("model_path", type=str, help="Directory where the model is stored.")
    parser.add_argument("hypothesis_path_file", type=str, help="Output CSV with the hypothesis.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    args = parser.parse_args()

    """ 0. Init variables """
    col_query_id = "query_id" #查询的id
    col_query = "query" #查询
    col_query_locale = "query_locale" #查询的国家
    col_product_id = "product_id" #产品id
    col_product_title = "product_title" #产品的title
    col_product_locale = "product_locale" #产品的国家
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ 1. Load data """
    df = pd.read_csv(args.test_path_file) #test的地址
    df_product_catalogue = pd.read_csv(args.product_catalogue_path_file) #product的地址
    df_product_catalogue.fillna('', inplace=True) #null变成“”
    df = df[df[col_query_locale] == args.locale]
    df_product_catalogue = df_product_catalogue[df_product_catalogue[col_product_locale] == args.locale]
    df = pd.merge(
        df,
        df_product_catalogue,
        how='left',
        left_on=[col_product_id, col_query_locale],
        right_on=[col_product_id, col_product_locale],) #test与product的内容合并
    features_query = df[col_query].to_list()  #query那边的特征
    features_product = df[col_product_title].to_list() #product那边的特征
    n_examples = len(features_query)  ##样本数量
    scores = np.zeros(n_examples) #scores的初始值全部设置为0

    if args.locale == "us": #英文使用AutoModelForSequenceClassification模型
        """ 2. Prepare Cross-encoder model """
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device) #加载模型
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        """ 3. Generate hypothesis """
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, n_examples, args.batch_size)): #n_examples/batch_size轮
                j = min(i + args.batch_size, n_examples)
                features_query_ = features_query[i:j]
                features_product_ = features_product[i:j]
                features = tokenizer(features_query_, features_product_, padding=True, truncation=True,
                                     return_tensors="pt").to(device)
                scores[i:j] = np.squeeze(model(**features).logits.cpu().detach().numpy()) ##us部分的得分
                i = j
    else: #西语和日语使用AutoModel模型，模型得到的时emb，还要对emb做处理
        """ 2. Prepare Sentence transformer model """
        model = AutoModel.from_pretrained(args.model_path).to(device) #加载模型
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        # CLS Pooling - Take output from first token
        def cls_pooling(model_output):
            return model_output.last_hidden_state[:, 0]

        # Encode text
        def encode(texts):
            # Tokenize sentences
            encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input, return_dict=True)
            # Perform pooling
            embeddings = cls_pooling(model_output)
            return embeddings

        model.eval()

        """ 3. Generate hypothesis """
        with torch.no_grad():
            for i in tqdm(range(0, n_examples, args.batch_size)):
                j = min(i + args.batch_size, n_examples)
                features_query_ = features_query[i:j]
                features_product_ = features_product[i:j]
                query_emb = encode(features_query_)
                product_emb = encode(features_product_)
                scores[i:j] = torch.diagonal(torch.mm(query_emb, product_emb.transpose(0, 1)).to('cpu')) #写入score
                i = j

    """ 4. Prepare hypothesis file """
    col_scores = "scores"
    df_hypothesis = pd.DataFrame({
        col_query_id: df[col_query_id].to_list(),
        col_product_id: df[col_product_id].to_list(),
        col_scores: scores,
    })
    df_hypothesis = df_hypothesis.sort_values(by=[col_query_id, col_scores], ascending=False)
    df_hypothesis[[col_query_id, col_product_id]].to_csv(
        args.hypothesis_path_file,
        index=False,
        sep=',',
    )


if __name__ == "__main__":
    main()

