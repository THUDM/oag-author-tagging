import os
import json
from sympy import epath
from tqdm import tqdm
import joblib
import nltk
from nltk.corpus import stopwords as stop_words
from gensim .models import LdaModel
from gensim.corpora.dictionary import Dictionary
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)
# nltk.download('stopwords')


def get_bert_emb(model_type="sbert", role="train"):
    if role == "train":
        file = "raw_data/t_author_paper.json"
    elif role == "test":
        file = "raw_data/p_author_paper_final.json"
    else:
        raise NotImplementedError
    with open(file, "r") as f:
        a_p = json.load(f)
    print("load data done")
    
    if model_type == "lda":
        out_dir = "out/lda/"
        lda = LdaModel.load(out_dir + "lda.model")
        common_dictionary = joblib.load(out_dir + "common_dictionary.pkl")
        for author, paper in tqdm(a_p.items()):
            if isinstance(paper, list):
                paper = " ".join(paper)[:512].strip().split()
            paper_bow = common_dictionary.doc2bow(paper)
            topic_vec = lda.get_document_topics(paper_bow)
            # print(topic_vec)
            cur_vec = np.zeros(100)
            # topic_vec = np.array([i[1] for i in topic_vec])
            for i in topic_vec:
                cur_vec[i[0]] = i[1]
            out_dir = "out/{}/{}/".format(model_type, role)
            os.makedirs(out_dir, exist_ok=True)
            assert len(cur_vec) == 100
            torch.save(cur_vec, out_dir + author + ".pt")
    elif model_type == "sbert":
        for author, paper in tqdm(a_p.items()):
            if isinstance(paper, list):
                paper = " ".join(paper)

            if model_type == "sbert":
                model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
                paper_embed_1 = model.encode(paper, convert_to_tensor=True).to(device)
                paper_embed_1 =torch.nn.functional.normalize(paper_embed_1, p=2, dim=0)
            elif model_type == "sbert-ft":
                model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
                model.load("out/sbert-ft/")
                paper_embed_1 = model.encode(paper, convert_to_tensor=True).to(device)
                paper_embed_1 =torch.nn.functional.normalize(paper_embed_1, p=2, dim=0)
            else:
                raise NotImplementedError
            
            # print(paper_embed_1.shape)
            
            out_dir = "out/{}/{}/".format(model_type, role)
            os.makedirs(out_dir, exist_ok=True)
            torch.save(paper_embed_1, out_dir + author + ".pt")
    else:
        raise NotImplementedError


def mv_author_files(role="train", model_type="sbert"):
    if role == "train":
        file = "raw_data/t_author_paper.json"
    elif role == "test":
        file = "raw_data/p_author_paper_final.json"
    else:
        raise NotImplementedError
    with open(file, "r") as f:
        a_p = json.load(f)
    for author, paper in tqdm(a_p.items()):
        out_dir = "out/{}/{}/".format(model_type, role)
        os.makedirs(out_dir, exist_ok=True)
        file_src = "out/{}/".format(model_type) + author + ".pt"
        file_dst = "out/{}/{}/{}.pt".format(model_type, role, author)
        try:
            os.rename(file_src, file_dst)
        except:
            print("file not found:", file_src)


def cauculate_matrix(role="train", model_type="sbert"):
    pts=os.listdir("out/{}/{}/".format(model_type, role))
    matrix=[]
    for i in  pts:
        i="out/{}/{}/".format(model_type, role)+i
        if model_type == "sbert" or model_type == "sbert-ft":
            matrix.append(torch.load(i).cpu().numpy())
        elif model_type == "ctm" or model_type == "lda":
            matrix.append(torch.load(i))
        else:
            raise NotImplementedError
    matrix=np.array(matrix) 
    print(matrix.shape)
    matrix=matrix.reshape(matrix.shape[0],matrix.shape[1])
    return  matrix


def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

def cauculate_similary_matrix(datatype1,datatype2, model_type):
    matrix1=cauculate_matrix(datatype1, model_type)
    matrix2=cauculate_matrix(datatype2, model_type)
    #norm1 = np.linalg.norm(matrix1,axis=-1,keepdims=True)
    #norm2 = np.linalg.norm(matrix2,axis=-1,keepdims=True)
    #consin_similary_matrix=np.dot(matrix1/norm1,(matrix2/norm2).T)
    return  get_cos_similar_matrix(matrix1,matrix2)


def run_testdata_train_data(model="sbert"):
    with open("raw_data/t_author_interest.json","r",encoding="utf-8")as fp:
        t_a_i=json.load(fp=fp)

    train_vail_similary_matrix=cauculate_similary_matrix(datatype1="test",datatype2="train", model_type=model)
    test_names_order = []
    with open("raw_data/task2_test_final.txt") as rf:
        for line in rf:
            test_names_order.append(line.strip())
    pts = os.listdir("out/{}/test/".format(model))
    file_to_idx = {pt[:-3]: idx for idx, pt in enumerate(pts)}
    pts_train = os.listdir("out/{}/train/".format(model))
    train_authors = [pt[:-3] for pt in pts_train]
    out_dir = "out/"
    os.makedirs(out_dir, exist_ok=True)
    wf_name = "author_interest_{}.txt".format(model)
    wf = open(os.path.join(out_dir, wf_name), "w")
    wf.write("<task2>\n")
    wf.write("authorname	interest1	interest2	interest3	interest4	interest5\n")
    for author in tqdm(test_names_order):
        idx = file_to_idx[author]
        similary=train_vail_similary_matrix[idx]
        sorted_author_index=np.array(similary).argsort()[-2:]
        predict_interest=[t_a_i[train_authors[index]] for index in sorted_author_index]
        predict_interest=[i for item in predict_interest for i in item]
        wf.write(author + "\t" + "\t".join(predict_interest) + "\n")
    wf.write("</task2>\n")
    wf.close()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"]="true"
    get_bert_emb(model_type="sbert", role="train")
    get_bert_emb(model_type="sbert", role="test")
    run_testdata_train_data(model="sbert")
