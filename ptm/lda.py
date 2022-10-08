import os
import json
import joblib
from sympy import im
from tqdm import tqdm
from gensim.models import LdaModel
from gensim import models
from gensim.corpora.dictionary import Dictionary


def train_lda():
    with open("raw_data/t_author_paper.json") as rf:
        a_p = json.load(rf)
    documents = []
    for author, paper in tqdm(a_p.items()):
        if isinstance(paper, list):
            paper = " ".join(paper)[:512].strip().split()
        documents.append(paper)
    
    common_dictionary = Dictionary(documents)
    common_corpus = [common_dictionary.doc2bow(text) for text in documents]

    lda = LdaModel(common_corpus, num_topics=100)
    out_dir = "out/lda/"
    os.makedirs(out_dir, exist_ok=True)
    lda.save(out_dir + "lda.model")
    joblib.dump(common_dictionary, out_dir + "common_dictionary.pkl")


def test_lda_topic_vec():
    with open("raw_data/p_author_paper_final.json") as rf:
        a_p = json.load(rf)
    out_dir = "out/lda/"
    lda = LdaModel.load(out_dir + "lda.model")
    common_dictionary = joblib.load(out_dir + "common_dictionary.pkl")
    for author, paper in tqdm(a_p.items()):
        if isinstance(paper, list):
            paper = " ".join(paper)[:512].strip().split()
        paper_bow = common_dictionary.doc2bow(paper)
        topic_vec = lda.get_document_topics(paper_bow)
        print(topic_vec)


def create_lsi_model(num_topics=100):
    with open("raw_data/t_author_paper.json") as rf:
        a_p = json.load(rf)
    documents = []
    for author, paper in tqdm(a_p.items()):
        if isinstance(paper, list):
            paper = " ".join(paper)[:512].strip().split()
        documents.append(paper)
    common_dictionary = Dictionary(documents)

    corpus = [common_dictionary.doc2bow(line) for line in documents]

    print ("create lsi model ...")
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]
    lsi_model = models.LsiModel(corpus_tfidf,id2word=common_dictionary,num_topics = num_topics)
    #lsi_model = models.LsiModel(corpus,id2word=dictionary,num_topics = num_topics)
    corpus_lsi = lsi_model[corpus_tfidf]
    print(corpus_lsi[0])
    #corpus_lsi = lsi_model[corpus]
    # return tfidf_model,lsi_model

    out_dir = "out/lsi/"
    os.makedirs(out_dir, exist_ok=True)
    lsi_model.save(out_dir + "lsi.model")
    joblib.dump(common_dictionary, out_dir + "common_dictionary.pkl")
    # joblib.dump(tfidf_model, out_dir + "tfidf_model.pkl")
    tfidf_model.save(out_dir + "tfidf_model.model")


if __name__ == "__main__":
    # train_lda()
    # test_lda_topic_vec()
    create_lsi_model()
