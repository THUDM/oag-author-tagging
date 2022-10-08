import json
import codecs
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from lsi_model import extract_training_paper_for_interest, shuffle_list, clean_data, create_dictionary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)


def sbert_main(t_author_paper,author_interest,p_author_paper,train_author,vali_author,author_test,flag):

    print ("sbert main ...")
    interest_paper = extract_training_paper_for_interest(train_author,t_author_paper,author_interest)
    # interest_seq, dictionary,corpus =  create_dictionary(interest_paper)

    paper_seq = []
    interest_seq = []
    for interest,paper in interest_paper.items():
        interest_seq.append(interest)
        text = ' '.join(paper)
        text = clean_data(text)
        text = ' '.join(text)
        paper_seq.append(text)
        # print(text)
        # print()

    print("load model ...")
    model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
    print("model loaded ...")
    # print("interest paper", interest_paper)
    # print("paper seq", paper_seq[-10:])
    sent_embs = model.encode(paper_seq)
    print("sent_embs:", sent_embs.shape)
    # print(sent_embs.shape)
    # corpus_simi_matrix = cosine_similarity(sent_embs)

    print ("predict interest ...")
    predict_author_interest = {}
    predict_author_interest_score = {}

    flag = 0

    with codecs.open("./raw_data/p_author_paper_ex_cite.json","r","utf-8") as fid:
        cite_author_paper = json.load(fid)

    for author,paper in tqdm(p_author_paper.items()):
    #exclude_author = []
    #for author in author_vali:
        paper = p_author_paper[author]
        if len(paper) <= 10:
            paper.extend(shuffle_list(cite_author_paper[author],40))

        interest = []
        test_text = clean_data(' '.join(paper))
        test_text = ' '.join(test_text)
        # print("test_text:", test_text)
        cur_emb = model.encode([test_text])
        test_simi = cosine_similarity(cur_emb,sent_embs)
        # print("test_simi:", test_simi)

        result = list(enumerate(test_simi[0]))
        # print("result:", result)
        result.sort(key=lambda x:x[1],reverse=True)
        # print("result sorted:", result)
        # print("result sorted:", result[:10])

        interest_score = {}
        for v in result:
            #interest_score.setdefault(interest_seq[v[0]],math.exp(v[1]))
            interest_score.setdefault(interest_seq[v[0]],v[1])
        predict_author_interest_score.setdefault(author,interest_score)
        # print("predict_author_interest_score:", author, predict_author_interest_score[author])

        for v in result[:5]:
            interest.append(interest_seq[v[0]])
        predict_author_interest.setdefault(author,interest)
        # raise
    
    return predict_author_interest_score