import re
import os
import json
import codecs
import random
import pickle
import utility

import numpy as np
import pandas as pd
import xgboost as xgb
import data_io as dio
from pypinyin import lazy_pinyin
from utility import homepage_neg
from utility import homepage_pos
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split


def one_sample_homepage_features(data, search_res, labeled=True):
    """
    Input: data - the row of dataframe. search_res - the list of search results info
    Ouput: features
    """
    features = []
    p = re.compile(r'University|university|大学|Institute|School|school|College')
    pos_p = re.compile('|'.join(homepage_pos))
    neg_p = re.compile('|'.join(homepage_neg))
    name = data['name'].replace('?', '')
    name_p = re.compile(r'|'.join(name.lower().split(' ')))
    
    if p.match(data["org"]):
        in_school = 1 # 2
    else:
        in_school = 0
    
    if search_res == None:
        return []
    for i in range(len(search_res)):
        title = ' '.join(lazy_pinyin(search_res[i][0]))
        url = search_res[i][1]
        rank = [i] # 1
        # for j in range(10):
        #     if i == j:
        #         rank.append(1)
        #     else:
        #         rank.append(0)
        if labeled:
            if url == data.homepage:
                label = 1
            else:
                # subsample
                if random.random() < 0.3:
                # if rank < 2:
                    label = 0
                else:
                    continue
        else:
            # if rank >= 2:
                # continue
            label = url
        feature = []
        feature.append(label)
        feature.extend(rank)
        content = search_res[i][2]
        is_cited = search_res[i][3] # 3
        pos_words_num = len(pos_p.findall(url)) # 4
        neg_words_num = len(neg_p.findall(url)) # 5
        edu = 1 if 'edu' in url else 0 # 6
        org = 1 if 'org' in url else 0 # 7
        gov = 1 if 'gov' in url else 0 # 8
        name_in = 1 if len(name_p.findall(title.lower())) != 0 else 0 # 9
        linkedin = 1 if 'linkedin' in url else 0 # 10
        title_len = len(title) # 11
        content_len = len(content) # 12
        org_len = len(data["org"]) + 1 # 13
        name_title = check_name_in_text(name, title) # 14
        name_content = check_name_in_text(name, content) # 15
        mail_content = 1 if 'mail' in content.lower() else 0 # 16
        address_content = 1 if 'address' in content.lower() else 0 # 17
        feature.extend([in_school, is_cited, pos_words_num, neg_words_num, edu,\
         org, gov, linkedin, title_len, content_len, org_len, name_title, name_content]) 
        # for i in homepage_neg:
        #     if i in title.lower():
        #         feature.append(1)
        #     else:
        #         feature.append(0)
        # for i in homepage_pos:
        #     if i in title.lower():
        #         feature.append(1)
        #     else:
        #         feature.append(0)
        features.append(feature)
    return features

def check_name_in_text(name, text):
    """
    Sample: for the name of "Bai Li", \
    # www.xx.com/li.jpg get 0.5
    www.xx.org/bai_li.jpg get 1
    www.xx.org.avatar.jpg get 0
    """
    score = 0
    text = ' '.join(lazy_pinyin(text))
    for i in re.split(r'[ -]', name):
        if i.lower() in text.lower():
            score += 1
    return score / len(name.split(' '))

def extract_homepage_features(labeled=True, full_data=False):
    
    features = []
    if labeled:
        if full_data:
            raw_data = dio.read_former_task1_ans('./full_data/full_data_ans.txt', raw='./full_data/full_data.tsv', skiprows=False)
            search_info = json.load(open('./full_data/all_search_info.json'))
            # another = dio.read_task1('./task1/training.txt')
            # raw_data = pd.concat([raw_data, another])
            # another = dio.load_search_res(True)
            # search_info.update(another)
            print(raw_data.shape)
        else:
            raw_data = dio.read_task1('./task1/training.txt')
            search_info = dio.load_search_res(labeled)
    else:
        raw_data = dio.read_task1('./task1/validation.txt')
        search_info = dio.load_search_res(labeled)
    for i, r in raw_data.iterrows():
        samples = one_sample_homepage_features(r, search_info[r["id"]], labeled)
        if samples != []:
            features.extend(samples)
    if full_data:
        labeled = 'all'
    with open('./data/%s_features.svm.txt'%(labeled), 'w') as f:
        for feature in features:
            line = str(feature[0]) + ' '
            line = line + ' '.join([str(i) + ':' + str(feature[i]) for i in range(1, len(feature))]) + '\n'
            f.write(line)

def homepage_xgb_model(model_path, training_set='True'):
    training_set = './data/%s_features.svm.txt'%(training_set)
    model = xgb.XGBClassifier( learning_rate =0.1,
         n_estimators=200,
         max_depth=5,
         min_child_weight=1,
         gamma= 0.3,
         subsample= 0.7,
         colsample_bytree=0.7,
         objective= 'binary:logistic',
         scale_pos_weight=1)
    X, y = load_svmlight_file(training_set)
    model.fit(X,y)
    pickle.dump(model, open(model_path, 'wb'))
    return model

def load_homepage_model(model_path):
    model = pickle.load(open(model_path, 'rb'))
    return model

def predict_one_homepage(model, data):
    """
    Input: model - the trained xgb model. data - features generated by `one_sample_homepage_features`
    Output: the predict url of homepage
    """
    features = np.array([x[1:] for x in data ])
    urls = [i[0] for i in data]
    # return urls[0]
    # pred = model.predict_prob(features)[: ,1]
    pred = model.predict_proba(features)
    # print(pred)
    url = urls[pred[: ,1].argmax()]
    # if pred[: ,1].max() < 0.05:
    #     url = urls[0]
    return url

def check_homepage_validity(name, res):
    """
    Check if the homepage is simtisfied basic rules.
    Input: name-name of expert res-homepage info list
    """
    title, url, detail, cited = res
    if url.endswith('pdf') or url.endswith('doc') or 'linkedin' in url.lower() or 'researchgate' in url.lower() or 'citations' in url.lower():
        return False
    # to check if the title or detail contains the name
    
    
    title = ' '.join(lazy_pinyin(title))
    name = name.replace('?', '')
    p = re.compile(r'|'.join(name.lower().split(' ')))
    if len(p.findall(title.lower())) == 0:
        return False
    
    #if 'wikipedia' in title.lower():
     #   return False
    return True

def simple_guess_homepage(data, res):
    """
    Use simple rules to guess homepage
    res - homepage search results 
    """
    for i in res:
        if check_homepage_validity(data['name'], i):
            return i[1]
    return res[0][1]

def predcit_homepage_simple(data, res):
    """
    Assign homepage values using simple rules
    """
    for index, row in data.iterrows():
        homepage = simple_guess_homepage(row, res[row['id']])
        data.set_value(index, 'homepage', homepage)
    return data
def score_homepage_simple(data, res):
    """
    Score homepage generated by simple rules
    """
    score = 0
    for index, row in data.iterrows():
        homepage = simple_guess_homepage(row, res[row['id']])
        if homepage == row['homepage']:
            score += 1
    return score / data.shape[0]

def predict_homepage(model, data, res):
    """
    Assign homepage value to input data, using the input model.
    """
    for index, row in data.iterrows():
        features = one_sample_homepage_features(row, res[row['id']], labeled=False)
        homepage = predict_one_homepage(model, features)
        data.set_value(index, 'homepage', homepage)

    

def score_homepage(model, data, res):
    """
    To get the score of homepage result to input data, using the input model.
    """
    score = 0   
    for index, row in data.iterrows():
        features = one_sample_homepage_features(row, res[row['id']], labeled=False)
        homepage = predict_one_homepage(model, features)
        if homepage == row['homepage']:
            score += 1
    print(score)
    return score / data.shape[0]


def main(model_path='./model/temp.dat'):
    
    extract_homepage_features(labeled=True, full_data=True)
    # extract_homepage_features(labeled=False, full_data=False)
    model = homepage_xgb_model(model_path, training_set='all')
    data = dio.read_former_task1_ans('./full_data/full_data_ans.txt', raw='./full_data/full_data.tsv', skiprows=False)
    search_info = json.load(open('./full_data/all_search_info.json'))
    print('Training set:', score_homepage(model, data, search_info))
    # test = dio.read_task1('./task1/training.txt')
    # res = dio.load_search_res(True)
    # print('Test set:', score_homepage(model, test, res))
    
    return model