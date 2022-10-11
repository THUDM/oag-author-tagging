from operator import truth
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='aca', help='used method')
args = parser.parse_args()


def read_smpcup2017(path):
    try:
        truth1, truth2, truth3 = [], [], []
        task_start = ['<task1>', '<task2>', '<task3>']
        task_end = ['</task1>', '</task2>', '</task3>']
        flag = [False, False, False]
        truth = [truth1, truth2, truth3]

        resReader = open(path, 'r', encoding='utf-8')
        for line in resReader:
            line = line.replace("\n", "").replace(", ", ",")
            if line in task_start:
                x = task_start.index(line)
                flag[x] = True
                continue
            elif line in task_end:
                continue
            else:
                count = 2
                for item in flag[::-1]:
                    if item:
                        truth[count].append(line)
                        break
                    else:
                        count = count - 1
        return truth
    except Exception as e:
        return -1


def task1(lst, truth_lst):
    try:
        score = 0.0

        tru_list = []
        valid_set = set()

        for item in truth_lst:
            item = item.split(',')
            valid_set.add(item[0])
            tru_list.append(item)

        sub_list = []
        for item in lst:
            item = item.split(',')
            for i in range(1, len(item)):
                item[i] = item[i].lower()

            if item[0] in valid_set:
                sub_list.append(item)
        length = len(tru_list)
        if len(sub_list) != length:
            return "file length incorrect in task1"
        else:
            for i in range(length):
                set1 = set(tru_list[i])
                set2 = set(sub_list[i][:4])
                if len(set2) != 4:
                    return "incorrect format detected around: " + str(sub_list[i])
                if len(set1 & set2) >= 1:
                    score += (len(set1 & set2)-1) / 3

        score = score/length
        return score

    except Exception as e:
        return "error in task 1: " + str(e)


def task1_2(lst, truth_lst, max_length=3, separator=','):
    try:
        score = 0.0

        tru_list = []
        valid_set = set()

        for item in truth_lst:
            item = item.split(separator)
            for i in range(1, len(item)):
                item[i] = item[i].lower()
            valid_set.add(item[0])
            tru_list.append(item)

        sub_list = []
        for item in lst:
            item = item.split(separator)
            for i in range(1, len(item)):
                item[i] = item[i].lower()
            if item[0] in valid_set:
                sub_list.append(item)

        length = len(tru_list)
        if len(sub_list) != length:
            return "file length incorrect in task2"
        else:
            for i in range(length):
                set1 = set(tru_list[i])
                set2 = set(sub_list[i][:max_length + 1])
                # print(set1 & set2, set1, set2)
                if len(set1 & set2) >= 1:
                    score += (len(set1 & set2)-1) / 3
        score = score/length
        return score

    except Exception as e:
        return "error in task 2: " + str(e)


def task_3(lst, truth_lst, separator=',', valid_err_msg = False):
    try:
        valid_set = set()

        tru_df = []
        for item in truth_lst:
            item = item.replace(' ', '').split(separator)
            valid_set.add(item[0])
            tru_df.append(item)
        truth = pd.DataFrame(tru_df, columns=['userid', 'tru'])

        truth['tru'] = truth['tru'].astype(float)

        sub_list = []

        for item in lst:
            item = item.replace(' ', '').split(separator)
            if len(item) != 2:
                return "incorrect format around: " + str(item)
            if item[0] in valid_set:
                sub_list.append(item)
            else:
                if valid_err_msg:
                    return "unidentified name: " + str(item)
            if float(item[1]) < 0:
                return "negative number detected around: " + str(item)
        sub = pd.DataFrame(sub_list, columns=['userid', 'growthvalue'])

        sub['growthvalue'] = sub['growthvalue'].astype(float)

        if len(sub['userid']) != len(truth['userid']):
            return "file length incorrect in task3"

        df = pd.merge(sub, truth, on='userid', how='outer')
        if len(df) != len(truth) or len(df) != len(sub):
            return "file length didn't match in task3"

        df['minus'] = df['growthvalue'] - df['tru']
        df['minus'] = df['minus'].apply(abs)
        df['max'] = np.where(df['growthvalue'] >= df['tru'], df['growthvalue'], df['tru'])
        df['sum'] = df['minus'] / df['max']
        df['sum'] = df['sum'].fillna(0.0)
        final = 1 - df['sum'].sum() / len(df)

        return final

    except Exception as e:
        return "error in task 3: " + str(e)

def ma_task_1(lst, truth_lst):
    try:
        score = 0.0

        tru_lst = []
        for item in truth_lst:
            item = item.split('\t')
            tru_lst.append(item)

        sub_lst = []
        for item in lst:
            item = item.split('\t')
            sub_lst.append(item)

        length = len(tru_lst)
        if len(sub_lst) != length:
            return "file length incorrect in task1"

        for i in range(length):
            if tru_lst[i][0] != sub_lst[i][0]:
                return "id order incorrect in task1"
            if len(sub_lst[i]) != 7:
                return "incorrect format around :" + str(sub_lst[i][0])
            if tru_lst[i][2] == 'fm':
                tru_lst[i][2] = 'f'

            title_tru = set(tru_lst[i].pop(3).split(';'))
            title_sub = set(sub_lst[i].pop(3).split(';'))
            jaccard_index = len(title_tru & title_sub) / len(title_tru | title_sub)

            s = 0
            for n in range(1, len(sub_lst[i])):
                if sub_lst[i][n] == tru_lst[i][n]:
                    s += 1

            score += s + jaccard_index

        return score / (6 * length)

    except Exception as e:
        return "error in task 1: " + str(e)


def eval_task2_only(pred_file):
    preds = read_smpcup2017(pred_file)
    truths = read_smpcup2017("raw_data/scholar_final.txt")
    # print(preds)
    # print(truths)
    score = task1_2(preds[1], truths[1], max_length=3, separator='\t')
    print(score)
    return score


if __name__ == "__main__":
    if args.method not in ["sbert", "lsi", "aca"]:
        raise NotImplementedError

    eval_task2_only("./out/author_interest_{}.txt".format(args.method))
    # eval_task2_only("out/task2_ctm_sim.txt")
    # eval_task2_only("out/task2_sbert-ft_sim.txt")
    # eval_task2_only("out/task2_lda_sim.txt")
