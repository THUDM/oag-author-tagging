import json
from tqdm import tqdm
from collections import defaultdict as dd


def gen_training_sim_data():
    with open("raw_data/t_author_paper.json", "r") as f:
        a_p = json.load(f)
    with open("raw_data/t_author_interest.json", "r") as f:
        a_i = json.load(f)
    i_to_authors = dd(set)

    for a in tqdm(a_i):
        for i in a_i[a]:
            i_to_authors[i].add(a)
    all_authors = set(a_i.keys())

    pos_author_pairs = []
    neg_author_pairs = []

    for a in tqdm(a_p):
        cur_sim_a_to_cnt = dd(int)
        cur_interest = set(a_i[a])
        for i in cur_interest:
            for a2 in i_to_authors[i]:
                if a2 != a:
                    cur_sim_a_to_cnt[a2] += 1
        sorted_authors = sorted(cur_sim_a_to_cnt.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_authors) == 0:
            continue

        authors_remain = all_authors - set([a] + [x[0] for x in sorted_authors])
        if len(authors_remain) == 0:
            continue
        pos_author_pairs.append((a, sorted_authors[0][0]))
        neg_author_pairs.append((a, list(authors_remain)[0]))
    
    print(len(pos_author_pairs), len(neg_author_pairs))
    with open("raw_data/sbert_training_data.json", "w") as f:
        json.dump({"pos": pos_author_pairs, "neg": neg_author_pairs}, f)


if __name__ == "__main__":
    gen_training_sim_data()
