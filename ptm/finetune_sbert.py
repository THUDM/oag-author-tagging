import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)


def fine_tune_sbert():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
    # InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
    with open("raw_data/sbert_training_data.json", "r") as f:
        train_pairs = json.load(f)

    with open("raw_data/t_author_paper.json", "r") as f:
        a_p = json.load(f)
    a_p_new = {}
    for a in a_p:
        if isinstance(a_p[a], list):
            a_p_new[a] = " ".join(a_p[a])[:512]
        else:
            a_p_new[a] = a_p[a][:512]
    
    train_examples = []
    for pair in train_pairs["pos"]:
        train_examples.append(InputExample(texts=[a_p_new[pair[0]], a_p_new[pair[1]]], label=1.0))
    for pair in train_pairs["neg"]:
        train_examples.append(InputExample(texts=[a_p_new[pair[0]], a_p_new[pair[1]]], label=0.0))
    
    np.random.seed(0)
    np.random.shuffle(train_examples)
    train_dataloader = DataLoader(train_examples[:10000], shuffle=True, batch_size=16)
    valid_examples = train_examples[10000:]

    valid_texts1 = [example.texts[0] for example in valid_examples]
    valid_texts2 = [example.texts[1] for example in valid_examples]
    valid_labels = [example.label for example in valid_examples]
    evaluator = evaluation.EmbeddingSimilarityEvaluator(valid_texts1, valid_texts2, valid_labels)

    train_loss = losses.CosineSimilarityLoss(model)

    #Tune the model
    os.makedirs("out/sbert-ft/", exist_ok=True)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=20, warmup_steps=100, evaluator=evaluator, output_path="out/sbert-ft/")
    
    # model.save('out/sbert-ft/', 'sbert_finetune.pt')


if __name__ == "__main__":
    fine_tune_sbert()
