#!/usr/bin/env python
# coding: utf-8


import json
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, AdamW
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from opendelta import AdapterModel

seed=514
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


max_epoch = 20
path = 'emilyalsentzer/Bio_ClinicalBERT'


tok = AutoTokenizer.from_pretrained(path)
classifier = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2)

device = torch.device('cuda:0')
classifier = classifier.to(device)
delta = AdapterModel(classifier, bottleneck_dim=16 if "base" in path else 32)
delta.freeze_module(classifier, exclude=["adapter", "pooler", "classifier"])

criterion = nn.CrossEntropyLoss()
optimizer = Adam([p for p in classifier.parameters()], lr=1e-5, eps=1e-6, betas=(0.9, 0.999))
scheduler = ExponentialLR(optimizer, .67**(1/5000))


def train(dataset, batch_size = 8):

    bar = tqdm(range(0, len(dataset), batch_size))
    
    accuracy = torch.BoolTensor([]).to(device)

    for idx in bar:
        dataset_batch = dataset[idx:idx+batch_size]
        premises = [data["premise"] for data in dataset_batch]
        hypotheses = [data["hypothesis"] for data in dataset_batch]
        labels = [data["label"] for data in dataset_batch]

        inputs = tok(hypotheses, premises, padding=True, return_tensors='pt')
        
        for key in inputs.keys():
            inputs[key] = inputs[key][:, :512].to(device)
        scores = classifier(**inputs)[-1]
        labels = torch.LongTensor(labels).to(device)

        classifier.zero_grad()

        loss = criterion(scores, labels)
        
        predictions = scores.argmax(-1)
        
        accuracy = torch.cat((accuracy, predictions == labels), 0)

        loss.backward()

        optimizer.step()
        scheduler.step()

        bar.set_description(f'@Train #Loss={loss:.4} #Accuracy={accuracy.float().mean():.4}')
        
def evaluate(dataset, batch_size = 8):
    
    with torch.no_grad():

        bar = tqdm(range(0, len(dataset), batch_size))

        accuracy = torch.BoolTensor([]).to(device)

        for idx in bar:
            dataset_batch = dataset[idx:idx+batch_size]
            premises = [data["premise"] for data in dataset_batch]
            hypotheses = [data["hypothesis"] for data in dataset_batch]
            labels = [data["label"] for data in dataset_batch]

            inputs = tok(premises, hypotheses, padding=True, return_tensors='pt')

            for key in inputs.keys():
                inputs[key] = inputs[key][:, :512].to(device)
            scores = classifier(**inputs)[-1]
            labels = torch.LongTensor(labels).to(device)

            loss = criterion(scores, labels)

            predictions = scores.argmax(-1)

            accuracy = torch.cat((accuracy, predictions == labels), 0)

            bar.set_description(f'@Evaluate #Loss={loss:.4} #Accuracy={accuracy.float().mean():.4}')


def load_dataset(fname):
    
    labels = ['Contradiction', 'Entailment']

    dataset = []

    items = json.load(open(fname))

    for key in items.keys():
        item = items[key]
        if item['Type'] == "Single":
            section = item['Section_id']
            primary = item['Primary_id']
            ids = item['Primary_evidence_index']
            lines = json.load(open(f"data/CT/{primary}.json"))[section]
            premise = f"{section}: " + " ".join([lines[idx].strip() for idx in ids])
            hypothesis = item['Statement']
            label = item['Label']

            data = {
                "premise":premise,
                "hypothesis":hypothesis,
                "label":labels.index(label),
            }

            dataset.append(data)
            
    return dataset


dataset_train = load_dataset("data/train.json")
dataset_dev = load_dataset("data/dev.json")


for _ in range(max_epoch):
    train(dataset_train)
    evaluate(dataset_dev)



