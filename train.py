import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertModel, BertTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

txt = "phishing is the target of today"
tokenized_text = tokenizer(txt)
tokenized_text

indexed_tokens = torch.tensor(tokenized_text["input_ids"]).unsqueeze(0)
attention_mask = torch.tensor(tokenized_text["attention_mask"]).unsqueeze(0)

#Embeddigns
output = model(indexed_tokens, attention_mask)
output[1].shape

from data_utils import data_extraction
path_to_json = "sample.json"
data = data_extraction(path_to_json)

text, labels, urls = data

#encoding the labels and converting into tensors
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels_idx = le.fit_transform(labels)
labels_idx = torch.LongTensor(labels_idx)

from torch.utils.data import DataLoader, Dataset

from dataloader import extracted_data
trainset = extracted_data(text, urls, labels_idx, tokenizer)
train_loader = DataLoader(trainset, batch_size=8)

epochs = 5
optimizer = torch.optim.AdamW(model.parameters(),0.00001)
loss_fn = nn.CrossEntropyLoss()

from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained(model_name)
model.to(DEVICE)

from tqdm import tqdm
model.train()
for  epoch in range(epochs):
    print(epoch)
    
    loop=tqdm(enumerate(train_loader),leave=False,total=len(train_loader))
    for batch, dl in loop:
        ids=dl['text_input_ids'].squeeze(1)
        token_type_ids=dl['text_token_type_ids'].squeeze(1)
        mask= dl['text_attention_mask'].squeeze(1)
        label=dl['labels']
        label = label.unsqueeze(1)
        
        optimizer.zero_grad()
        
        import pdb; pdb.set_trace()

        output=model(
            input_ids=ids,
            attention_mask=mask)
        label = label.type_as(output)

        loss=loss_fn(output,label)
        loss.backward()
        
        optimizer.step()

