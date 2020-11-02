import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm
import math 
import copy
import random

#-----------------------------------------Model Definition-------------------------------------
class SimpleRNN(nn.Module): 
    def __init__(self, vocab_size, input_size, hidden_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        o, (h, c) = self.lstm(x)
        o = self.linear(o)
        return o

#------------------------------------------Dataset----------------------------------------------
class TrnWikiDataset(Dataset): 
    def __init__(self, trn_file, val_file):
        self.word2idx, self.word2idx_ = {}, {}
#         self.word2idx['UNK'] = [0, 9999]
        self.word2idx_['UNK'] = 0
        self.trn_ids = []
        with open(val_file) as f: 
            for l in f: 
                for w in l.split(' '):
                    if w not in self.word2idx.keys():
                        self.word2idx[w] = [len(self.word2idx), 1]
                    else:
                        self.word2idx[w][1] += 1 
        with open(trn_file) as f: 
            for l in f: 
                for w in l.split(' '):
                    if w not in self.word2idx.keys():
                        self.word2idx[w] = [len(self.word2idx), 1]
                    else:
                        self.word2idx[w][1] += 1
            for k in self.word2idx.keys():
                if self.word2idx[k][1] >= 5:
                    self.word2idx_[k] = len(self.word2idx_)
            print("Length of dictionary after deleting rare terms: ", len(self.word2idx_))
        with open(trn_file) as f:
            for l in f:
                ids = [] 
                for w in l.split(' '): 
                    if self.word2idx_.__contains__(w) == False:
                        ids.append(self.word2idx_['UNK'])
                    else:
                        ids.append(self.word2idx_[w])
                self.trn_ids.append(torch.tensor(ids).type(torch.int64))

    def __len__(self):
        return len(self.trn_ids)

    def __getitem__(self, idx): 
        return self.trn_ids[idx][:-1], self.trn_ids[idx][1:]


class ValWikiDataset(Dataset):
    def __init__(self, word2idx, val_file):
        self.word2idx = word2idx
        self.val_ids = []
        with open(val_file) as f: 
            for l in f:
                ids = [] 
                for w in l.split(' '):
                    if self.word2idx.__contains__(w) == False:
                        ids.append(self.word2idx['UNK'])
                    else: 
                        ids.append(self.word2idx[w])
                self.val_ids.append(torch.tensor(ids).type(torch.int64))
    def __len__(self):
        return len(self.val_ids)
    def __getitem__(self, idx): 
        return self.val_ids[idx][:-1], self.val_ids[idx][1:]

#------------------------------------------Training/Evaluate------------------------------------------
def train(model, dataLoader, optimizer, criterion, clip, use_cuda=False):
    if use_cuda == True:
        model.cuda()
    model.train()
    epoch_loss = 0
    total_cnt = 0
    cor_cnt = 0
    train_predictions = []
    for i, (x, y) in tqdm(enumerate(dataLoader)):
        if use_cuda == True:
            x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        pred_y = model(x)
        total_cnt += len(x.cpu().numpy().reshape(-1))
        cor_cnt += torch.eq(torch.argmax(pred_y, dim=2), y).sum().item()
        loss = criterion(pred_y.view(-1,20378), y.view(-1).type(torch.LongTensor).cuda())
        loss.backward()

        # Perpexity calcualtion
        pred_y = F.softmax(pred_y, dim=2).data
        pred_y, y = pred_y.cpu().detach().numpy(), y.cpu().numpy()
        for i in range(len(pred_y)): 
            tp = []
            for j in range(len(pred_y[i])):
                tp.append(pred_y[i][j][y[i][j]])
            train_predictions.append(tp)
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()        
    return epoch_loss / len(dataLoader.dataset), cor_cnt / total_cnt * 100, perplexity(train_predictions)

def evaluate(model, dataLoader, criterion, use_cuda=False):
    model.eval()
    val_predictions = []
    if use_cuda == True:
        model.cuda()
    epoch_loss = 0
    total_cnt = 0
    cor_cnt = 0
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(dataLoader)):
            if use_cuda == True:
                x, y = x.cuda(), y.cuda()
            pred_y = model(x) 
            total_cnt += len(x.cpu().numpy().reshape(-1))
            cor_cnt += torch.eq(torch.argmax(pred_y, dim=2), y).sum().item()
            loss = criterion(pred_y.view(-1, 20378), y.view(-1).type(torch.LongTensor).cuda())
            epoch_loss += loss.item()

            # Perpexity calcualtion
            pred_y = F.softmax(pred_y, dim=2).data
            pred_y, y = pred_y.cpu().detach().numpy(), y.cpu().numpy()
            for i in range(len(pred_y)): 
                vp = []
                for j in range(len(pred_y[i])):
                    vp.append(pred_y[i][j][y[i][j]])
                val_predictions.append(vp)
    return epoch_loss / len(dataLoader.dataset), cor_cnt / total_cnt * 100, perplexity(val_predictions)

def perplexity(predictions):
    num_sentence = len(predictions)
    predictions = [item for sublist in predictions for item in sublist]
    predictions = np.asarray(predictions)
    denominator = len(predictions) + num_sentence
    numerator = np.sum(np.log(predictions))
    return math.exp(-numerator/denominator)
    
#------------------------------------------Training Section------------------------------------------
N_EPOCHS = 10
CLIP = 0.25
best_valid_loss = float('inf')
trn_fn = '/kaggle/input/trn-wiki.txt'
val_fn = '/kaggle/input/dev-wiki.txt'
train_dataset = TrnWikiDataset(trn_fn, val_fn)
print('len', len(train_dataset))
val_dataset = ValWikiDataset(train_dataset.word2idx_, val_fn)
batch_size = 1
train_dataLoader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
val_dataLoader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=0)
model = SimpleRNN(len(train_dataset.word2idx_), 32, 32)
optimizer1 = optim.SGD(model.parameters(), lr=0.5)
optimizer2 = optim.Adam(model.parameters(), lr=0.001)
optimizer3 = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
optimizer4 = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer5 = optim.SGD(model.parameters(), lr=0.5, momentum=0.8)
criterion = nn.CrossEntropyLoss()
for epoch in range(N_EPOCHS):   	
    train_loss, train_acc, mt_ppl = train(model, train_dataLoader, optimizer1, criterion, CLIP, use_cuda = True)
    valid_loss, valid_acc, mv_ppl = evaluate(model, val_dataLoader, criterion, use_cuda = True)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
    print('Epoch:', epoch)
    print(f'\tTrain Loss: {train_loss:.3f}, Train Acc: {train_acc :.3f} | Train PPL: {math.exp(train_loss):7.3f} | My Train PPL: {mt_ppl:7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}, Valid Acc: {valid_acc:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} | My Val PPL: {mv_ppl:7.3f}')
print()
    