import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm.notebook import tqdm
import math 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
#-----------------------------------------Model Definition-------------------------------------
class SimpleRNN_Multibatch(nn.Module): 
    def __init__(self, vocab_size, input_size, hidden_size, num_layers=1):
        super(SimpleRNN_Multibatch, self).__init__()
        self.embed = nn.Embedding(vocab_size, input_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, x_length):        
        x = self.embed(x)
        X = pack_padded_sequence(x, x_length, batch_first=True)
        o, (h, c) = self.lstm(X)
        o, _ = pad_packed_sequence(o, batch_first = True)
        o = self.linear(o)
        return o
#------------------------------------------Dataset----------------------------------------------
class TrnWikiDataset(Dataset): 
    def __init__(self, trn_file, val_file):
        self.word2idx, self.word2idx_ = {}, {}
        self.word2idx_['UNK'] = 1
        self.word2idx_['PAD'] = 0
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
#         print('val y')
        return self.val_ids[idx][:-1], self.val_ids[idx][1:]
    
def collate_fn(batch):
    x_length = np.asarray([len(item[0]) for item in batch])
    perm_idx = np.asarray((-x_length).argsort()).astype(int)
    x_length = x_length[perm_idx]
    batch = [batch[i] for i in perm_idx]
    max_len = max([len(item[0]) for item in batch])
    in_sent = np.zeros((len(batch), max_len))
    target = np.zeros((len(batch), max_len))
    for i in range(len(in_sent)):
        in_sent[i][:len(batch[i][0])]= batch[i][0]
    for i in range(len(target)):
        target[i][:len(batch[i][1])]= batch[i][1]
    return (torch.tensor(in_sent).type(torch.LongTensor), torch.tensor(target).type(torch.LongTensor), x_length)
#------------------------------------------Training/Evaluate------------------------------------------
def train(model, dataLoader, optimizer, criterion, clip, t_wl, use_cuda=False):
    if use_cuda == True:
        model.cuda()
    model.train()
    epoch_loss = 0
    total_cnt = 0
    cor_cnt = 0
    train_predictions = []
    for i, (x, y, l) in tqdm(enumerate(dataLoader)):
        if use_cuda == True:
            x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        pred_y = model(x, l)

        total_cnt += len(x.cpu().numpy().reshape(-1))
        cor_cnt += torch.eq(torch.argmax(pred_y, dim=2), y).sum().item()
        loss = criterion(pred_y.view(-1, model.vocab_size), y.view(-1))
        loss.backward()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / t_wl, cor_cnt / total_cnt * 100

def evaluate(model, dataLoader, criterion, v_wl, use_cuda=False):
    model.eval()
    val_predictions = []
    if use_cuda == True:
        model.cuda()
    epoch_loss = 0
    total_cnt = 0
    cor_cnt = 0
    with torch.no_grad():
        for i, (x, y, l) in tqdm(enumerate(dataLoader)):
            if use_cuda == True:
                x, y = x.cuda(), y.cuda()
            pred_y = model(x, l) 
            total_cnt += len(x.cpu().numpy().reshape(-1))
            cor_cnt += torch.eq(torch.argmax(pred_y, dim=2), y).sum().item()
            loss = criterion(pred_y.view(-1, model.vocab_size), y.view(-1))
            epoch_loss += loss.item()
    return epoch_loss / v_wl, cor_cnt / total_cnt * 100
#------------------------------------------Training Section------------------------------------------
N_EPOCHS = 10
CLIP = 3
best_valid_loss = float('inf')
trn_fn = '/kaggle/input/dataforlangmodeling/trn-wiki.txt'
val_fn = '/kaggle/input/dataforlangmodeling/dev-wiki.txt'
train_dataset = TrnWikiDataset(trn_fn, val_fn)
len_all_tw = 0
for i in range(len(train_dataset)):
    len_all_tw += len(train_dataset[i][0])
print('len_val_all_words', len_all_tw)
print('PAD Index:', train_dataset.word2idx_['PAD'])
print('UNK Index:', train_dataset.word2idx_['UNK'])
val_dataset = ValWikiDataset(train_dataset.word2idx_, val_fn)
len_all_vw = 0
for i in range(len(val_dataset)):
    len_all_vw += len(val_dataset[i][0])
print('len_val_all_words', len_all_vw)

batch_size = 16
train_dataLoader = DataLoader(train_dataset, batch_size, collate_fn=collate_fn, shuffle=True, num_workers=0)
val_dataLoader = DataLoader(val_dataset, batch_size, collate_fn=collate_fn, shuffle=True, num_workers=0)
num_layers = 1
model = SimpleRNN_Multibatch(len(train_dataset.word2idx_), 32, 32, num_layers)
optimizer = optim.SGD(model.parameters(), lr=0.5)
criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
record = True
for epoch in range(N_EPOCHS):       
    train_loss, train_acc = train(model, train_dataLoader, optimizer, criterion, CLIP, len_all_tw, use_cuda = True)
    valid_loss, valid_acc = evaluate(model, val_dataLoader, criterion, len_all_vw, use_cuda = True)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
    print('Epoch:', epoch)
    print(f'\tTrain Loss: {train_loss:.3f}, Train Acc: {train_acc :.3f} | My Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}, Valid Acc: {valid_acc:.3f} |  My Val. PPL: {math.exp(valid_loss):7.3f}')
print()

