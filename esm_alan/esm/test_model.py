import os
import esm
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.utils.data import random_split
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
from model import Net, Net_orig
import argparse

class LoadData(Dataset):
    def __init__(self,train, orig, transform=None):
        self.train=train
        self.orig=orig
        self.transform=transform
        self.true_false=[0,0]
        self.data_path=[]
        self.fasta_path = '/data/alant/ESM/fasta_files/0.fasta'
        self.embed_path = '/data/alant/ESM/train_embeddings'
        if not self.train:
            self.fasta_path = '/data/alant/ESM/fasta_files/20000.fasta'
            self.embed_path = '/data/alant/ESM/test_embeddings'
        for header,sequence in esm.data.read_fasta(self.fasta_path):
            if (header.split('-')[-1])=='0':
                self.true_false[0]+=1
            else:
                self.true_false[1]+=1
            self.data_path.append([header,sequence])
        print("0's: {}, 1's: {}".format(*self.true_false))

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self,idx):
        header,sequence=self.data_path[idx]
        target = header.split('-')[-1]
        embedding = torch.load(os.path.join(self.embed_path, 'mutant', '{}.pt'.format(header[1:])))['mean_representations'][33]
        if (self.orig):
            header = header.split('-')
            embedding = torch.cat((embedding, torch.load(
                os.path.join(self.embed_path, 'orig', '{}.pt'.format(header[0][1:] + '-' + header[1])))[
                'mean_representations'][33]))
        return (embedding,int(target))


def plot_roc(_preds, _labels, orig):
    # plot ROC curve
    _preds = torch.reshape(_preds, (-1,))
    fpr, tpr, threshold = metrics.roc_curve(_labels.cpu().numpy(), _preds.cpu().numpy())
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('ROC')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if orig:
        plt.savefig('roc_orig.png')
    else:
        plt.savefig('roc.png')
    plt.clf()
    print ('Curve plotted')

def test_model(orig,train):
    if orig:
        dataset=LoadData(train=False,orig=orig)
    else:
        dataset=LoadData(False,orig=orig)

    test = DataLoader(dataset, batch_size=32,shuffle=False)

    if not orig:
        model=Net()
    else:
        model=Net_orig()
    if not orig:
        model.load_state_dict(torch.load('/home/alant/ESM/weights/weights.pt'))
    else:
        model.load_state_dict(torch.load('/home/alant/ESM/weights/weights_orig.pt'))
    print ("model loaded")

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _preds=torch.tensor([]).to(device)
    _labels=torch.tensor([]).to(device)
    model.to(device)
    total=0
    correct=0

    #test the model
    with torch.no_grad():
        for (data,targets) in  test:
            data=data.to(device)
            targets=targets.to(device)
            test_outputs = model(data)
            confidence, predicted = torch.max(test_outputs.data, 1)
            _preds=torch.cat((_preds,test_outputs[:,-1:]))
            _labels=torch.cat((_labels,targets))
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        print('Testing Accuracy: {:.1f}'.format(correct / total * 100))
        plot_roc(_preds,_labels,orig)

if __name__=='__main__':
    print ('Test via the train file. (Specify --train = 0)')
