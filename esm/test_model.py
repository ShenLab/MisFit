import os
import esm
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
from model import Net, Net_orig
import argparse

def get_dataset(train,orig):
    datapoints=[]
    
    benign=0
    pathnogenic=0
    fasta_path='/data/alant/ESM/fasta_files/0.fasta'
    embed_path='/data/alant/ESM/train_embeddings'
    if not train:
        fasta_path='/data/alant/ESM/fasta_files/20000.fasta'
        embed_path='/data/alant/ESM/test_embeddings'
    for header,sequence in esm.data.read_fasta(fasta_path):
        target=header.split('-')[-1]
       
        if target=='0':
            benign+=1
        else:
            pathnogenic+=1
        embedding=torch.load(os.path.join(embed_path,'mutant','{}.pt'.format(header[1:])))['mean_representations'][33]
        if (orig):
            header=header.split('-')

            embedding=torch.cat((embedding,torch.load(os.path.join(embed_path,'orig','{}.pt'.format(header[0][1:]+'-'+header[1])))['mean_representations'][33]))
        datapoints.append((embedding,int(target)))
    print (benign,pathnogenic)
    return datapoints


def test_model(orig,train):
    if  orig:
        dataset=get_dataset(train=True,orig=orig)
    else:
        dataset=get_dataset(False,orig)
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
    _preds=torch.tensor([]).cuda()
    _labels=torch.tensor([]).cuda()
    model.cuda()
    training_acc=[]
    total=0
    correct=0
    with torch.no_grad():
        for (data,targets) in  test:
            data=data.cuda()
            targets=targets.cuda()
            test_outputs = model(data)
           # print (test_outputs)
            confidence, predicted = torch.max(test_outputs.data, 1)
            _preds=torch.cat((_preds,test_outputs[:,-1:]))
            _labels=torch.cat((_labels,targets))
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print (_labels)
    _preds=torch.reshape(_preds,(-1,))
    print (_preds.cpu().numpy())
    print ('Testing Accuracy: {:.1f}'.format(correct/total*100))
    fpr, tpr, threshold = metrics.roc_curve(_labels.cpu().numpy(), _preds.cpu().numpy())
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('ROC')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if orig:
      plt.savefig('roc_orig.png')
    else:
      plt.savefig('roc.png')
    
