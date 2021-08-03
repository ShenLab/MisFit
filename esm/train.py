import argparse
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
from model import Net , Net_orig
from test_model import get_dataset,test_model

def create_parser():
    parser=argparse.ArgumentParser(description='Train or test a simple neural network on ESM embeddings')
    parser.add_argument('--train',default=False,type=bool,help='Specifies Train of test. Default False (Test)')
    parser.add_arguemtn('--origin',default=True,type=bool,help='Specifies whether or not to append Reference Sequences. Default False')
    return parser
if __name__ == '__main__':
    args=create_parser().parse_args()
    orig=args.train
    if args.train:
        dataset=get_dataset(train=True,orig=orig)

        train,test=train_test_split(dataset,test_size=.2)

        train_loader = DataLoader(train, batch_size=32)
        test_loader = DataLoader(test, batch_size=32)
    #    for target,label in test_loader:
    #      print (target.size())
        print ("Done with preprocessing")
        if orig:
            model=Net_orig()
        else:
            model=Net()
    #    print (model)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        train_loss_progress=[]
        test_acc_progress=[]
        epochs=100
        _targets=torch.tensor([]).to(device)
        _labels=torch.tensor([]).to(device)
        model.to(device)
        for epoch in range(epochs):
            model.train()
            running_c=0
            train_loss=0
            for (data, targets) in train_loader:
                data,targets=data.to(device),targets.to(device)
                optimizer.zero_grad()
                output = model(data)
         #       print (data.ndim)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                running_c += torch.sum(predicted == targets.data)
            train_loss = train_loss / len(train_loader.dataset)
            train_loss_progress.append(train_loss)
            print('Epoch: {} \tTraining Loss: {:.6f} ||| Accuracy: {:.1f}%'.format(epoch + 1, train_loss,running_c/len(train_loader.dataset)*100))

            correct = 0
            total = 0
            test_loss=0.0
            model.eval()
            with torch.no_grad():
                for (data,targets) in  test_loader:
                    data,targets=data.to(device),targets.to(device)
                    test_outputs = model(data)
                    _, predicted = torch.max(test_outputs.data, 1)
                    total += targets.size(0)
                    _targets=torch.cat((_targets,_))
                    _labels=torch.cat((_labels,targets))
                    correct += (predicted == targets).sum().item()


            test_acc_progress.append(100 * correct / total)

            print('Accuracy of the network on the test set: {:.1f}%'.format(100 * correct / total))
        if not orig:
            torch.save(model.state_dict(),'/home/alant/ESM/weights/weights.pt')
        else:
            torch.save(model.state_dict(),'/home/alant/ESM/weights/weights_orig.pt')
        x_range=np.arange(1,epochs+1)
        fig,axs=plt.subplots(2)
        axs[0].plot(x_range,train_loss_progress,c='b',label='Train Loss')
        axs[1].plot(x_range,test_acc_progress,c='r',label='Test Accuracy')
        axs[0].legend()
        axs[1].legend()
        plt.savefig('summary.png')
    elif not args.train:
        test_model(orig=orig,train=False)
