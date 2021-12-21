import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from model import Net , Net_orig
from test_model import LoadData,test_model
from torch.utils.data import random_split as random_split

def create_parser():
    parser=argparse.ArgumentParser(description='Train or test a simple neural network on ESM embeddings')
    parser.add_argument('--train',type=int,default=1,help='Specifies Train of test. Default True (Train)')
    parser.add_argument('--origin',type=int,default=0,help='Specifies whether or not to append Reference Sequences. Default False')
    parser.add_argument('--epochs',default=100,type=int,help='Specifices number of training epochs. Default 100')
    return parser


def trainer(args):
    orig=args.origin
    if args.train:
        print("Training with {} epochs. Original Sequences Added: {}".format(args.epochs,orig))
        dataset=LoadData(train=True,orig=orig)
        train_len=int(len(dataset)*.8)
        test_len = len(dataset)-train_len
        train,test=random_split(dataset,[train_len,test_len])
       
        train_loader = DataLoader(train, batch_size=32)
        test_loader = DataLoader(test, batch_size=32)

        print ("Done with preprocessing")
        if orig:
            model=Net_orig()
        else:
            model=Net()
        print ("model loaded")
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        train_loss_progress=[]
        test_acc_progress=[]
        epochs=args.epochs
        model.to(device)

        for epoch in range(epochs):
            model.train()
            running_c=0
            train_loss=0
            for (data, targets) in train_loader:
                data,targets=data.to(device),targets.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                running_c += torch.sum(predicted == targets.data)

            train_loss = train_loss / len(train_loader.dataset)
            train_loss_progress.append(train_loss)

            print('Epoch: {} \tTraining Loss: {:.6f} ||| Accuracy: {:.1f}%'.format(epoch + 1, train_loss,running_c/len(train_loader.dataset)*100))

            correct, total, test_loss = 0, 0, 0.0
            model.eval()

            #val
            with torch.no_grad():
                for (data,targets) in test_loader:

                    data,targets=data.to(device),targets.to(device)
                    test_outputs = model(data)
                    _, predicted = torch.max(test_outputs.data, 1)
                    total += targets.size(0)
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
        plt.clf()
    print ('Testing Model...')
    if (orig):
       orig=True
    else:
       orig=False
    test_model(orig=orig,train=False)


if __name__ == '__main__':
    trainer(args=create_parser().parse_args())
