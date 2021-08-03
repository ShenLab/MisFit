import torch
import torch.nn as nn
import torchvision


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hidden1 = nn.Sequential(

                 nn.Linear(1280*2, 800),
                 nn.ReLU(),
#                 nn.Dropout(.4),
#                 nn.Linear(1600,1000),
#                 nn.Tanh(),
#                 nn.Dropout(.4),
#                 nn.Linear(3000,1500),
#                 nn.Tanh(),
#                 nn.Dropout(.4),
            #     nn.Linear(1000, 400),
                 
        #         nn.Tanh(),
#                 nn.Dropout(.4),
                 
                 nn.Linear(800, 300),
                 nn.ReLU(),
 #                nn.Dropout(.4),
                 nn.Linear(300, 100),
                 nn.ReLU(),
 #                nn.Dropout(.4),
                 nn.Linear(100, 2),
                 
                 nn.Sigmoid()

        )
#        self.lstm=nn.Sequential(
#            nn.LSTM(1280,10),
  #          nn.Linear(1280,1280),
  #          nn.Tanh(),
 #           nn.LSTM(1280,3)
 #          )


    def forward(self, x):
#        print (type(x))
  #      output=self.lstm(x.unsqueeze(0))
 #       print(output) 
        output = (self.hidden1(x))

        return output


if __name__==('__main__'):
   print (Net())
