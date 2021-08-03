import torch
import torch.nn as nn
import torchvision


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
     

        self.hid1=nn.Linear(1280, 400)
        self.act1=nn.ReLU()
                 
        self.hid2=nn.Linear(400, 200)
        self.act2=nn.ReLU()
                 
        self.hid3=nn.Linear(200, 100)
        self.act3=nn.ReLU()
                 
                 
        self.hid4=nn.Linear(100, 2)
                 
        self.out=nn.Sigmoid()


    def forward(self, x):
        output=self.hid1(x)
        output=self.act1(output)
        
        output=self.hid2(output)
        output=self.act2(output)
        
        output=self.hid3(output)
        output=self.act3(output)
        
        output=self.hid4(output)
        output=self.out(output)
        
        return output


if __name__==('__main__'):
   print (Net())
