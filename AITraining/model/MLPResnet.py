import torch
import torch.nn as nn
import torchvision
import numpy as np
from einops.layers.torch import Rearrange
from torchvision import models
import torch.nn as nn
class MLPResnet(nn.Module):

    def __init__(self, inputdim,  hidden_dim, outputdim,dropout = 0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(inputdim, hidden_dim),
            nn. GELU (),         # Todo
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, outputdim),
            nn.Dropout(dropout)
        )
        #  这里大小 变为 B*dim--> B*hidden_dim-->B*outputdim     B*256  这里需要reshape 16*16
        self.resnet50 = models.resnet50(pretrained=True)   #print(net.conv1.weight)    net.fc = nn.Linear(2048, 2)
        self.resnet50.conv1=nn.Conv2d(1,64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.resnet50.fc = nn.Sequential(
            nn.Linear(self.resnet50.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 26),
            nn.LogSoftmax(dim=1)
        )


    def forward(self, x):
        x = self.net(x)  #  x: B*25;   B*256
        print(x.shape)
        x=torch.reshape(x,(-1,1,16,16))
        print(x.shape)
        x=self.resnet50(x)
        #print(x.shape)
        return x

if __name__ == "__main__":
    img = torch.ones([10, 5])
    print(img.shape)
    model = MLPResnet(inputdim=5,hidden_dim=1024,outputdim=256)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    from torch.autograd import Variable
    from torchsummary import summary
    summary(model,5) #模型参数，输入数据的格式
    out_img = model(img)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]

