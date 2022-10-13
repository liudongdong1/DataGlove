import torch
import torch.nn as nn
import torchvision
import numpy as np
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size,hidden_size2,num_classes,dropout = 0.):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),         # Todo
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size2),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_size2, hidden_size),
            nn.GELU(), 
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),         # Todo
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),         # Todo
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),         # Todo
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),         # Todo
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),         # Todo
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),


            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x=self.net(x)
        return x


if __name__ == "__main__":
    data = torch.ones([10,25])
    print(data.shape)
    model = NeuralNet(input_size=25,hidden_size=1024,hidden_size2=524,num_classes=26)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(data)

    print("Shape of out :", out_img.shape)  # [10,30,10]
