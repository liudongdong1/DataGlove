from torch import nn
from einops.layers.torch import Rearrange
import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange

# code from https://github.com/rishikksh20/MLP-Mixer-pytorch
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn. GELU (),         # Todo
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x=self.net(x)
        return x

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):

    def __init__(self, in_channels, dim, num_classes, num_patch, image_size, depth, token_dim, channel_dim):
        super().__init__()

        #assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch = num_patch
        # self.to_patch_embedding = nn.Sequential(
        #     nn.Conv2d(in_channels, dim, patch_size, patch_size),    #3,512,16,16  -> 512,14,14
        #     Rearrange('b c h w -> b (h w) c'),                     # 196,512
        # )
        self.to_patch_embedding=nn.Sequential(
            nn.Conv1d(in_channels, dim,1,stride=1),   # 1*5*5   =  1*64*5
            Rearrange('b d n -> b n d')                    #  [100, 128,5 ]-> [100, 5, 128]
        )
        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        #print(x.shape)
        x = self.to_patch_embedding(x)
        #print("to_patch_embedding",x.shape)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)
        x = x.mean(dim=1)
        return self.mlp_head(x)





class CharRNN(nn.Module):
    
    def __init__(self, tokens, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)   #注意这里
        self.dropout=nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(self.chars))
      
    
    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
        r_output,hidden=self.lstm(x,hidden)
        out=self.dropout(r_output)
        out=out.contiguous().view(-1,self.n_hidden)
        out=self.fc(out)
        return out, hidden
    
    
    def init_hidden(self, batch_size,train_on_gpu=False):
        ''' Initializes hidden state '''
        weight = next(self.parameters()).data
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden


if __name__ == "__main__":
    img = torch.ones([1, 5, 5])
    print(img.shape)
    model = MLPMixer(in_channels=5, image_size=224, num_patch=5, num_classes=10,
                     dim=512, depth=8, token_dim=256, channel_dim=2048)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(img)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]