import torch
import torch.nn as nn


class SFENet(nn.Module):
    def __init__(self):
        super(SFENet, self).__init__()
        self.SFE1 = nn.Conv2d(3, 64, (3, 3), padding='same')
        self.SFE2 = nn.Conv2d(64, 64, (3, 3), padding='same')
        
    def forward(self, lr):
        F_1 = self.SFE1(lr)
        F0 = self.SFE2(F_1)
        
        return F_1, F0
    
class ResidualDenseBlock(nn.Module):
    def __init__(self, C, G):
        super(ResidualDenseBlock, self).__init__()
        self.C = C
        self.G = G
        
        self.convolutions = nn.ModuleList(
            [self.ConvBlock(64, self.G)] + \
            [self.ConvBlock(64+self.G*i, self.G) for i in range(1, self.C-1)]
        )
        
        self.conv1d = nn.Conv2d(64+self.G*(self.C-1), 64, (1, 1), padding='same')
        
    def ConvBlock(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (3, 3), padding='same'),
            nn.ReLU()
        )
        
    def forward(self, inputs):
        x = inputs
        fs = []
        
        # contiguous memory
        for conv in self.convolutions:
            fs.append(x)
            x = conv(torch.cat(fs, dim=1))
        fs.append(x)
        
        # local feature fusion
        x = torch.cat(fs, dim=1)
        x = self.conv1d(x)
        
        # local residual learning
        outputs = x + inputs
        
        return outputs
    
class ResidualDenseNetwork(nn.Module):
    def __init__(self, D, C, G):
        super(ResidualDenseNetwork, self).__init__()
        self.D = D
        self.C = C
        self.G = G
        self.sfe = SFENet()
        self.rdbs = nn.ModuleList([
            ResidualDenseBlock(self.C, self.G) for _ in range(self.D)]
        )
        self.gff = nn.Sequential(
            nn.Conv2d(64*self.D, 64, (1, 1), padding='same'),
            nn.Conv2d(64, 64, (3, 3), padding='same')
        )
        
    def forward(self, inputs):
        f_1, f0 = self.sfe(inputs)
        
        x = f0
        fs = []
        for rdb in self.rdbs:
            x = rdb(x)
            fs.append(x)
        fs = torch.cat(fs, dim=1)
        fs = self.gff(fs)
        outputs = f_1 + fs
        
        return outputs