import torch.nn as nn


class UpSample(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.conv2d = nn.Conv2d(64, ratio*ratio*3, (1, 1), padding='same')
        self.pixel_shuffle = nn.PixelShuffle(ratio)
        
    def forward(self, inputs):
        x = self.conv2d(inputs)
        outputs = self.pixel_shuffle(x)
        return outputs
    