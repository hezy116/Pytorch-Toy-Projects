'''
A Simplified Residual U-Net implementation for noise prediction in DDPM
'''

import torch
import torch.nn as nn
import math

Time_emb_dim = 32
Image_channels = 1

class SinusoidalPositionEmbeddings(nn.Module):
    """
    (正弦位置嵌入) Used to generate a time embedding tensor from time t
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """
    ResBlock is used to extract features from the input image and time embedding.
    It strcture is as follows:
        1. ReLU + Conv1 x
        2. x + MLP(time_emb)
        3. ReLU + Conv2 x

    In step 2, addition of MLP(time_emb) is used to inforce/deforce the information in the feature channels.
    MLP is used to decide the inforce/deforce amount based on time_emb tensor.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.relu = nn.ReLU()

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(Time_emb_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x, time_emb):
        residual = self.shortcut(x)

        h = self.relu(x)
        h = self.conv1(h)

        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
        h += time_emb

        h = self.relu(h)
        h = self.conv2(h)

        return residual + h

class ResUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_emb_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(Time_emb_dim),
            nn.Linear(Time_emb_dim, Time_emb_dim),
            nn.ReLU(),
            nn.Linear(Time_emb_dim, Time_emb_dim),
        )

        '''Initial convolution'''
        # channel -> 64
        self.init_conv = nn.Conv2d(Image_channels, 64, kernel_size=3, padding=1)

        '''Encoder'''
        # size 32 -> 16
        self.down1_res1 = ResBlock(64, 64)
        self.down1_res2 = ResBlock(64, 64)
        self.down1_pool = nn.MaxPool2d(kernel_size=2)
        # channel 64 -> 128, size 16 -> 8
        self.down2_res1 = ResBlock(64, 128)
        self.down2_res2 = ResBlock(128, 128)
        self.down2_pool = nn.MaxPool2d(kernel_size=2)

        '''Bottleneck'''
        self.mid_res1 = ResBlock(128, 128)
        self.mid_res2 = ResBlock(128, 128)

        '''Decoder'''
        # size 8 -> 16
        self.up1_sample = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.up1_res1 = ResBlock(128 + 128, 128)
        self.up1_res2 = ResBlock(128 + 128, 128)
        # channel 128 -> 64, size 16 -> 32
        self.up2_sample = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2_res1 = ResBlock(64 + 64, 64)
        self.up2_res2 = ResBlock(64 + 64, 64)

        '''Output'''
        self.act = nn.ReLU()
        self.final_conv = nn.Conv2d(64, Image_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        time_emb = self.time_emb_mlp(t)

        x = self.init_conv(x)
        skips = []

        '''Encoder'''
        x = self.down1_res1(x, time_emb); skips.append(x)
        x = self.down1_res2(x, time_emb); skips.append(x)
        x = self.down1_pool(x)

        x = self.down2_res1(x, time_emb); skips.append(x)
        x = self.down2_res2(x, time_emb); skips.append(x)
        x = self.down2_pool(x)

        '''Bottleneck'''
        x = self.mid_res1(x, time_emb)
        x = self.mid_res2(x, time_emb)

        '''Decoder'''
        x = self.up1_sample(x)
        x = torch.cat([x, skips.pop()], dim=1) # cat on channel dimension
        x = self.up1_res1(x, time_emb)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.up1_res2(x, time_emb)

        x = self.up2_sample(x)
        x = torch.cat([x, skips.pop()], dim=1) # cat on channel dimension
        x = self.up2_res1(x, time_emb)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.up2_res2(x, time_emb)

        '''Output'''
        x = self.act(x)
        x = self.final_conv(x)
        return x

