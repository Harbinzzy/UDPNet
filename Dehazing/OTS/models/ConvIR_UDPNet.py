import torch
import torch.nn as nn
import torch.nn.functional as F
from .convirlayers import *

class fusion2(nn.Module):
    def __init__(self, out_channel):
        super(fusion2, self).__init__()
        self.rpi = torch.zeros((16 * 16, (int(16 * 0.5) + 16) ** 2), dtype=torch.long)
        self.ocab = OCAB(dim=out_channel, window_size=16, overlap_ratio=0.5, num_heads=2, norm_layer=nn.LayerNorm)
        self.main = nn.Sequential(
            BasicConv(1, 16, kernel_size=3, stride=1, relu=True),
            BasicConv(16, 32, kernel_size=1, stride=1, relu=True),
            BasicConv(32, 32, kernel_size=3, stride=1, relu=True),
            BasicConv(32, 64, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(64, affine=True)
        )
    def forward(self, x, depth):
        # print(out1.shape)
        out1  = self.ocab(x,self.main(depth), self.rpi)
        out2  = self.ocab(self.main(depth),x, self.rpi)
        return out1+ out2 + x
    
class fusion3(nn.Module):
    def __init__(self, out_channel):
        super(fusion3, self).__init__()
        self.rpi = torch.zeros((16 * 16, (int(16 * 0.5) + 16) ** 2), dtype=torch.long)
        self.ocab = OCAB(dim=out_channel, window_size=16, overlap_ratio=0.5, num_heads=2, norm_layer=nn.LayerNorm)
        self.main = nn.Sequential(
            BasicConv(1, 32, kernel_size=3, stride=1, relu=True),
            BasicConv(32, 64, kernel_size=1, stride=1, relu=True),
            BasicConv(64, 64, kernel_size=3, stride=1, relu=True),
            BasicConv(64, 128, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(128, affine=True)
        )
    def forward(self, x, depth):
        # print(out1.shape)
        out1  = self.ocab(x,self.main(depth), self.rpi)
        out2  = self.ocab(self.main(depth),x, self.rpi)
        return out1+ out2 + x
    
class fusion1(nn.Module):
    def __init__(self, out_channel):
        super(fusion1, self).__init__()
        self.rpi = torch.zeros((16 * 16, (int(16 * 0.5) + 16) ** 2), dtype=torch.long)
        self.ocab = OCAB(dim=out_channel, window_size=16, overlap_ratio=0.5, num_heads=2, norm_layer=nn.LayerNorm)
        self.main = nn.Sequential(
            BasicConv(1, 8, kernel_size=3, stride=1, relu=True),
            BasicConv(8, 16, kernel_size=1, stride=1, relu=True),
            BasicConv(16, 16, kernel_size=3, stride=1, relu=True),
            BasicConv(16, 32, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(32, affine=True)
        )
    def forward(self, x, depth):
        # print(out1.shape)
        out1  = self.ocab(x,self.main(depth), self.rpi)
        out2  = self.ocab(self.main(depth),x, self.rpi)
        return out1+ out2 + x

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res-1)]
        layers.append(ResBlock(out_channel, out_channel, filter=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res-1)]
        layers.append(ResBlock(channel, channel, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(4, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))

class DepthRefinement(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64):
        super(DepthRefinement, self).__init__()
        self.refinement = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, depth):
        return self.refinement(depth)
    
class DepthGuidedFusionModule(nn.Module):
    def __init__(self, in_channels=4, out_channels=64):
        super(DepthGuidedFusionModule, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU()
        )
        
        self.semantic_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_block(x)
        
        attention_weights = self.semantic_attention(conv_out)
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)
        attention_out = x * attention_weights
        
        return attention_out

class ConvIR(nn.Module):
    def __init__(self):
        super(ConvIR, self).__init__()
    
        num_res = 8
        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.fusion = nn.ModuleList([
            fusion1(base_channel),
            fusion2(base_channel*2),  
            fusion3(base_channel*4),
        ]) 

        self.feat_extract = nn.ModuleList([
            BasicConv(4, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.depth_refinement = DepthRefinement()
        self.attention = DepthGuidedFusionModule()

    def forward(self, x):
        depth = x[:,3:,:,:]
        depth_2 = F.interpolate(depth, scale_factor=0.5)
        depth_4 = F.interpolate(depth_2, scale_factor=0.5)
        optimized_depth = self.depth_refinement(depth)
        x = torch.cat([x[:, :3, :, :], optimized_depth], dim=1)
        x = self.attention(x)
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        # 256
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        res1 = self.fusion[0](res1, depth)
        # 128
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        res2 = self.fusion[1](res2, depth_2)
        # 64
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)
        z = self.fusion[2](z, depth_4)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        # 128
        z = self.feat_extract[3](z)
        x_4 = x_4[:,:3,:,:]
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        # 256
        z = self.feat_extract[4](z)
        x_2 = x_2[:,:3,:,:]
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        x = x[:,:3,:,:]
        outputs.append(z+x)

        return outputs


def build_net():
    return ConvIR()