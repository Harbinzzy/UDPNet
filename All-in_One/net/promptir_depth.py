## PromptIR: Prompting for All-in-One Blind Image Restoration
## Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2306.13090


import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from einops.layers.torch import Rearrange
import time
from options import options as opt

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

def trunc_normal_(tensor, std=0.02):
    nn.init.trunc_normal_(tensor, std=std)

def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class OCAB(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_heads,
                 qkv_bias=True, qk_scale=None, mlp_ratio=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # self.input_resolution = input_resolution  # (h, w)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.depth_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size),
                                stride=window_size,
                                padding=(self.overlap_win_size - window_size) // 2)
        num_relative_position = (window_size + self.overlap_win_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_position, num_heads))
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

    def forward(self, x, depth, rpi):
        # x: (b, c, h, w)，depth: (b, 1, h, w)
        b, c, h, w = x.shape
        orig_h, orig_w = h, w
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            depth = F.pad(depth, (0, pad_w, 0, pad_h))
            h, w = x.shape[2], x.shape[3]

        shortcut = x 

        x_feat = x.permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)
        x_feat = self.norm1(x_feat)  # (b, h, w, c)

        qkv = self.qkv(x_feat)  # (b, h, w, 3*c)
        qkv = qkv.reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2)  # (3, b, c, h, w)
        kv = qkv[1:3]  # (2, b, c, h, w)
        k = kv[0].permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)
        v = kv[1].permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)

        q = self.depth_proj(depth)  # (b, c, h, w)
        q = q.permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)

        q_windows = window_partition(q, self.window_size)  # (num_windows*b, window_size, window_size, c)
        q_windows = q_windows.view(-1, self.window_size * self.window_size, c)  # (num_windows*b, ws*ws, c)

        kv_cat = torch.cat([k, v], dim=-1)  # (b, h, w, 2*c)
        kv_cat = kv_cat.permute(0, 3, 1, 2).contiguous()  # (b, 2*c, h, w)
        kv_windows = self.unfold(kv_cat)  # (b, 2*c * (overlap_win_area), num_windows)
        kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch',
                               nc=2, ch=c, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous()
        k_windows, v_windows = kv_windows[0], kv_windows[1]  # (num_windows*b, overlap_win_area, c)

        b_windows, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = c // self.num_heads

        q_windows = q_windows.reshape(b_windows, nq, self.num_heads, d).permute(0, 2, 1, 3)  # (num_windows*b, num_heads, nq, d)
        k_windows = k_windows.reshape(b_windows, n, self.num_heads, d).permute(0, 2, 1, 3)
        v_windows = v_windows.reshape(b_windows, n, self.num_heads, d).permute(0, 2, 1, 3)

        q_windows = q_windows * self.scale
        attn = (q_windows @ k_windows.transpose(-2, -1))  # (num_windows*b, num_heads, nq, n)

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, ws*ws, ows*ows)
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)

        attn_windows = (attn @ v_windows).transpose(1, 2).reshape(b_windows, nq, c)  # (num_windows*b, ws*ws, c)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)  # (num_windows*b, ws, ws, c)
        x_out = window_reverse(attn_windows, self.window_size, h, w)  # (b, h, w, c)
        x_out = x_out.permute(0, 3, 1, 2).contiguous()  # (b, c, h, w)

        x_proj = self.proj(x_out.flatten(2).transpose(1, 2))  # (b, h*w, c)
        x_proj = x_proj.transpose(1, 2).view(b, c, h, w)         # (b, c, h, w)
        x_out = x_proj + shortcut

        x_flat = x_out.flatten(2).transpose(1, 2)  # (b, h*w, c)
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        x_out = x_flat.transpose(1, 2).view(b, c, h, w)

        x_out = x_out[:, :, :orig_h, :orig_w]

        return x_out

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
            nn.Sigmoid()  # Attention weights between 0 and 1
        )

    def forward(self, x):
        conv_out = self.conv_block(x)
        
        attention_weights = self.semantic_attention(conv_out)
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)
        attention_out = x * attention_weights
        
        return attention_out

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



class resblock(nn.Module):
    def __init__(self, dim):

        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=4, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x




##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        

    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt


class fusion2(nn.Module):
    def __init__(self, out_channel):
        super(fusion2, self).__init__()
        self.rpi = torch.zeros((16 * 16, (int(16 * 0.5) + 16) ** 2), dtype=torch.long)
        self.ocab = OCAB(dim=out_channel, window_size=16, overlap_ratio=0.5, num_heads=2, norm_layer=nn.LayerNorm)
        self.main = nn.Sequential(
            BasicConv(1, 16, kernel_size=3, stride=1, relu=True),
            BasicConv(16, 32, kernel_size=1, stride=1, relu=True),
            BasicConv(32, 32, kernel_size=3, stride=1, relu=True),
            BasicConv(32, out_channel, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_channel, affine=True)
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
            BasicConv(64, out_channel, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_channel, affine=True)
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
            BasicConv(16, out_channel, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_channel, affine=True)
        )
    def forward(self, x, depth):
        # print(out1.shape)
        out1  = self.ocab(x,self.main(depth), self.rpi)
        out2  = self.ocab(self.main(depth),x, self.rpi)
        return out1+ out2 + x


##########################################################################
##---------- PromptIR -----------------------

class PromptIR(nn.Module):
    def __init__(self, 
        inp_channels=4, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        decoder = False
        # fix_prompt=False
    ):

        super(PromptIR, self).__init__()
        # print("inp_channels:", type(inp_channels))  # 调试代码，确认类型
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        
        self.decoder = decoder
        
        if self.decoder:
            self.prompt1 = PromptGenBlock(prompt_dim=64,prompt_len=5,prompt_size = 64,lin_dim = 96)
            self.prompt2 = PromptGenBlock(prompt_dim=128,prompt_len=5,prompt_size = 32,lin_dim = 192)
            self.prompt3 = PromptGenBlock(prompt_dim=320,prompt_len=5,prompt_size = 16,lin_dim = 384)
        
        
        self.chnl_reduce1 = nn.Conv2d(64,64,kernel_size=1,bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128,128,kernel_size=1,bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320,256,kernel_size=1,bias=bias)



        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64,dim,kernel_size=1,bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2

        self.reduce_noise_channel_2 = nn.Conv2d(int(dim*2**1) + 128,int(dim*2**1),kernel_size=1,bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.reduce_noise_channel_3 = nn.Conv2d(int(dim*2**2) + 256,int(dim*2**2),kernel_size=1,bias=bias)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.reduce_channels = nn.Conv2d(384, 192, kernel_size=1)  # 预定义1x1卷积
        self.up4_3 = Upsample(int(dim*2**2)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1)+192, int(dim*2**2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(dim*2**2) + 512, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim*2**2)+512,int(dim*2**2),kernel_size=1,bias=bias)


        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(dim*2**1) + 224, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim*2**1)+224,int(dim*2**2),kernel_size=1,bias=bias)


        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.noise_level1 = TransformerBlock(dim=int(dim*2**1)+64, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim*2**1)+64,int(dim*2**1),kernel_size=1,bias=bias)


        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
                    
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.depth_refinement = DepthRefinement(in_channels=1, out_channels=1, num_features=64)

        self.depth_attention = DepthGuidedFusionModule(in_channels=4, out_channels=64)

        self.fusion = nn.ModuleList([
            fusion1(out_channel=48),
            fusion2(out_channel=96),  
            fusion3(out_channel=192),
        ]) 

    def forward(self, inp_img,noise_emb = None):
        
        depth = inp_img[:,3:,:,:]
        depth_2 = F.interpolate(depth, scale_factor=0.5)
        depth_4 = F.interpolate(depth_2, scale_factor=0.5)
        optimized_depth = self.depth_refinement(depth)
        inp_img = torch.cat([inp_img[:, :3, :, :], optimized_depth], dim=1)
        inp_img = self.depth_attention(inp_img)

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        # print("Shape of out_enc_level1:", out_enc_level1.shape)
        out_enc_level1 = self.fusion[0](out_enc_level1, depth)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        # print("Shape of out_enc_level2:", out_enc_level2.shape)
        out_enc_level2 = self.fusion[1](out_enc_level2, depth_2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        # print("Shape of out_enc_level3:", out_enc_level3.shape)
        out_enc_level3 = self.fusion[2](out_enc_level3, depth_4) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4)
        if self.decoder:
            dec3_param = self.prompt3(latent)

            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)
        if latent.shape[1] == 384:
            latent = self.reduce_channels(latent)     
        inp_dec_level3 = self.up4_3(latent)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        if self.decoder:
           
            dec1_param = self.prompt1(out_dec_level2)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)


        out_dec_level1 = self.output(out_dec_level1) + inp_img[:, :3, :, :]

        # print("Shape after patch embedding:", inp_enc_level1.shape)
        # print("Shape after encoder level 1:", out_enc_level1.shape)
        # print("Shape after encoder level 2:", out_enc_level2.shape)
        # print("Shape after encoder level 3:", out_enc_level3.shape)
        # print("Shape after latent:", latent.shape)
        # print("Shape after up4_3:", inp_dec_level3.shape)
        # print("Shape after up3_2:", inp_dec_level2.shape)
        # print("Shape after up2_1:", inp_dec_level1.shape)
        # print("Shape of final output:", out_dec_level1.shape)
        # print("Shape of inp_img:", inp_img)

        return out_dec_level1
