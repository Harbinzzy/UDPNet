import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            DeepPoolLayer(in_channel, out_channel) if filter else nn.Identity(),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class DeepPoolLayer(nn.Module):
    def __init__(self, k, k_out):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [8,4,2]
        dilation = [7,9,11]
        pools, convs, dynas = [],[],[]
        for j, i in enumerate(self.pools_sizes):
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
            dynas.append(MultiShapeKernel(dim=k, kernel_size=3, dilation=dilation[j]))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.dynas = nn.ModuleList(dynas)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            if i == 0:
                y = self.dynas[i](self.convs[i](self.pools[i](x)))
            else:
                y = self.dynas[i](self.convs[i](self.pools[i](x)+y_up))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_sizes)-1:
                y_up = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        resl = self.relu(resl)
        resl = self.conv_sum(resl)

        return resl

class dynamic_filter(nn.Module):
    def __init__(self, inchannels, kernel_size=3, dilation=1, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        self.dilation = dilation

        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Tanh()
    
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.pad = nn.ReflectionPad2d(self.dilation*(kernel_size-1)//2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.inside_all = nn.Parameter(torch.zeros(inchannels,1,1), requires_grad=True)

    def forward(self, x):
        identity_input = x
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)     

        n, c, h, w = x.shape  
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size, dilation=self.dilation).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = low_filter.shape
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
       
        low_filter = self.act(low_filter)
    
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_low = low_part * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)

        out_low = out_low * self.lamb_l[None,:,None,None]

        out_high = (identity_input) * (self.lamb_h[None,:,None,None] + 1.) 

        return out_low + out_high


class cubic_attention(nn.Module):
    def __init__(self, dim, group, dilation, kernel) -> None:
        super().__init__()

        self.H_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim,1,1))
        self.beta = nn.Parameter(torch.ones(dim,1,1))

    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta


class spatial_strip_att(nn.Module):
    def __init__(self, dim, kernel=3, dilation=1, group=2, H=True) -> None:
        super().__init__()

        self.k = kernel
        pad = dilation*(kernel-1) // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel//2, 1) if H else (1, kernel//2)
        self.dilation = dilation
        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group*kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Tanh()
        self.inside_all = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.lamb_l = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(dim), requires_grad=True)
        gap_kernel = (None,1) if H else (1, None) 
        self.gap = nn.AdaptiveAvgPool2d(gap_kernel)

    def forward(self, x):
        identity_input = x.clone()
        filter = self.ap(x)
        filter = self.conv(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel, dilation=self.dilation).reshape(n, self.group, c//self.group, self.k, h*w)
        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1//self.k, self.k, p*q).unsqueeze(2)
        filter = self.filter_act(filter)
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)

        out_low = out * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)
        out_low = out_low * self.lamb_l[None,:,None,None]
        out_high = identity_input * (self.lamb_h[None,:,None,None]+1.)

        return out_low + out_high


class MultiShapeKernel(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1, group=8):
        super().__init__()

        self.square_att = dynamic_filter(inchannels=dim, dilation=dilation, group=group, kernel_size=kernel_size)
        self.strip_att = cubic_attention(dim, group=group, dilation=dilation, kernel=kernel_size)

    def forward(self, x):

        x1 = self.strip_att(x)
        x2 = self.square_att(x)

        return x1+x2

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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