import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
train_size = (1,3,256,256)

class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5,4,3,2,1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )
           
    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2]*self.base_size[0]//train_size[-2]
            self.kernel_size[1] = x.shape[3]*self.base_size[1]//train_size[-1]
            
            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0]*x.shape[2]//train_size[-2])
            self.max_r2 = max(1, self.rs[0]*x.shape[3]//train_size[-1])

        if self.fast_imp:   # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0]>=h and self.kernel_size[1]>=w:
                out = F.adaptive_avg_pool2d(x,1)
            else:
                r1 = [r for r in self.rs if h%r==0][0]
                r2 = [r for r in self.rs if w%r==0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:,:,::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h-1, self.kernel_size[0]//r1), min(w-1, self.kernel_size[1]//r2)
                out = (s[:,:,:-k1,:-k2]-s[:,:,:-k1,k2:]-s[:,:,k1:,:-k2]+s[:,:,k1:,k2:])/(k1*k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1,r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum(dim=-2)
            s = torch.nn.functional.pad(s, (1,0,1,0)) # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])

            s1, s2, s3, s4 = s[:,:,:-k1,:-k2],s[:,:,:-k1,k2:], s[:,:,k1:,:-k2], s[:,:,k1:,k2:]
            out = s4+s1-s2-s3
            out = out / (k1*k2)
    
        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = ((w - _w)//2, (w - _w + 1)//2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')
        
        return out
    
    
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


class Gap(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()

        self.fscale_d = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.fscale_h = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap = AvgPool2d(base_size=210)

    def forward(self, x):
        x_d = self.gap(x)
        x_h = (x - x_d) * (self.fscale_h[None, :, None, None] + 1.)
        x_d = x_d  * self.fscale_d[None, :, None, None]
        return x_d + x_h


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        self.filter = filter

        self.dyna = dynamic_filter(in_channel//2) if filter else nn.Identity()
        self.dyna_2 = dynamic_filter(in_channel//2, kernel_size=5) if filter else nn.Identity()

        self.localap = Patch_ap(in_channel//2, patch_size=2)
        self.global_ap = Gap(in_channel//2)


    def forward(self, x):
        out = self.conv1(x)
       
        if self.filter:

            k3, k5 = torch.chunk(out, 2, dim=1)
            out_k3 = self.dyna(k3)
            out_k5 = self.dyna_2(k5)
            out = torch.cat((out_k3, out_k5), dim=1)
            
        non_local, local = torch.chunk(out, 2, dim=1)
        non_local = self.global_ap(non_local)
        local = self.localap(local)
        out = torch.cat((non_local, local), dim=1)
        out = self.conv2(out)
        return out + x

class Unet(nn.Module):
    def __init__(self, in_channel, out_channel, num_res):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_res-1):
            self.layers.append(ResBlock(in_channel, out_channel))
        self.layers.append(ResBlock(in_channel, out_channel, filter=True))
        self.down = nn.Conv2d(in_channel, in_channel, kernel_size=2, stride=2, groups=in_channel)
        self.num_res = num_res

        self.conv = nn.Conv2d(in_channel*2, in_channel, kernel_size=1, stride=1)
    def forward(self, x):
        res = x.clone()

        for i, layer in enumerate(self.layers):
            if i == self.num_res//4:
                skip = x
                x = self.down(x)
            if i == self.num_res - self.num_res//4:
                x = F.upsample(x, res.shape[2:], mode='bilinear')
                x = self.conv(torch.cat((x, skip), dim=1))
            x = layer(x)

        return x + res

class dynamic_filter(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.pad = nn.ReflectionPad2d(kernel_size//2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))

        self.modulate = SFconv(inchannels)

    def forward(self, x):
        identity_input = x # 3,32,64,64
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)     

        n, c, h, w = x.shape  
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = low_filter.shape
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
       
        low_filter = self.act(low_filter)
    
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)
        out_high = identity_input - low_part
        out = self.modulate(low_part, out_high)
        return out

class SFconv(nn.Module):
    def __init__(self, features, M=2, r=2, L=32) -> None:
        super().__init__()
        
        d = max(int(features/r), L)
        self.features = features

        self.fc = nn.Conv2d(features, d, 1, 1, 0)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, 1, 1, 0)
            )
        self.softmax = nn.Softmax(dim=1)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap = AvgPool2d(base_size=210)

        self.out = nn.Conv2d(features, features, 1, 1, 0)
    def forward(self, low, high):
        emerge = low + high
        emerge = self.gap(emerge)

        fea_z = self.fc(emerge)

        high_att = self.fcs[0](fea_z)
        low_att = self.fcs[1](fea_z)
        
        attention_vectors = torch.cat([high_att, low_att], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        high_att, low_att = torch.chunk(attention_vectors, 2, dim=1)

        fea_high = high * high_att
        fea_low = low * low_att
        
        out = self.out(fea_high + fea_low) 
        return out

class Patch_ap(nn.Module):
    def __init__(self, inchannel, patch_size):
        super(Patch_ap, self).__init__()

        # self.ap = nn.AdaptiveAvgPool2d((1,1))
        self.ap = AvgPool2d(base_size=210)

        self.patch_size = patch_size
        self.channel = inchannel * patch_size**2
        self.h = nn.Parameter(torch.zeros(self.channel))
        self.l = nn.Parameter(torch.zeros(self.channel))

    def forward(self, x):

        patch_x = rearrange(x, 'b c (p1 w1) (p2 w2) -> b c p1 w1 p2 w2', p1=self.patch_size, p2=self.patch_size)
        patch_x = rearrange(patch_x, ' b c p1 w1 p2 w2 -> b (c p1 p2) w1 w2', p1=self.patch_size, p2=self.patch_size)

        low = self.ap(patch_x)
        high = (patch_x - low) * self.h[None, :, None, None]
        out = high + low * self.l[None, :, None, None]
        out = rearrange(out, 'b (c p1 p2) w1 w2 -> b c (p1 w1) (p2 w2)', p1=self.patch_size, p2=self.patch_size)

        return out

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
