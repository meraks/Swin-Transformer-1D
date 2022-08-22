# -*- coding: utf-8 -*-
'''
@File    :   SReT.py
@Time    :   2022/08/22 20:28:55
@Author  :   HUANG Dong
@Version :   1.0
@Contact :   huang_dong@tju.edu.cn
@Desc    :   Refers Sliced Recursive Transformer (SReT), https://github.com/szq0214/SReT
'''
import torch
from torch import nn
from torch.nn.init import trunc_normal_
from timm.models.layers import DropPath

import math
from einops import rearrange
from functools import partial


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out


class Non_proj(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim=dim, hidden_dim=mlp_hidden_dim, dropout=drop)
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()

    def forward(self, x, recursive_index):
        x = self.coefficient1(x) + self.coefficient2(self.mlp(self.norm1(x)))
        return x


class Group_Attention(nn.Module):
    def __init__(self, dim, num_groups1=8, num_groups2=4, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.num_groups1 = num_groups1
        self.num_groups2 = num_groups2
        head_dim = math.ceil(dim / num_heads)
        inner_dim = head_dim * num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, recursive_index):
        B, N, C = x.shape
        if recursive_index == False:
            num_groups = self.num_groups1
        else:
            num_groups = self.num_groups2
            if num_groups != 1:
                idx = torch.randperm(N)
                x = x[:, idx, :]
                inverse = torch.argsort(idx)
        Np = int(math.ceil(N / num_groups) * num_groups)
        x = torch.nn.functional.pad(x, (0, 0, 0, Np-N))

        qkv = self.qkv(x).\
            view(B, num_groups, Np // num_groups, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(2, 3).reshape(B, num_groups, Np // num_groups, C)
        x = x.permute(0, 3, 1, 2).view(B, C, Np).transpose(1, 2)
        if recursive_index == True and num_groups != 1:
            x = x[:, inverse, :]
        x = x[:, :N, :]
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Transformer_Block(nn.Module):
    def __init__(self, dim, num_groups1, num_groups2, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,

                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Group_Attention(
            dim, num_groups1=num_groups1, num_groups2=num_groups2, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim=dim, hidden_dim=mlp_hidden_dim, dropout=drop)
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()

    def forward(self, x, recursive_index):
        x = self.coefficient1(x) + self.coefficient2(self.drop_path(self.attn(self.norm1(x), recursive_index)))
        x = self.coefficient3(x) + self.coefficient4(self.drop_path(self.mlp(self.norm2(x))))
        return x


class Transformer(nn.Module):
    def __init__(self, base_dim, depth, recursive_num, groups1, groups2, heads, mlp_ratio, np_mlp_ratio,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        blocks = [
            Transformer_Block(
                dim=embed_dim,
                num_groups1=groups1,
                num_groups2=groups2,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_prob[i],
                act_layer=nn.GELU,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for i in range(recursive_num)]

        recursive_loops = int(depth/recursive_num)
        non_projs = [
            Non_proj(
                dim=embed_dim, num_heads=heads, mlp_ratio=np_mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_prob[i], norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU)
            for i in range(depth)]
        RT = []
        for rn in range(recursive_num):
            for rl in range(recursive_loops):
                RT.append(blocks[rn])
                RT.append(non_projs[rn*recursive_loops+rl])

        self.blocks = nn.ModuleList(RT)

    def forward(self, x):
        # l = x.shape[2]
        x = rearrange(x, 'b c l -> b l c')

        for i, blk in enumerate(self.blocks):
            if (i+2)%4 == 0: # mark the recursive layers
                recursive_index = True
            else:
                recursive_index = False
            x = blk(x, recursive_index)

        x = rearrange(x, 'b l c -> b c l')

        return x


class conv_head_pooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride,
                 padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()
        self.conv = nn.Conv1d(in_feature, out_feature, kernel_size=stride + 1,
                              padding=stride // 2, stride=stride,
                              padding_mode=padding_mode, groups=in_feature)

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_embedding(nn.Module):
    def __init__(self, chans, samples, dropoutRate=0.5, kernLength=63,
                 F1=16, D=8):
        super().__init__()
        F2 = F1 * D
        self.seq_embed = nn.Sequential(
            nn.Conv1d(chans, F1, kernLength // 2, padding=(kernLength // 4)),  # conv1
            nn.Conv1d(F1, F1, kernLength // 2, padding=(kernLength // 4)),
            nn.BatchNorm1d(F1),
            nn.Conv1d(F1, F2, chans // 2, groups=F1),  # conv
            nn.Conv1d(F2, F2, chans // 2, groups=F2),
            nn.BatchNorm1d(F2),
            nn.ReLU(),
            nn.AvgPool1d(4),
            nn.Dropout(dropoutRate),
            nn.Conv1d(F2, F2, kernel_size=31 // 2, padding=31 // 4),  # conv3
            nn.Conv1d(F2, F2, kernel_size=31 // 2, padding=31 // 4),
            nn.BatchNorm1d(F2),
            nn.ReLU(),
            nn.AvgPool1d(5),
            nn.Dropout(dropoutRate),
        )
        self.embed_dim = F2
        self.seq_len = (samples - chans + 1) // (4 * 5)

    def forward(self, x):
        x = self.seq_embed(x)
        return x


class SReT(nn.Module):
    def __init__(self, input_shape, output_shape, patch_size=16, stride=8,
                 base_dims=[16, 16, 32], depth=[4, 10, 6], recursive_num=[2, 5, 3],
                 heads=[6, 12, 12], groups1=[8, 4, 1], groups2=[2, 1, 1],
                 mlp_ratio=3.6, np_mlp_ratio=1,
                 attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.1):
        super().__init__()

        chans: int = input_shape[0]
        samples: int = input_shape[-1]

        total_block = sum(depth)
        padding = 0
        block_idx = 0

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = output_shape

        self.patch_embed = conv_embedding(chans, samples, dropoutRate=0.5, kernLength=63,
                 F1=16, D=6)
        embed_dim, embed_seq_len = self.patch_embed.embed_dim, self.patch_embed.seq_len
        self.pos_embed = nn.Parameter(
            torch.randn(1, embed_dim, embed_seq_len),
            requires_grad=True
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])

        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]
            self.transformers.append(
                Transformer(base_dims[stage], depth[stage], recursive_num[stage], groups1[stage], groups2[stage],
                            heads[stage],
                            mlp_ratio, np_mlp_ratio,
                            drop_rate, attn_drop_rate, drop_path_prob)
            )
            if stage < len(heads) - 1:
                self.pools.append(
                    conv_head_pooling(base_dims[stage] * heads[stage],
                                      base_dims[stage + 1] * heads[stage + 1],
                                      stride=2
                                      )
                )

        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Classifier head
        if output_shape > 0:
            self.head = nn.Linear(base_dims[-1] * heads[-1], output_shape)
        else:
            self.head = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)

        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)

        for stage in range(len(self.pools)):
            x = self.transformers[stage](x)
            x = self.pools[stage](x)
        x = self.transformers[-1](x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.norm(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x



if __name__ == '__main__':
    chans, samples = 30, 2000
    x = torch.rand(3, chans, samples)
    net = SReT(input_shape=(chans, samples), output_shape=9)
    print(net(x).shape)