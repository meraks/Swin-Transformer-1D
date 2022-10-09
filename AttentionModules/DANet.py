import numpy as np
import torch
from torch import nn
from torch.nn import init
from .SelfAttention import ScaledDotProductAttention
from .SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
# from SelfAttention import ScaledDotProductAttention
# from SimplifiedSelfAttention import SimplifiedScaledDotProductAttention

class PositionAttentionModule(nn.Module):

    def __init__(self,d_model=512,kernel_size=3,H=7,W=7):
        super().__init__()
        self.cnn=nn.Conv2d(d_model,d_model,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.pa=ScaledDotProductAttention(d_model,d_k=d_model,d_v=d_model,h=1)
    
    def forward(self,x):
        bs,c,h,w=x.shape
        y=self.cnn(x)
        y=y.view(bs,c,-1).permute(0,2,1) #bs,h*w,c
        y=self.pa(y,y,y) #bs,h*w,c
        return y


class ChannelAttentionModule(nn.Module):
    
    def __init__(self,d_model=512,kernel_size=3,H=7,W=7):
        super().__init__()
        self.cnn=nn.Conv2d(d_model,d_model,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.pa=SimplifiedScaledDotProductAttention(H*W,h=1)
    
    def forward(self,x):
        bs,c,h,w=x.shape
        y=self.cnn(x)
        y=y.view(bs,c,-1) #bs,c,h*w
        y=self.pa(y,y,y) #bs,c,h*w
        return y


class DAModule(nn.Module):

    def __init__(self,d_model=512,kernel_size=3,H=7,W=7):
        super().__init__()
        self.position_attention_module=PositionAttentionModule(d_model=512,kernel_size=3,H=7,W=7)
        self.channel_attention_module=ChannelAttentionModule(d_model=512,kernel_size=3,H=7,W=7)
    
    def forward(self,input):
        bs,c,h,w=input.shape
        p_out=self.position_attention_module(input)
        c_out=self.channel_attention_module(input)
        p_out=p_out.permute(0,2,1).view(bs,c,h,w)
        c_out=c_out.view(bs,c,h,w)
        return p_out+c_out


class PositionAttentionModule1D(nn.Module):

    def __init__(self, d_model=512, kernel_size=3, L=7):
        super().__init__()
        self.cnn = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.pa = ScaledDotProductAttention(d_model, d_k=d_model, d_v=d_model, h=1)

    def forward(self, x):
        bs, c, l = x.shape
        y = self.cnn(x)
        y = y.view(bs,c,-1).permute(0, 2, 1)  # bs,l,c
        y = self.pa(y, y, y)  # bs,l,c
        return y


class ChannelAttentionModule1D(nn.Module):

    def __init__(self, d_model=512, kernel_size=3, L=7):
        super().__init__()
        self.cnn = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.pa = SimplifiedScaledDotProductAttention(L, h=1)

    def forward(self, x):
        bs, c, l = x.shape
        y = self.cnn(x)
        y = y.view(bs, c, -1)  # bs,c,l
        y = self.pa(y, y, y)  # bs,c,l
        return y


class DAModule1D(nn.Module):

    def __init__(self, d_model=512, kernel_size=3, L=7):
        super().__init__()
        self.position_attention_module = PositionAttentionModule1D(d_model=d_model, kernel_size=kernel_size, L=L)
        self.channel_attention_module = ChannelAttentionModule1D(d_model=d_model, kernel_size=kernel_size, L=L)

    def forward(self, input):
        bs, c, l = input.shape
        p_out = self.position_attention_module(input)
        c_out = self.channel_attention_module(input)
        p_out = p_out.permute(0, 2, 1).view(bs, c, l)
        c_out = c_out.view(bs, c, l)
        return p_out + c_out


DAModule2D = DAModule
__all__ = ['DAModule1D', 'DAModule2D']

if __name__ == '__main__':
    # input=torch.randn(50,512,7,7)
    # danet=DAModule(d_model=512,kernel_size=3,H=7,W=7)
    # print(danet(input).shape)

    input = torch.randn(50, 512, 2000)
    danet = DAModule1D(d_model=512, kernel_size=3, L=2000)
    print(danet(input).shape)
