# Swin-Transformer-1D

这是Swin-Transformer（下称ST） 1D的实现，参考了[MSRA版本的原始ST](https://github.com/microsoft/Swin-Transformer)以及[Vedio-Swin-Transformer版本](https://github.com/SwinTransformer/Video-Swin-Transformer)实现的。

受到Vedio-Swin-Transformer（下称VST）启发，VST实现了3D的窗口自注意力（window attention）以及3D的掩码（mask）设置。本ST-1D除了实现所有ST的功能，并添加了对于序列长度不能整除窗口长度（window size）需要补零的功能，并将其考虑进窗口自注意力以及掩码的设计中。

目前包括的模型：
1. Swin-Transformer: ST-V1、ST-V2及ST-ACmix。
2. Sliced Recursive Transformer (SReT): V1(原始版本)。
3. 添加了一些1D的通道注意力模块，[FightingCV](http://link.zhihu.com/?target=https%3A//github.com/xmu-xiaoma666/External-Attention-pytorch)提供了2D通道注意力的实现。

未来会陆续增加热点模型的1D实现（指追随着CV的顶会），不再增加新的Repositories。

欢迎大家提issue！欢迎大家点star！谢谢！

### 更新日志
- 2022.08.12：添加了Swin-Transformer-ACmix-1D的实现，参考[3]实现。
- 2022.08.14：添加了Swin-Transformer-V2-1D的实现，参考MSRA版本的V2代码实现。
- 2022.08.22：添加了Sliced Recursive Transformer 1D的实现，参考CMU & MBZUAI[4]的版本实现。
- 2022.10.10：添加了诸多1D通道注意力的实现。目前包括：ExternalAttention、SelfAttention、SEAttention、SKAttention、CBAM、BAM、ECAAttention、DANet、PSA、EMSA、SplitAtttention。

### 参考

[1] Swin-Transformer: <br />
github: https://github.com/microsoft/Swin-Transformer <br />
paper: https://arxiv.org/pdf/2103.14030.pdf

[2] Vedio-Swin-Transformer: <br />
github: https://github.com/SwinTransformer/Video-Swin-Transformer<br />
paper: https://arxiv.org/pdf/2106.13230.pdf

[3] ACmix: <br />
github: https://github.com/LeapLabTHU/ACmix <br />
paper: https://arxiv.org/pdf/2111.14556.pdf

[4] Sliced Recursive Transformer (SReT): <br />
github: https://github.com/szq0214/SReT <br />
paper: https://arxiv.org/abs/2111.05297
