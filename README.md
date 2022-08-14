# Swin-Transformer-1D
Swin-Transformer 1D implements.

Refers: https://github.com/SwinTransformer/Video-Swin-Transformer

ACmix refers: https://github.com/LeapLabTHU/ACmix

## 中文版本

### 简介

这是Swin-Transformer（下称ST） 1D的实现，参考了[MSRA版本的原始ST](https://github.com/microsoft/Swin-Transformer)以及[Vedio-Swin-Transformer版本](https://github.com/SwinTransformer/Video-Swin-Transformer)实现的。

出于项目需要，因此需要实现1D的ST，受到Vedio-Swin-Transformer（下称VST）启发，VST实现了3D的窗口自注意力（window attention）以及3D的掩码（mask）设置。本ST-1D除了实现所有ST的功能，并添加了对于序列长度不能整除窗口长度（window size）需要补零的功能，并将其考虑进窗口自注意力以及掩码的设计中。

欢迎大家提issue！谢谢！

### 更新日志：
- 2022.08.14：添加了Swin-Transformer-V2-1D的实现，参考MSRA版本的V2代码实现。

### 参考

Swin-Transformer: <br />
github: https://github.com/microsoft/Swin-Transformer <br />
paper: https://arxiv.org/pdf/2103.14030.pdf

Vedio-Swin-Transformer: <br />
github: https://github.com/SwinTransformer/Video-Swin-Transformer<br />
paper: https://arxiv.org/pdf/2106.13230.pdf


ACmix: <br />
github: https://github.com/LeapLabTHU/ACmix <br />
paper: https://arxiv.org/pdf/2111.14556.pdf

