根据 [hyper-connection 论文](https://arxiv.org/pdf/2409.19606) 说法，PreNorm 缓解了 gradient vanishing，但会导致 representation collapse，即层与层之间的 hidden states 越来越相似。
早期 transformers 架构是 PostNorm，但因为 gradient vanishing 问题转向了 PreNorm，然后冒出来这个 representation collapse。hyper-connection 意在解决这个 tradeoff。

PreNorm 输入套 Norm: $x_{l+1} = x_l + \text{F}(\text{LN}(x_l))$

PostNorm 输出套 Norm: $x_{l+1} = \text{LN}(x_l + \text{F}(x_l))$ 
