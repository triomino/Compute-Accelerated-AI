根据 [hyper-connection 论文](https://arxiv.org/pdf/2409.19606) 说法，PreNorm 缓解了 gradient vanishing，但会导致 representation collapse，即层与层之间的 hidden states 越来越相似。
早期 transformers 架构是 PostNorm，但因为 gradient vanishing 问题转向了 PreNorm，然后冒出来这个 representation collapse。hyper-connection 意在解决这个 tradeoff。

PreNorm 输入套 Norm: $x_{l+1} = x_l + \text{F}(\text{LN}(x_l))$

PostNorm 输出套 Norm: $x_{l+1} = \text{LN}(x_l + \text{F}(x_l))$ 

### Math
这里有个有趣的小知识，mHC 里面用到双随机矩阵，为什么行列和为 1 的矩阵会被命名为“双随机矩阵”？其实[随机矩阵](https://en.wikipedia.org/wiki/Stochastic_matrix)翻译成转移矩阵更贴近其含义，每行是一个概率分布，$p_{ij}$是状态$i$到状态$j$的概率，描述了一个有限状态空间的马尔可夫链。
