### Intention
根据 [hyper-connection 论文](https://arxiv.org/pdf/2409.19606) 说法，PreNorm 缓解了 gradient vanishing，但会导致 representation collapse，即层与层之间的 hidden states 越来越相似。
早期 transformers 架构是 PostNorm，但因为 gradient vanishing 问题转向了 PreNorm，然后冒出来这个 representation collapse。hyper-connection 意在解决这个 tradeoff。

PostNorm 即输出套 Norm: $x_{l+1} = \text{LN}(x_l + \text{F}(x_l))$ 。为什么 PostNorm 会导致 gradient vanishing？一通推导之后能看到 LN 梯度反传的时候会砍掉梯度方向和输入方向的分量，多层叠加后梯度就没了。直观来说，LN 是对激活值很强的一个压制，它让早期层的 hidden states 对后面层的影响变弱了。

PreNorm 即输入套 Norm: $x_{l+1} = x_l + \text{F}(\text{LN}(x_l))$ 。为什么 PreNorm 能缓解 gradient vanishing？因为 LN 作用到增量上了，残差的梯度是不变的。直观来说，PreNorm 里早期层 hidden states 对后面层影响没有明显衰减，所以会有 representation collapse 问题。

### mHC

### Math
这里有个有趣的小知识，mHC 里面用到双随机矩阵，为什么行列和为 1 的矩阵会被命名为“双随机矩阵”？其实[随机矩阵](https://en.wikipedia.org/wiki/Stochastic_matrix)翻译成转移矩阵更贴近其含义，每行是一个概率分布， $p_{ij}$ 是状态 $i$ 到状态 $j$ 的概率，描述了一个有限状态空间的马尔可夫链。
