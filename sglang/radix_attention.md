## Radix Attention
类似于 trie 的想法，把序列对应的 kvcache 组成一个树管理。[论文](https://arxiv.org/pdf/2312.07104) 里说它是逐 token 管理的，相当于 page attention 里 block_size=1. 这个感觉 kernel 性能会比较差，不过代码里倒是能配 page size。

考虑和 vllm 的 kvblock+hash 方式实现 prefix cache 对比，能想到 RadixAttention 的一个优势是在分叉位置没有 O(block_size) 的浪费，如果不开 beam search 这种分叉浪费没多少其实。其他优势，说实话 vllm 的 block hash+prefix hash 那种扭曲的方式也有（其实这种方式近似变成树形管理了）。RadixAttention 的设计是更优雅的，但不足以成为选择 SGLang 的决定性因素。

