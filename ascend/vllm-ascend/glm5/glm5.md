# GLM-5 性能调优
## 项目背景
智谱在今年二月发布 [GLM-5](https://arxiv.org/abs/2602.15763)，推理流量出现极大增长，现有算力无法承载，因此在寻求扩容，除此之外其他厂商也有内部部署 GLM-5 诉求。基于此背景昇腾产品部联合 OTT 综合在 vLLM-Ascend 上进行 GLM-5 推理性能优化，期望达到性能 xxx。
## GLM-5 模型结构
从推理角度看，GLM-5 和 DeepSeek V3.2 一致，均使用 DSA(DeepSeek Sparse Attention) 和 MoE 结构，这一点从 vLLM 的 GLM-5 官方 [PR](https://github.com/vllm-project/vllm/pull/34124) 直接复用 DeepSeek 的类就能看出来。
### 整体架构
### MLA
### DSA
DSA 是 DeepSeek 基于 MLA(Multi-head Latent Attention) 演进的稀疏 Attention 结构。
### MoE

## 时间线
2.xx 提前拿到模型进行适配
2.17 0day 发布，包含 xxx 优化
3.xx 全量优化完成，性能达到 xxx
## 模型优化
DeepSeek V3.2 在去年 10 月预发布，12 月正式发布，因此 VLLM-Ascend 在 0.13.0 版本上基本已经完成了 DeepSeek V3.2 的优化适配。我们预期 GLM-5 可以复用这些优化点，但是由于算子 shape 泛化等问题导致部分特性开启失败。最开始的 0day 适配只包含直接能启用的特性，TPOT 仅 xxms，直到所有性能优化特性开启后 TPOT 达到 xxms。
### W8A8 量化

### MLAPO 算子
（这一段介绍这个优化是怎么启发得到的）在进入 DSA 的 Indexer 之前存在 qkv 降维升维、rope、更新 kvcache 等算子，里面的 vector 算子存在访存瓶颈，所以考虑把 vector 融合下。基于这个想法，昇腾 CANN 做了一个 [MLAPO 融合算子](找链接)把 MLA 前处理融合成一个算子。首先融合本身可以减少 kernel launch 时延以及减少访存。其次 MLAPO 通过对Vector和Cube计算单元的并行处理及流水优化，基本可以将用时较短的Vector耗时完全掩盖，进一步缩短MLA前处理的时延。
#### 原理
（图示+代码，可参考 https://www.hiascend.com/developer/techArticles/20250526-1）
#### 效果
优化前后 profiling 截图对比
TPOT 提升：xxms->xxms
### MTP/Speculative Decoding
[Speculative Decoding（投机解码）](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf) 早在 22 年就已经被提出了。现代 transformer 架构在 decode 阶段存在的问题是必须逐 token 推理，下一个 token 推理依赖上一个 token 输出。投机解码的想法是用一个小模型快速采样若干 token，然后在大模型上一步验证这几个 token。小模型学到的数据分布和大模型相似，乐观情况下一次 decode 加上几次小模型推理的时间可以产生 3-4 个 token，推理速度极大提升。
#### 原理
（图示+代码，可参考）
#### 效果
优化前后 profiling 截图对比
TPOT 提升：xxms->xxms
### mulsAdd
TPOT
### 共享专家多流
TTFT+TPOT
### matmul nd 转 nz
TPOT
### SFA kvcache 重复更新问题
TTFT+TPOT
### FlashComm1/Sequence Parallel
TTFT
### PCP/DCP
0.17.0rc1 dsv3.2 和 FC1 是一个开关。
### Lightning Indexer C8
可以讲分析工作，实际没用
### 重计算
recompute_scheduler_enable
### kvcache 内存优化（0.13.0 问题）
吞吐量
### 主模型精度修正（0.13.0 问题）
TPOT