## vLLM 源码笔记：DeepSeek-V3.2

> **版本信息**：本文档基于 vLLM 0.13.0 版本代码进行分析。

> **分析前提**：本文档的分析基于以下配置假设：
> - Tensor Parallel (TP) > 1
> - Data Parallel (DP) > 1  
> - Expert Parallel (EP) = TP × DP

---

### 1 MLA (Multi-Head Latent Attention)

#### 1.1 QKV 计算

MLA 通过低秩压缩技术减少 KV Cache 的存储需求。核心计算流程如下：

**低秩映射公式**：

$$\mathbf{h} \in \mathbb{R}^{d} \rightarrow \mathbf{q}_c \in \mathbb{R}^{r_q}, \mathbf{kv}_{\text{lora}} \in \mathbb{R}^{r_{kv} + d_r}$$

其中 $d$ 为 hidden_size (7168)，$r_q$ 为 q_lora_rank (1536)，$r_{kv}$ 为 kv_lora_rank (512)，$d_r$ 为 rope 维度 (64)。

先计算 q_lora 和 kv_lora，然后进行升维投影。lora 这层硬编码不开 TP。

```python
# File: vllm/model_executor/layers/mla.py:129
# qkv 低秩映射
qkv_lora = self.fused_qkv_a_proj(hidden_states)[0] # 7168->1536+512+64
q_c, kv_lora = qkv_lora.split(
    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
    dim=-1,
) # 1536+512+64->1536, 512+64
q_c = self.q_a_layernorm(q_c)
q = self.q_b_proj(q_c)[0] # 1536->128*192
kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1) # 512+64->512,64
kv_c_normed = self.kv_a_layernorm(kv_c)
q = q.view(-1, self.num_heads, self.qk_head_dim)
k_pe = k_pe.unsqueeze(1) # (N,1,head_size)
```

**Q 的升维投影**：

$$\mathbf{q} = \mathbf{W}_q^{(b)} \cdot \text{RMSNorm}(\mathbf{q}_c) \in \mathbb{R}^{h \times (d_{qk}^{\text{nope}} + d_r)}$$

其中 $h$ 为 num_heads (128)，$d_{qk}^{\text{nope}}$ = 128，$d_r$ = 64。

其中 fused_qkv_a_proj 是把两个矩阵乘法合并了。
```python
# File: vllm/model_executor/models/deepseek_v2.py:959
if self.q_lora_rank is not None:
    self.fused_qkv_a_proj = MergedColumnParallelLinear(
        self.hidden_size, # 7168
        [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], # 1536, 512+64
        bias=False,
        quant_config=quant_config,
        prefix=f"{prefix}.fused_qkv_a_proj",
        disable_tp=True,
    )
```
MergedColumnParallelLinear 是多个矩阵 TP 切分，每个矩阵取一个分片后合并做一次乘法。

#### 1.2 RoPE

只对每个 head 后 64 维做 RoPE。位置编码只作用于 query 和 key 的 rope 部分。

**RoPE 公式**：

$$\mathbf{q}_i^{\text{rope}} = \text{RoPE}(\mathbf{q}_i), \quad i \in [d_{qk}^{\text{nope}}, d_{qk}^{\text{nope}} + d_r)$$

$$\mathbf{k}_{\text{pe}} = \text{RoPE}(\mathbf{k}_{\text{pe}})$$

```python
# File: vllm/model_executor/layers/mla.py:154
q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
    positions, q[..., self.qk_nope_head_dim :], k_pe
) # qk_nope_head_dim=128, qk_rope_head_dim=64
```

#### 1.3 Indexer

V3.2 专属模块，轻量的 L² 计算每个 q 最高的分数。结果存到内部 buffer 里，这里返回的没用。
```python
# File: vllm/model_executor/layers/mla.py:158
if self.indexer and self.is_sparse:
    _topk_indices = self.indexer(
        hidden_states, q_c, positions, self.indexer_rope_emb
    )
```

#### 1.4 Attention

先看外层调用。use_direct_call=true 走 python 实现，否则作为一个融合算子，需要看底层 backend(flash infer/triton MLA/flash attn) 实现。
```python
# File: vllm/attention/layer.py (MLAAttention.forward)
        if self.use_direct_call:
            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.layer_name]
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]

            if self.attn_backend.accept_output_buffer:
                output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
                self.impl.forward(
                    self,
                    q,
                    kv_c_normed,
                    k_pe,
                    self_kv_cache,
                    attn_metadata,
                    output=output,
                )
                return output
            else:
                return self.impl.forward(
                    self, q, kv_c_normed, k_pe, self_kv_cache, attn_metadata
                )
        else:
            if self.attn_backend.accept_output_buffer:
                output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
                torch.ops.vllm.unified_mla_attention_with_output(
                    q,
                    kv_c_normed,
                    k_pe,
                    output,
                    self.layer_name,
                )
                return output
            else:
                return torch.ops.vllm.unified_mla_attention(
                    q,
                    kv_c_normed,
                    k_pe,
                    self.layer_name,
                )
```
其中 kv_c_normed 是压缩后的 kv，每个 token 512 维，在用的时候会先 up projection 到 kv_head * head_size 再融合 k_pe。

**Attention 计算**：

$$\text{KV}_{\text{full}} = \text{UpProj}(\text{KV}_c) \in \mathbb{R}^{h \times (d_{qk}^{\text{nope}} + d_v)}$$

$$\text{Attention}(\mathbf{Q}, \text{KV}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

#### 1.5 Output Projection

需要注意的是这里隐含了 all reduce。

```python
# File: vllm/model_executor/layers/mla.py:173
return self.o_proj(attn_out)[0]
# o_proj 定义 (vllm/model_executor/models/deepseek_v2.py:1001)
self.o_proj = RowParallelLinear(
    self.num_heads * self.v_head_dim,
    self.hidden_size,
    bias=False,
    quant_config=quant_config,
    prefix=f"{prefix}.o_proj",
)
```

**输出投影公式**：

$$\mathbf{o} = \mathbf{W}_o \cdot \text{attn\_out} \in \mathbb{R}^{d}$$

其中 $\mathbf{W}_o \in \mathbb{R}^{d \times h \cdot d_v}$，RowParallelLinear 会执行 all-reduce 操作。

**Tips**：RowParallelLinear 和 ColumnParallelLinear 在 MLP 结构里大量使用，ColumnParallelLinear 做切片下的 up projection（Tensor Parallel），过完激活函数再用 RowParallelLinear 做切片下的 down projection 然后 reduce 求和。因此 ColumnParallelLinear 类 gather 默认 false，RowParallelLinear 类 reduce 默认 true。

---

### 2 MoE (Mixture of Experts)

#### 2.1 Sequence Parallel (SP)

和 MLA 那一层的 SP 优化不是一个概念。在 MLA 结束后，在一个 DP Group 里，每张卡的 hidden_states 是相同的，然后在这里把 hidden_states 按 token 那一维切分，分布到不同 TP rank 上。这样后面跑 MoE gate 或者共享专家的时候 token 不会重复，减小激活值和计算量。

**SP 切分公式**：

$$\mathbf{H}_{\text{tp\_rank}} = \text{Split}(\mathbf{H}, \text{dim}=0)[\text{tp\_rank}]$$

假设 TP size = $T$，sequence length = $L$：

$$\mathbf{H} \in \mathbb{R}^{L \times d} \rightarrow \mathbf{H}_i \in \mathbb{R}^{\lceil L/T \rceil \times d}, \quad i = 0, 1, ..., T-1$$

```python
# File: vllm/model_executor/models/deepseek_v2.py:346
# Chunk the hidden states so they aren't replicated across TP ranks.
# This avoids duplicate computation in self.experts.
# TODO: We can replace the all_reduce at the end of attn with a
# reduce_scatter instead of chunking here.
if self.is_sequence_parallel:
    hidden_states = sequence_parallel_chunk(hidden_states)

# File: vllm/model_executor/models/utils.py:778
# 切分实现
def sequence_parallel_chunk_impl(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    # all_gather needs the sequence length to be divisible by tp_size
    seq_len = x.size(0)
    remainder = seq_len % tp_size
    if remainder != 0:
        pad_len = tp_size - remainder
        y = nn.functional.pad(x, (0, 0, 0, pad_len))
    else:
        y = x

    chunk = y.shape[0] // tp_size
    start = tp_rank * chunk
    return torch.narrow(y, 0, start, chunk)
```

vLLM 官方在这有个性能优化的注释，这里的切分可以挪到 MLA 最后用 reduce_scatter 实现。（现在 MLA 最后是 all reduce 合并结果）

#### 2.2 Gate

gate 是线性映射，从隐藏层生成 256 个专家分数 router_logits，是 MoE 的第一步。

**Router 计算**：

$$\mathbf{r} = \mathbf{W}_g \cdot \mathbf{h} \in \mathbb{R}^{E}$$

$$\mathbf{w} = \text{TopK-Softmax}(\mathbf{r}, k) \in \mathbb{R}^{E}$$

其中 $E$ 为专家总数 (n_routed_experts = 256)，$k$ 为 top_k。

```python
# File: vllm/model_executor/models/deepseek_v2.py:261
self.gate = ReplicatedLinear(
    config.hidden_size,
    config.n_routed_experts,
    bias=False,
    quant_config=None,
    prefix=f"{prefix}.gate",
)
```

#### 2.3 Mixture of Experts

MoE 有个很关键的环境变量 VLLM_ALL2ALL_BACKEND。默认 allgather_reducescatter 会用 ag_rs 替代 all_to_allv，这样整个 MoE 的逻辑就很 simple，self.quant_method 不会被替换为 FusedMoEModularMethod，没有抽象替换。

```python
# File: vllm/config/__init__.py (或 vllm/envs.py)
VLLM_ALL2ALL_BACKEND: Literal[
    "naive",
    "pplx",
    "deepep_high_throughput",
    "deepep_low_latency",
    "allgather_reducescatter",
    "flashinfer_all2allv",
] = "allgather_reducescatter"
```

##### 2.3.1 默认实现

默认实现下，self.quant_method 不会被替换为 FusedMoEModularMethod，就是个简单的 FFN。会走 do_naive_dispatch_combine=True 的逻辑。

###### 主干逻辑

```python
# File: vllm/model_executor/layers/fused_moe/layer.py:1889
    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # self.quant_method 是根据 all2all 后端决定的，默认情况下是 UnquantizedFusedMoEMethod，不是 FusedMoEModularMethod，所以 do_naive_dispatch_combine=True
        do_naive_dispatch_combine: bool = self.dp_size > 1 and not isinstance(
            self.quant_method, FusedMoEModularMethod
        )

        ctx = get_forward_context()
        sp_ctx = (
            ctx.dp_metadata.sp_local_sizes(self.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

        with sp_ctx:
            # dispatch 逻辑，底层是 all gather
            if do_naive_dispatch_combine:
                hidden_states_combined, router_logits = get_ep_group().dispatch(
                    hidden_states, router_logits, self.is_sequence_parallel
                )
            # Run shared experts before matrix multiply.
            # because matrix multiply maybe modify the hidden_states.
            if has_separate_shared_experts and not use_shared_experts_stream:
                assert self.shared_experts is not None
                shared_output = self.shared_experts(hidden_states)

            # NOTE: Similar with DP, PCP also needs dispatch and combine. For
            # simplicity, AgRsAll2All was added separately for PCP here. Maybe
            # we should modify All2AllManager abstract to better support PCP.
            if self.pcp_size > 1:
                hidden_states = get_pcp_group().all_gather(
                    hidden_states,
                    dim=0,
                )
                router_logits = get_pcp_group().all_gather(
                    router_logits,
                    dim=0,
                )

            # Matrix multiply.
            final_hidden_states = self.quant_method.apply(
                layer=self,
                x=hidden_states_combined
                if do_naive_dispatch_combine
                else hidden_states,
                router_logits=router_logits,
            )

            if has_separate_shared_experts:
                assert self.shared_experts is not None

                if use_shared_experts_stream:
                    # Run shared experts in parallel on a separate stream
                    # NOTE: We start the separate stream here and mark the
                    # sync end point immediately after it is done. This is
                    # important to avoid excessive stream allocations by the cuda
                    # graph replay later.
                    with torch.cuda.stream(self.shared_experts_stream):
                        # Note that hidden_states clone() is necessary here to avoid
                        # conflict with the main stream
                        shared_output = self.shared_experts(hidden_states_clone)
                    current_stream().wait_stream(self.shared_experts_stream)

                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, tuple)
                final_hidden_states, zero_expert_result = final_hidden_states

            def combine_output(states: torch.Tensor) -> torch.Tensor:
                if do_naive_dispatch_combine:
                    states = get_ep_group().combine(states, self.is_sequence_parallel)

                if self.pcp_size > 1:
                    states = get_pcp_group().reduce_scatter(
                        states,
                        dim=0,
                    )

                return states

            if self.shared_experts is not None:
                return (
                    final_hidden_states[0],
                    combine_output(final_hidden_states[1]),
                )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, torch.Tensor)
                return (combine_output(final_hidden_states), zero_expert_result)
            else:
                return combine_output(final_hidden_states)
```

上面这个默认实现还是很简单的 dispatch(allgather) + fused_experts + combine(reducescatter)。

唯一需要注意的是 dispatch 和 combine 都有是否开启 SP 的入参。显然 dispatch 和 combine 是在 EP 域内做的，之前 SP 只在 DP 域内（EP = DP × TP）。
- 如果之前如果没做 SP，DP 域内每个卡上数据是一样的，dispatch 就不需要在 DP 域内 gather 数据。
- 如果之前如果没做 SP，combine 就不需要把 token 分到整个 EP 域，在 DP 域内每个 TP rank 的 hidden_states 一样。

```python
# File: vllm/distributed/device_communicators/all2all.py:102
class AgRsAll2AllManager(All2AllManagerBase):
    """
    An implementation of all2all communication based on
    all-gather (dispatch) and reduce-scatter (combine).
    """

    def __init__(self, cpu_group):
        super().__init__(cpu_group)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather hidden_states and router_logits from all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None

        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]
        hidden_states, router_logits = dist_group.all_gatherv(
            [hidden_states, router_logits],
            dim=0,
            sizes=sizes,
        )
        return hidden_states, router_logits

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        Reduce-scatter hidden_states across all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None

        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        hidden_states = dist_group.reduce_scatterv(hidden_states, dim=0, sizes=sizes)
        return hidden_states

    def destroy(self):
        pass
```

###### Fused Experts

**Expert 前向公式**：

$$\mathbf{y}_i = \text{FFN}_i(\mathbf{x}_i) = \mathbf{W}_2^{(i)} \cdot \sigma(\mathbf{W}_1^{(i)} \cdot \mathbf{x}_i) \odot (\mathbf{W}_3^{(i)} \cdot \mathbf{x}_i)$$

$$\mathbf{y} = \sum_{i \in \text{topk}} w_i \cdot \mathbf{y}_i$$

其中 $\sigma$ 为激活函数（SiLU），$\odot$ 为逐元素乘法，$w_i$ 为 router weight。

主干代码：
```python
# File: vllm/model_executor/layers/fused_moe/fused_moe.py:1740
def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    ... 
    # 前面一堆校验和参数配置
    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.size()

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[
                : tokens_in_chunk * topk_ids.size(1)
            ]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        qcurr_hidden_states, a1q_scale = moe_kernel_quantize_input(
            A=curr_hidden_states,
            A_scale=a1_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape,
        )

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            curr_topk_ids,
            config["BLOCK_SIZE_M"],
            global_num_experts,
            expert_map,
            ignore_invalid_experts=True,
        )

        invoke_fused_moe_kernel(
            qcurr_hidden_states,
            w1,
            intermediate_cache1,
            a1q_scale,
            w1_scale,
            w1_zp,
            curr_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            top_k_num,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=w1_bias,
        )

        # Activation function with multiplication
        if activation == "silu":
            torch.ops._C.silu_and_mul(
                intermediate_cache2, intermediate_cache1.view(-1, N)
            )
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(
                intermediate_cache2, intermediate_cache1.view(-1, N)
            )
        elif activation == "swigluoai":
            # alpha = 1.702, limit = 7.0
            torch.ops._C.swigluoai_and_mul(
                intermediate_cache2, intermediate_cache1.view(-1, N)
            )
        # Activation function without multiplication
        elif activation == SILU_NO_MUL:
            intermediate_cache2 = F.silu(intermediate_cache1.view(-1, N))
        elif activation == GELU_NO_MUL:
            intermediate_cache2 = F.gelu(intermediate_cache1.view(-1, N))
        elif activation == RELU2_NO_MUL:
            intermediate_cache2 = torch.square(F.relu(intermediate_cache1.view(-1, N)))
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}.")

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            A=intermediate_cache2,
            A_scale=a2_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape,
        )

        if expert_map is not None:
            intermediate_cache3.zero_()

        invoke_fused_moe_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            a2q_scale,
            w2_scale,
            w2_zp,
            curr_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=w2_bias,
        )

        ops.moe_sum(
            intermediate_cache3.view(*intermediate_cache3.size()),
            out_hidden_states[begin_chunk_idx:end_chunk_idx],
        )

    return out_hidden_states
```

然后来看最核心的专家计算逻辑。不看量化，核心关注下面几个步骤：

1. **moe_align_block_size**：重排 token_id，按照专家顺序排，并且 pad token 到后续乘法的 batch size。要配合后面乘法的 kernel 调用能更好理解这个的作用。vLLM 注释里有个例子：

   Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
   block_size = 4, and num_experts = 4:
   - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
         with each expert needing to process 3 tokens.
     - As block_size is 4, we pad 1 token for each expert.
     - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
     - Then append padding tokens [12, 12, 12, 12] for each block.
     - After sorting by expert index, we obtain token_ids
         [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
         Tokens 12 are non-existent (padding) and are ignored in
         the subsequent matrix multiplication.
     - The padding ensures that the total number of tokens is now divisible
         by block_size for proper block matrix operations.

2. **invoke_fused_moe_kernel**(第一次调用，up projection)：

   介绍下关键的 fused_moe_kernel 逻辑：先从 topk_ids 数组取左矩阵下标。注意这里 offs_token//top_k 是因为每个 token_id 之前 moe_align_block_size 里都复制了 k 分，比如 0→(0,1,...,k-1)，1→(k,...,2k-1)。

```python
# File: vllm/model_executor/layers/fused_moe/fused_moe.py (fused_moe_kernel)
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
        a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
```

然后是右矩阵下标。因为 fused_moe_kernel 每个 thread 只算一个 block，每个 block 是同一个专家，所以取这个专家就行。

```python
# File: vllm/model_executor/layers/fused_moe/fused_moe.py (fused_moe_kernel)
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )
```

然后是矩阵乘法。会分块相乘再求和，只按矩阵相乘那一维分块。

```python
# File: vllm/model_executor/layers/fused_moe/fused_moe.py (fused_moe_kernel)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    # acc used to enable fp8_fast_accum
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
```

需要注意的是 A 内存是不连续的，因为 token 下标不连续。问了几个 AI 说 triton load A 到寄存器里就连续了，所以还是一次矩阵乘法，只不过会有不连续内存 gather 到一起的开销。所以 vLLM MoE 非默认实现里会有 permute/unpermute，提前把 A 的内存排成连续。

还有一些不怎么重要的细节，比如 expert=-1 的时候直接全零输出，这里不看了。

3. **activation**：

   这里就是带不带 gate 的区别。with_mul/no_mul 两种。如果是带门控的，前面 up projection 算出来是 2×N 的结果（intermediate_cache1），先 view 成 N 然后切分计算 silu(x[:middle]) × (x[middle:])。

**SiLU 激活公式**：

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

**SwiGLU 公式**：

$$\text{SwiGLU}(\mathbf{x}, \mathbf{W}, \mathbf{V}, \mathbf{b}, \mathbf{c}) = \text{Swish}(\mathbf{x}\mathbf{W} + \mathbf{b}) \odot (\mathbf{x}\mathbf{V} + \mathbf{c})$$

```python
# File: vllm/model_executor/layers/fused_moe/fused_moe.py (fused_experts_impl)
        # Activation function with multiplication
        if activation == "silu":
            torch.ops._C.silu_and_mul(
                intermediate_cache2, intermediate_cache1.view(-1, N)
            )
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(
                intermediate_cache2, intermediate_cache1.view(-1, N)
            )
        elif activation == "swigluoai":
            # alpha = 1.702, limit = 7.0
            torch.ops._C.swigluoai_and_mul(
                intermediate_cache2, intermediate_cache1.view(-1, N)
            )
        # Activation function without multiplication
        elif activation == SILU_NO_MUL:
            intermediate_cache2 = F.silu(intermediate_cache1.view(-1, N))
        elif activation == GELU_NO_MUL:
            intermediate_cache2 = F.gelu(intermediate_cache1.view(-1, N))
        elif activation == RELU2_NO_MUL:
            intermediate_cache2 = torch.square(F.relu(intermediate_cache1.view(-1, N)))
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}.")
```

4. **invoke_fused_moe_kernel**(第二次调用, down projection)：

   这里是可以在 kernel 里把 router score 乘进去的。

```python
# File: vllm/model_executor/layers/fused_moe/fused_moe.py (fused_moe_kernel)
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
```

5. **ops.moe_sum**：

   每个 token 在这个 EP rank 上所有专家的计算结果进行汇总。

###### 通信后端

MoE dispatch 默认后端是 all gather + reduce scatter（vllm/distributed/device_communicators/all2all.py）。如果用 all to allv 后端要显式指定并且安装额外依赖。比如 pplx, deepep, flashinfer_all2allv 等。

**为什么默认不用原生的 all_to_allv？**
- 通用和兼容性。
- 小规模数据 AG+RS 性能更好，all2all 还有额外头开销影响时延（是吗？）。而且 NVLink 的带宽很高，推理的通信量不像训练那么大，AG+RS 性能和 all2all 区别不大。
- AG+RS 可以和前后操作做通信-计算掩盖。这块要再研究一下。

用了 all gather 后有个很奇怪的问题，前面有 SP 逻辑把 token 分发到不同 TP rank 上，这里又 all gather 把所有 token 在 EP group 里全合起来，看起来有点多余。也许是减小 gate 和共享专家的激活值和计算量？

##### 2.3.2 模块化替换

非默认实现比如 pplx/deepep/flashinfer_all2allv 都被 vLLM 做成 [模块化抽象](https://github.com/vllm-project/vllm/blob/main/docs/design/fused_moe_modular_kernel.md) 了。主干逻辑在 FusedMoEModularKernel：

```python
# File: vllm/model_executor/layers/fused_moe/modular_kernel.py
@final
class FusedMoEModularKernel(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool = False,
        activation: str = "silu",
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inplace and self.shared_experts is None and not disable_inplace():
            output = hidden_states
        else:
            output = torch.zeros_like(hidden_states)
        # 给共享专家开一个流
        use_shared_experts_stream, hidden_states_clone = (
            self._maybe_setup_shared_experts_stream(hidden_states)
        )
        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts
        # 量化、dispatch(all2all)
        a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_weights = self._prepare(
            hidden_states,
            topk_weights,
            topk_ids,
            global_num_experts,
            expert_map,
            apply_router_weight_on_input,
        )
        # expert
        fused_out = self._fused_experts(
            in_dtype=hidden_states.dtype,
            a1q=a1q,
            a1q_scale=a1q_scale,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_tokens_meta=expert_tokens_meta,
        )
        # combine
        return self._finalize(
            output,
            fused_out,
            hidden_states,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            hidden_states_clone=hidden_states_clone,
            use_shared_experts_stream=use_shared_experts_stream,
        )
```

---

### 代码问题

在 MoE 里有这么一段代码，根据是不是 TPU 调用 torch.ops.vllm.moe_forward 和 self.forward_impl：

```python
# File: vllm/model_executor/layers/fused_moe/layer.py:301
class FusedMoE(CustomOp):
    def forward_native():
        if current_platform.is_tpu():
            # TODO: Once the OOM issue for the TPU backend is resolved, we
            # will switch to using the moe_forward custom op.
            fused_output = self.forward_impl(hidden_states, router_logits)
            assert not isinstance(fused_output, tuple)
        else:
            fused_output = torch.ops.vllm.moe_forward(
                hidden_states, router_logits, self.layer_name
            )
```

但是事实上 torch.ops.vllm.moe_forward 就是 self.forward_impl，在文件末尾有这个注册逻辑：

```python
# File: vllm/model_executor/layers/fused_moe/layer.py (文件末尾)
direct_register_custom_op(
    op_name="moe_forward",
    op_func=moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)
```

这里就产生了一个问题：为什么相同的函数调用在 if-else 两个分支里，看上去可以合并？

Gemini 的回答是这两个函数在 torch.compile 下有区别。torch.ops.vllm.moe_forward 是被注册的 custom op，torch.compile 会把它看做一个黑盒，一个原子操作，而 self.forward_impl 内部是对 torch.compile 可见的，编译的时候会对里面的图做优化。
