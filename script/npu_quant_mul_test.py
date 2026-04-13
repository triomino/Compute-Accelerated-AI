"""
Reproduction script: Token-parallel vs TP-parallel precision difference
in W8A8 static quantized matmul (o_proj layer).

This script simulates what happens in the o_proj layer under two modes:
  1. Token-parallel (DSA-CP mode): weight is NOT split, input is split on dim=0 (tokens).
     Each "rank" processes its own subset of tokens with the full weight.
  2. TP-parallel: weight is split along rows (input dim), input is split on dim=-1 (hidden).
     Each "rank" computes a partial result, then results are summed (simulating all_reduce).

Both modes use W8A8 static quantization:
  - Activation: per-tensor static quantization (int8)
  - Weight: per-channel int8, with deq_scale for dequantization

Usage: Run on a single NPU card.
  python repro_token_vs_tp_quant.py
"""

import torch
import torch_npu


def quantize_static(x: torch.Tensor, input_scale: torch.Tensor,
                    input_scale_reciprocal: torch.Tensor,
                    input_offset: torch.Tensor) -> torch.Tensor:
    """
    Simulate static per-tensor quantization:  x_int8 = round(x / input_scale) + input_offset
    Using the same op as vllm-ascend: torch.ops.vllm.quantize
    If vllm quantize op is not available, use a manual fallback.
    """
    try:
        return torch.ops.vllm.quantize(x, input_scale, input_scale_reciprocal, input_offset)
    except Exception:
        # Manual fallback: symmetric-like quantization
        x_scaled = x * input_scale_reciprocal
        x_int8 = torch.clamp(torch.round(x_scaled) + input_offset.to(x_scaled.dtype), -128, 127).to(torch.int8)
        return x_int8


def setup_fake_layer(output_size: int, input_size: int, params_dtype: torch.dtype, device: str):
    """
    Create fake W8A8 static quantization parameters, mimicking what
    process_weights_after_loading() produces.

    Weight shape after transpose: [input_size, output_size] (stored as int8)
    deq_scale shape: [output_size] (float32 for bf16 models)
    input_scale: scalar (per-tensor)
    input_offset: scalar (int8)
    quant_bias: [output_size] (int32)
    """
    # Simulate a real weight: random bf16 weight -> quantize to int8
    weight_fp = torch.randn(output_size, input_size, dtype=params_dtype, device=device)
    weight_scale = weight_fp.abs().max(dim=1, keepdim=True).values / 127.0  # per-channel scale [output_size, 1]

    weight_int8 = torch.clamp(torch.round(weight_fp / weight_scale), -128, 127).to(torch.int8)
    # After process_weights_after_loading, weight is transposed: [input_size, output_size]
    weight_int8_T = weight_int8.t().contiguous()

    # input_scale: simulate a per-tensor activation scale (scalar)
    input_scale_val = 0.05  # typical scale
    input_scale = torch.tensor([input_scale_val], dtype=params_dtype, device=device)
    # Expand to input_size as done in process_weights_after_loading
    aclnn_input_scale = input_scale.repeat(input_size)
    aclnn_input_scale_reciprocal = 1.0 / aclnn_input_scale
    input_offset = torch.zeros(input_size, dtype=params_dtype, device=device)

    # deq_scale = input_scale * weight_scale (per-channel dequantization scale)
    deq_scale = (input_scale_val * weight_scale.squeeze()).to(torch.float32).to(device)

    # quant_bias: typically int32, set to zero for simplicity
    quant_bias = torch.zeros(output_size, dtype=torch.int32, device=device)

    return {
        "weight_int8_T": weight_int8_T,  # [input_size, output_size]
        "weight_fp": weight_fp,  # [output_size, input_size] original
        "deq_scale": deq_scale,  # [output_size]
        "quant_bias": quant_bias,  # [output_size]
        "aclnn_input_scale": aclnn_input_scale,
        "aclnn_input_scale_reciprocal": aclnn_input_scale_reciprocal,
        "input_offset": input_offset,
        "params_dtype": params_dtype,
    }


def w8a8_quant_matmul(x_bf16: torch.Tensor, layer: dict, tp_rank: int = 0) -> torch.Tensor:
    """
    Perform W8A8 static quantized matmul, replicating AscendW8A8LinearMethod.apply()

    Steps:
      1. Quantize activation: bf16 -> int8  (per-tensor static scale)
      2. npu_quant_matmul(x_int8, weight_int8, deq_scale) -> bf16 output
    """
    x_int8 = quantize_static(
        x_bf16,
        layer["aclnn_input_scale"],
        layer["aclnn_input_scale_reciprocal"],
        layer["input_offset"],
    )
    quant_bias = layer["quant_bias"] if tp_rank == 0 else None
    output = torch_npu.npu_quant_matmul(
        x_int8,
        layer["weight_int8_T"],
        layer["deq_scale"],
        bias=quant_bias,
        output_dtype=layer["params_dtype"],
    )
    return output


def token_parallel_matmul(hidden_states: torch.Tensor, layer: dict, tp_size: int = 2) -> torch.Tensor:
    """
    Token-parallel mode (DSA-CP / ShardedCPRowParallelOp):
      - Weight is NOT split (full weight)
      - hidden_states is split along dim=0 (token dimension)
      - Each "rank" independently quantizes & matmuls its chunk
      - Results are concatenated along dim=0

    This is what happens when enable_dsa_cp_with_layer_shard() is True
    and ShardedCPRowParallelOp is used for o_proj.
    """
    num_tokens = hidden_states.shape[0]
    chunk_size = num_tokens // tp_size
    outputs = []

    for rank in range(tp_size):
        start = rank * chunk_size
        end = start + chunk_size
        x_chunk = hidden_states[start:end]  # [chunk_size, hidden_dim]

        # Each rank uses the full weight, tp_rank=0 (ShardedCPRowParallelOp fakes tp_rank=0)
        out = w8a8_quant_matmul(x_chunk, layer, tp_rank=0)
        outputs.append(out)

    return torch.cat(outputs, dim=0)  # [num_tokens, output_size]


def tp_parallel_matmul(hidden_states: torch.Tensor, layer: dict, tp_size: int = 2) -> torch.Tensor:
    """
    TP-parallel mode (standard RowParallelLinear):
      - Weight is split along rows (input dimension): weight_rank = weight[:, input_size//tp*rank : input_size//tp*(rank+1)]
      - hidden_states is split along dim=-1 (hidden dimension)
      - Each "rank" quantizes its partition & does matmul -> partial output
      - Partial outputs are summed (all_reduce)

    For RowParallelLinear with W8A8 static quant:
      - weight_int8_T shape: [input_size, output_size], split along dim=0 -> [input_size//tp, output_size]
      - deq_scale: [output_size], split? No — deq_scale = input_scale * weight_scale, where weight_scale is per output channel.
        In TP row-parallel, weight is split along INPUT dim, but deq_scale is per OUTPUT channel.
        However, the input_scale changes because only a partition of the input is being quantized.
        In practice, the static input_scale is pre-computed for the partition.
        For this repro, we re-derive the per-partition quantization parameters.
    """
    num_tokens, hidden_dim = hidden_states.shape
    partition_size = hidden_dim // tp_size
    output_size = layer["weight_int8_T"].shape[1]
    params_dtype = layer["params_dtype"]

    # Build per-rank sub-layers (simulating weight sharding done at model load time)
    partial_outputs = []
    for rank in range(tp_size):
        start = rank * partition_size
        end = start + partition_size

        x_partition = hidden_states[:, start:end]  # [num_tokens, partition_size]

        # Weight partition: weight_int8_T is [input_size, output_size], split dim=0
        weight_partition = layer["weight_int8_T"][start:end, :].contiguous()  # [partition_size, output_size]

        # For static quant, input_scale is pre-computed per-tensor for the full input.
        # When the input is partitioned, the same static scale is used for the partition.
        # (In real deployment, input_scale is calibrated offline and doesn't change per partition.)
        # The aclnn_input_scale is expanded to input_size; take the partition slice.
        sub_input_scale = layer["aclnn_input_scale"][start:end]
        sub_input_scale_reciprocal = layer["aclnn_input_scale_reciprocal"][start:end]
        sub_input_offset = layer["input_offset"][start:end]

        # Quantize the partition
        x_int8 = quantize_static(x_partition, sub_input_scale, sub_input_scale_reciprocal, sub_input_offset)

        # deq_scale needs to be recalculated for this partition:
        # Original: deq_scale = input_scale_scalar * weight_scale_per_channel
        # The weight_scale_per_channel doesn't change (it's per output channel),
        # and input_scale_scalar is the same. So deq_scale is the same.
        # But the matmul result is only a partial sum — the final dequantized output
        # is sum of partial (x_part_int8 @ w_part_int8) * deq_scale across ranks.
        # Actually, npu_quant_matmul computes: output = (x_int8 @ weight_int8) * deq_scale
        # For TP: result = sum_over_ranks[ (x_part_int8 @ w_part_int8) * deq_scale ]
        # But this is NOT the same as (x_full_int8 @ w_full_int8) * deq_scale !!
        # The deq_scale should only be applied once after the sum.
        #
        # In practice, npu_quant_matmul always applies deq_scale internally.
        # For TP row-parallel with W8A8 static:
        #   rank0: out0 = npu_quant_matmul(x0, w0, deq_scale) — applies deq_scale
        #   rank1: out1 = npu_quant_matmul(x1, w1, deq_scale) — applies deq_scale
        #   final = out0 + out1
        # This double-applies deq_scale? No — actually the int matmul gives
        # int_result = sum_k(x_int8[k] * w_int8[k]), and deq_scale converts it back to float.
        # For TP: int_result_partial = sum_{k in partition}(x_int8[k] * w_int8[k])
        # npu_quant_matmul output = int_result_partial * deq_scale  (in float)
        # After all_reduce: final = sum(int_result_partial * deq_scale) = (sum int_result_partial) * deq_scale
        # This is correct mathematically for the integer part. The key difference is in QUANTIZATION:
        # Token-parallel: one scale for all hidden_dim elements
        # TP-parallel: one scale for partition_size elements (same scale value, but input range is different!)
        #
        # Wait — the input_scale is STATIC (pre-calibrated). It's the same scalar regardless of partition.
        # So quantize(x_full, scale) splits into quantize(x_part, scale) which uses THE SAME scale.
        # The int8 values are the same whether you quantize the full vector or parts independently!
        # (Because it's per-tensor static scale, not per-partition.)
        #
        # So the difference must come from the npu_quant_matmul kernel's internal rounding behavior
        # when accumulating partial sums vs full sums.

        quant_bias = layer["quant_bias"] if rank == 0 else None
        out_partial = torch_npu.npu_quant_matmul(
            x_int8,
            weight_partition,
            layer["deq_scale"],
            bias=quant_bias,
            output_dtype=params_dtype,
        )
        partial_outputs.append(out_partial)

    # Simulate all_reduce (sum)
    return sum(partial_outputs)


def compute_fp_reference(hidden_states: torch.Tensor, layer: dict) -> torch.Tensor:
    """Compute the reference output using full-precision (bf16) matmul."""
    # weight_fp is [output_size, input_size], so output = hidden_states @ weight_fp.T
    return torch.matmul(hidden_states, layer["weight_fp"].t())


def main():
    device = "npu:0"
    torch.npu.set_device(device)

    # Typical o_proj dimensions for a large model
    # hidden_size = num_heads * v_head_dim (input to o_proj)
    # output_size = hidden_size (output of o_proj)
    # Let's use realistic-ish dimensions:
    num_heads = 16
    v_head_dim = 128
    input_size = num_heads * v_head_dim   # 2048
    output_size = input_size              # 2048 (hidden_size)
    num_tokens = 64
    tp_size = 2
    params_dtype = torch.bfloat16

    print("=" * 70)
    print("W8A8 Static Quant Matmul: Token-Parallel vs TP-Parallel Precision")
    print("=" * 70)
    print(f"  num_tokens   = {num_tokens}")
    print(f"  input_size   = {input_size}")
    print(f"  output_size  = {output_size}")
    print(f"  tp_size      = {tp_size}")
    print(f"  params_dtype = {params_dtype}")
    print()

    # Setup fake layer (quantized weights + scales)
    layer = setup_fake_layer(output_size, input_size, params_dtype, device)

    # Generate random input (simulating attention output)
    torch.manual_seed(42)
    hidden_states = torch.randn(num_tokens, input_size, dtype=params_dtype, device=device)

    # 1. Full-precision reference
    ref_output = compute_fp_reference(hidden_states, layer)

    # 2. Token-parallel quantized matmul
    token_par_output = token_parallel_matmul(hidden_states, layer, tp_size=tp_size)

    # 3. TP-parallel quantized matmul
    tp_par_output = tp_parallel_matmul(hidden_states, layer, tp_size=tp_size)

    # 4. Single-rank full quantized matmul (no parallelism, baseline)
    single_output = w8a8_quant_matmul(hidden_states, layer, tp_rank=0)

    # Compare results
    print("-" * 70)
    print("Comparison: Token-Parallel vs TP-Parallel")
    print("-" * 70)

    diff_token_tp = (token_par_output - tp_par_output).float()
    diff_token_ref = (token_par_output - ref_output).float()
    diff_tp_ref = (tp_par_output - ref_output).float()
    diff_single_ref = (single_output - ref_output).float()

    # Token-parallel should be identical to single (since weight is not split,
    # just tokens are split — quantization is independent per token)
    diff_token_single = (token_par_output - single_output).float()

    print(f"  token_par vs single (should be 0):  max_abs_diff = {diff_token_single.abs().max().item():.6e}")
    print()
    print(f"  token_par vs tp_par:                max_abs_diff = {diff_token_tp.abs().max().item():.6e}")
    print(f"  token_par vs tp_par:                mean_abs_diff = {diff_token_tp.abs().mean().item():.6e}")
    print(f"  token_par vs tp_par:                rmse = {diff_token_tp.pow(2).mean().sqrt().item():.6e}")
    print()
    print(f"  token_par vs fp_ref:                max_abs_diff = {diff_token_ref.abs().max().item():.6e}")
    print(f"  token_par vs fp_ref:                mean_abs_diff = {diff_token_ref.abs().mean().item():.6e}")
    print()
    print(f"  tp_par vs fp_ref:                   max_abs_diff = {diff_tp_ref.abs().max().item():.6e}")
    print(f"  tp_par vs fp_ref:                   mean_abs_diff = {diff_tp_ref.abs().mean().item():.6e}")
    print()
    print(f"  single vs fp_ref:                   max_abs_diff = {diff_single_ref.abs().max().item():.6e}")
    print(f"  single vs fp_ref:                   mean_abs_diff = {diff_single_ref.abs().mean().item():.6e}")

    # Check if token_par and tp_par produce identical results
    print()
    print("-" * 70)
    print("Analysis")
    print("-" * 70)
    if diff_token_tp.abs().max().item() == 0:
        print("  Token-parallel and TP-parallel produce IDENTICAL results.")
        print("  The precision issue may come from a different source.")
    else:
        print("  Token-parallel and TP-parallel produce DIFFERENT results!")
        print("  This confirms the precision gap between the two parallelism strategies.")
        print()
        # Show some examples of differing elements
        diff_mask = diff_token_tp.abs() > 0
        num_diff = diff_mask.sum().item()
        total = diff_token_tp.numel()
        print(f"  Differing elements: {num_diff}/{total} ({100*num_diff/total:.2f}%)")

        # Show which mode is closer to fp reference
        token_par_mse = diff_token_ref.pow(2).mean().item()
        tp_par_mse = diff_tp_ref.pow(2).mean().item()
        print()
        print(f"  MSE(token_par, fp_ref) = {token_par_mse:.6e}")
        print(f"  MSE(tp_par, fp_ref)    = {tp_par_mse:.6e}")
        if tp_par_mse < token_par_mse:
            print("  => TP-parallel is CLOSER to fp reference (better precision).")
        else:
            print("  => Token-parallel is CLOSER to fp reference (better precision).")

    print()
    print("=" * 70)
    print("Root cause hypothesis:")
    print("  In W8A8 static quant, npu_quant_matmul internally computes:")
    print("    output = (x_int8 @ w_int8) * deq_scale")
    print("  The integer matmul (x_int8 @ w_int8) accumulates in int32.")
    print()
    print("  Token-parallel: accumulates over full input_size (2048) elements.")
    print("  TP-parallel:    accumulates over input_size/tp (1024) elements,")
    print("                  then dequantizes to bf16, then sums in bf16.")
    print()
    print("  Splitting the accumulation changes the dequantization boundary:")
    print("  - Full accumulation: int32_full * deq_scale -> bf16")
    print("  - Split accumulation: (int32_part0 * deq_scale + int32_part1 * deq_scale) -> bf16")
    print("  These are mathematically equivalent in exact arithmetic, but")
    print("  the bf16 addition introduces rounding that the int32 accumulation avoids.")
    print("  So token-parallel (full accumulation) should theoretically be MORE precise")
    print("  for each individual element, but the question is whether the static")
    print("  per-tensor quantization scale is well-suited for the full input range.")
    print("=" * 70)


if __name__ == "__main__":
    main()
