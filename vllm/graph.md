## 静态图 FULL_DECODE_ONLY
总体是 `torch.compile`+`fusion pass`+`CUDAGraph` 三层搞图模式。
正常自己玩一个 `torch.compile` 就完事了。但是看起来 vllm 觉得 `torch.compile` 封装太多东西了，想把一部分 `fusion pass` 和 `CUDAGraph` 摘出来自己管理。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.subgraph_rewriter import replace_pattern


class TinyBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size, bias=False)

    def forward(self, x):
        x = self.linear1(x)

        # 故意写成 sigmoid(x) * x
        # fusion pass 会把它替换成 F.silu(x)
        x = torch.sigmoid(x) * x

        x = self.linear2(x)
        return x


def silu_pattern(x):
    return torch.sigmoid(x) * x


def silu_replacement(x):
    return F.silu(x)


def fusion_backend(gm: torch.fx.GraphModule, example_inputs):
    print("\n========== FX graph before fusion ==========")
    print(gm.graph)

    matches = replace_pattern(gm, silu_pattern, silu_replacement)

    print(f"\nFusion matches: {len(matches)}")

    gm.graph.lint()
    gm.recompile()

    print("\n========== FX graph after fusion ==========")
    print(gm.graph)

    # 继续交给 Inductor 编译
    return torch._inductor.compile(gm, example_inputs)


class CUDAGraphRunner:
    def __init__(self, fn, static_input: torch.Tensor):
        self.fn = fn
        self.static_input = static_input
        self.graph = torch.cuda.CUDAGraph()
        self.static_output = None

        # warmup：让 torch.compile / Inductor / Triton 编译、autotune、显存分配
        # 都发生在 CUDA Graph capture 之前
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())

        with torch.no_grad():
            with torch.cuda.stream(warmup_stream):
                for _ in range(5):
                    _ = self.fn(self.static_input)

        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        # 正式 capture
        with torch.no_grad():
            with torch.cuda.graph(self.graph):
                self.static_output = self.fn(self.static_input)

        torch.cuda.synchronize()

    def __call__(self, x: torch.Tensor):
        assert x.shape == self.static_input.shape
        assert x.dtype == self.static_input.dtype
        assert x.device == self.static_input.device

        # 把真实输入 copy 到固定地址的 static buffer
        self.static_input.copy_(x)

        # replay 固定 CUDA kernel 序列
        self.graph.replay()

        # static_output 是复用内存；clone 方便外面保存结果
        return self.static_output.clone()


def main():
    assert torch.cuda.is_available(), "This demo requires CUDA."

    torch.manual_seed(0)

    device = "cuda"
    dtype = torch.float16

    batch_size = 16
    hidden_size = 1024

    model = TinyBlock(hidden_size).to(device=device, dtype=dtype).eval()

    compiled_model = torch.compile(
        model,
        backend=fusion_backend,
        fullgraph=True,
        dynamic=False,
    )

    # 固定地址 input buffer
    static_input = torch.empty(
        batch_size,
        hidden_size,
        device=device,
        dtype=dtype,
    )

    # 先跑一次，触发 torch.compile
    x0 = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)

    with torch.no_grad():
        static_input.copy_(x0)
        _ = compiled_model(static_input)

    torch.cuda.synchronize()

    # capture compiled model
    runner = CUDAGraphRunner(compiled_model, static_input)

    # 测试两次不同输入
    x1 = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    x2 = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)

    with torch.no_grad():
        y1_ref = compiled_model(x1)
        y2_ref = compiled_model(x2)

        y1_graph = runner(x1)
        y2_graph = runner(x2)

    torch.cuda.synchronize()

    print("\n========== correctness ==========")
    print("y1 max diff:", (y1_ref - y1_graph).abs().max().item())
    print("y2 max diff:", (y2_ref - y2_graph).abs().max().item())

    print("\n========== summary ==========")
    print("torch.compile: enabled")
    print("FX fusion pass: sigmoid(x) * x -> silu(x)")
    print("CUDA Graph: explicit capture/replay")
    print("dtype:", dtype)
    print("device:", torch.cuda.get_device_name())


if __name__ == "__main__":
    main()
```
