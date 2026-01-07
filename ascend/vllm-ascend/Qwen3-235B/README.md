# Qwen3-235B on Ascend

## Environment

- **Hardware**: 8 Ascend 910B cards on x86 machine
- **Software**: vllm-ascend 0.13.0rc1

## Build Docker Image

To build the docker image, you may need to clone this repository first:

```bash
# Clone the repository
git clone https://github.com/triomino/Compute-Accelerated-AI.git
cd Compute-Accelerated-AI/ascend/vllm-ascend/Qwen3-235B

# Build the image
docker build -t vllm-ascend-qwen3-235b:latest .
```

## Why vllm_ascend_fix.patch?

This patch applies the fix from [Pull Request #5522](https://github.com/vllm-project/vllm-ascend/pull/5522). While the PR originally aimed to solve a precision problem, it also happens to resolve the crash problem described in [Issue #5541](https://github.com/vllm-project/vllm-ascend/issues/5541), although the developers were unaware of this connection at the time of submission.

## Why xgrammar_fix.patch?

This patch is required to make the structured outputs function work correctly. It is a fix derived from [Issue #5524](https://github.com/vllm-project/vllm-ascend/issues/5524).

## How to serve Qwen3-235B

### 1. Run Container

Run the container with the following command:

```bash
image=vllm-ascend-qwen3-235b:latest
docker run \
    --name vllm-ascend-13 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    --net=host \
    --shm-size=80g \
    --privileged=true \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /data:/data \
    -itd $image bash
```

### 2. Start Server

Inside the container, set up the environment and start the server:

```bash
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1

TS=$(date +"%Y%m%d_%H%M%S")
mkdir -p logs
LOG="logs/serve_${TS}.log"
model=/data/models/Qwen3-235B-A22B-W8A8

vllm serve $model \
    --served-model-name "Qwen3-235B-A22B-W8A8"  \
    --host 0.0.0.0 \
    --port 1025 \
    --async-scheduling \
    --tensor-parallel-size 8 \
    --data-parallel-size 1 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --max-num-seqs 128 \
    --max-model-len 8192 \
    --max-num-batched-tokens 40960 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --quantization ascend \
    2>&1 | tee $LOG
```

### 3. References

For more details, please refer to the official documentation:
https://docs.vllm.ai/projects/ascend/en/main/tutorials/Qwen3-235B-A22B.html
