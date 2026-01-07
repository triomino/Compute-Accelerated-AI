FROM quay.io/ascend/vllm-ascend:v0.13.0rc1

COPY vllm_ascend_fix.patch /vllm_ascend_fix.patch
COPY xgrammar_fix.patch /xgrammar_fix.patch

# Apply vllm_ascend_fix.patch to /vllm-workspace/vllm-ascend
WORKDIR /vllm-workspace/vllm-ascend
RUN git apply /vllm_ascend_fix.patch

# Apply xgrammar_fix.patch to /usr/local/python3.11.13/lib/python3.11/site-packages/xgrammar/kernels/apply_token_bitmask_inplace_cpu.py
RUN patch /usr/local/python3.11.13/lib/python3.11/site-packages/xgrammar/kernels/apply_token_bitmask_inplace_cpu.py < /xgrammar_fix.patch
