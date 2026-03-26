import os
import shutil
import re
from safetensors.torch import load_file, save_file
from tqdm import tqdm

# ===================== 配置路径 =====================
SOURCE_DIR = "/mnt/xds/sfs/DeepSeek-V3.2-w8a8-mtp-QuaRot"
TARGET_DIR = "/mnt/xds/sfs/DeepSeek-V3.2-pruned"
KEEP_LAYERS = list(range(5))  # 保留 0~4 层
# ====================================================

# 匹配 transformer 层权重的正则表达式
layer_pattern = re.compile(r"model\.layers\.(\d+)\.")

# 创建目标目录
os.makedirs(TARGET_DIR, exist_ok=True)

# 遍历源目录所有文件
for filename in tqdm(os.listdir(SOURCE_DIR), desc="Processing files"):
    src_path = os.path.join(SOURCE_DIR, filename)
    tgt_path = os.path.join(TARGET_DIR, filename)

    # 处理 safetensors 权重文件
    if filename.endswith(".safetensors"):
        print(f"Processing weights: {filename}")
        weights = load_file(src_path)
        pruned_weights = {}

        for key, tensor in weights.items():
            match = layer_pattern.match(key)
            if match:
                # 处理 transformer 层，仅保留指定层
                layer_idx = int(match.group(1))
                if layer_idx in KEEP_LAYERS:
                    pruned_weights[key] = tensor
            else:
                # 保留非 transformer 层的全部权重
                pruned_weights[key] = tensor

        # 保存剪枝后的权重文件
        save_file(pruned_weights, tgt_path)
        print(f"Saved pruned weights: {tgt_path}")

    # 直接复制其他文件（配置文件、分词器文件等）
    else:
        if os.path.isfile(src_path):
            shutil.copy2(src_path, tgt_path)
            print(f"Copied: {filename}")

print("\nPruning completed.")
print(f"Source model: {SOURCE_DIR}")
print(f"Pruned model: {TARGET_DIR}")
print(f"Reserved layers: model.layers.0 ~ {max(KEEP_LAYERS)}")
