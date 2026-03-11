# vLLM-Ascend PD分离一键部署工具

一键部署 vLLM-Ascend PD (Prefill-Decode) 分离服务，支持多节点分布式部署。

## 功能特性

- **集中配置**: 单个 YAML 文件管理所有节点配置
- **一键部署**: 自动生成分发脚本并远程部署
- **容器支持**: 自动管理 Docker 容器生命周期
- **并行部署**: 多节点并行部署，提高效率
- **状态监控**: 实时查看各节点部署状态
- **日志收集**: 便捷查看各节点运行日志

## 快速开始

### 1. 安装依赖

```bash
cd vllm-pd-deployer
pip install -r requirements.txt
```

### 2. 配置 SSH 免密登录

确保部署机器可以免密 SSH 登录到所有目标节点：

```bash
ssh-copy-id root@<node-ip>
```

### 3. 编辑配置文件

复制示例配置并修改：

```bash
cp config.example.yaml config.yaml
vim config.yaml
```

主要需要修改：
- 各节点的 IP 地址 (`host`)
- SSH 登录信息 (`user`, `key_file`)
- Docker 容器配置 (`container_name`, `image`, `workdir`)
- DP/TP 配置 (`dp_size`, `tp_size`)

### 4. 验证配置

```bash
python deploy.py validate --config config.yaml
```

### 5. 一键部署

```bash
python deploy.py deploy --config config.yaml
```

### 6. 查看状态

```bash
python deploy.py status --config config.yaml
```

## 命令详解

### deploy - 部署服务

```bash
# 部署所有服务（P层 + D层 + Proxy）
python deploy.py deploy --config config.yaml

# 仅部署 P 层
python deploy.py deploy --config config.yaml --target prefill

# 仅部署 D 层
python deploy.py deploy --config config.yaml --target decode

# 仅部署 Proxy
python deploy.py deploy --config config.yaml --target proxy

# 仅生成脚本，不实际部署
python deploy.py deploy --config config.yaml --dry-run

# 指定脚本输出目录
python deploy.py deploy --config config.yaml --output ./my-scripts
```

### stop - 停止服务

```bash
# 停止所有服务
python deploy.py stop --config config.yaml

# 停止指定层
python deploy.py stop --config config.yaml --target prefill
```

### status - 查看状态

```bash
python deploy.py status --config config.yaml
```

输出示例：
```
┌───────────┬───────────┬─────────┬─────────┬──────────────────────┐
│ Node      │ Role      │ Status  │ Health  │ Message              │
├───────────┼───────────┼─────────┼─────────┼──────────────────────┤
│ p-node-1  │ prefill   │ running │ unknown │ Deployed successfully│
│ p-node-2  │ prefill   │ running │ unknown │ Deployed successfully│
│ d-node-1  │ decode    │ running │ unknown │ Deployed successfully│
│ d-node-2  │ decode    │ running │ unknown │ Deployed successfully│
│ proxy     │ proxy     │ running │ unknown │ Deployed successfully│
└───────────┴───────────┴─────────┴─────────┴──────────────────────┘
```

### logs - 查看日志

```bash
# 查看节点日志（默认最后100行）
python deploy.py logs --config config.yaml --node p-node-1

# 查看指定行数
python deploy.py logs --config config.yaml --node p-node-1 --tail 50
```

### validate - 验证配置

```bash
python deploy.py validate --config config.yaml
```

### generate - 仅生成脚本

```bash
# 生成所有脚本到指定目录
python deploy.py generate --config config.yaml --output ./generated
```

## 配置文件详解

### 全局配置 (global)

```yaml
global:
  model:
    path: "/mnt/sfs_turbo/GLM-5-w8a8-new"  # 模型权重路径
    name: "dsv3"                           # 服务模型名称
    max_model_len: 120000                  # 最大序列长度
    quantization: "ascend"                 # 量化方式
  
  vllm:
    gpu_memory_utilization: 0.9            # GPU显存利用率
    max_num_seqs: 16                       # 最大并发序列数
    max_num_batched_tokens: 8192           # 最大批处理token数
  
  env:                                     # 全局环境变量
    HCCL_OP_EXPANSION_MODE: "AIV"
    # ... 其他环境变量
```

### P层配置 (prefill)

```yaml
prefill:
  dp_size: 2              # 总数据并行度
  tp_size: 16             # 张量并行度（每张卡数）
  
  nodes:
    - name: "p-node-1"
      host: "192.168.0.32"     # 节点IP
      ssh:
        user: "root"
        key_file: "~/.ssh/id_rsa"
      docker:
        enabled: true
        container_name: "vllm-ascend-p1"
        image: "vllm-ascend:glm5-0.13.0-a3"
        workdir: "/mnt/sfs_turbo/glm5_PD/P/glm5_dp2_tp16/node1"
      local_dp_size: 1      # 该节点的DP实例数
      dp_rank_start: 0      # DP起始rank（从0开始，连续不重复）
      dp_rpc_port: 12899    # DP通信端口
      vllm_start_port: 9100 # vLLM服务起始端口
      nic_name: "enp23s0f3" # 网卡名称
```

### D层配置 (decode)

与 P 层类似，`role` 自动识别为 decode。

### Proxy 配置

```yaml
proxy:
  enabled: true
  host: "192.168.0.100"     # Proxy 所在机器
  port: 8000                # Proxy 监听端口
  # hosts/ports 会自动从 P/D 配置生成
```

## 典型使用场景

### 场景1: 调整 DP/TP 配置

只需修改 `config.yaml`：

```yaml
prefill:
  dp_size: 4    # 从 2 改为 4
  tp_size: 8    # 从 16 改为 8
```

重新部署即可：
```bash
python deploy.py deploy --config config.yaml
```

### 场景2: 更换机器

修改对应节点的 `host`：

```yaml
nodes:
  - name: "p-node-1"
    host: "192.168.0.50"  # 新IP地址
```

其他配置保持不变。

### 场景3: 添加新节点

在 `nodes` 列表中添加新节点配置：

```yaml
nodes:
  # ... 现有节点
  - name: "d-node-5"
    host: "192.168.0.60"
    # ... 其他配置
    local_dp_size: 4
    dp_rank_start: 16     # 确保rank连续
```

记得更新 `decode.dp_size` 以匹配总 DP 数。

## 目录结构

```
vllm-pd-deployer/
├── deploy.py                 # 主入口脚本
├── config.example.yaml       # 配置文件示例
├── requirements.txt          # Python依赖
├── README.md                 # 本文档
│
├── templates/                # 脚本模板
│   ├── server.sh.j2
│   ├── run_dp_template.sh.j2
│   ├── launch_online_dp.py.j2
│   └── proxy.sh.j2
│
├── core/                     # 核心模块
│   ├── config.py             # 配置解析
│   ├── ssh_client.py         # SSH客户端
│   ├── docker_manager.py     # Docker管理
│   ├── generator.py          # 脚本生成器
│   └── deployer.py           # 部署编排器
│
└── generated/                # 生成的脚本（自动创建）
    ├── prefill/
    │   ├── p-node-1/
    │   │   ├── server.sh
    │   │   ├── run_dp_template.sh
    │   │   └── launch_online_dp.py
    │   └── p-node-2/
    └── decode/
        └── ...
```

## 依赖要求

- Python >= 3.8
- SSH 免密登录到所有目标节点
- Docker 已在目标节点安装（如使用容器）

## 故障排查

### 1. SSH 连接失败

```bash
# 测试 SSH 连接
ssh -i ~/.ssh/id_rsa root@<node-ip>

# 检查密钥权限
chmod 600 ~/.ssh/id_rsa
```

### 2. 配置验证失败

```bash
# 查看详细错误
python deploy.py validate --config config.yaml -v
```

常见错误：
- 端口冲突
- DP rank 不连续或重复
- 缺少必需字段

### 3. 部署失败

```bash
# 查看详细日志
python deploy.py deploy --config config.yaml -v

# 查看节点日志
python deploy.py logs --config config.yaml --node <node-name>
```

### 4. 容器启动失败

手动检查容器状态：
```bash
ssh root@<node-ip> "docker logs <container-name>"
```

## 注意事项

1. **配置备份**: 修改配置前备份 `config.yaml`
2. **端口规划**: 确保端口不冲突，特别是 `vllm_start_port` 范围
3. **rank 连续性**: DP rank 必须连续且不重复
4. **SSH 超时**: 大文件传输可能需要调整 SSH 超时时间
5. **Docker 权限**: 确保用户有权限操作 Docker

## License

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v1.0.0 (2026-03-11)
- 初始版本发布
- 支持 PD 分离部署
- 支持 Docker 容器管理
- 支持多节点并行部署
