# OpenPI JAX PI0.5 训练与 SO-101 部署

涉及数据格式、字段映射、关节状态/动作转换、训练参数和真机 WebSocket 部署的完整说明见
[SO101_OpenPI_完整说明.md](SO101_OpenPI_完整说明.md)。

本目录使用 OpenPI 官方 JAX PI0.5：A100 从官方 `pi05_base` 做全参数微调，部署时运行
OpenPI WebSocket policy server；连接机械臂的本地主机只运行轻量 `openpi-client` 和 LeRobot
硬件驱动。

训练数据固定为：

```text
/data5/zjh/data/so101_pickup_battery_merged
100 episodes / 39000 frames / 30 FPS
```

模型输入使用 `front`、`wrist` 和6维关节状态。前5个关节在训练管线中转换为相对动作，
夹爪保持绝对动作；推理输出会在服务端自动还原为6维绝对目标。

## A100 环境

首次创建 OpenPI 环境：

```bash
cd /home/zjh/openpi
export UV_CACHE_DIR=/data5/zjh/.cache/uv
export TMPDIR=/data5/zjh/openpi/tmp
mkdir -p "$TMPDIR"
uv sync --python 3.11 --no-dev
```

训练环境使用 `/home/zjh/lerobot_openpi_v044` 的 LeRobot v0.4.4 读取 v3.0 数据；它与
OpenPI 固定的 Transformers/JAX 依赖兼容，不会修改现有 `/home/zjh/lerobot` 环境。

先计算 OpenPI 自己的 normalization stats：

```bash
cd /home/zjh/openpi
./examples/so101/compute_norm_stats.sh
```

输出位置：

```text
/data5/zjh/openpi/assets/pi05_so101_pickup_battery/so101_pickup_battery_merged/norm_stats.json
```

## 双卡全参数训练

`batch_size=18` 是全局 batch，每卡9；`fsdp_devices=2` 会把模型和优化器状态分片到2卡。

```bash
tmux new -s openpi_so101

cd /home/zjh/openpi
export CUDA_VISIBLE_DEVICES=0,1
export EXP_NAME=full_sft_30k_bs18_fsdp2

./examples/so101/run_train.sh
```

两张卡必须是同一型号，且每卡启动前至少保留约60GB空闲显存。当前服务器的0/1是同型号
PCIe A100；1号卡即使仍有约11.6GB的 VLLM 占用，也有约69GB空闲。
启动前检查：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader
```

`run_train.sh` 默认会检查每卡至少60,000MiB空闲显存。只在已经实测确认显存足够时，才可用
`MIN_GPU_FREE_MIB` 调低这个保护阈值。

当前0/1双卡已用真实 batch18 完成一次前向和反向测试：首次编译加执行约118秒，未发生 OOM。
训练脚本默认使用 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`，适配1号卡仍有约11.6GB占用的情况。

按 `Ctrl-b d` 离开 tmux，重新查看：

```bash
tmux attach -t openpi_so101
```

checkpoint 位于：

```text
/data5/zjh/openpi/checkpoints/pi05_so101_pickup_battery/full_sft_30k_bs18_fsdp2/{5000,10000,...,25000,29999}
```

中断后继续：

```bash
CUDA_VISIBLE_DEVICES=0,1 RESUME=true EXP_NAME=full_sft_30k_bs18_fsdp2 \
  ./examples/so101/run_train.sh
```

## 远程真机部署

这是当前项目为 SO-101 补齐的远程部署链路，不使用 LeRobot 原来的异步 action queue：

```text
A100 OpenPI JAX policy server
        |
    SSH tunnel, localhost:8000
        |
本地 SO-101 client: 两路相机 + 串口 + 标定 + 安全限幅
```

client 是**同步分块控制**：读取当前 front/wrist/state，等待 A100 返回一段动作，本地以 `CONTROL_FPS`
执行该段动作后才重新取图。动作段内的关节目标以30Hz发送，但视觉策略不是30Hz闭环。

### 远程部署涉及的改动文件

| 文件 | 改动/职责 |
| --- | --- |
| `examples/so101/run_policy_server.sh` | A100 启动官方 `serve_policy.py`；设置 JAX/HF/缓存环境变量；修正顶层 `--port` 必须在 `policy:checkpoint` 前的参数顺序。 |
| `examples/so101/open_ssh_tunnel.sh` | 本地 `ssh -N -L 8000:127.0.0.1:8000`，并启用 SSH 30 秒保活。 |
| `examples/so101/run_so101_client.sh` | 把环境变量转换为 client 的串口、相机、控制和安全参数。 |
| `examples/so101/so101_client.py` | 新增 SO-101 硬件 client：双相机、六维关节状态、同步 chunk 执行、安全限幅、时延 JSONL 日志。 |
| `packages/openpi-client/src/openpi_client/websocket_client_policy.py` | 支持配置 WebSocket ping；SO-101 client 禁用它，避免首次 JAX 编译超过默认20秒时错误断开。 |
| `src/openpi/policies/so101_policy.py` | 将 front/wrist/六维 state/action 映射到官方 PI0.5 接口，推理输出截取为6维。 |

`scripts/serve_policy.py` 和 WebSocket 协议仍是 OpenPI 官方实现；这里增加的是 SO-101 硬件适配和部署包装。

### 首次同步到本地 5060

以下命令在**本地机械臂主机**执行。假设本地项目放在 `~/桌面/openpi`：

```bash
export LOCAL_OPENPI="$HOME/桌面/openpi"

rsync -avP --exclude='__pycache__' \
  zjh@114.214.255.57:/home/zjh/openpi/examples/so101/ \
  "$LOCAL_OPENPI/examples/so101/"

rsync -avP --exclude='__pycache__' \
  zjh@114.214.255.57:/home/zjh/openpi/packages/openpi-client/ \
  "$LOCAL_OPENPI/packages/openpi-client/"
```

首次安装一次本地通信包；以后同步源码后 editable install 会直接使用新文件：

```bash
conda activate lerobot312
cd "$HOME/桌面/openpi/packages/openpi-client"
python -m pip install -e .
```

### 三个终端与启动顺序

#### 终端 A：A100，启动推理服务

不需要手动进入 conda 环境。脚本内部使用 `/home/zjh/openpi/.venv` 的 `uv run --no-dev`。

```bash
cd /home/zjh/openpi

export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR=/data5/zjh/openpi/checkpoints/pi05_so101_pickup_battery/full_sft_30k_bs18_fsdp2/20000

./examples/so101/run_policy_server.sh
```

`CUDA_VISIBLE_DEVICES=0` 选择物理0号卡；若0号卡忙，可改为一张空闲卡，例如 `1`。当最终 checkpoint
`29999` 存在后，只需把 `CHECKPOINT_DIR` 改为该目录。服务端模型加载和**第一次** PI0.5 请求的 JAX/XLA
编译可能耗时数分钟；保持该终端运行，不要在编译时反复重启服务。

#### 终端 B：本地 5060，建立 SSH 隧道

不需要 Python 或 conda 环境，只需要 `ssh`。保持终端运行：

```bash
cd "$HOME/桌面/openpi/examples/so101"

A100_SSH_HOST=zjh@114.214.255.57 \
./open_ssh_tunnel.sh
```

它将本机 `127.0.0.1:8000` 转发到 A100 的 `127.0.0.1:8000`。出现 `channel open failed: Connection refused`
表示 A100 服务还没有开始监听8000，等待服务端加载完毕即可；SSH 隧道本身无需重开。

#### 终端 C：本地 5060，连接 SO-101 并执行策略

此终端必须使用本地的 LeRobot 硬件环境：

```bash
conda activate lerobot312
cd "$HOME/桌面/openpi/examples/so101"

export SO101_PORT=/dev/ttyACM0
export SO101_ID=zjh_follower_arm
export FRONT_CAMERA_INDEX=0
export WRIST_CAMERA_INDEX=2
export TASK="Grab blue battery to the bin"

export CONTROL_FPS=30
export ACTIONS_PER_CHUNK=50
export SO101_MAX_RELATIVE_TARGET=5
export SO101_LOG_DIR="$HOME/openpi/so101_logs"

./run_so101_client.sh
```

启动顺序固定为：A100 服务端 -> 本地 SSH 隧道 -> 本地 SO-101 client。环境变量只对当前终端有效；新开
终端后需要重新设置。

### 部署参数说明

| 参数 | 默认值 | 含义与建议 |
| --- | --- | --- |
| `CHECKPOINT_DIR` | 必填 | A100 上的 OpenPI checkpoint 路径，不是本机路径。切换模型时只改这里后重启 A100 服务。 |
| `CUDA_VISIBLE_DEVICES` | 必填 | A100 服务使用的单张物理 GPU 编号。 |
| `SERVER_PORT` | `8000` | A100 policy server 端口；本地和 A100 必须与隧道端口一致。 |
| `A100_SSH_HOST` | 必填 | A100 的 `用户名@IP`。 |
| `SO101_PORT` | `/dev/ttyACM0` | 本地 follower arm 的串口。 |
| `SO101_ID` | `zjh_follower_arm` | 标定 JSON 文件名，必须与采集数据时的机械臂一致。 |
| `FRONT_CAMERA_INDEX` / `WRIST_CAMERA_INDEX` | `0` / `2` | 本地 OpenCV 相机编号。前/腕顺序必须与训练数据一致。 |
| `CAMERA_WIDTH/HEIGHT/FPS` | `1280/720/30` | 相机实际采集配置；设备不支持时必须按设备能力调整。 |
| `TASK` | `Grab blue battery to the bin` | 发给 PI0.5 的任务文本，应与训练数据任务语义一致。 |
| `CONTROL_FPS` | `30` | 本地 `send_action()` 的目标频率。当前实测可达约30Hz。 |
| `ACTIONS_PER_CHUNK` | `50` | 本地从 A100 返回的50步动作中实际执行多少步。50步在30Hz下约1.67秒；20步约0.67秒，视觉更新更频繁但会更频繁等待远程推理。 |
| `SO101_MAX_RELATIVE_TARGET` | `5` | 单次下发前允许的目标角度相对当前位置的最大变化。`3` 更保守，`8`/`10` 逐步放宽；`none` 关闭保护，不建议。 |
| `SO101_LOG_DIR` | `~/openpi/so101_logs` | 本地每次部署生成 `run_时间/latency.jsonl` 的目录。 |

当前实测 A100 远程链路大约为：模型推理 `63-101ms`、往返 `129-181ms`、网络和序列化估算
`27-28ms`、本地动作下发约`2ms`、实际控制频率约`29.9Hz`。因此网络不是主要瓶颈；`ACTIONS_PER_CHUNK=50`
时视觉策略更新周期约为 `1.8s`，即约`0.54Hz`。想提高视觉响应性时，先试 `ACTIONS_PER_CHUNK=20`，不要
直接降到1，否则会在每一步等待远程推理，机械臂反而会停顿变慢。

### 推理时延日志

每个完整执行的 chunk 都会写入本地：

```text
~/openpi/so101_logs/run_YYYYMMDD_HHMMSS/
  run_info.json
  latency.jsonl
```

`latency.jsonl` 每行是一段动作的记录，常用字段：

| 字段 | 含义 |
| --- | --- |
| `round_trip_ms` | 本机发 observation 到收到 A100 action 的总往返时延。 |
| `server_policy_ms` | A100 内 PI0.5 模型采样时间。 |
| `transport_and_serialization_ms` | `round_trip_ms - server_total_ms`，SSH 上下行、两端序列化和客户端等待的估算总和。 |
| `sensor_to_first_command_ms` | 取观测、预处理、远程往返后首次下发动作的时延。 |
| `mean_send_action_ms` | 单个 SO-101 关节目标下发的平均耗时。 |
| `actual_control_hz` | 该动作段实际控制频率。 |
| `chunk_cycle_ms` | 取图、推理、执行整个动作段的总周期。 |

查看最近一次：

```bash
LATEST_RUN="$(ls -dt "$HOME"/openpi/so101_logs/run_* | head -1)"
tail -n 5 "$LATEST_RUN/latency.jsonl"
```

若首轮 JAX 编译尚未结束，日志可能只有 `run_info.json`，因为尚未收到并执行完整动作段。正常 `Ctrl-C`
停止时，client 还会在终端打印 round-trip、policy 和控制频率的 P50/P95 汇总。

### 真机安全与常见问题

首次测试必须准备物理急停，并保持 `SO101_MAX_RELATIVE_TARGET=5`。确认 task、相机编号、标定文件、
关节顺序和角度单位全部正确前，不要设置为 `none`。

| 现象 | 优先检查 |
| --- | --- |
| SSH terminal 报 `Connection refused` | A100 服务是否已完成模型加载并监听8000；不要把本地路径填入 `CHECKPOINT_DIR`。 |
| client 报 WebSocket keepalive timeout | 同步最新 `examples/so101` 和 `packages/openpi-client` 到本机，重新执行 editable install；首次 JAX 编译期间保持 client 等待。 |
| 机械臂动作很慢但 `actual_control_hz` 接近30 | 先检查 `ACTIONS_PER_CHUNK` 导致的同步等待，以及模型/数据质量；不是串口频率瓶颈。 |
| 大量 safety clamp warning | 先降低 `SO101_MAX_RELATIVE_TARGET`，检查任务文本、相机、标定、关节顺序和训练数据。 |
