# SO-101 + OpenPI JAX PI0.5 完整说明

本文说明当前项目从数据集到训练、checkpoint 和远程真机部署的完整链路。当前实现使用
Physical Intelligence 官方 OpenPI JAX PI0.5 (`pi05_base`) 做**全参数微调**；没有使用
LoRA，也没有加载此前 LeRobot/PyTorch 训练得到的20k checkpoint。

训练数据、OpenPI 环境和模型缓存都在 A100 服务器；SO-101、两个相机和标定文件留在本地主机。

## 1. 当前结论与目录

| 项目 | 当前值 |
| --- | --- |
| 原始/合并数据集 | `/data5/zjh/data/so101_pickup_battery_merged` |
| 数据规模 | 100 episodes、39,000 frames、30 FPS |
| 训练模型 | 官方 `gs://openpi-assets/checkpoints/pi05_base/params` |
| 微调方式 | 全参数微调，不冻结、不使用 LoRA |
| 训练卡 | `CUDA_VISIBLE_DEVICES=0,1` |
| 全局 batch | 18，等于每卡9 |
| FSDP | `fsdp_devices=2` |
| 训练步数 | 30,000 |
| checkpoint 根目录 | `/data5/zjh/openpi/checkpoints/pi05_so101_pickup_battery` |
| 最终 checkpoint | `full_sft_30k_bs18_fsdp2/29999` |
| normalization stats | `/data5/zjh/openpi/assets/pi05_so101_pickup_battery/so101_pickup_battery_merged/norm_stats.json` |

OpenPI 虚拟环境是 `/home/zjh/openpi/.venv`，使用 Python 3.11。模型的官方 base 权重约12GB，
已经下载并缓存到：

```text
/data5/zjh/.cache/openpi/openpi-assets/checkpoints/pi05_base/params
```

以后训练不会重新下载这份权重。

## 2. 从数据到真机的链路

```text
LeRobot v3.0 数据集
  data/*.parquet: state、action、时间和任务索引
  videos/*.mp4: front、wrist 两路视频
        |
        v
LeRobotSO101DataConfig / SO101Inputs
  字段映射、动作相对化、quantile normalization、图像缩放、32维 padding
        |
        v
OpenPI JAX PI0.5 全参数训练
  官方 pi05_base -> 5000/10000/.../29999 checkpoint
        |
        v
OpenPI 官方 WebSocket policy server (A100)
  加载 checkpoint 内的权重和 norm stats
        |
        v
SSH tunnel
        |
        v
本地 SO101 client
  读双相机和6维关节状态 -> 发送 observation
  收到6维绝对关节目标 -> 30Hz 同步执行50步 action chunk
```

其中模型推理、归一化和动作还原在 A100 的 OpenPI policy server 中完成；串口控制、相机读取、
本地标定和物理安全限幅在连接 SO-101 的本地主机完成。

## 3. 你的数据集到底是什么格式

数据集的版本由 `/data5/zjh/data/so101_pickup_battery_merged/meta/info.json` 指定：

```json
"codebase_version": "v3.0"
```

这表示它是 **LeRobot 数据集格式 v3.0**。这里有两个容易混淆的版本：

| 名称 | 当前值 | 含义 |
| --- | --- | --- |
| LeRobot 数据格式 | v3.0 | 磁盘上的 `data/`、`videos/`、`meta/` 组织方式 |
| LeRobot Python 包 | v0.4.4 | A100 上读取数据的代码版本 |

OpenPI 没有一种独立的“PI 专属 LeRobot 数据文件格式”。它通过数据配置读取 LeRobot 数据，
再转换成 PI0.5 模型统一的输入结构。我们使用独立的
`/home/zjh/lerobot_openpi_v044` worktree，是因为它能读取 v3.0 数据，同时与 OpenPI 固定的
JAX/Transformers 依赖兼容；不会改动你之前使用的 `/home/zjh/lerobot`。

当前 `meta/info.json` 中的重要 feature 是：

| LeRobot key | 类型/形状 | 含义 |
| --- | --- | --- |
| `observation.images.front` | video, `720 x 1280 x 3` | 前视相机 |
| `observation.images.wrist` | video, `720 x 1280 x 3` | 腕部相机 |
| `observation.state` | float32, 6 | 当前六个关节位置 |
| `action` | float32, 6 | 六个关节的目标位置 |
| `task_index` | int64 | 到 `meta/tasks.parquet` 查询任务文本 |

六维的顺序在 state、action 和真机 client 中都必须一致：

```text
0 shoulder_pan.pos
1 shoulder_lift.pos
2 elbow_flex.pos
3 wrist_flex.pos
4 wrist_roll.pos
5 gripper.pos
```

采集数据使用的是角度单位，因此 state 和 action 都是**角度**。本地 client 也显式使用
`use_degrees=True`。不要在某一端单独改成弧度，否则训练与部署的单位会不一致。

## 4. 为什么要“转换”数据

这里实际有四类不同的转换；没有把数据集从 LeRobot v3.0 重编码成别的文件格式。

### 4.1 文件格式没有转换

训练继续直接读取原数据集的 Parquet 和视频：

```text
data/chunk-*/file-*.parquet
videos/observation.images.front/chunk-*/file-*.mp4
videos/observation.images.wrist/chunk-*/file-*.mp4
meta/info.json
meta/tasks.parquet
meta/episodes/
```

`pyav` 是视频**解码后端**，不是数据格式转换。它用于避开此前 `torchcodec`/FFmpeg 动态库
版本不匹配的问题；训练仍然使用原 AV1 视频文件。

### 4.2 字段名称映射

OpenPI 模型希望的内部字段名不是 LeRobot 的原始字段名。训练时会做如下映射：

| LeRobot 原始字段 | OpenPI 中间字段 | PI0.5 最终相机名 |
| --- | --- | --- |
| `observation.images.front` | `observation/image` | `base_0_rgb` |
| `observation.images.wrist` | `observation/wrist_image` | `left_wrist_0_rgb` |
| `observation.state` | `observation/state` | `state` |
| `action` | `actions` | `actions` |
| LeRobot task 文本 | `prompt` | `prompt` |

PI0.5 的通用模型接口保留三个图像槽位。SO-101 只有两路相机，因此第三路
`right_wrist_0_rgb` 使用零图像并把 mask 设为 `False`，模型不会把它当成真实观测。

### 4.3 绝对动作与相对动作

你的 LeRobot `action` 是**绝对关节目标角度**。例如 action 的第一维是“shoulder_pan 要到达的
绝对角度”。这是 SO-101 串口控制所需要的格式。

为了符合 OpenPI 单臂策略的常用处理，训练前会做：

```text
action[0:5] = absolute_joint_target[0:5] - current_state[0:5]
action[5]   = absolute_gripper_target[5]
```

也就是说，前5个机械臂关节训练为相对变化量，夹爪仍是绝对目标。模型输出时执行严格反操作：

```text
absolute_joint_target[0:5] = predicted_delta[0:5] + current_state[0:5]
absolute_gripper_target[5] = predicted_value[5]
```

所以：**部署时本地程序只发送当前原始六维角度，不需要自己手工计算 delta，也不需要自己做
归一化。** 服务端拿到当前 state 后会自动把预测 delta 还原成可发送给 SO-101 的绝对角度。

### 4.4 归一化、图像缩放和 padding

PI0.5 的模型固定使用 `action_dim=32`，而 SO-101 只有6维。因此训练输入会：

1. 对六维 state 和 action 采用 PI0.5 使用的 quantile normalization。
2. 把 state 从6维补零到32维，把 action 从6维补零到32维。
3. 把两路图像缩放并 padding 到 `224 x 224`。
4. 将任务文本和归一化后的 state 编码为 PI0.5 prompt token。

模型输出的32维 action 会先反归一化、还原前5维相对动作，再由 `SO101Outputs` 截成前6维。
因此 SO-101 client 最终收到的始终是 `(50, 6)` 的绝对关节角轨迹。

`norm_stats.json` 包含 state/action 的 `mean`、`std`、`q01`、`q99`。统计由全39,000帧计算，
且使用与训练相同的50步 action horizon 和相对动作定义。图像不做数值归一化统计，视觉输入保持
identity 语义。

## 5. 关节信息是否需要转换

需要被模型使用，但你不需要在真机端手工转换。

训练数据的 `observation.state` 已经是六维关节角；训练时它会经过 mapping、normalization 和
padding。推理时 `so101_client.py` 从本地机器人读取相同顺序、相同单位的六维当前位置：

```python
state = [
    shoulder_pan.pos,
    shoulder_lift.pos,
    elbow_flex.pos,
    wrist_flex.pos,
    wrist_roll.pos,
    gripper.pos,
]
```

它把这个原始 state 和两张图一起发送给 A100。A100 policy server 根据 checkpoint 中保存的
normalization stats 完成所有内部变换。

当前不是末端位姿训练。模型学习的是**关节空间 state -> 关节空间 action**。若要改为末端位姿：

1. 数据集 action 必须保存末端的 XYZ/旋转表示，而不仅是关节角。
2. 必须定义可靠的逆运动学（IK），把策略输出的末端轨迹转换为六个关节目标。
3. 需要同步改 state/action 映射、normalization、真机 client 和安全边界。

因此当前抓取任务继续用关节角是正确且实现复杂度最低的方案。

## 6. 涉及的文件与职责

### 6.1 你平时直接使用的文件

| 文件 | 作用 | 平时是否需要改 |
| --- | --- | --- |
| `examples/so101/README.md` | 简版训练/部署命令 | 查看即可 |
| `examples/so101/run_train.sh` | 双卡训练入口、GPU/缓存/W&B 环境变量 | 通常不改 |
| `examples/so101/compute_norm_stats.sh` | 生成 normalization stats | 数据集改变后运行一次 |
| `examples/so101/run_policy_server.sh` | A100 上启动官方 WebSocket 服务 | 部署时设置 checkpoint |
| `examples/so101/open_ssh_tunnel.sh` | 本机到 A100 的 SSH 隧道 | 设置 A100 地址 |
| `examples/so101/run_so101_client.sh` | 本地 SO-101 启动入口 | 部署时设置串口、相机等 |
| `examples/so101/so101_client.py` | 本地相机、串口、WebSocket、同步控制循环 | 硬件逻辑已实现，通常不改 |

### 6.2 为 SO-101 新增或修改的核心训练文件

| 文件 | 修改内容 |
| --- | --- |
| `src/openpi/policies/so101_policy.py` | `SO101Inputs`、`SO101Outputs`；双相机、6维 state/action 映射、输出截断 |
| `src/openpi/policies/so101_policy_test.py` | 验证图像映射、6维输出和 delta 到 absolute 的还原顺序 |
| `src/openpi/training/config.py` | `LeRobotSO101DataConfig` 和 `pi05_so101_pickup_battery` 训练配置 |
| `src/openpi/training/data_loader.py` | 支持本地 root、`pyav`、LeRobot v3 task 表；统计时跳过不需要的视频解码 |
| `scripts/compute_norm_stats.py` | 将 SO-101 统计加载路径传入数据加载器 |
| `scripts/train.py` | 使用 `JAX_COMPILATION_CACHE_DIR`，把 XLA 编译缓存放到 `/data5` |
| `pyproject.toml` 和 `uv.lock` | Python 3.11/3.12 约束、LeRobot v0.4.4、NumPy 兼容依赖和 headless OpenCV |

OpenPI 官方的 `scripts/serve_policy.py` 没有改动。它仍是官方通用 WebSocket policy server；
我们只是给它增加了一个名为 `pi05_so101_pickup_battery` 的数据/机器人配置。

本地 `so101_client.py` 是需要补齐的硬件实现。OpenPI 官方 `openpi-client` 只提供 WebSocket
通讯和图像工具，不知道你的 SO-101 串口、相机和校准文件在哪里；该文件已经使用 LeRobot 的
`SO101Follower` 接上了这些硬件部分。

## 7. 当前训练参数在哪里改

主配置位于 `src/openpi/training/config.py` 的 `pi05_so101_pickup_battery`。当前值：

| 参数 | 当前值 | 作用 |
| --- | --- | --- |
| `pi05=True` | true | 使用 PI0.5 而不是 PI0 |
| `action_dim` | 32 | PI0.5 固定内部动作维度；SO-101 有效维度仍是6 |
| `action_horizon` | 50 | 每次预测50个未来动作 |
| `weight_loader` | `pi05_base/params` | 从官方 base 开始训练 |
| `batch_size` | 18 | 全局 batch；双卡时每卡9 |
| `fsdp_devices` | 2 | 两卡分片模型/优化器状态 |
| `num_train_steps` | 30,000 | 总更新步数 |
| `num_workers` | 4 | 视频/Parquet 数据加载 worker 数 |
| `warmup_steps` | 1,000 | 学习率 warmup |
| `peak_lr` | `2.5e-5` | 峰值学习率 |
| `decay_steps` | 30,000 | cosine decay 结束步数 |
| `decay_lr` | `2.5e-6` | 最低学习率 |
| `save_interval` | 5,000 | checkpoint 间隔 |
| `keep_period` | 5,000 | 长期保留的 checkpoint 间隔 |
| `freeze_filter` | 默认 Nothing | 不冻结任何参数，即全参微调 |
| `ema_decay` | 0.99 | 保存 EMA 参数，推理时可获得更平稳权重 |

`run_train.sh` 中的运行参数：

| 参数/环境变量 | 当前值 | 含义 |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0,1` | 使用两张同型号 PCIe A100 |
| `MIN_GPU_FREE_MIB` | 60000 | 每卡启动前的最小空闲显存检查 |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | 0.80 | JAX 最多占每卡约80%显存 |
| `HF_HOME` | `/data5/zjh/hf_cache` | Hugging Face 缓存目录 |
| `UV_CACHE_DIR` | `/data5/zjh/.cache/uv` | Python 包缓存目录 |
| `WANDB_*` | `/data5/zjh/wandb/openpi/...` | W&B 日志/缓存目录 |
| `JAX_COMPILATION_CACHE_DIR` | `/data5/zjh/openpi/jax_cache` | XLA 编译缓存 |
| `TMPDIR` | `/data5/zjh/openpi/tmp` | 临时文件目录 |

已经验证：双卡0/1、真实 batch18 和完整反向传播可以跑通；首次编译加一轮执行约118秒，
测试 loss 为 `0.04554`，没有 OOM。不要把 PCIe A100 的0/1与 SXM A100 的3/4/5混在同一个
JAX FSDP 训练里，JAX 会因为硬件类型不同而拒绝执行。

## 8. 从零开始训练的实际命令

训练前不需要再次运行 stats，也不需要再次下载 base。首次 W&B 登录一次：

```bash
cd /home/zjh/openpi
uv run --no-dev wandb login
```

启动训练：

```bash
tmux new -s openpi_so101

cd /home/zjh/openpi
export CUDA_VISIBLE_DEVICES=0,1
export EXP_NAME=full_sft_30k_bs18_fsdp2

./examples/so101/run_train.sh
```

脚本先检查两张卡型号一致、每卡空闲显存不少于60GB；通过后才启动训练。训练日志每100步输出一次，
checkpoint 在5000、10000、15000、20000、25000和最终29999步写入：

```text
/data5/zjh/openpi/checkpoints/pi05_so101_pickup_battery/full_sft_30k_bs18_fsdp2/
```

离开 tmux：`Ctrl-b d`。重新查看：

```bash
tmux attach -t openpi_so101
```

如果进程中断且已有 checkpoint，恢复命令是：

```bash
cd /home/zjh/openpi

CUDA_VISIBLE_DEVICES=0,1 \
RESUME=true \
EXP_NAME=full_sft_30k_bs18_fsdp2 \
./examples/so101/run_train.sh
```

若只在第一个5000步 checkpoint 之前中断，没有可恢复的训练状态；此时重新使用原启动命令即可。

## 9. 数据更新后如何继续训练

补录数据必须保持以下约束：

1. LeRobot v3.0 格式。
2. 相同的 feature 名称、关节顺序、角度单位、30 FPS、front/wrist 相机定义。
3. task 文本与部署的 `TASK` 一致，或在部署时使用数据集中实际的任务文本。

新数据合并到 `/data5/zjh/data/so101_pickup_battery_merged` 后，必须重新生成 stats：

```bash
cd /home/zjh/openpi
./examples/so101/compute_norm_stats.sh
```

这是因为 state/action 的范围和 quantile 已变化。继续训练已有 checkpoint 时，建议把新统计和新训练
视为一个新实验名；否则已有 checkpoint 里的旧 stats 与新数据范围不一致。重新从官方 base 开始训练
最干净；若确实继续旧模型，需要明确评估统计变化带来的影响。

## 10. 部署前要准备什么

### A100 服务器已经具备

1. OpenPI 环境和官方 base 缓存。
2. `examples/so101/run_policy_server.sh`。
3. 最终训练 checkpoint。
4. 任选一张空闲 GPU 用于推理，通常使用0号卡。

### 本地主机需要提供

1. SO-101 通过 USB 串口连接，例如 `/dev/ttyACM0`。
2. 采集数据时同一台 SO-101 的标定文件，默认：
   `~/.cache/huggingface/lerobot/calibration/robots/so_follower/zjh_follower_arm.json`。
3. front 和 wrist 两个相机；当前编号为0和2。
4. 本机 LeRobot 环境，且能导入 `SO101Follower`。
5. `openpi-client`：

```bash
cd ~/openpi/packages/openpi-client
pip install -e .
```

如果本机还没有本目录，可以从服务器复制 `examples/so101` 和 `packages/openpi-client`，但**不需要**
复制训练数据或12GB base 权重；模型始终留在 A100。

## 11. 完整部署顺序

### 11.1 A100: 启动官方 OpenPI WebSocket server

训练结束后在 A100 上执行：

```bash
cd /home/zjh/openpi

export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR=/data5/zjh/openpi/checkpoints/pi05_so101_pickup_battery/full_sft_30k_bs18_fsdp2/29999

./examples/so101/run_policy_server.sh
```

这个命令使用官方 `scripts/serve_policy.py policy:checkpoint`。它从 checkpoint 本身读取训练时保存的
模型参数和 `assets/so101_pickup_battery_merged` 下的 norm stats，确保部署不会误用别的统计文件。

### 11.2 本地主机: 建立 SSH 隧道

新开终端并保持运行：

```bash
cd ~/openpi/examples/so101

A100_SSH_HOST=zjh@114.214.255.57 \
./open_ssh_tunnel.sh
```

它把本机 `127.0.0.1:8000` 转发到 A100 的 server 端口8000。服务端无需直接暴露公网端口。

### 11.3 本地主机: 启动 SO-101 client

再开一个终端：

```bash
conda activate lerobot312
cd ~/openpi/examples/so101

SO101_PORT=/dev/ttyACM0 \
SO101_ID=zjh_follower_arm \
FRONT_CAMERA_INDEX=0 \
WRIST_CAMERA_INDEX=2 \
SO101_MAX_RELATIVE_TARGET=5 \
TASK="Grab blue battery to the bin" \
./run_so101_client.sh
```

首次测试保留 `SO101_MAX_RELATIVE_TARGET=5`，并准备物理急停。它限制单个控制周期允许的最大目标变化，
可避免模型、相机或标定不一致时突然大幅运动。确认各项一致前不要设置为 `none`。

## 12. 推理时实际发生什么

`so101_client.py` 是同步分块执行，而不是此前 LeRobot async queue 模式：

1. 读取 front、wrist 图像和当前六维关节状态。
2. 本地将图像 resize/pad 到224，并通过 SSH 隧道发送 observation 和 task 文本。
3. A100 运行 PI0.5，一次返回最多50个六维绝对关节目标。
4. 本地按30Hz依次发送这50个目标，大约执行 `50 / 30 = 1.67` 秒。
5. action chunk 执行完成后，再读取新观测并请求下一块动作。

因此网络推理延迟主要表现为两个 chunk 之间的停顿，不会在一个 chunk 的中途覆盖旧轨迹。控制频率由：

```bash
CONTROL_FPS=30
ACTIONS_PER_CHUNK=50
```

决定。`ACTIONS_PER_CHUNK` 可设为1到50；较小会更频繁请求 A100、响应更及时，但更容易受到网络和
推理时间影响。当前先保持50，保证同步执行稳定。

### 12.1 推理时延如何计算和查看

本地 client 会为每一个 action chunk 记录时延。日志默认写在**本地主机**，而不是 A100：

```text
~/openpi/so101_logs/run_YYYYMMDD_HHMMSS/
  run_info.json
  latency.jsonl
```

`latency.jsonl` 每行是一个 JSON 对象，即一次“取观测 -> A100 推理 -> 执行一段动作”的完整记录。
启动时终端会打印实际日志路径，正常按 `Ctrl-C` 停止时还会打印全部 chunk 的 mean、P50、P95。
可以显式指定日志根目录：

```bash
export SO101_LOG_DIR="$HOME/openpi/so101_logs"
./run_so101_client.sh
```

每个 chunk 的时延分解如下：

```text
observation_ms                  本机读取两路相机和当前关节 state
preprocess_ms                   本机图像 resize/pad 和请求打包
round_trip_ms                   本机发出 request 到收到 action 的总往返时间
  server_total_ms               A100 收到 request 后到发回 response 的处理时间
    server_policy_ms            PI0.5 policy 内部推理时间
  transport_and_serialization_ms
                               round_trip_ms - server_total_ms；SSH 上下行、两端序列化和客户端等待的估算合计
mean_send_action_ms             单步 SO-101 `send_action` 平均耗时
execution_ms                    本 chunk 实际执行动作的总时间
actual_control_hz               actions_executed / execution_ms
chunk_cycle_ms                  本次从读取观测到执行完 action chunk 的总时间
```

其中最直接的闭环首动作时延是：

```text
sensor_to_first_command_ms = observation_ms + preprocess_ms + round_trip_ms
```

这里的 `round_trip_ms` 是最可信的远程指标，因为它只依赖本机时钟，不需要校准 A100 和本机的系统时间。
无法从两台未校时的机器直接得到“上行单程延迟”或“下行单程延迟”。

同步分块模式下，真正的视觉闭环更新周期约为：

```text
chunk_cycle_ms = observation_ms + preprocess_ms + round_trip_ms + execution_ms
```

例如 `ACTIONS_PER_CHUNK=50`、`CONTROL_FPS=30` 时，单是 `execution_ms` 理论上就约为
`50 / 30 = 1.67s`。这不是网络慢，而是本机在同步执行这50个已有动作；下一张图像只会在这段动作
执行完后才送到 A100。若要让视觉更新更频繁，可以把 `ACTIONS_PER_CHUNK` 降到10或20，但会更频繁地
暴露网络和推理抖动，先用50做基准测量后再调整。

查看最近一次测试的最后几条记录：

```bash
LATEST_RUN="$(ls -dt "$HOME"/openpi/so101_logs/run_* | head -1)"
tail -n 5 "$LATEST_RUN/latency.jsonl"
```

快速只看最关键字段：

```bash
LATEST_RUN="$(ls -dt "$HOME"/openpi/so101_logs/run_* | head -1)"
python - <<'PY' "$LATEST_RUN/latency.jsonl"
import json
import sys

for line in open(sys.argv[1], encoding="utf-8"):
    x = json.loads(line)
    print(
        f"chunk={x['chunk']} rtt={x['round_trip_ms']:.1f}ms "
        f"policy={x['server_policy_ms']}ms "
        f"transport={x['transport_and_serialization_ms']}ms "
        f"control={x['actual_control_hz']:.1f}Hz "
        f"cycle={x['chunk_cycle_ms']:.1f}ms"
    )
PY
```

判断瓶颈时优先看：

| 现象 | 结论和下一步 |
| --- | --- |
| `server_policy_ms` 高 | A100 推理/首次 JAX 编译是瓶颈；模型服务保持运行，第二次以后再测 |
| `round_trip_ms` 高但 `server_total_ms` 低 | 网络、SSH 隧道、图像序列化或本机链路是主要开销 |
| `actual_control_hz` 明显低于 `CONTROL_FPS` | 本地串口读写/安全限幅导致的 SO-101 下发瓶颈，查看 `mean_send_action_ms` |
| `actual_control_hz` 正常但 `chunk_cycle_ms` 很长 | 同步 chunk 本身过长；降低 `ACTIONS_PER_CHUNK` 以增加视觉更新频率 |

这些计时不会修改动作、平滑策略或安全限幅，只在本地记录性能数据。

## 13. 常用部署环境变量

| 变量 | 默认值 | 何时改 |
| --- | --- | --- |
| `CHECKPOINT_DIR` | 必填 | 服务端切换训练 checkpoint |
| `CUDA_VISIBLE_DEVICES` | 服务端自行设置 | 选择推理 GPU |
| `A100_SSH_HOST` | 必填 | A100 SSH 地址变化时 |
| `SO101_PORT` | `/dev/ttyACM0` | 串口变化时 |
| `SO101_ID` | `zjh_follower_arm` | 必须匹配校准 JSON 文件名 |
| `FRONT_CAMERA_INDEX` | 0 | 前视相机编号变化时 |
| `WRIST_CAMERA_INDEX` | 2 | 腕部相机编号变化时 |
| `CAMERA_WIDTH/HEIGHT/FPS` | 1280/720/30 | 设备实际能力变化时 |
| `CONTROL_FPS` | 30 | 想降低或提高本地发送频率时 |
| `ACTIONS_PER_CHUNK` | 50 | 想调整每次预测/执行的时长时 |
| `SO101_MAX_RELATIVE_TARGET` | 5 | 真机安全限幅；首次不要取消 |
| `SO101_LOG_DIR` | `~/openpi/so101_logs` | 本地部署时延 JSONL 的根目录 |
| `TASK` | `Grab blue battery to the bin` | 必须与训练任务语义一致 |

## 14. 验证结果与常见问题

已经完成的验证：

1. 读取真实 LeRobot v3.0 数据的一条样本：两路图像、6维 state、`(50, 6)` action 和任务文本正确。
2. 完整训练 batch：图像为 `(18, 3, 224, 224)`，state 为 `(18, 32)`，action 为 `(18, 50, 32)`。
3. 全数据 normalization stats 已生成，数值全部有限。
4. 官方 base 的12GB checkpoint 已成功恢复。
5. 双卡0/1真实 batch18 的前向和反向已完成，loss/梯度均有限。
6. SO-101 输入、输出和 delta-to-absolute 变换单元测试已通过。

若遇到问题，优先检查：

| 现象 | 优先检查 |
| --- | --- |
| 训练启动即 OOM | `nvidia-smi`；0/1每卡至少60GB空闲；不要混用 PCIe 和 SXM A100 |
| `torchcodec`/FFmpeg 报错 | 配置必须保持 `video_backend=pyav` |
| 训练与部署动作方向/幅度异常 | 关节顺序、角度单位、SO101 ID 对应的校准文件、相机编号 |
| 服务端找不到模型 | `CHECKPOINT_DIR` 必须是 A100 上的 checkpoint 目录，不是本机路径 |
| 真机没有动作 | 先检查 SSH tunnel、`127.0.0.1:8000`、串口和相机是否已连接 |
| 真机动作过大 | 保持 `SO101_MAX_RELATIVE_TARGET=5`，检查 task/相机/标定，不要先取消限幅 |

## 15. 你以后通常只需要做什么

1. 新数据保持 LeRobot v3.0 和相同关节语义，合并后重新跑 stats。
2. 确认0/1显存后运行 `run_train.sh`。
3. 根据 W&B 和真机效果选择5000/10000/.../29999 checkpoint。
4. A100 运行 `run_policy_server.sh`，本机开 SSH tunnel 后运行 `run_so101_client.sh`。
5. 首次真机始终保持安全限幅和物理急停。

除非你改变相机数量、关节顺序、动作单位或改为末端位姿，否则不需要再改 Python 核心代码。

## 16. 当前 PI0.5 到底是什么模型

当前训练的是 OpenPI 官方 JAX 实现中的 `Pi0Config(pi05=True)`，从官方
`pi05_base/params` 加载预训练参数。它不是“图像输入后只接一个小的 action head”，而是一个
视觉、语言、状态和动作轨迹共同工作的 flow-matching VLA。当前参数默认都是可训练的。

### 16.1 模型组成

| 部分 | 当前工程实现 | 作用 |
| --- | --- | --- |
| 视觉编码器 | SigLIP `So400m/14` | 将每张 `224 x 224` 图像编码为视觉 token |
| 语言/视觉骨干 | Gemma 2B，18 层、宽度2048 | 融合图像 token、任务文本和离散化 state token |
| 动作 expert | Gemma 300M，18 层、宽度1024 | 在视觉/语言条件下生成连续动作的流场 |
| 动作输入投影 | `32 -> 1024` 线性层 | 把带噪的动作轨迹变为 action expert token |
| 时间条件 | sin/cos timestep embedding + 两层 MLP + adaRMSNorm | 告诉 action expert 当前 flow/diffusion 去噪时刻 |
| 动作输出投影 | `1024 -> 32` 线性层 | 预测每个动作维度的速度场 |

代码位置是 `src/openpi/models/pi0.py`、`src/openpi/models/pi0_config.py` 和
`src/openpi/models/gemma.py`。OpenPI 将 Gemma 2B 和 300M 写成两个 expert：两者都有18层，
视觉/语言 prefix 由2B expert 编码，50个动作 token 由300M action expert 生成，并通过 attention
读取 prefix 的上下文。PI0.5 与 PI0 的两个关键差异是：

1. state 不作为单独连续 token 注入 action suffix，而是在归一化后被量化为256个 bin，拼进文本：
   `Task: <任务>, State: <6个离散值>; Action:`。
2. flow timestep 通过 adaRMSNorm 调制 action expert；PI0 使用的是另一套 action/time MLP 融合方式。

这套配置的内部动作形状固定为：

```text
state:   [batch, 32]
actions: [batch, 50, 32]
```

SO-101 实际只有6维。我们不是改变 PI0.5 的32维接口，而是把6维 state/action 补零到32维；模型输出
经过反归一化和绝对动作还原后，`SO101Outputs` 只截取前6维发给机器人。

### 16.2 SO-101 输入输出如何接到 PI0.5

```text
front RGB --------------> base_0_rgb ---------+
wrist RGB --------------> left_wrist_0_rgb ---+--> SigLIP --> Gemma 2B prefix
第三相机槽位 -----------> 全零图，mask=False -+

6维关节 state --> quantile normalization --> 256-bin state token --> 任务文本 prefix

50 x 6 absolute action
  --> 前5维减当前 state，夹爪保留 absolute
  --> quantile normalization + 补零到 50 x 32
  --> 加噪并训练 PI0.5 预测 flow velocity

推理时反向执行：50 x 32 --> 反归一化 --> 前5维加回当前 state --> 截取 50 x 6 absolute action
```

实现对应关系：

| 文件 | 实现内容 |
| --- | --- |
| `src/openpi/policies/so101_policy.py` | front/wrist 映射到 PI0.5 三相机接口；第三路为 mask=False；输出截成6维 |
| `src/openpi/training/config.py` 的 `LeRobotSO101DataConfig` | LeRobot v3 字段重命名、前5维绝对动作转 delta、推理时 delta 还原 |
| `src/openpi/transforms.py` | quantile normalize/unnormalize、padding、delta/absolute transform |
| `src/openpi/models/tokenizer.py` | PI0.5 的任务文本和离散 state token 生成 |
| `examples/so101/so101_client.py` | 读真机的原始6维角度与两路图像；不做手工归一化 |

`SO101Inputs`、`SO101Outputs` 是机器人适配层，不是另一个神经网络。真正训练和推理的模型仍然是官方
JAX PI0.5。

### 16.3 PI0.5 如何学习和采样动作

训练时对真实动作块 `a` 采样高斯噪声 `n` 和时间 `t`：

```text
x_t = t * n + (1 - t) * a
u_t = n - a
loss = mean((model(x_t, observation, t) - u_t)^2)
```

也就是模型学习从带噪动作 `x_t` 指向真实动作 `a` 的连续速度场，不是逐关节分类，也不是直接回归单步
动作。推理时从一段高斯噪声动作开始，先缓存视觉/语言 prefix 的 KV，再默认进行10次 Euler 积分，得到
一个 `50 x 32` 动作块。最终只把前6维绝对关节目标交给 SO-101。

## 17. 一次训练从读取数据到保存 checkpoint 的过程

下面是当前配置的真实执行顺序；“step”表示一次全局 batch 的参数更新。

1. 启动 `scripts/train.py pi05_so101_pickup_battery`，创建 JAX PI0.5 网络并从官方 `pi05_base`
   恢复预训练参数。`freeze_filter` 为默认的 `Nothing`，所以视觉编码器、Gemma 2B、300M action expert、
   state/action 投影层都会训练；没有启用 LoRA。
2. DataLoader 从100条 episode 的39,000帧中随机取时间点。每个样本读取该时刻的 front、wrist、6维 state，
   并取后续50帧 action；30 FPS 下每个目标 action chunk 覆盖约1.67秒。
3. 在 CPU DataLoader worker 内在线完成：视频解码（PyAV）、LeRobot key 重命名、SO-101 双相机映射、
   前5维 action 转 delta、state/action quantile normalization、图像缩放/填充到224、state/action 补零到32、
   state+task 的 PaliGemma token 化。不会将这些中间结果写回数据集。
4. 一个全局 batch 为18，数据按两张GPU分配为每卡9个样本。JAX 用 bfloat16 跑前向，PI0.5 对50步、32维
   action 做 flow-matching MSE。
5. 反向传播得到全部可训练参数的梯度；Optax 先做全局 grad norm=1.0 的裁剪，再用 AdamW 更新。学习率前
   1000 step warmup 到 `2.5e-5`，随后 cosine decay，在30000 step 降到 `2.5e-6`。
6. 同时以 `ema_decay=0.99` 维护参数 EMA。推理 checkpoint 导出的 `params` 是 EMA 权重；用于恢复训练的
   `train_state` 包含当前模型参数、EMA 和 AdamW optimizer state。DataLoader 的随机游标不单独保存，
   恢复后会重新创建并继续随机采样数据集，这通常不影响离线 SFT。
7. 每100 step 把 loss、grad norm、param norm 写入终端和 W&B；保存点会异步写入 Orbax checkpoint。

### 17.1 当前 checkpoint 周期

当前配置的 `save_interval=5000`，所以**每5000 step 保存一次**，目录名为：

```text
5000
10000
15000
20000
25000
29999   # 30,000 step 训练循环的最终保存点
```

训练代码还会无条件保存最后一个 step，因此最终目录是 `29999`，不是 `30000`。这是训练循环从0开始计数
导致的目录名；实际使用时始终选择最后的 `29999`。`keep_period=5000` 会保留这些5000间隔的 checkpoint，
checkpoint 内同时保存推理参数、可恢复训练状态和本次训练使用的 `norm_stats.json`；DataLoader 不保存
随机游标，恢复后重新随机采样。

## 18. 双卡 JAX FSDP 的实际实现

当前启动命令为：

```bash
CUDA_VISIBLE_DEVICES=0,1 ./examples/so101/run_train.sh
```

这不是 PyTorch `torchrun` 或 `accelerate` 的两个独立进程；是**一个 Python/JAX 进程**同时看见两张
GPU，由 JAX 建立 device mesh。当前配置是：

```text
visible devices = 2
fsdp_devices   = 2
mesh           = (batch=1, fsdp=2)
global batch   = 18
local batch    = 18 / 2 = 9 per GPU
```

具体过程：

1. `sharding.make_mesh(2)` 创建名为 `batch, fsdp` 的 mesh；本次 `batch=1`，所以没有额外的 data-parallel
   副本，两个GPU构成一个 FSDP 组。
2. 初始化时 `fsdp_sharding()` 将大于4MiB的矩阵/张量沿可整除的最大轴切分到 `fsdp` 轴；小张量、向量和
   标量会复制到两卡。模型参数、AdamW optimizer state、EMA 参数因此分片保存，降低单卡显存占用。
3. DataLoader 把 batch 沿 `(batch, fsdp)` 轴切分，两个GPU各拿9条样本。activation 也有同样的 sharding
   constraint；JAX 在需要处自动进行 collective 通信，得到等价于全局18条样本的梯度更新。
4. `train_step` 被 `jax.jit` 编译成一个多GPU可执行图。首次会花较长时间 XLA 编译；之后使用
   `/data5/zjh/openpi/jax_cache` 缓存，形状和代码不变时重启会快很多。
5. Orbax 将两卡分片状态保存为一个逻辑 checkpoint；恢复时仍使用 `CUDA_VISIBLE_DEVICES=0,1` 和
   `RESUME=true`，不需要手工合并权重。

`run_train.sh` 会强制恰好两张、型号一致且每张空闲显存至少60GB的GPU。当前0/1均为PCIe A100，适合组成
同一 mesh；不要混入3/4/5号 SXM A100。若未来要改成三卡，需要同时把 `fsdp_devices`、启动脚本的“两卡”
检查和全局 batch 的整除关系一起改，而不是只修改 `CUDA_VISIBLE_DEVICES`。

## 19. 数据转换和归一化：哪些预先做，哪些在线做

当前数据已经是 LeRobot v3.0，因此**不需要**在训练前把视频/Parquet 重新转换成“OpenPI 数据集”。需要在
训练前预先完成的只有：

1. 将补录 episode 合并到目标 LeRobot 数据集目录（仅数据有新增时）。
2. 运行 `./examples/so101/compute_norm_stats.sh`，生成：

   ```text
   /data5/zjh/openpi/assets/pi05_so101_pickup_battery/
   so101_pickup_battery_merged/norm_stats.json
   ```

这个脚本遍历所有 state 和50步 action chunk，先应用与训练相同的字段映射和“前5维 absolute -> delta”规则，
再统计 `mean`、`std`、`q01`、`q99`。PI0.5 使用 quantile normalization，把主要数值范围缩放到约
`[-1, 1]`；统计时不解码视频，因为图像不参与数值统计。

下列内容是**每次取样在线执行**的，不会产生第二份转换后数据：

| 在线处理 | 训练时 | 部署推理时 |
| --- | --- | --- |
| 解码 front/wrist 图像 | PyAV 从数据集视频读取 | 本机 OpenCV 相机读取 |
| key 映射、SO-101 6维适配 | 是 | 是 |
| 前5维 absolute/delta 转换 | absolute -> delta | predicted delta -> absolute |
| 使用已保存的 norm stats 归一化/反归一化 | 是 | 是，且从 checkpoint 内 assets 读取 |
| 图像224 resize/pad、state/action padding、state token 化 | 是 | 是 |

因此：**统计量是训练前预计算的；实际归一化和数据变换是在线应用的。** 数据增加、关节顺序/单位改变、
action horizon 改变，或 absolute/delta 规则改变时，必须重新计算 `norm_stats.json` 后再开始一个新训练实验。
只改训练步数、batch size 或 GPU 数量则不需要重算统计。
