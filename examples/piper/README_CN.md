# PIPER 插 USB 任务：从数据集到 π0.5 训练与部署

本文记录当前项目的完整流程：使用 PIPER 机械臂将 USB 插入排插插座的数据集，在官方 JAX 版 OpenPI π0.5 上加入六维关节 effort（力矩/电流反馈）条件，完成数据转换、归一化、训练、保存 checkpoint，并在另一台电脑上部署推理服务。

任务指令：

~~~text
Insert the USB into the USB port on the power strip.
~~~

## 1. 最终结果

当前训练使用的模型和配置：

~~~text
模型：官方 JAX π0.5
任务：PIPER 插 USB
条件输入：6 维关节 effort
训练方式：双 LoRA + effort MLP
全局 batch size：24
训练步数：30,000
最终 checkpoint：
/data5/zjh/openpi/checkpoints/pi05_piper_usb_effort_lora/usb_effort_lora_bs24_30k/29999
~~~

checkpoint 的 params 已包含基础模型参数、LoRA 参数、effort 投影层参数以及其他训练得到的参数，不需要另外提供 pi05_base。

## 2. 数据集

原始数据集：

~~~text
/data5/usb_0808
~~~

已确认元数据：

~~~text
LeRobot 数据格式：v2.1
robot_type：piper
episodes：100
frames：42,406
帧率：30 FPS
相机：front、side
observation.state：7 维（6 个关节位置 + 1 个夹爪）
action：7 维（6 个关节动作 + 1 个夹爪）
observation.effort：6 维（6 个机械臂关节）
~~~

原始 feature 名称：

~~~text
observation.state
observation.effort
observation.images.front
observation.images.side
action
~~~

effort 顺序：

~~~text
effort_joint_1
effort_joint_2
effort_joint_3
effort_joint_4
effort_joint_5
effort_joint_6
~~~

数据元信息里的 robot_type 可以确认这是 PIPER 数据，但仅凭 LeRobot 元信息不能进一步确认机械臂的具体硬件子型号。具体型号应结合机械臂本体、控制板或采集程序确认。

## 3. LeRobot 版本兼容与数据转换

当前 OpenPI 环境固定使用：

~~~text
Python 3.11.14
JAX 0.5.3
Flax 0.10.2
LeRobot 0.4.4
~~~

这里的 LeRobot 0.4.4 和数据里的 codebase_version v3.0 不是同一个版本号：

| 名称 | 当前值 | 含义 |
| --- | --- | --- |
| LeRobot Python 软件包版本 | 0.4.4 | 负责读取数据、提供 Dataset API 的代码版本 |
| LeRobot 数据格式版本 | v3.0 | meta、data、videos 等磁盘文件的组织格式 |

LeRobot 0.4.4 的代码内部声明的 CODEBASE_VERSION 是 v3.0。因此正确组合就是“LeRobot 0.4.4 软件 + v3.0 数据”，两者不冲突。

原始数据是 LeRobot v2.1，而 OpenPI 使用的 LeRobot v0.4.4 读取 v3.0 数据布局。因此原始数据不能直接作为本项目的训练输入。

训练使用的转换副本：

~~~text
/data5/usb_0808_pi05_v3
~~~

转换副本元信息：

~~~text
codebase_version：v3.0
episodes：100
frames：42,406
fps：30
robot_type：piper
~~~

转换前后的数据内容没有改变，仍然包含 state、action、effort 和两路图像；主要变化是 LeRobot 的磁盘组织和元信息格式。原始 /data5/usb_0808 保持不变，转换副本单独保存，避免破坏原始数据。

如果以后需要重新转换，建议先复制一份 v2.1 数据，再使用 LeRobot 提供的 v2.1 -> v3.0 转换脚本对副本操作，不要直接在唯一的原始数据上转换。

本地转换的参考流程如下。转换脚本对目标目录原地修改，因此先复制原始数据：

~~~bash
cp -a /data5/usb_0808 /data5/usb_0808_pi05_v3

python /path/to/lerobot/src/lerobot/scripts/convert_dataset_v21_to_v30.py \
  --repo-id=usb_0808_pi05_v3 \
  --root=/data5/usb_0808_pi05_v3 \
  --push-to-hub=false
~~~

转换完成后检查：

~~~bash
grep -n 'codebase_version' /data5/usb_0808_pi05_v3/meta/info.json
~~~

结果应为 v3.0。实际训练前还应确认 `observation.effort`、两路视频和 7 维 action 仍然存在。

## 4. 训练环境

当前 OpenPI 源码：

~~~text
/home/zjh/openpi
~~~

当前 openpi/pyproject.toml 从 PyPI 固定安装 LeRobot：

~~~toml
"lerobot==0.4.4"
~~~

uv.lock 同时记录了 PyPI wheel 的来源和哈希。朋友执行 uv sync 时会自动下载 0.4.4，不需要额外传 LeRobot 源码目录，也不会解析到 0.6.x。

检查环境：

~~~bash
cd /home/zjh/openpi
uv run --no-dev python -c "import jax; print(jax.devices())"
~~~

## 5. π0.5 的 state 文本格式

π0.5 会把 state 量化成 0 到 255 的离散 bin，并放进训练好的文本 prompt 中：

~~~text
Task: Insert the USB into the USB port on the power strip., State: 120 37 208 ...;
Action:
~~~

这段文本格式是预训练 π0.5 已经学到的格式。本项目没有把 effort 拼进这段 state 文本中，也没有修改离散 state token，以尽量保留 π0.5 的预训练输入结构。

归一化后的 state 继续按照原有逻辑量化到 256 个 bin；effort 走独立的连续分支。

## 6. effort 条件模型设计

本项目首版结构：

~~~text
原始 effort
    ↓
归一化 effort
    ↓
展平为 [history_length * 6]
    ↓
Linear(6 * history_length, 2 * expert_width)
    ↓
Swish
    ↓
Linear(2 * expert_width, expert_width)
    ↓
得到 1 个连续 effort token
    ↓
插入 noisy action tokens 之前
~~~

当前默认配置：

~~~python
effort_dim = 6
effort_history_length = 1
effort_history = (0,)
~~~

也就是每个时间点只使用当前六维 effort。推理时可以传 [6]，PIPER 输入适配器会自动变成 [1, 6]。

如果以后使用历史 effort，需要同时修改模型和数据配置：

~~~python
effort_history = (-2, -1, 0)
effort_history_length = 3
~~~

此时模型输入形状必须是 [batch, 3, 6]。

π0.5 原有的 adaRMS 时间条件保持不变，新增 effort MLP 随训练随机初始化并学习。这个设计对应 TA-VLA 的当前 effort（EXPERT / DePost）路线；历史拼接则对应 EXPERT_HIS_C / Dec-1 路线。

## 7. 代码改动

模型侧主要修改：

~~~text
src/openpi/models/model.py
    Observation 增加可选 effort 字段。

src/openpi/models/pi0_config.py
    增加 effort_dim 和 effort_history_length 配置。

src/openpi/models/pi0.py
    创建 effort MLP，并在 action expert 的动作 token 前插入 effort token。
~~~

数据和策略侧主要修改：

~~~text
src/openpi/training/config.py
    增加 LeRobotPiperDataConfig、pi05_piper_usb_effort 和
    pi05_piper_usb_effort_lora 配置。

src/openpi/training/data_loader.py
    根据 effort_history 从 observation.effort 取当前或历史帧。

src/openpi/policies/piper_policy.py
    把 PIPER 的相机、state、effort 输入转换成 OpenPI 模型输入，
    并把模型输出转换回 PIPER 的 7 维动作。
~~~

当 effort_dim=0 时，不创建 effort 投影层，原来的 π0/π0.5 任务保持原布局，因此 effort 改动不会强制影响其他任务。

训练数据字段会映射为：

~~~text
observation.images.front  -> observation/image
observation.images.side   -> observation/wrist_image
observation.state         -> state
observation.effort        -> effort
action                    -> actions
~~~

## 8. action 的训练空间

原始 PIPER 数据记录的是绝对关节目标。训练时：

~~~text
前 6 个机械臂关节：绝对动作 -> 相对 state 的 delta 动作
夹爪：保持绝对动作
~~~

具体为：

~~~text
action_delta[:6] = action[:6] - state[:6]
action_delta[6]   = action[6]
~~~

推理输出阶段会把前六维 delta 加回当前 state，恢复成 PIPER 可以执行的绝对关节目标。

最终策略输出：

~~~text
[action_horizon, 7] = [50, 7]
~~~

部署端不需要自己做归一化或 delta/absolute 转换，只需要按正确关节顺序提供原始输入，并执行返回的动作。

## 9. 计算归一化统计量

训练前执行：

~~~bash
cd /home/zjh/openpi
./examples/piper/compute_norm_stats.sh
~~~

脚本使用 pi05_piper_usb_effort 配置，计算：

~~~text
state：7 维
actions：7 维
effort：6 维
~~~

输出：

~~~text
/data5/zjh/openpi/assets/pi05_piper_usb_effort/usb_0808_pi05_v3/norm_stats.json
~~~

如果仍使用相同数据集重新训练，不需要每次重复计算。保存 checkpoint 时，归一化统计量会复制到 checkpoint 的 assets/usb_0808_pi05_v3/ 下，部署时优先读取 checkpoint 内的统计量。

## 10. 全参数微调

全参数配置名：

~~~text
pi05_piper_usb_effort
~~~

主要参数：

~~~text
π0.5 全参数训练
全局 batch size：24
双卡 FSDP：2
训练步数：30,000
峰值学习率：2.5e-5
EMA：开启
~~~

双卡训练命令，以 1、2 号卡为例：

~~~bash
cd /home/zjh/openpi

CUDA_VISIBLE_DEVICES=1,2 \
BATCH_SIZE=24 \
EXP_NAME=usb_effort_full_bs24_30k \
./examples/piper/run_train.sh
~~~

如果同名实验目录已经存在：

~~~bash
OVERWRITE=true
~~~

表示删除并重新创建这个实验目录，会覆盖之前同名实验。继续已有训练使用：

~~~bash
RESUME=true
~~~

不要同时设置 OVERWRITE=true 和 RESUME=true。

全参数 batch 24 通常要求每张 GPU 至少约 72GB 可用显存。显存不足时，XLA 的 OOM 往往会伴随多卡 rendezvous timeout；后者通常是 OOM 造成的次级错误。

## 11. LoRA 微调

LoRA 配置名：

~~~text
pi05_piper_usb_effort_lora
~~~

当前采用双 LoRA：

~~~text
PaliGemma attention/FFN：rank 16
Action Expert attention/FFN：rank 32
~~~

训练状态：

~~~text
原始 LLM 主权重：冻结
LoRA 参数：训练
effort MLP：训练
action input/output projection：训练
π0.5 time MLP：训练
图像编码器：按当前 freeze filter 保持可训练
EMA：关闭
~~~

推荐双卡 1、2 号卡、全局 batch 24：

~~~bash
cd /home/zjh/openpi

CUDA_VISIBLE_DEVICES=1,2 \
BATCH_SIZE=24 \
EXP_NAME=usb_effort_lora_bs24_30k \
./examples/piper/run_train_lora.sh
~~~

LoRA batch 24 和 batch 32 的差异通常小于显存和稳定性风险，首版建议优先使用 batch 24。

LoRA 双卡通常要求每张 GPU 至少约 60GB 可用显存。GPU 利用率为 0% 但显存仍被占用，通常表示其他服务的模型或 KV cache 仍驻留在 GPU 中，不代表显存可用。

单卡训练必须显式设置较小 batch，例如：

~~~bash
CUDA_VISIBLE_DEVICES=0 \
BATCH_SIZE=4 \
EXP_NAME=usb_effort_lora_single_bs4_30k \
./examples/piper/run_train_lora.sh
~~~

单卡 batch 16/24/32 是否可行取决于显卡型号和剩余显存，不能按双卡经验直接设置。

## 12. checkpoint 保存位置

checkpoint 目录格式：

~~~text
/data5/zjh/openpi/checkpoints/<config_name>/<exp_name>/<step>
~~~

当前 LoRA 模型：

~~~text
/data5/zjh/openpi/checkpoints/pi05_piper_usb_effort_lora/usb_effort_lora_bs24_30k/29999
~~~

典型保存步数：

~~~text
5000、10000、15000、20000、25000、29999
~~~

EXP_NAME 只是实验名称，用来区分不同 batch、学习率或训练方式，不会改变模型结构。

纯推理只需要：

~~~text
29999/params/
29999/assets/
~~~

29999/train_state/ 主要用于继续训练，体积较大，部署时可以不传。

## 13. 部署到另一台电脑

### 13.1 需要传输的目录

推荐目录结构：

~~~text
piper_pi05_deploy/
├── openpi/                    # 修改后的整个源码，不含 .venv 也可以
└── checkpoint/
    └── 29999/
        ├── params/
        ├── assets/
        └── _CHECKPOINT_METADATA
~~~

不要只传几个 .py 文件；OpenPI 还有大量相互依赖的原始源码。最稳妥的是传整个 openpi 源码目录，但排除：

~~~text
.venv/
.git/
__pycache__/
~~~

不需要传训练数据集。部署时归一化统计量已经从 checkpoint/assets 加载。

LeRobot 0.4.4 会由 uv 根据 pyproject.toml 和 uv.lock 从 PyPI 自动下载，不需要作为单独目录传输。

### 13.2 朋友电脑建立环境

在朋友电脑上：

~~~bash
cd /path/to/piper_pi05_deploy/openpi

uv venv --python 3.11
uv sync --frozen --no-dev
~~~

`--frozen` 表示严格使用仓库中的 uv.lock，不在部署电脑上重新解析或改写依赖版本。

检查 GPU：

~~~bash
uv run --no-dev python -c "import jax; print(jax.devices())"
~~~

如果 uv sync 无法下载 LeRobot，先检查网络和 PyPI 访问；如果 JAX 看不到 GPU，先检查 NVIDIA 驱动和 CUDA 兼容性。

### 13.3 启动本地策略服务

~~~bash
cd /path/to/piper_pi05_deploy/openpi

CUDA_VISIBLE_DEVICES=0 \
uv run --no-dev scripts/serve_policy.py \
  --port=8000 \
  --default-prompt="Insert the USB into the USB port on the power strip." \
  policy:checkpoint \
  --policy.config=pi05_piper_usb_effort_lora \
  --policy.dir=/path/to/piper_pi05_deploy/checkpoint/29999
~~~

这是 JAX 本地推理服务。模型加载在执行命令的机器 GPU 上，不会自动控制机器人。

### 13.4 机器人客户端输入

机器人和模型在同一台电脑时：

~~~python
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(
    host="127.0.0.1",
    port=8000,
)
~~~

两台电脑在同一局域网时，把 host 改成模型服务器的局域网 IP。机器人端只需要安装轻量的 openpi-client，不需要安装完整 JAX 训练环境。

每次推理发送：

~~~python
observation = {
    "observation/image": front_image,
    "observation/wrist_image": side_image,
    "observation/state": state_7d,
    "effort": effort_6d,
    "prompt": "Insert the USB into the USB port on the power strip.",
}

result = client.infer(observation)
actions = result["actions"]
~~~

输入要求：

~~~text
front_image：前视相机图像，建议 uint8、224 x 224、HWC
side_image：侧视相机图像，建议 uint8、224 x 224、HWC
state_7d：6 个关节位置 + 夹爪，原始单位
effort_6d：6 个关节 effort，原始单位，不要提前归一化
~~~

输出：

~~~text
actions.shape = [50, 7]
~~~

输出动作已经经过归一化还原以及六个关节的 delta -> absolute 转换，机器人控制程序按数据集中的关节顺序执行即可。上线前仍应设置关节限位、速度限制和急停逻辑。

## 14. 部署时不能混用的内容

以下组合是不正确的：

~~~text
官方未修改 OpenPI 源码
+ pi05_piper_usb_effort_lora checkpoint
~~~

因为官方源码不会创建 effort MLP 和 effort token。

也不要把这个 JAX checkpoint 直接交给只支持 model.safetensors 的 PyTorch 部署脚本。当前 checkpoint 是 Orbax/JAX 格式，应该使用上面的 JAX serve_policy.py 路径。

其他官方任务仍可使用同一份源码，只要使用自己的配置，例如 pi05_droid 或 pi05_libero；这些配置的 effort_dim=0，不会启用 effort 分支。

## 15. 训练前后检查清单

训练前：

~~~text
[ ] 数据是 /data5/usb_0808_pi05_v3，且 codebase_version=v3.0
[ ] LeRobot 使用 v0.4.4 配套版本
[ ] 已生成 norm_stats.json
[ ] CUDA_VISIBLE_DEVICES 拼写正确
[ ] 选中 GPU 的显存确实空闲
[ ] batch size 能被 GPU 数量整除
~~~

训练后：

~~~text
[ ] checkpoint/params 存在
[ ] checkpoint/assets/usb_0808_pi05_v3/norm_stats.json 存在
[ ] 使用与训练一致的配置名
[ ] 推理输入包含 effort
[ ] 先低速、限位、带急停进行实机测试
~~~

当前已经验证过真实 PIPER batch 的主要形状：

~~~text
state：  [B, 32]
effort： [B, 1, 6]
actions：[B, 50, 32]
~~~

其中 32 是 π0.5 内部 action 表示宽度；PIPER 对外的实际动作维度仍然是 7。

## 16. 代码验证

在当前开发环境中，可以运行针对 effort、PIPER 适配器和 checkpoint 加载的测试：

~~~bash
cd /home/zjh/openpi

uv run pytest \
  src/openpi/models/model_test.py \
  src/openpi/policies/piper_policy_test.py \
  src/openpi/training/weight_loaders_test.py

uv run ruff check \
  src/openpi/models/model.py \
  src/openpi/models/pi0.py \
  src/openpi/models/pi0_config.py \
  src/openpi/policies/piper_policy.py \
  src/openpi/training/config.py \
  src/openpi/training/data_loader.py
~~~

已经实际验证过真实 PIPER batch 可以加载，主要形状为：

~~~text
state：  [B, 32]
effort： [B, 1, 6]
actions：[B, 50, 32]
images： 224 x 224 x 3
~~~
