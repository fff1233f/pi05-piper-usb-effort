# PIPER USB insertion with effort-conditioned PI0.5

中文完整流程见 [README_CN.md](README_CN.md)。

This example fine-tunes the official JAX PI0.5 model on the PIPER task
`Insert the USB into the USB port on the power strip.`

## Dataset and environment

The original dataset is kept unchanged at `/data5/usb_0808`. Its metadata declares LeRobot v2.1, which cannot be
loaded by the LeRobot v0.4.4 package pinned in this OpenPI environment. Training uses the converted v3.0 copy:

```text
/data5/usb_0808_pi05_v3
robot_type: piper
100 episodes / 42406 frames / 30 FPS
state/action: 7 dimensions (6 arm joints + gripper)
effort: 6 arm-joint dimensions
cameras: front + side
```

The project pins `lerobot==0.4.4` from PyPI. Do not install a newer LeRobot checkout into this environment, because
OpenPI's pinned JAX, Flax and Transformers versions are tested together with LeRobot v0.4.4.

## Effort conditioning

PI0.5 keeps its pretrained discrete state prompt unchanged:

```text
Task: Insert the USB ..., State: 120 37 208 ...;
Action:
```

The normalized state values are quantized into 256 bins before tokenization. Effort is not added to this text. The
selected design projects normalized effort through a two-layer MLP into one continuous action-expert token:

```text
normalized effort history -> flatten -> MLP -> one effort token -> noisy action tokens
```

The default config uses only the current six-dimensional effort (`effort_history=(0,)`). This matches TA-VLA's
`EXPERT / DePost` route. To concatenate three historical samples into the same token, set both fields consistently:

```python
effort_history=(-2, -1, 0)
effort_history_length=3
```

That variant corresponds to TA-VLA's `EXPERT_HIS_C / Dec-1` route.

## Normalization and training

Compute state, action and effort normalization statistics first:

```bash
cd /home/zjh/openpi
./examples/piper/compute_norm_stats.sh
```

The output is written below:

```text
/data5/zjh/openpi/assets/pi05_piper_usb_effort/usb_0808_pi05_v3/norm_stats.json
```

Then launch two-GPU full fine-tuning:

```bash
cd /home/zjh/openpi
CUDA_VISIBLE_DEVICES=0,1 EXP_NAME=usb_effort_v1 ./examples/piper/run_train.sh
```

The default global batch size is 24 (12 samples per GPU with two GPUs), and training runs for 30,000 steps.
The launcher checks that both GPUs are the same model and each has at least 72,000 MiB free before starting.

The official `pi05_base` checkpoint initializes the existing model parameters. The new effort projection layers are
randomly initialized and learned during fine-tuning.

At inference time, provide `effort` as `[history_length, 6]`. A current-only vector with shape `[6]` is also accepted
and promoted to `[1, 6]` by the PIPER input adapter.

## Dual-LoRA training

The `pi05_piper_usb_effort_lora` config adds rank-16 LoRA adapters to PaliGemma and rank-32 adapters to the action
expert. Their original LLM weights are frozen. The effort MLP, action projections, PI0.5 time MLP and image encoder
remain trainable. EMA is disabled so training does not keep a second full parameter copy.

Run the default two-GPU LoRA setup with global batch size 32 and 30,000 steps:

```bash
CUDA_VISIBLE_DEVICES=1,2 \
EXP_NAME=usb_effort_lora_bs32_30k \
./examples/piper/run_train_lora.sh
```

Both GPUs must have at least 60,000 MiB free. The LoRA config reuses the state, action and effort normalization
statistics produced by the full-finetuning config.
