import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

PIPER_ACTION_DIM = 7
PIPER_EFFORT_DIM = 6


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Map the USB insertion LeRobot dataset into OpenPI's PIPER policy inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        front_image = _parse_image(data["observation/image"])
        side_image = _parse_image(data["observation/wrist_image"])
        effort = np.asarray(data["effort"], dtype=np.float32)
        if effort.ndim == 1:
            effort = effort[None, :]
        if effort.ndim != 2 or effort.shape[-1] != PIPER_EFFORT_DIM:
            raise ValueError(
                f"Expected effort shape [history, {PIPER_EFFORT_DIM}], got {effort.shape}."
            )

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "effort": effort,
            "image": {
                "base_0_rgb": front_image,
                "left_wrist_0_rgb": side_image,
                "right_wrist_0_rgb": np.zeros_like(front_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :PIPER_ACTION_DIM]}
