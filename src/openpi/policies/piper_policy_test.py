import numpy as np
import pytest

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import piper_policy


def test_piper_inputs_maps_cameras_state_effort_and_actions():
    transform = piper_policy.PiperInputs(model_type=_model.ModelType.PI05)
    result = transform(
        {
            "observation/image": np.zeros((3, 480, 640), dtype=np.float32),
            "observation/wrist_image": np.zeros((480, 640, 3), dtype=np.uint8),
            "observation/state": np.arange(7, dtype=np.float32),
            "effort": np.arange(12, dtype=np.float32).reshape(2, 6),
            "actions": np.zeros((50, 7), dtype=np.float32),
            "prompt": "Insert the USB into the USB port on the power strip.",
        }
    )

    assert result["image"]["base_0_rgb"].shape == (480, 640, 3)
    assert result["image"]["left_wrist_0_rgb"].shape == (480, 640, 3)
    assert not result["image_mask"]["right_wrist_0_rgb"]
    assert result["state"].shape == (7,)
    assert result["effort"].shape == (2, 6)
    assert result["actions"].shape == (50, 7)


def test_piper_inputs_promotes_current_effort_to_one_step_history():
    transform = piper_policy.PiperInputs(model_type=_model.ModelType.PI05)
    result = transform(
        {
            "observation/image": np.zeros((480, 640, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((480, 640, 3), dtype=np.uint8),
            "observation/state": np.zeros(7, dtype=np.float32),
            "effort": np.zeros(6, dtype=np.float32),
        }
    )
    assert result["effort"].shape == (1, 6)


def test_piper_inputs_rejects_wrong_effort_dimension():
    transform = piper_policy.PiperInputs(model_type=_model.ModelType.PI05)
    with pytest.raises(ValueError, match="Expected effort shape"):
        transform(
            {
                "observation/image": np.zeros((480, 640, 3), dtype=np.uint8),
                "observation/wrist_image": np.zeros((480, 640, 3), dtype=np.uint8),
                "observation/state": np.zeros(7, dtype=np.float32),
                "effort": np.zeros(5, dtype=np.float32),
            }
        )


def test_piper_outputs_restore_absolute_arm_targets_before_truncating():
    output_transforms = transforms.Group(outputs=[piper_policy.PiperOutputs()]).push(
        outputs=[transforms.AbsoluteActions(transforms.make_bool_mask(6, -1))]
    )
    state = np.arange(1, 8, dtype=np.float32)
    actions = np.zeros((2, 32), dtype=np.float32)
    actions[:, :7] = 0.5

    result = transforms.compose(output_transforms.outputs)({"state": state, "actions": actions})

    expected_arm_targets = np.broadcast_to(state[:6] + 0.5, (2, 6))
    np.testing.assert_allclose(result["actions"][:, :6], expected_arm_targets)
    np.testing.assert_allclose(result["actions"][:, 6], 0.5)
    assert result["actions"].shape == (2, 7)
