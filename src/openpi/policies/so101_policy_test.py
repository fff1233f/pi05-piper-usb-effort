import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import so101_policy


def test_so101_inputs_maps_two_cameras_and_actions():
    transform = so101_policy.SO101Inputs(model_type=_model.ModelType.PI05)
    result = transform(
        {
            "observation/image": np.zeros((3, 720, 1280), dtype=np.float32),
            "observation/wrist_image": np.zeros((720, 1280, 3), dtype=np.uint8),
            "observation/state": np.arange(6, dtype=np.float32),
            "actions": np.zeros((50, 6), dtype=np.float32),
            "prompt": "Grab blue battery to the bin",
        }
    )

    assert result["image"]["base_0_rgb"].shape == (720, 1280, 3)
    assert result["image"]["left_wrist_0_rgb"].shape == (720, 1280, 3)
    assert not result["image_mask"]["right_wrist_0_rgb"]
    assert result["state"].shape == (6,)
    assert result["actions"].shape == (50, 6)


def test_so101_outputs_keeps_six_action_dimensions():
    actions = np.zeros((50, 32), dtype=np.float32)
    result = so101_policy.SO101Outputs()({"actions": actions})
    assert result["actions"].shape == (50, 6)


def test_so101_outputs_restore_absolute_arm_targets_before_truncating():
    output_transforms = transforms.Group(outputs=[so101_policy.SO101Outputs()]).push(
        outputs=[transforms.AbsoluteActions(transforms.make_bool_mask(5, -1))]
    )
    state = np.arange(1, 7, dtype=np.float32)
    actions = np.zeros((2, 32), dtype=np.float32)
    actions[:, :6] = 0.5

    result = transforms.compose(output_transforms.outputs)({"state": state, "actions": actions})

    expected_arm_targets = np.broadcast_to(state[:5] + 0.5, (2, 5))
    np.testing.assert_allclose(result["actions"][:, :5], expected_arm_targets)
    np.testing.assert_allclose(result["actions"][:, 5], 0.5)
    assert result["actions"].shape == (2, 6)
