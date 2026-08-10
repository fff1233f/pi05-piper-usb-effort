import numpy as np

from openpi.training import weight_loaders


def test_merge_params_keeps_random_effort_projection_weights():
    loaded_params = {"existing": np.ones((2,), dtype=np.float32)}
    reference_params = {
        "existing": np.zeros((2,), dtype=np.float32),
        "effort_proj_in": {
            "kernel": np.full((2, 3), 2.0, dtype=np.float32),
            "bias": np.full((3,), 3.0, dtype=np.float32),
        },
        "effort_proj_out": {
            "kernel": np.full((3, 2), 4.0, dtype=np.float32),
            "bias": np.full((2,), 5.0, dtype=np.float32),
        },
        "unrelated_new_weight": np.full((1,), 6.0, dtype=np.float32),
    }

    result = weight_loaders._merge_params(  # noqa: SLF001
        loaded_params,
        reference_params,
        missing_regex=".*lora.*|.*effort_proj_.*",
    )

    np.testing.assert_array_equal(result["existing"], loaded_params["existing"])
    np.testing.assert_array_equal(
        result["effort_proj_in"]["kernel"], reference_params["effort_proj_in"]["kernel"]
    )
    np.testing.assert_array_equal(
        result["effort_proj_out"]["bias"], reference_params["effort_proj_out"]["bias"]
    )
    assert "unrelated_new_weight" not in result
