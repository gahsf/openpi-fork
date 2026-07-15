import numpy as np
import pytest

from openpi.models import model as model_lib
from openpi.shared import array_typing as at
from openpi.training import weight_loaders


def _reference_params():
    return {
        "base": {"kernel": np.ones((2, 2), dtype=np.float32)},
        "overview_action_conditioning": {"kernel": np.full((2,), 2, dtype=np.float32)},
        "PaliGemma": {
            "llm": {
                "layers": {
                    "attn": {
                        "conditional_q_lora_1": {"lora_a": np.full((2,), 3, dtype=np.float32)},
                        "conditional_o_lora_1": {"lora_a": np.full((2,), 4, dtype=np.float32)},
                    }
                }
            }
        },
    }


def _load(monkeypatch, loaded_params):
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(model_lib, "restore_params", lambda *args, **kwargs: loaded_params)
    return weight_loaders.CheckpointWeightLoader("/tmp/params").load(_reference_params())


def test_checkpoint_loader_merges_ocaev1_params_and_preserves_dtype_conversion(monkeypatch):
    loaded_params = {"base": {"kernel": np.ones((2, 2), dtype=np.float64)}}

    result = _load(monkeypatch, loaded_params)

    at.check_pytree_equality(expected=_reference_params(), got=result, check_shapes=True, check_dtypes=True)
    assert result["base"]["kernel"].dtype == np.float32
    np.testing.assert_array_equal(result["overview_action_conditioning"]["kernel"], np.full((2,), 2))


def test_checkpoint_loader_does_not_merge_missing_base_params(monkeypatch):
    result = _load(monkeypatch, loaded_params={})

    with pytest.raises(ValueError, match="different structure"):
        at.check_pytree_equality(expected=_reference_params(), got=result, check_shapes=True, check_dtypes=True)


def test_checkpoint_loader_does_not_hide_shape_mismatch(monkeypatch):
    loaded_params = {"base": {"kernel": np.ones((3, 2), dtype=np.float32)}}

    result = _load(monkeypatch, loaded_params)

    with pytest.raises(ValueError, match="Shape mismatch"):
        at.check_pytree_equality(expected=_reference_params(), got=result, check_shapes=True, check_dtypes=True)
