import argparse
import dataclasses
import json
import pathlib
import re

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma as gemma_lib
from openpi.models import model as model_lib
from openpi.models import overview_action_conditioning as conditioning_lib
from openpi.models import pi0 as pi0_lib
from openpi.training import config as config_lib
from openpi.training import data_loader as data_loader_lib

_ADAPTER_PATH = re.compile(r".*(overview_action_conditioning|conditional_(q|o)_lora_1).*")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose an OCAE LIBERO pilot checkpoint without modifying it.")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--base-params", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def _flatten(params) -> dict[str, np.ndarray]:
    return {
        "/".join(str(part) for part in path): np.asarray(value)
        for path, value in traverse_util.flatten_dict(params).items()
    }


def _compare_frozen_params(base_params, checkpoint_params) -> dict:
    base = _flatten(base_params)
    checkpoint = _flatten(checkpoint_params)
    missing = sorted(set(base) - set(checkpoint))
    unexpected = sorted(path for path in set(checkpoint) - set(base) if not _ADAPTER_PATH.fullmatch(path))
    adapter = sorted(path for path in set(checkpoint) - set(base) if _ADAPTER_PATH.fullmatch(path))

    changed = []
    dtype_casts = 0
    for path in sorted(set(base) & set(checkpoint)):
        actual = checkpoint[path]
        expected = base[path]
        if expected.dtype != actual.dtype:
            expected = expected.astype(actual.dtype)
            dtype_casts += 1
        if expected.shape != actual.shape or not np.array_equal(expected, actual):
            changed.append(path)

    return {
        "base_leaf_count": len(base),
        "checkpoint_leaf_count": len(checkpoint),
        "frozen_leaf_count": len(set(base) & set(checkpoint)),
        "adapter_leaf_count": len(adapter),
        "dtype_cast_leaf_count": dtype_casts,
        "missing_frozen_paths": missing,
        "unexpected_new_paths": unexpected,
        "changed_frozen_paths": changed,
        "equal": not missing and not unexpected and not changed,
    }


def _find(flat_params: dict[str, np.ndarray], suffix: str) -> np.ndarray:
    matches = [value for path, value in flat_params.items() if path.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one parameter ending in {suffix!r}, found {len(matches)}")
    return matches[0]


def _compute_conditioning_diagnostics(model, observation):
    observation = model_lib.preprocess_observation(None, observation, train=False)
    prefix = model.embed_prefix(observation)
    prefix_mask = pi0_lib.make_attn_mask(prefix.input_mask, prefix.ar_mask)
    positions = jnp.cumsum(prefix.input_mask, axis=1) - 1
    (prefix_out, _), _ = model.PaliGemma.llm(
        [prefix.tokens, None],
        mask=prefix_mask,
        positions=positions,
    )

    image_masks = tuple(observation.image_masks[name] for name in observation.images)
    language_mask = observation.tokenized_prompt_mask
    state = observation.state
    if model.action_conditioning_mode == "constant":
        prefix_out, image_masks, language_mask, state = conditioning_lib.make_constant_reference_inputs(
            prefix_out,
            image_masks,
            language_mask,
            state,
        )

    conditioning = model.overview_action_conditioning
    scene_context = conditioning.encode_overview(prefix_out, prefix, image_masks, language_mask)
    state_context = conditioning.encode_state(state)
    controller = conditioning.controller
    controller_input = jnp.concatenate(
        [controller.scene_norm(scene_context), controller.state_norm(state_context)], axis=-1
    )
    hidden = nnx.swish(controller.hidden_proj(controller_input))
    raw_gates = controller.gate_proj(hidden).reshape(
        scene_context.shape[0], controller.num_layers, 2, controller.num_heads
    )
    raw_gates = raw_gates * conditioning.config.gate_logit_scale
    return {
        "prefix_input": prefix_out,
        "state_input": state,
        "scene_context": scene_context,
        "state_context": state_context,
        "controller_hidden": hidden,
        "q_raw_gate": raw_gates[:, :, 0],
        "o_raw_gate": raw_gates[:, :, 1],
        "q_gate": jnp.tanh(raw_gates[:, :, 0]),
        "o_gate": jnp.tanh(raw_gates[:, :, 1]),
    }


def _sample_metrics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    flattened = values.reshape(values.shape[0], -1)
    distances = np.linalg.norm(flattened - flattened[:1], axis=1)
    norms = np.linalg.norm(flattened, axis=1)
    reference_norm = max(float(norms[0]), np.finfo(np.float32).eps)
    return {
        "shape": list(values.shape),
        "finite": bool(np.isfinite(values).all()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "max_abs": float(np.abs(values).max()),
        "mean_l2_norm": float(norms.mean()),
        "mean_l2_distance_from_first_sample": float(distances.mean()),
        "max_l2_distance_from_first_sample": float(distances.max()),
        "mean_relative_l2_distance_from_first_sample": float(distances.mean() / reference_norm),
    }


def _gate_metrics(gates: np.ndarray) -> dict:
    gates = np.asarray(gates, dtype=np.float32)
    return {
        **_sample_metrics(gates),
        "abs_gt_0_9_fraction": float(np.mean(np.abs(gates) > 0.9)),
        "abs_gt_0_99_fraction": float(np.mean(np.abs(gates) > 0.99)),
    }


def _raw_gate_metrics(raw_gates: np.ndarray) -> dict:
    raw_gates = np.asarray(raw_gates, dtype=np.float32)
    return {
        **_sample_metrics(raw_gates),
        "abs_gt_3_fraction": float(np.mean(np.abs(raw_gates) > 3)),
        "abs_gt_5_fraction": float(np.mean(np.abs(raw_gates) > 5)),
    }


def _value_metrics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    return {
        "shape": list(values.shape),
        "finite": bool(np.isfinite(values).all()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "max_abs": float(np.abs(values).max()),
        "l2_norm": float(np.linalg.norm(values)),
    }


def _effective_delta_ratio(
    gates: np.ndarray,
    lora_a: np.ndarray,
    lora_b: np.ndarray,
    base_kernel: np.ndarray,
    scale: float,
) -> dict:
    gates = np.asarray(gates, dtype=np.float32)
    lora_a = np.asarray(lora_a, dtype=np.float32)
    lora_b = np.asarray(lora_b, dtype=np.float32)
    base_kernel = np.asarray(base_kernel, dtype=np.float32)
    delta = np.einsum("lnir,lnro->lnio", lora_a, lora_b) * scale
    weighted_delta = gates[..., None, None] * delta[None, ...]
    ratios = np.sqrt(np.sum(np.square(weighted_delta), axis=(1, 2, 3, 4)))
    ratios /= np.linalg.norm(base_kernel)
    return {
        "finite": bool(np.isfinite(ratios).all()),
        "per_sample": ratios.tolist(),
        "mean": float(ratios.mean()),
        "min": float(ratios.min()),
        "max": float(ratios.max()),
    }


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    train_config = config_lib.get_config(args.config_name)
    if not train_config.model.overview_action_conditioning.enabled:
        raise ValueError(f"Config is not OCAE-enabled: {args.config_name}")

    params_path = args.checkpoint_dir / "params"
    checkpoint_params = model_lib.restore_params(params_path, restore_type=np.ndarray)
    base_params = model_lib.restore_params(args.base_params, restore_type=np.ndarray)
    frozen = _compare_frozen_params(base_params, checkpoint_params)
    del base_params

    conditioning = train_config.model.overview_action_conditioning
    if conditioning.conditioning_mode == "static":
        action_config = gemma_lib.get_config(train_config.model.action_expert_variant)
        q_gates = o_gates = np.ones(
            (args.batch_size, action_config.depth, action_config.num_heads),
            dtype=np.float32,
        )
        conditioning_path = None
    else:
        diagnostic_config = dataclasses.replace(train_config, batch_size=args.batch_size, num_workers=0)
        data_loader = data_loader_lib.create_data_loader(diagnostic_config, shuffle=False, num_batches=1)
        observation, _ = next(iter(data_loader))

        model = train_config.model.load(checkpoint_params)
        graphdef, state = nnx.split(model)

        @jax.jit
        def compute_conditioning_diagnostics(model_state, obs):
            return _compute_conditioning_diagnostics(nnx.merge(graphdef, model_state), obs)

        diagnostic_arrays = jax.device_get(compute_conditioning_diagnostics(state, observation))
        q_gates = diagnostic_arrays["q_gate"]
        o_gates = diagnostic_arrays["o_gate"]
        conditioning_path = {
            name: (_raw_gate_metrics(values) if name in ("q_raw_gate", "o_raw_gate") else _sample_metrics(values))
            for name, values in diagnostic_arrays.items()
            if name not in ("q_gate", "o_gate")
        }

    flat_checkpoint = _flatten(checkpoint_params)
    controller_params = {
        path: _value_metrics(value)
        for path, value in flat_checkpoint.items()
        if "overview_action_conditioning/controller/" in path
    }
    scale = conditioning.lora_alpha / conditioning.rank
    q_ratio = _effective_delta_ratio(
        q_gates,
        _find(flat_checkpoint, "conditional_q_lora_1/lora_a"),
        _find(flat_checkpoint, "conditional_q_lora_1/lora_b"),
        _find(flat_checkpoint, "q_einsum_1/w"),
        scale,
    )
    o_ratio = _effective_delta_ratio(
        o_gates,
        _find(flat_checkpoint, "conditional_o_lora_1/lora_a"),
        _find(flat_checkpoint, "conditional_o_lora_1/lora_b"),
        _find(flat_checkpoint, "attn_vec_einsum_1/w"),
        scale,
    )

    result = {
        "config_name": args.config_name,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "base_params": str(args.base_params.resolve()),
        "batch_size": args.batch_size,
        "conditioning_mode": conditioning.conditioning_mode,
        "frozen_params": frozen,
        "q_gate": _gate_metrics(q_gates),
        "o_gate": _gate_metrics(o_gates),
        "conditioning_path": conditioning_path,
        "controller_params": controller_params,
        "q_effective_delta_ratio": q_ratio,
        "o_effective_delta_ratio": o_ratio,
    }
    result["passed"] = bool(
        frozen["equal"]
        and result["q_gate"]["finite"]
        and result["o_gate"]["finite"]
        and q_ratio["finite"]
        and o_ratio["finite"]
        and q_ratio["mean"] > 0
        and o_ratio["mean"] > 0
        and q_ratio["max"] < 1
        and o_ratio["max"] < 1
        and (
            conditioning.conditioning_mode == "static"
            or (result["q_gate"]["abs_gt_0_9_fraction"] < 0.01 and result["o_gate"]["abs_gt_0_9_fraction"] < 0.01)
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
