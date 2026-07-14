import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models import overview_action_conditioning
from openpi.models.pi0 import Pi0


class _FakeImageEmbedder:
    def __call__(self, image, *, train):
        del train
        return image.reshape(image.shape[0], -1, image.shape[-1]), None


class _FakeLlm:
    def __call__(self, tokens, *, method):
        assert method == "embed"
        return jnp.repeat(tokens[..., None], 2, axis=-1).astype(jnp.float32)


class _FakePaliGemma:
    img = _FakeImageEmbedder()
    llm = _FakeLlm()


class _FakePi0:
    PaliGemma = _FakePaliGemma()


def _make_observation(*, with_language: bool) -> model_lib.Observation:
    return model_lib.Observation(
        images={
            "base": jnp.arange(8, dtype=jnp.float32).reshape(2, 1, 2, 2),
            "wrist": jnp.arange(12, dtype=jnp.float32).reshape(2, 1, 3, 2),
        },
        image_masks={
            "base": jnp.array([True, False]),
            "wrist": jnp.array([True, True]),
        },
        state=jnp.zeros((2, 2), dtype=jnp.float32),
        tokenized_prompt=jnp.array([[1, 2, 0, 0], [3, 4, 5, 0]]) if with_language else None,
        tokenized_prompt_mask=(
            jnp.array([[True, True, False, False], [True, True, True, False]]) if with_language else None
        ),
    )


def test_embed_prefix_records_segment_layout_and_masks():
    observation = _make_observation(with_language=True)

    prefix = Pi0.embed_prefix(_FakePi0(), observation)

    assert isinstance(prefix, overview_action_conditioning.PrefixEmbeddings)
    assert prefix.image_lengths == (2, 3)
    assert prefix.language_length == 4
    np.testing.assert_array_equal(prefix.tokens[:, :2], observation.images["base"].reshape(2, 2, 2))
    np.testing.assert_array_equal(prefix.tokens[:, 2:5], observation.images["wrist"].reshape(2, 3, 2))
    np.testing.assert_array_equal(prefix.tokens[:, 5:], jnp.repeat(observation.tokenized_prompt[..., None], 2, axis=-1))
    np.testing.assert_array_equal(
        prefix.input_mask,
        jnp.array(
            [
                [True, True, True, True, True, True, True, False, False],
                [False, False, True, True, True, True, True, True, False],
            ]
        ),
    )
    np.testing.assert_array_equal(prefix.ar_mask, jnp.zeros((9,), dtype=jnp.bool_))


def test_embed_prefix_without_language_records_zero_length():
    observation = _make_observation(with_language=False)

    prefix = Pi0.embed_prefix(_FakePi0(), observation)

    assert prefix.image_lengths == (2, 3)
    assert prefix.language_length == 0
    assert prefix.tokens.shape == (2, 5, 2)
    assert prefix.input_mask.shape == (2, 5)
    assert prefix.ar_mask.shape == (5,)


def test_prefix_embeddings_static_layout_survives_jit():
    prefix = overview_action_conditioning.PrefixEmbeddings(
        tokens=jnp.ones((1, 5, 2), dtype=jnp.float32),
        input_mask=jnp.ones((1, 5), dtype=jnp.bool_),
        ar_mask=jnp.zeros((5,), dtype=jnp.bool_),
        image_lengths=(2, 3),
        language_length=0,
    )

    result = jax.jit(lambda value: value)(prefix)

    assert result.image_lengths == (2, 3)
    assert result.language_length == 0
    assert len(jax.tree.leaves(result)) == 3
