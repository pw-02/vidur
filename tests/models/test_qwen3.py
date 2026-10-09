from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vidur.config import ReplicaConfig
from vidur.config.model_config import BaseModelConfig
from vidur.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from vidur.profiling.qk_norm import normalize_attention_heads
from vidur.utils.param_counter import ParamCounter


@pytest.mark.parametrize(
    "name,layers,heads,hidden,mlp",
    [
        ("Qwen/Qwen3-8B", 36, 32, 4096, 12288),
        ("Qwen/Qwen3-14B", 40, 40, 5120, 17408),
    ],
)
def test_configuration(name, layers, heads, hidden, mlp):
    model = BaseModelConfig.create_from_name(name)
    assert (
        model.num_layers,
        model.num_q_heads,
        model.embedding_dim,
        model.mlp_hidden_dim,
    ) == (layers, heads, hidden, mlp)
    assert model.num_kv_heads == 8
    assert model.embedding_dim // model.num_q_heads == 128
    assert model.use_qk_norm and not model.use_qkv_bias
    assert model.dtype_name == "bfloat16"
    assert model.max_position_embeddings == 40960
    assert model.vocab_size == 151936


def test_legacy_models_unchanged():
    model = BaseModelConfig.create_from_name("meta-llama/Meta-Llama-3-8B")
    assert not model.use_qk_norm
    assert model.dtype_name == "float16"


def test_headwise_norm_preserves_gqa_shapes():
    q = np.arange(1, 25, dtype=float).reshape(2, 12)
    k = np.arange(1, 9, dtype=float).reshape(2, 4)
    norm = lambda x: x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    nq, nk = normalize_attention_heads(q, k, 4, norm, norm)
    assert nq.shape == q.shape and nk.shape == k.shape
    np.testing.assert_allclose(nq, norm(q.reshape(2, 3, 4)).reshape(q.shape))
    np.testing.assert_allclose(nk, norm(k.reshape(2, 1, 4)).reshape(k.shape))


def test_qk_norm_parameters_are_counted():
    config = ReplicaConfig(model_name="Qwen/Qwen3-8B")
    counter = ParamCounter(config)
    with_norm = counter.get_num_parameters_per_layer()
    config.model_config.use_qk_norm = False
    assert with_norm - counter.get_num_parameters_per_layer() == 256


class ProfileReader(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


def predictor():
    result = object.__new__(ProfileReader)
    result._model_config = BaseModelConfig.create_from_name("Qwen/Qwen3-8B")
    result._replica_config = SimpleNamespace(tensor_parallel_size=1)
    result._block_size = 16
    return result


def test_old_profiles_rejected():
    model = predictor()
    model._read_input_file = lambda path: pd.DataFrame({"n_head": [32]})
    with pytest.raises(ValueError, match="normalization"):
        model._load_compute_df("unused")


def test_wrong_precision_rejected():
    model = predictor()
    model._read_input_file = lambda path: pd.DataFrame(
        {"qk_norm_in_attn_rope": [True], "dtype": ["float16"]}
    )
    with pytest.raises(ValueError, match="dtype"):
        model._load_compute_df("unused")


def test_qwen_profile_import(tmp_path):
    model = predictor()
    frame = pd.DataFrame(
        dict(
            n_head=[32],
            n_kv_head=[8],
            n_embd=[4096],
            n_expanded_embd=[12288],
            use_gated_mlp=[True],
            vocab_size=[151936],
            num_tensor_parallel_workers=[1],
            qk_norm_in_attn_rope=[True],
            dtype=["bfloat16"],
        )
    )
    model._read_input_file = lambda path: frame.copy()
    assert len(model._load_compute_df("unused")) == 1
    path = tmp_path / "attention.csv"
    pd.DataFrame(
        dict(
            n_q_head=[32],
            n_kv_head=[8],
            n_embd=[4096],
            block_size=[16],
            num_tensor_parallel_workers=[1],
            dtype=["bfloat16"],
        )
    ).to_csv(path, index=False)
    assert len(model._load_attention_df(path)) == 1


def test_network_profile_reader_does_not_require_compute_metadata(tmp_path):
    path = tmp_path / "network.csv"
    pd.DataFrame({"collective": ["all_reduce"], "num_workers": [1]}).to_csv(
        path, index=False
    )
    assert len(predictor()._read_input_file(path)) == 1


def test_qwen_compute_csv_import_without_mock(tmp_path):
    model = predictor()
    path = tmp_path / "mlp.csv"
    pd.DataFrame(
        dict(
            n_head=[32],
            n_kv_head=[8],
            n_embd=[4096],
            n_expanded_embd=[12288],
            use_gated_mlp=[True],
            vocab_size=[151936],
            num_tensor_parallel_workers=[1],
            qk_norm_in_attn_rope=[True],
            dtype=["bfloat16"],
        )
    ).to_csv(path, index=False)
    assert len(model._load_compute_df(path)) == 1
