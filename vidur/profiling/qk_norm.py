"""Headwise normalization shared by the Qwen3 reference profiler and CPU tests."""


def normalize_attention_heads(query, key, head_dim, q_norm, k_norm):
    """Apply each norm over head_dim, preserving flattened projection shapes."""
    query_shape, key_shape = query.shape, key.shape
    query = q_norm(query.reshape(-1, head_dim)).reshape(query_shape)
    key = k_norm(key.reshape(-1, head_dim)).reshape(key_shape)
    return query, key
