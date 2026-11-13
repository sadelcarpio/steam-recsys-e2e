"""numpy export of the user tower (`user_tower.npz`), part of every saved model.

The serving Lambda embeds a history of liked games without torch: it runs `UserTower.forward`
in numpy over these arrays (serving/src/steam_serving/online.py; parity with torch is tested
in inference/tests/test_online_parity.py). Item embeddings are not exported here: they depend
on the current game features, so the inference pipeline recomputes them every run.

    format_version   int64 scalar    USER_TOWER_NUMPY_FORMAT
    model_id         str scalar      the model these weights belong to
    history_length   int64 scalar    ModelConfig.history_length (5 = the mart's)
    padding_id       int64 scalar    PADDING_ID
    oov_id           int64 scalar    OOV_ID
    game_table       float32 [vocab_games, game_embedding_dim]  frozen game table
    seen_games       bool [vocab_games]  game_idx with training interactions (else OOV)
    w1, b1           float32 [hidden, in_dim], [hidden]  x @ w1.T + b1, relu
    w2, b2           float32 [output_dim, hidden], [output_dim]  then L2 normalization

Format 2 (a model with the A4 tag profile) adds, and x gets the profile appended
(in_dim = game_embedding_dim + 1 + attribute_embedding_dim):
    tag_values       uint16 (int64 if a tag id doesn't fit) [n]  game_idx -> tags, CSR values
    tag_offsets      int64 [vocab_games + 1]
    tag_weights      float32 [n]         Steam tag weights (sum 1 per game)
    tag_recency      float32 [history_length]  raw per-position parameter (weight = exp)
    tag_table        float32 [vocab_tags, attribute_embedding_dim]
Format 1 (no tag profile) stays byte-compatible with servers that only read format 1.

Format 3 (A3 attention encoder, `HistoryAttention`), with or without the tag profile arrays
above (present iff the model has the profile), adds `ATTENTION_ARRAYS`, and x becomes
[pooled, length / history_length, attention summary, tag profile?]:
    attn_heads       int64 scalar
    attn_ln_eps      float64 scalar      LayerNorm eps (torch default 1e-5)
    attn_summary     float32 [d]         learned summary token (d = game_embedding_dim)
    attn_position    float32 [history_length, d]  added to the history games (0 = most recent)
    attn_norm_w, attn_norm_b  float32 [d]  final LayerNorm of the summary token
    per block, stacked on a leading [layers] axis (pre-LN, see `_AttentionBlock`):
    attn_ln1_w, attn_ln1_b  [layers, d]; attn_qkv_w [layers, 3d, d], attn_qkv_b [layers, 3d]
    attn_out_w [layers, d, d], attn_out_b [layers, d]; attn_ln2_w, attn_ln2_b [layers, d]
    attn_ffn_w1 [layers, 2d, d], attn_ffn_b1 [layers, 2d]; attn_ffn_w2 [layers, d, 2d],
    attn_ffn_b2 [layers, d]
Padding never takes part (it is masked as a key and only the summary token is read), so a
port may drop the padding slots.
"""

from __future__ import annotations

import io

import numpy as np
from torch import nn

from steam_training.contracts import OOV_ID, PADDING_ID
from steam_training.model import TwoTowerModel

USER_TOWER_NUMPY_FORMAT = 1
USER_TOWER_TAG_PROFILE_FORMAT = 2
USER_TOWER_ATTENTION_FORMAT = 3
TAG_PROFILE_ARRAYS = ("tag_values", "tag_offsets", "tag_weights", "tag_recency", "tag_table")
ATTENTION_ARRAYS = (
    "attn_heads",
    "attn_ln_eps",
    "attn_summary",
    "attn_position",
    "attn_norm_w",
    "attn_norm_b",
    "attn_ln1_w",
    "attn_ln1_b",
    "attn_qkv_w",
    "attn_qkv_b",
    "attn_out_w",
    "attn_out_b",
    "attn_ln2_w",
    "attn_ln2_b",
    "attn_ffn_w1",
    "attn_ffn_b1",
    "attn_ffn_w2",
    "attn_ffn_b2",
)
USER_TOWER_ARRAYS = (
    "format_version",
    "model_id",
    "history_length",
    "padding_id",
    "oov_id",
    "game_table",
    "seen_games",
    "w1",
    "b1",
    "w2",
    "b2",
)


def user_tower_arrays(model: TwoTowerModel, model_id: str) -> dict[str, np.ndarray]:
    tower = model.user_tower
    first, second = tower.mlp[0], tower.mlp[2]
    assert isinstance(first, nn.Linear) and isinstance(second, nn.Linear), tower.mlp

    def numpy(tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    profile = tower.tag_profile is not None
    if tower.attention is not None:
        version = USER_TOWER_ATTENTION_FORMAT
    elif profile:
        version = USER_TOWER_TAG_PROFILE_FORMAT
    else:
        version = USER_TOWER_NUMPY_FORMAT
    arrays = {
        "format_version": np.int64(version),
        "model_id": np.str_(model_id),
        "history_length": np.int64(tower.history_length),
        "padding_id": np.int64(PADDING_ID),
        "oov_id": np.int64(OOV_ID),
        "game_table": numpy(tower.game_table).astype(np.float32),
        "seen_games": numpy(tower.game_ids.seen_games).astype(bool),
        "w1": numpy(first.weight).astype(np.float32),
        "b1": numpy(first.bias).astype(np.float32),
        "w2": numpy(second.weight).astype(np.float32),
        "b2": numpy(second.bias).astype(np.float32),
    }
    assert tuple(arrays) == USER_TOWER_ARRAYS
    if profile:
        arrays |= {
            # tag ids are small (~450 tags): uint16 saves most of the lookup's size
            "tag_values": _small_ids(numpy(tower.game_tag_values)),
            "tag_offsets": numpy(tower.game_tag_offsets).astype(np.int64),
            "tag_weights": numpy(tower.game_tag_weights).astype(np.float32),
            "tag_recency": numpy(tower.tag_recency).astype(np.float32),
            "tag_table": numpy(tower.tag_profile.weight).astype(np.float32),
        }
    if tower.attention is not None:
        arrays |= _attention_arrays(tower.attention, numpy)
    return arrays


def _attention_arrays(attention, numpy) -> dict[str, np.ndarray]:
    blocks = list(attention.blocks)
    eps = {block.attention_norm.eps for block in blocks} | {attention.norm.eps}
    assert len(eps) == 1, eps

    def stack(get) -> np.ndarray:
        return np.stack([numpy(get(block)) for block in blocks]).astype(np.float32)

    arrays = {
        "attn_heads": np.int64(blocks[0].heads),
        "attn_ln_eps": np.float64(eps.pop()),
        "attn_summary": numpy(attention.summary).astype(np.float32),
        "attn_position": numpy(attention.position).astype(np.float32),
        "attn_norm_w": numpy(attention.norm.weight).astype(np.float32),
        "attn_norm_b": numpy(attention.norm.bias).astype(np.float32),
        "attn_ln1_w": stack(lambda b: b.attention_norm.weight),
        "attn_ln1_b": stack(lambda b: b.attention_norm.bias),
        "attn_qkv_w": stack(lambda b: b.qkv.weight),
        "attn_qkv_b": stack(lambda b: b.qkv.bias),
        "attn_out_w": stack(lambda b: b.out.weight),
        "attn_out_b": stack(lambda b: b.out.bias),
        "attn_ln2_w": stack(lambda b: b.ffn_norm.weight),
        "attn_ln2_b": stack(lambda b: b.ffn_norm.bias),
        "attn_ffn_w1": stack(lambda b: b.ffn[0].weight),
        "attn_ffn_b1": stack(lambda b: b.ffn[0].bias),
        "attn_ffn_w2": stack(lambda b: b.ffn[2].weight),
        "attn_ffn_b2": stack(lambda b: b.ffn[2].bias),
    }
    assert tuple(arrays) == ATTENTION_ARRAYS
    return arrays


def _small_ids(values: np.ndarray) -> np.ndarray:
    if len(values) and int(values.max()) > np.iinfo(np.uint16).max:
        return values.astype(np.int64)
    return values.astype(np.uint16)


def encode_user_tower(model: TwoTowerModel, model_id: str) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **user_tower_arrays(model, model_id))  # loads without pickle
    return buffer.getvalue()
