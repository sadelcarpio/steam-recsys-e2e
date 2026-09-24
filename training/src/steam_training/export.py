"""numpy export of the user tower (`user_tower.npz`), part of every saved model.

The serving Lambda embeds a history of liked games without torch: it runs `UserTower.forward`
in numpy over these arrays (serving/src/steam_serving/online.py; parity with torch is tested
in inference/tests/test_online_parity.py). Item embeddings are not exported here: they depend
on the current game features, so the inference pipeline recomputes them every run.

    format_version   int64 scalar    USER_TOWER_NUMPY_FORMAT
    model_id         str scalar      the model these weights belong to
    history_length   int64 scalar    USER_HISTORY_LENGTH
    padding_id       int64 scalar    PADDING_ID
    oov_id           int64 scalar    OOV_ID
    game_table       float32 [vocab_games, game_embedding_dim]  frozen game table
    seen_games       bool [vocab_games]  game_idx with training interactions (else OOV)
    w1, b1           float32 [hidden, game_embedding_dim + 1], [hidden]  x @ w1.T + b1, relu
    w2, b2           float32 [output_dim, hidden], [output_dim]  then L2 normalization
"""

from __future__ import annotations

import io

import numpy as np
from torch import nn

from steam_training.contracts import OOV_ID, PADDING_ID, USER_HISTORY_LENGTH
from steam_training.model import TwoTowerModel

USER_TOWER_NUMPY_FORMAT = 1
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

    arrays = {
        "format_version": np.int64(USER_TOWER_NUMPY_FORMAT),
        "model_id": np.str_(model_id),
        "history_length": np.int64(USER_HISTORY_LENGTH),
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
    return arrays


def encode_user_tower(model: TwoTowerModel, model_id: str) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **user_tower_arrays(model, model_id))  # loads without pickle
    return buffer.getvalue()
