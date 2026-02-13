"""PI (Prandtl-Ishlinskii) 基础算子：适合实时流式与嵌入式迁移。"""

from __future__ import annotations

import numpy as np


def play_operator_update(x_t: float, state_prev: float, threshold: float) -> float:
    """单个 play operator 的因果更新。

    y_t = min(max(x_t - r, y_{t-1}), x_t + r)
    """
    return min(max(x_t - threshold, state_prev), x_t + threshold)


def update_play_bank(x_t: float, states: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """更新整组 play operators（逐算子循环，便于直译到 C/C++）。"""
    new_states = np.empty_like(states)
    for i in range(states.size):
        new_states[i] = play_operator_update(x_t, states[i], thresholds[i])
    return new_states