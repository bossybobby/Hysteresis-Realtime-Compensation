"""离子凝胶压力传感器迟滞信号生成模块（独立文件）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from pi_operators import update_play_bank


@dataclass
class PISimulatorConfig:
    fs: float = 200.0
    duration: float = 20.0
    noise_std: float = 0.03
    seed: Optional[int] = None


class PIHysteresisGenerator:
    """生成“真实压力 -> 带 PI 迟滞与噪声电压”数据。"""

    def __init__(self, thresholds: np.ndarray, weights: np.ndarray, cfg: PISimulatorConfig):
        self.thresholds = np.asarray(thresholds, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.cfg = cfg

        if self.thresholds.ndim != 1 or self.weights.ndim != 1:
            raise ValueError("thresholds/weights 必须为 1D 向量")
        if self.thresholds.size != self.weights.size:
            raise ValueError("thresholds 与 weights 长度必须一致")
        if np.any(self.weights < 0):
            raise ValueError("为便于 NNLS 反演，weights 应为非负")

    def _pressure_profile(self, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """生成更随机化的压力波形：对幅值、频率和相位引入小幅随机扰动，并加入低频漂移。"""
        # 基础参数
        amps = np.array([0.65, 0.25, 0.15])
        freqs = np.array([0.13, 0.05, 0.32])
        phases = np.array([0.0, 0.8, 1.7])

        # 随机抖动：幅值 0.8-1.2，频率 0.95-1.05，相位 ±0.6 rad
        amp_jitter = rng.uniform(0.8, 1.2, size=3)
        freq_jitter = rng.uniform(0.95, 1.05, size=3)
        phase_jitter = rng.uniform(-0.6, 0.6, size=3)

        a = amps * amp_jitter
        f = freqs * freq_jitter
        ph = phases + phase_jitter

        p = a[0] * np.sin(2 * np.pi * f[0] * t + ph[0])
        p += a[1] * np.sin(2 * np.pi * f[1] * t + ph[1])
        p += a[2] * np.sin(2 * np.pi * f[2] * t + ph[2])

        # 低频漂移与随机微扰，增强随机性但保持可控
        drift_amp = rng.uniform(0.0, 0.12)
        drift_phase = rng.uniform(0.0, 2 * np.pi)
        p += drift_amp * np.sin(2 * np.pi * 0.005 * t + drift_phase)

        # 小量随机高频分量（使得不同生成结果更不相同）
        hf_amp = rng.uniform(0.0, 0.03)
        hf_phase = rng.uniform(0.0, 2 * np.pi)
        p += hf_amp * np.sin(2 * np.pi * 1.2 * t + hf_phase)

        # 归一化到 0-1
        p = (p - p.min()) / (p.max() - p.min())
        return p

    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(self.cfg.duration * self.cfg.fs)
        t = np.arange(n, dtype=float) / self.cfg.fs
        rng = np.random.default_rng(self.cfg.seed)
        p_true = self._pressure_profile(t, rng)
        states = np.zeros_like(self.thresholds)
        v_hyst = np.zeros_like(p_true)

        for k, x_t in enumerate(p_true):
            states = update_play_bank(x_t, states, self.thresholds)
            v_hyst[k] = self.weights @ states

        v_noisy = v_hyst + rng.normal(0.0, self.cfg.noise_std, size=n)
        return t, p_true, v_noisy