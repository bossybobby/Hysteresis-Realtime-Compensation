"""压力信号与噪声观测生成模块（无迟滞版本）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class SignalSimulatorConfig:
    fs: float = 200.0
    duration: float = 20.0
    noise_std: float = 0.03
    seed: Optional[int] = None


class NoisySignalGenerator:
    """生成“真实压力 -> 带高斯噪声观测”数据。"""

    def __init__(self, cfg: SignalSimulatorConfig):
        self.cfg = cfg

    def _pressure_profile(self, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        amps = np.array([0.65, 0.25, 0.15])
        freqs = np.array([0.13, 0.05, 0.32])
        phases = np.array([0.0, 0.8, 1.7])

        amp_jitter = rng.uniform(0.8, 1.2, size=3)
        freq_jitter = rng.uniform(0.95, 1.05, size=3)
        phase_jitter = rng.uniform(-0.6, 0.6, size=3)

        a = amps * amp_jitter
        f = freqs * freq_jitter
        ph = phases + phase_jitter

        p = a[0] * np.sin(2 * np.pi * f[0] * t + ph[0])
        p += a[1] * np.sin(2 * np.pi * f[1] * t + ph[1])
        p += a[2] * np.sin(2 * np.pi * f[2] * t + ph[2])

        drift_amp = rng.uniform(0.0, 0.12)
        drift_phase = rng.uniform(0.0, 2 * np.pi)
        p += drift_amp * np.sin(2 * np.pi * 0.005 * t + drift_phase)

        hf_amp = rng.uniform(0.0, 0.03)
        hf_phase = rng.uniform(0.0, 2 * np.pi)
        p += hf_amp * np.sin(2 * np.pi * 1.2 * t + hf_phase)

        p = (p - p.min()) / (p.max() - p.min())
        return p

    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(self.cfg.duration * self.cfg.fs)
        t = np.arange(n, dtype=float) / self.cfg.fs
        rng = np.random.default_rng(self.cfg.seed)

        p_true = self._pressure_profile(t, rng)
        noisy_obs = p_true + rng.normal(0.0, self.cfg.noise_std, size=n)
        return t, p_true, noisy_obs
