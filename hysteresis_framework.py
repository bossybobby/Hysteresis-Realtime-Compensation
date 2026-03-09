"""实时流式压力信号降噪框架（以卡尔曼滤波为主）。"""

from __future__ import annotations

import numpy as np

from signal_generator import NoisySignalGenerator, SignalSimulatorConfig


class RealTimeKalmanDenoiser:
    """一维卡尔曼滤波器（严格因果）。"""

    def __init__(self, process_var: float = 1e-4, measure_var: float = 2.5e-3):
        if process_var <= 0.0 or measure_var <= 0.0:
            raise ValueError("process_var 与 measure_var 必须为正数")

        self.q = float(process_var)
        self.r = float(measure_var)
        self.x_hat = 0.0
        self.p = 1.0
        self._initialized = False

    def update(self, z_t: float) -> float:
        """输入当前观测值，输出当前去噪估计。"""
        if not self._initialized:
            self.x_hat = float(z_t)
            self._initialized = True
            return self.x_hat

        # 预测（状态模型：x_k = x_{k-1} + w_k）
        x_pred = self.x_hat
        p_pred = self.p + self.q

        # 更新
        k_gain = p_pred / (p_pred + self.r)
        self.x_hat = x_pred + k_gain * (float(z_t) - x_pred)
        self.p = (1.0 - k_gain) * p_pred
        return self.x_hat


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def run_demo(show_plot: bool = True, save_path: str = "kalman_denoise_demo.png") -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plt = None

    cfg = SignalSimulatorConfig(fs=200.0, duration=24.0, noise_std=0.05, seed=123)
    generator = NoisySignalGenerator(cfg)
    t, p_true, noisy_obs = generator.generate()

    denoiser = RealTimeKalmanDenoiser(process_var=5e-5, measure_var=cfg.noise_std**2)
    p_hat = np.zeros_like(p_true)
    for k, z in enumerate(noisy_obs):
        p_hat[k] = denoiser.update(float(z))

    e_in = rmse(p_true, noisy_obs)
    e_out = rmse(p_true, p_hat)

    print("=== Kalman Denoising Metrics ===")
    print(f"RMSE(原始带噪观测 vs 真实压力): {e_in:.6f}")
    print(f"RMSE(卡尔曼去噪估计 vs 真实压力): {e_out:.6f}")
    print(f"Improvement ratio: {e_in / max(e_out, 1e-12):.2f}x")

    if plt is None:
        return

    try:
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    plt.figure(figsize=(11, 6))
    plt.plot(t, p_true, label="原始压力（真实值）", linewidth=2)
    plt.plot(t, noisy_obs, label="带噪观测", alpha=0.7)
    plt.plot(t, p_hat, label="卡尔曼滤波去噪估计", linewidth=1.8)
    plt.xlabel("时间 (s)")
    plt.ylabel("幅值")
    plt.title("实时压力信号降噪（Kalman Filter）")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    if show_plot:
        plt.show()
    plt.close()


if __name__ == "__main__":
    run_demo(show_plot=False)
