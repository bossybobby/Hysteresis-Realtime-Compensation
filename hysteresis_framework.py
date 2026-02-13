"""实时流式离子凝胶压力传感器迟滞补偿框架。"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from pi_operators import update_play_bank
from signal_generator import PIHysteresisGenerator, PISimulatorConfig
import matplotlib.pyplot as plt 

# --------------------------------
# 1) Offline identification (NNLS)
# --------------------------------

def build_play_feature_matrix(signal: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """通过一次顺序扫描构建 PI 特征矩阵 Phi。

    Phi[k, i] = 第 i 个 play operator 在时刻 k 的输出。
    """
    x = np.asarray(signal, dtype=float)
    r = np.asarray(thresholds, dtype=float)
    n = x.size
    m = r.size

    phi = np.zeros((n, m), dtype=float)
    states = np.zeros(m, dtype=float)

    for k in range(n):
        states = update_play_bank(x[k], states, r)
        phi[k, :] = states

    return phi


def identify_inverse_weights_nnls(v_meas: np.ndarray, p_true: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """辨识逆模型权重：p ≈ Phi(v) @ w_inv，其中 w_inv >= 0。"""
    phi_v = build_play_feature_matrix(v_meas, thresholds)
    w_inv, _ = nnls(phi_v, p_true)
    return w_inv


# --------------------------------
# 2) Real-time compensation engine
# --------------------------------

class RealTimePICompensator:
    """实时 PI 迟滞补偿引擎（严格因果）。"""

    def __init__(self, thresholds: np.ndarray, inv_weights: np.ndarray, alpha: float = 0.2):
        self.thresholds = np.asarray(thresholds, dtype=float)
        self.inv_weights = np.asarray(inv_weights, dtype=float)

        if self.thresholds.size != self.inv_weights.size:
            raise ValueError("thresholds 与 inv_weights 维度不一致")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("IIR alpha 需满足 0 < alpha <= 1")

        self.alpha = float(alpha)
        self.states = np.zeros_like(self.thresholds)
        self.y_lp = 0.0
        self._lp_initialized = False

    def _lowpass(self, x_t: float) -> float:
        """一阶 IIR: y[k] = alpha*x[k] + (1-alpha)*y[k-1]。"""
        if not self._lp_initialized:
            self.y_lp = x_t
            self._lp_initialized = True
        else:
            self.y_lp = self.alpha * x_t + (1.0 - self.alpha) * self.y_lp
        return self.y_lp

    def update(self, v_in: float) -> float:
        """输入当前电压采样，输出当前补偿压力估计。"""
        v_f = self._lowpass(v_in)
        self.states = update_play_bank(v_f, self.states, self.thresholds)
        return float(self.inv_weights @ self.states)


# -----------------------------
# 3) Demo / validation routine
# -----------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def run_demo(show_plot: bool = True, save_path: str = "hysteresis_demo.png") -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plt = None

    # 真实系统（用于模拟数据）
    model_thresholds = np.linspace(0.01, 0.25, 18)
    model_weights = np.exp(-5.2 * model_thresholds)
    model_weights /= model_weights.sum()

    cfg = PISimulatorConfig(fs=200.0, duration=24.0, noise_std=0.025, seed=123)
    generator = PIHysteresisGenerator(model_thresholds, model_weights, cfg)
    t, p_true, v_meas = generator.generate()

    # 离线辨识：构造逆模型 p <- v
    id_thresholds = np.linspace(0.005, 0.28, 24)
    w_inv = identify_inverse_weights_nnls(v_meas, p_true, id_thresholds)

    # 在线引擎：逐点 update
    engine = RealTimePICompensator(id_thresholds, w_inv, alpha=0.22)
    p_hat = np.zeros_like(p_true)
    for k, v in enumerate(v_meas):
        p_hat[k] = engine.update(float(v))

    e_in = rmse(p_true, v_meas)
    e_out = rmse(p_true, p_hat)

    print("=== Hysteresis Compensation Metrics ===")
    print(f"RMSE(原始带迟滞噪声电压 vs 压力): {e_in:.6f}")
    print(f"RMSE(实时补偿压力估计 vs 压力): {e_out:.6f}")
    print(f"Improvement ratio: {e_in / max(e_out, 1e-12):.2f}x")

    plt.figure(figsize=(11, 6))
    # 最简单的中文支持：直接设置 matplotlib rcParams，适用于大多数系统（需要已安装中文字体，如 SimHei）
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    plt.plot(t, p_true, label="原始压力（真实值）", linewidth=2)
    plt.plot(t, v_meas, label="带迟滞+噪声电压", alpha=0.75)
    plt.plot(t, p_hat, label="实时补偿后的压力估计", linewidth=1.8)
    plt.xlabel("时间 (s)")
    plt.ylabel("幅值")
    plt.title("离子凝胶传感器迟滞补偿（流式 PI + NNLS）")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    if show_plot:
        plt.show()
    plt.close()


if __name__ == "__main__":
    run_demo(show_plot=False)