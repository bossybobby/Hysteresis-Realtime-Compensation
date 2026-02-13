```
# Ion-gel-hysteresis-algorithm

用于离子凝胶压力传感器迟滞（Hysteresis）补偿的 Python 实时流式算法框架。

## 模块结构
- `signal_generator.py`：独立信号生成模块（PI 迟滞 + 高斯噪声）。
- `pi_operators.py`：PI 核心 play operator 因果更新逻辑。
- `hysteresis_framework.py`：离线辨识 + 实时补偿引擎 + 演示入口。

## 功能概览
- **数据模拟模块**：使用 Prandtl-Ishlinskii (PI) play operators 注入迟滞，并叠加高斯噪声。
- **离线参数辨识**：基于固定阈值算子构建特征矩阵，使用 `scipy.optimize.nnls` 辨识逆模型权重。
- **实时补偿引擎**：`RealTimePICompensator.update(v_in)` 严格因果，只依赖当前输入和内部状态。
- **实时去噪**：引擎内部集成一阶 IIR 低通滤波器。
- **验证与可视化**：输出 RMSE 并绘制对比图。

## 运行方式
```bash
python hysteresis_framework.py
```

运行后将：

1. 在终端打印补偿前后 RMSE。
2. 在仓库目录输出图像 `hysteresis_demo.png`。

## 迁移到 C++/嵌入式提示

- `pi_operators.py` 中 `play_operator_update` 和 `update_play_bank` 使用简洁逐算子循环，便于直接映射为 C/C++ for-loop。
- `RealTimePICompensator.update()` 仅做：
  1) 当前采样低通，2) play 状态更新，3) 点积输出，满足实时任务调度需求。
- 状态变量只有 `states` 与 `y_lp`，易于放入 MCU/RTOS 的对象或结构体中。

```

```
