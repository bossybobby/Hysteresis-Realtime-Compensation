# Ion-gel-hysteresis-algorithm

当前版本已简化为**基础实时降噪算法**，仅保留卡尔曼滤波流程，不再包含迟滞补偿建模。

## 模块结构
- `signal_generator.py`：生成“真实压力 + 高斯噪声观测”的模拟数据。
- `hysteresis_framework.py`：实时一维卡尔曼滤波去噪引擎，支持读取 `tcdata.txt` 并输出处理前后图像。
- `pi_operators.py`：历史文件，当前简化流程未使用。

## 使用你的数据（推荐）
准备 `tcdata.txt`，格式为两列：
1. 第一列：时间
2. 第二列：电容值

运行：
- `hysteresis_framework.py`：实时一维卡尔曼滤波去噪引擎 + 演示入口。
- `pi_operators.py`：历史文件，当前简化流程未使用。

## 功能概览
- **数据模拟模块**：生成多频叠加压力曲线并叠加高斯噪声。
- **实时去噪引擎**：`RealTimeKalmanDenoiser.update(z_t)` 严格因果，仅使用当前观测和内部状态。
- **验证与可视化**：输出 RMSE 与增益比例，并绘制对比图。

## 运行方式
```bash
python hysteresis_framework.py --data-file tcdata.txt
```

将输出：
- `tcdata_before.png`：处理前原始电容信号。
- `tcdata_after.png`：卡尔曼滤波后的电容信号。

可调参数：
```bash
python hysteresis_framework.py --data-file tcdata.txt --process-var 1e-5 --measure-var 1e-3
```

## 无数据文件时
如果未找到 `tcdata.txt`，程序会自动回退到内置模拟 demo，并输出 `kalman_denoise_demo.png`。

## 迁移到 C++/嵌入式提示
1. 在终端打印去噪前后 RMSE。
2. 在仓库目录输出图像 `kalman_denoise_demo.png`。

## 迁移到 C++/嵌入式提示

- 核心更新仅包含卡尔曼两步：预测与更新，可直接映射为 MCU 的标量运算。
- 关键状态只有 `x_hat`（估计值）和 `p`（估计协方差），内存占用极低。
- `update()` 接口为逐采样输入输出，适配定时中断或实时线程。
