# -*- coding: utf-8 -*-
"""
Verasonics 脚本结构的 Python 复刻（教学用，最小平面波成像）

目标：用 Python 的 dict 复刻 Verasonics 脚本的「定义结构体 + 排事件序列」思维，
让你在不装 MATLAB 的情况下，亲手体验一遍「TX 发射 → Receive 接收 → Recon 重建」。

对照关系（Verasonics -> 本脚本）：
    Resource.Parameters -> 物理常量（声速、阵元数）
    Trans               -> 换能器几何（阵元位置）
    TX                  -> 发射（平面波角度）
    Receive             -> 接收（时间轴、通道）
    Recon               -> 波束形成（DAS）
    PData               -> 像素网格
    Event               -> 事件序列（把发射/接收/重建串起来）

运行：  & "E:\develop\Python\python.exe" verasonics_style_demo.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块


# ======================================================================
# ① Resource.Parameters —— 系统物理参数
# ======================================================================
Resource = {
    "Parameters": {
        "numTransmit": 64,        # 发射通道数（阵元数）
        "numRcvChannels": 64,     # 接收通道数
        "speedOfSound": 1540.0,   # 声速 m/s（对应 Verasonics 的 1540）
        "sampleRate": 40e6,       # 采样率 Hz
    }
}
c = Resource["Parameters"]["speedOfSound"]
fs = Resource["Parameters"]["sampleRate"]
N = Resource["Parameters"]["numTransmit"]

# ======================================================================
# ② Trans —— 换能器（阵元几何）
# ======================================================================
pitch = 0.3e-3  # 阵元间距 m
Trans = {
    "name": "L11-5v (仿真)",
    "frequency": 5.0e6,                      # 中心频率 Hz
    "numelements": N,
    "ElementPos": (np.arange(N) - (N - 1) / 2) * pitch,  # 阵元 x 坐标
}
elem_x = Trans["ElementPos"]

# ======================================================================
# ③ 散射体（相当于「被测组织」，Verasonics 里是真实组织/仿体）
#    这里用几个已知点散射体 + 一个消声区，模拟「已知真值」场景
# ======================================================================
scatterers = np.array([
    [0.0e-3, 15e-3, 1.0],   # 点1：x, z, 幅度
    [3.0e-3, 20e-3, 0.7],   # 点2
    [-3.0e-3, 25e-3, 0.5],  # 点3（深点、弱点）
])

# ======================================================================
# ④ TX —— 发射：平面波（focus=0，给 Delay 加线性梯度实现斜发射）
# ======================================================================
def make_TX(theta_deg):
    """定义一个平面波发射事件。theta 是发射角（度）。
    Verasonics 里：TX.focus=0（不聚焦）+ TX.Delay 线性梯度 = 斜平面波。
    这里用「发射角 theta」直接表达同一个物理概念。"""
    theta = np.deg2rad(theta_deg)
    TX = {
        "focus": 0,                        # 0 = 平面波（不聚焦）
        "Apod": np.ones(N),                # 切趾（全 1 = 不加权）
        "steer_angle": theta_deg,          # 发射角（对应 Delay 线性梯度）
        "Delay": elem_x * np.sin(theta),   # 发射延时：x*sinθ 的线性梯度
    }
    return TX

# ======================================================================
# ⑤ Receive —— 接收：时间轴 + 每个通道采集 RF
# ======================================================================
z_max = 40e-3
t_axis = np.arange(0, 2 * z_max / c, 1 / fs)   # 接收时间轴（往返）
sigma = 0.05e-6                                  # 高斯脉冲宽度（时间域）

Receive = {
    "startDepth": 2e-3,
    "endDepth": z_max,
    "tAxis": t_axis,
    "sigma": sigma,
}

# ======================================================================
# ⑥ 发射-接收过程（make_rf）：TX 发射 + Receive 采集
# ======================================================================
def acquire_rf(TX):
    """执行一次「发射+接收」，返回 N 通道的 RF 数据 (N, T)。
    这是 Verasonics 里 Event 的 tx+rcv 部分。"""
    theta = np.deg2rad(TX["steer_angle"])
    rf = np.zeros((N, len(t_axis)))
    for xk, zk, amp in scatterers:
        # 发射延时：平面波到散射体
        tau_tx = (xk * np.sin(theta) + zk * np.cos(theta)) / c
        for i in range(N):
            # 接收延时：散射体到阵元 i
            tau_rx = np.sqrt((xk - elem_x[i]) ** 2 + zk ** 2) / c
            tau = tau_tx + tau_rx
            rf[i] += amp * np.exp(-((t_axis - tau) ** 2) / (2 * sigma ** 2))
    return rf

# ======================================================================
# ⑦ PData —— 像素网格（成像网格）
# ======================================================================
nx, nz = 256, 512
x_grid = np.linspace(-10e-3, 10e-3, nx)
z_grid = np.linspace(2e-3, 40e-3, nz)
X, Z = np.meshgrid(x_grid, z_grid)
PData = {"X": X, "Z": Z, "nz": nz, "nx": nx}

# ======================================================================
# ⑧ Recon —— 波束形成（DAS）
# ======================================================================
def recon_das(rf, TX):
    """延时求和波束形成。对应 Verasonics 的 Recon（Image Reconstruction）。
    注意：image /= N 是阵元数归一化（昨天 1B 踩过的坑）。"""
    theta = np.deg2rad(TX["steer_angle"])
    image = np.zeros((nz, nx))
    for i in range(N):
        tau = (X * np.sin(theta) + Z * np.cos(theta)) / c \
            + np.sqrt((X - elem_x[i]) ** 2 + Z ** 2) / c
        idx = np.round(tau * fs).astype(int)
        idx = np.clip(idx, 0, len(t_axis) - 1)
        image += rf[i, idx]
    image /= N  # 阵元数归一化
    return image

# ======================================================================
# ⑨ Event —— 事件序列：把「发射→接收→重建」串成一个循环
# ======================================================================
def run_sequence(angles_deg):
    """事件序列：对每个角度，发射→接收→重建，最后相干复合。
    这就是 Verasonics 的 Event 序列 + 多角度复合（CPWC）。"""
    angles = angles_deg
    images = []
    for a in angles:
        TX = make_TX(a)            # Event.tx：定义发射
        rf = acquire_rf(TX)        # Event.rcv：接收
        img = recon_das(rf, TX)    # Event.recon：重建
        images.append(img)
    # 相干复合（coherent compounding）：多角度取平均
    compound = np.mean(images, axis=0)
    return compound

# ======================================================================
# 运行：单角度 vs 多角度复合
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Verasonics 风格最小平面波成像（Python 复刻）")
    print("=" * 60)

    # 单角度（K=1）
    img_k1 = run_sequence([0.0])

    # 多角度复合（K=9，在 ±16° 内取 9 个角）
    angles_9 = np.linspace(-16, 16, 9)
    img_k9 = run_sequence(angles_9)

    # 打印关键信息（对照 Verasonics 结构体）
    print(f"\n[Resource] 声速={c} m/s, 阵元数={N}, 采样率={fs/1e6} MHz")
    print(f"[Trans]   {Trans['name']}, 中心频率={Trans['frequency']/1e6} MHz")
    print(f"[TX]      focus=0(平面波), 角度范围 ±16°")
    print(f"[Receive] 深度 {Receive['startDepth']*1e3:.0f}-{Receive['endDepth']*1e3:.0f} mm")
    print(f"[PData]   网格 {nz}x{nx}")
    print(f"[Event]   K=1(单角度) vs K=9(复合)\n")

    print(f"单角度 K=1  峰值幅度 = {img_k1.max():.4f}")
    print(f"复合   K=9  峰值幅度 = {img_k9.max():.4f}")
    print(f"\n(注：复合后旁瓣下降、峰值更接近真值，对应 '少角度→多角度' 的增益)")

    # 画图对比
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_k1, cmap="gray", aspect="auto",
                   extent=[x_grid[0]*1e3, x_grid[-1]*1e3, z_grid[-1]*1e3, z_grid[0]*1e3])
    axes[0].set_title("K=1 (单角度平面波)")
    axes[0].set_xlabel("x (mm)"); axes[0].set_ylabel("z (mm)")
    axes[1].imshow(img_k9, cmap="gray", aspect="auto",
                   extent=[x_grid[0]*1e3, x_grid[-1]*1e3, z_grid[-1]*1e3, z_grid[0]*1e3])
    axes[1].set_title("K=9 (多角度复合)")
    axes[1].set_xlabel("x (mm)"); axes[1].set_ylabel("z (mm)")
    plt.tight_layout()
    plt.savefig("verasonics_style_demo.png", dpi=150)
    print("\n已保存对比图: verasonics_style_demo.png")
