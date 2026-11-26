# -*- coding: utf-8 -*-
"""
主任务 1B：正确参照系补「保组织」指标 —— 点恢复率 vs K 曲线
============================================================

背景（昨天的诚实边界）：
  昨天的点恢复率用了「峰值归一化」(img/max)，这会把绝对幅值的丢失
  抹平（除以自己峰值后 max 恒为 1），所以「少角度是否丢点」看不出来。
  gCNR 也有分布可分度陷阱。二者参照系（K=75 图）有误，不采信。

今天的修正：
  参照系换成「理想高斯点图」——在每个散射体位置放一个理想高斯点
  （幅度=1，sigma 同 RF 脉冲），作为「物理真值」。
  点恢复率 = compound(K) 在点位置的峰值 / 理想点图在点位置的峰值。
  关键：不做峰值归一化，直接比绝对幅值。

要回答的问题：
  少角度（K 小）时，真实点目标被保留了多少？有没有把真实点抹掉？

用法：
  python scan_k_preserve.py
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ===== 仿真参数（复用昨天） =====
c = 1540.0
fs = 40e6
N = 64
pitch = 0.3e-3
sigma = 0.05e-6
elem_x = (np.arange(N) - (N - 1) / 2) * pitch

# 固定散射体（3 个点，含深点看「保点」）
scatterers = np.array([
    [0.0e-3, 15e-3, 1.0],   # 点1：中心浅点
    [3.0e-3, 20e-3, 0.7],   # 点2：偏右中深
    [-3.0e-3, 25e-3, 0.5],  # 点3：偏左更深（弱）
])

z_max = 40e-3
t_axis = np.arange(0, 2 * z_max / c, 1 / fs)


def make_rf(theta):
    """生成多角度发射的 RF（无噪声/衰减，与昨天基线一致）。"""
    rf = np.zeros((N, len(t_axis)))
    for xk, zk, amp in scatterers:
        for i in range(N):
            tau = (xk * np.sin(theta) + zk * np.cos(theta)) / c \
                + np.sqrt((xk - elem_x[i]) ** 2 + zk ** 2) / c
            rf[i] += amp * np.exp(-((t_axis - tau) ** 2) / (2 * sigma ** 2))
    return rf


nx, nz = 256, 512
x_grid = np.linspace(-10e-3, 10e-3, nx)
z_grid = np.linspace(2e-3, 40e-3, nz)
X, Z = np.meshgrid(x_grid, z_grid)


def das(rf, theta):
    image = np.zeros((nz, nx))
    for i in range(N):
        tau = (X * np.sin(theta) + Z * np.cos(theta)) / c \
            + np.sqrt((X - elem_x[i]) ** 2 + Z ** 2) / c
        idx = np.round(tau * fs).astype(int)
        idx = np.clip(idx, 0, len(t_axis) - 1)
        image += rf[i, idx]
    image /= N  # 阵元数归一化：抵消 N 阵元相干累加的固有增益，使幅值与散射体强度同尺度
    return image


def compound(K, theta_max_deg=16.0):
    if K <= 1:
        angles = np.deg2rad([0.0])
    else:
        angles = np.deg2rad(np.linspace(-theta_max_deg, theta_max_deg, K))
    img = np.zeros((nz, nx))
    for th in angles:
        img += das(make_rf(th), th)
    img /= len(angles)
    return img


# ===== 绝对点恢复率（正确参照系：理想峰值 = 散射体幅度本身） =====
def point_recovery_absolute(img, xk, zk, amp, box_x=1.5e-3, box_z=2e-3):
    """
    绝对点恢复率：在点位置的小邻域内取重建图峰值，除以「散射体幅度 amp」。
    物理含义：理想情况下该点峰值就是它的散射强度 amp，重建能恢复到 amp 的
    百分之几，就是恢复率（<=1，越接近 1 越不丢点）。
    不做峰值归一化，直接比绝对幅值。
    """
    m = (np.abs(X - xk) <= box_x) & (np.abs(Z - zk) <= box_z)
    peak_rec = np.abs(np.asarray(img, dtype=np.float64)[m]).max()
    if amp == 0:
        return float("nan")
    return peak_rec / amp


def main():
    K_list = [1, 3, 5, 9, 15, 25, 37, 49, 61, 75]

    # 每个点、每个 K 的绝对恢复率
    # recovery[k][i] = 第 i 个点在第 k 个 K 下的恢复率
    recovery = {k: [] for k in K_list}

    print("点恢复率（绝对幅值，参照系=理想峰值=散射体幅度）：")
    print(f"{'K':>4} | " + " | ".join([f"点{i+1}" for i in range(len(scatterers))]) + " | 平均")
    for k in K_list:
        img = compound(k)
        for (xk, zk, amp) in scatterers:
            r = point_recovery_absolute(img, xk, zk, amp)
            recovery[k].append(r)
        avg = float(np.nanmean(recovery[k]))
        row = " | ".join([f"{recovery[k][i]:.4f}" for i in range(len(scatterers))])
        print(f"{k:4d} | {row} | {avg:.4f}")

    # ===== 画「点恢复率 vs K」曲线（每个点一条 + 平均一条） =====
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for i, (xk, zk, amp) in enumerate(scatterers):
        ys = [recovery[k][i] for k in K_list]
        ax.plot(K_list, ys, "o-", color=colors[i], linewidth=2, markersize=6,
                label=f"点{i+1} (x={xk*1e3:.0f}mm, z={zk*1e3:.0f}mm, 幅={amp})")

    # 平均恢复率
    avg_ys = [float(np.nanmean(recovery[k])) for k in K_list]
    ax.plot(K_list, avg_ys, "o-", color="#333333", linewidth=3, markersize=8,
            label="三点平均", alpha=0.9)

    ax.axhline(1.0, color="gray", ls="--", alpha=0.6, label="理想恢复率=1（不丢点）")
    ax.set_xlabel("K（平面波数 / 发射次数）")
    ax.set_ylabel("点目标恢复率（绝对幅值）")
    ax.set_title("点恢复率 vs K：少角度是否保留真实点目标？\n（参照系=理想峰值=散射体幅度，不峰值归一化）", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    out_png = "K_curve_preserve.png"
    plt.savefig(out_png, dpi=150)
    print(f"\n已保存曲线图 {out_png}")

    # ===== 一句话结论 =====
    r1 = recovery[1][0]    # K=1 时点1的恢复率
    r75 = recovery[75][0]  # K=75 时点1的恢复率
    print("\n===== 结论 =====")
    print(f"K=1 时点1恢复率 = {r1:.4f}；K=75 时点1恢复率 = {r75:.4f}")
    if r1 < 0.9:
        print(f"→ 少角度（K=1）确实把真实点目标削弱了（恢复率 {r1:.4f} < 0.9），"
              f"说明少角度有「丢点」风险。")
    else:
        print(f"→ 少角度（K=1）点目标恢复率 {r1:.4f}，说明真实点基本被保留，"
              f"少角度「不丢点」。")
    print("（注：本图用绝对幅值，未做峰值归一化，能真实反映幅值丢失）")


if __name__ == "__main__":
    main()
