# -*- coding: utf-8 -*-
"""
多角度 K 曲线扫描：K = 1..75 多指标曲线 + 临界 K 结论
=====================================================

物理背景（先定义后例子）：
  单平面波（K=1）不聚焦，横向分辨率受孔径限制、旁瓣多。
  多平面波相干复合（CPWC）：用 K 个不同角度发射，每次 DAS 一张图，
  再相干叠加。多个角度从不同方向"看"同一散射体，叠加等效于
  "合成更大孔径（合成聚焦）"，横向分辨率改善、旁瓣压低。
  K 越大质量越好，但有天花板（饱和）——找到饱和拐点 = 临界 K。

本脚本做什么：
  1. 复用 11/6 的 make_rf(theta) + das(rf, theta) 多角度复合代码；
  2. 在固定角度范围 ±16° 内，取 K ∈ {1,3,5,9,15,25,37,49,61,75} 个角度；
  3. 每个 K 合成一张复合 B 图，目标图 = K=75 满配图；
  4. 对每个 K 算 4 个指标：PSNR / SSIM / CNR / 点目标恢复率；
  5. 画「指标 vs K」曲线，找临界 K。

用法：
  python scan_k.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ===== 复用 11/6 的仿真参数 =====
c = 1540.0            # 声速 m/s
fs = 40e6             # 采样频率
N = 64                # 阵元数
pitch = 0.3e-3        # 阵元间距
sigma = 0.05e-6       # 高斯脉冲宽度
elem_x = (np.arange(N) - (N - 1) / 2) * pitch

# 固定散射体（3 个点，含一个深点便于看"保点"效果）
scatterers = np.array([
    [0.0e-3, 15e-3, 1.0],   # 点1：中心浅点
    [3.0e-3, 20e-3, 0.7],   # 点2：偏右中深
    [-3.0e-3, 25e-3, 0.5],  # 点3：偏左更深（弱）
])

z_max = 40e-3
t_axis = np.arange(0, 2 * z_max / c, 1 / fs)

# ===== 多角度发射：生成 RF =====
def make_rf(theta):
    """theta（弧度）：平面波发射角度。返回 (N, T) 的 RF。"""
    rf = np.zeros((N, len(t_axis)))
    for xk, zk, amp in scatterers:
        for i in range(N):
            tau = (xk * np.sin(theta) + zk * np.cos(theta)) / c \
                + np.sqrt((xk - elem_x[i]) ** 2 + zk ** 2) / c
            rf[i] += amp * np.exp(-((t_axis - tau) ** 2) / (2 * sigma ** 2))
    return rf


# ===== 带角度的 DAS =====
nx, nz = 256, 512
x_grid = np.linspace(-10e-3, 10e-3, nx)
z_grid = np.linspace(2e-3, 40e-3, nz)
X, Z = np.meshgrid(x_grid, z_grid)


def das(rf, theta):
    """带角度的 DAS 重建，返回 (nz, nx) 的 B 图。"""
    image = np.zeros((nz, nx))
    for i in range(N):
        tau = (X * np.sin(theta) + Z * np.cos(theta)) / c \
            + np.sqrt((X - elem_x[i]) ** 2 + Z ** 2) / c
        idx = np.round(tau * fs).astype(int)
        idx = np.clip(idx, 0, len(t_axis) - 1)
        image += rf[i, idx]
    return image


# ===== 复合：K 个角度相干叠加 =====
def compound(K, theta_max_deg=16.0):
    """
    K 个角度在 ±theta_max 内均匀取，各自 DAS 后相干叠加（平均）。
    返回 (nz, nx) 的复合 B 图。
    """
    if K <= 1:
        angles = np.deg2rad([0.0])
    else:
        angles = np.deg2rad(np.linspace(-theta_max_deg, theta_max_deg, K))
    img = np.zeros((nz, nx))
    for th in angles:
        img += das(make_rf(th), th)
    img /= len(angles)
    return img


# ===== 指标（复用 us-recon/metrics 的算法，内联以便独立运行）=====
def psnr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def ssim(a, b, C1=1e-4, C2=9e-4):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = np.mean((a - mu_a) * (b - mu_b))
    var_a = max(var_a, 0.0)
    var_b = max(var_b, 0.0)
    l = (2 * mu_a * mu_b + C1) / (mu_a ** 2 + mu_b ** 2 + C1)
    cc = (2 * np.sqrt(var_a) * np.sqrt(var_b) + C2) / (var_a + var_b + C2)
    s = (cov + C2 / 2) / (np.sqrt(var_a) * np.sqrt(var_b) + C2 / 2)
    return l * cc * s


def cnr(img, in_mask, bg_mask):
    img = np.asarray(img, dtype=np.float64)
    mean_in = img[in_mask].mean()
    mean_bg = img[bg_mask].mean()
    sigma_bg = img[bg_mask].std()
    if sigma_bg == 0:
        return float("inf") if mean_in != mean_bg else 0.0
    return (mean_in - mean_bg) / sigma_bg


def gcnr(img, in_mask, bg_mask, bins=256):
    """广义 CNR：两区域强度分布的交叠面积，越接近 1 越可区分。
    不依赖均值/方差假设，对稀疏点目标更敏感（顶刊惯例用 gCNR 测对比度）。"""
    img = np.asarray(img, dtype=np.float64)
    vals_in = img[in_mask]
    vals_bg = img[bg_mask]
    range_min = min(vals_in.min(), vals_bg.min())
    range_max = max(vals_in.max(), vals_bg.max())
    hist_in, _ = np.histogram(vals_in, bins=bins, range=(range_min, range_max))
    hist_bg, _ = np.histogram(vals_bg, bins=bins, range=(range_min, range_max))
    hist_in = hist_in / hist_in.sum()
    hist_bg = hist_bg / hist_bg.sum()
    overlap = np.sum(np.minimum(hist_in, hist_bg))
    return 1.0 - overlap


def point_recovery(img_rec, img_ref, in_mask):
    peak_rec = np.abs(np.asarray(img_rec, dtype=np.float64)[in_mask]).max()
    peak_ref = np.abs(np.asarray(img_ref, dtype=np.float64)[in_mask]).max()
    if peak_ref == 0:
        return float("nan")
    return peak_rec / peak_ref


# ===== 构造「目标区」和「背景区（消声区）」掩码 =====
def build_masks():
    """
    目标区：圈住 3 个散射体各自附近的小方块（并集）。
    消声区：一块「已知不放散射体」的区域，理论应全黑，有信号就是伪影。
    """
    def rect(x0, x1, z0, z1):
        return (X >= x0) & (X <= x1) & (Z >= z0) & (Z <= z1)

    # 目标区：每个散射体周围 ±1.5mm（x）±2mm（z）
    in_mask = np.zeros((nz, nx), dtype=bool)
    for xk, zk, _ in scatterers:
        in_mask |= rect(xk - 1.5e-3, xk + 1.5e-3, zk - 2e-3, zk + 2e-3)

    # 消声区（已知无散射体）：x∈[-8,-4]mm, z∈[10,18]mm（避开三个点）
    bg_mask = rect(-8e-3, -4e-3, 10e-3, 18e-3)

    return in_mask, bg_mask


def main():
    # K 档位
    K_list = [1, 3, 5, 9, 15, 25, 37, 49, 61, 75]

    # 满配目标图 = K=75
    img_ref = compound(75)
    img_ref_norm = img_ref / (np.abs(img_ref).max() + 1e-8)

    in_mask, bg_mask = build_masks()

    results = {k: {} for k in K_list}
    for k in K_list:
        img = compound(k)
        # 归一化到 [0,1]（峰值归一，和 ref 一致）
        img_norm = img / (np.abs(img).max() + 1e-8)

        results[k]["psnr"] = psnr(img_norm, img_ref_norm)
        results[k]["ssim"] = ssim(img_norm, img_ref_norm)
        results[k]["gcnr"] = gcnr(img_norm, in_mask, bg_mask)
        # 点目标恢复率：对 3 个点分别算，取平均
        recs = []
        for xk, zk, _ in scatterers:
            m = (np.abs(X - xk) <= 1.5e-3) & (np.abs(Z - zk) <= 2e-3)
            recs.append(point_recovery(img_norm, img_ref_norm, m))
        results[k]["recovery"] = float(np.nanmean(recs))

        print(f"K={k:3d}  PSNR={results[k]['psnr']:8.3f}  "
              f"SSIM={results[k]['ssim']:.5f}  "
              f"gCNR={results[k]['gcnr']:.4f}  "
              f"点恢复率={results[k]['recovery']:.4f}")

    # ===== 画 4 条「指标 vs K」曲线 =====
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics = [
        ("psnr", "PSNR (dB)", "越大越好"),
        ("ssim", "SSIM", "越大越好，趋近1"),
        ("gcnr", "gCNR", "越大越好（对比度）"),
        ("recovery", "点目标恢复率", "越接近1越好（保点）"),
    ]
    for ax, (key, ylab, note) in zip(axes.ravel(), metrics):
        ys = [results[k][key] for k in K_list]
        ax.plot(K_list, ys, "o-", linewidth=2, markersize=6)
        ax.set_xlabel("K（平面波数 / 发射次数）")
        ax.set_ylabel(ylab)
        ax.set_title(f"{ylab}  ({note})")
        ax.grid(True, alpha=0.3)
        # 标出最大值对应的 K
        if key == "recovery":
            best_k = K_list[int(np.nanargmin(np.abs(np.array(ys) - 1.0)))]
            ax.axvline(best_k, color="gray", ls="--", alpha=0.5)
        else:
            best_k = K_list[int(np.nanargmax(ys))]
            ax.axvline(best_k, color="gray", ls="--", alpha=0.5)

    fig.suptitle("多角度 K 曲线：少角度 → 多角度的多指标变化", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = "K_curve.png"
    plt.savefig(out_png, dpi=150)
    print(f"\n已保存曲线图 {out_png}")

    # ===== 临界 K：找指标饱和的拐点（用「增量法」，避免 K=75 自己和自己比 PSNR=inf 的问题）=====
    # 判据：相邻 K 的 PSNR 增幅 < 阈值（默认 1 dB）即视为饱和。
    # 临界 K = 最后一个「相对前一个 K 提升 >= 1 dB」的 K。
    def find_critical_k(K_list, results, thresh_db=1.0):
        crit = K_list[0]
        for i in range(1, len(K_list)):
            k_prev, k_cur = K_list[i - 1], K_list[i]
            gain = results[k_cur]["psnr"] - results[k_prev]["psnr"]
            if gain >= thresh_db:
                crit = k_cur
            else:
                break
        return crit

    crit_k = find_critical_k(K_list, results, thresh_db=1.0)
    psnr_ref = results[61]["psnr"]  # 用 K=61 当"接近满配"的参照（K=75 是 inf）
    print(f"\n=== 临界 K 结论 ===")
    print(f"判据：相邻 K 的 PSNR 增幅 < 1 dB 视为饱和")
    print(f"K=61 的 PSNR = {psnr_ref:.3f} dB（K=75 为自己比自己，PSNR=inf 不参与判据）")
    print(f"临界 K = {crit_k}  →  用 K={crit_k}（约 1/{int(75/crit_k)} 的发射数据）即可逼近满配质量")
    print(f"相比满配，省下的发射次数 ≈ {75 - crit_k} 次")


if __name__ == "__main__":
    main()
