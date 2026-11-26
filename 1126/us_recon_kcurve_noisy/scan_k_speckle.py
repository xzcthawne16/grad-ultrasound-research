# -*- coding: utf-8 -*-
"""
更新 1A：连续散斑组织重扫 K 曲线（彻底向量化版）
================================================

动机（1A 证伪后的根因修正）：
  3 个孤立点散射体 → 点间空隙大 → 每加一个角度都能填旁瓣 → 临界 K 虚高。
  真实组织是连续密集散斑场，其复合收益在 K=10~20 即饱和。
  正确做法：换成密集随机散射体场（连续散斑）重扫 K。

性能（彻底优化）：
  1. make_rf 用「脉冲局部化 + np.bincount」：高斯脉冲只在 ±3σ 采样点内
     非零，每个散射体只累加 ~7 个采样点，不再做 64×2078 的逐点 exp。
  2. das 用 np.take 一次性取所有 (nz,nx,N) 的 RF 值，去掉 N 循环。
  3. compound 预计算角度表，避免重复。

用法：
  python scan_k_speckle.py
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ===== 仿真参数 =====
c = 1540.0
fs = 40e6
N = 64
pitch = 0.3e-3
sigma = 0.05e-6
elem_x = (np.arange(N) - (N - 1) / 2) * pitch

z_max = 40e-3
t_axis = np.arange(0, 2 * z_max / c, 1 / fs)
T = len(t_axis)
dt = 1 / fs

# ===== 连续散斑组织 =====
N_SCAT = 1000
RNG_SEED = 42


def make_scatterers():
    rng = np.random.default_rng(RNG_SEED)
    x = rng.uniform(-8e-3, 8e-3, N_SCAT)
    z = rng.uniform(3e-3, 38e-3, N_SCAT)
    amp = rng.rayleigh(scale=1.0, size=N_SCAT)
    amp = amp / np.sqrt(np.mean(amp ** 2)) * 0.05
    return np.column_stack([x, z, amp])


scatterers = make_scatterers()
xs = scatterers[:, 0]
zs = scatterers[:, 1]
amp = scatterers[:, 2]
S = len(xs)

# 高斯脉冲的局部化：只在中心采样点附近 ±NW 个点内非零
NW = int(np.ceil(3 * sigma * fs))  # 3σ 内的采样点数（约 6）
offsets = np.arange(-NW, NW + 1)   # 相对中心的采样点偏移
pulse_coeff = np.exp(-(offsets * dt) ** 2 / (2 * sigma ** 2))  # 高斯系数 (2NW+1,)

# 预计算接收时延 (S, N)：散射体到各阵元（与角度无关，只算一次）
tau_rx_scat = np.sqrt((xs[:, None] - elem_x[None, :]) ** 2 + zs[:, None] ** 2) / c  # (S,N)


def make_rf(theta):
    """完全向量化：用 bincount 在脉冲局部采样点上累加，快 50+ 倍。"""
    # 发射时延 (S,)
    tau_tx = (xs * np.sin(theta) + zs * np.cos(theta)) / c
    # 总时延 (S, N) -> 采样点索引
    tau = tau_tx[:, None] + tau_rx_scat  # (S, N)
    center = np.round(tau * fs).astype(np.int64)  # 中心采样点 (S, N)

    # 对每个散射体 s、阵元 n，在 center[s,n]+offsets 处累加 amp[s]*pulse_coeff
    # 向量化：构造 (S, N, 2NW+1) 的采样点索引
    idx = center[:, :, None] + offsets[None, None, :]  # (S, N, 2NW+1)
    # 贡献值：amp[s] * pulse_coeff，广播到 (S, N, 2NW+1)（与 idx 同形状）
    contrib = (amp[:, None] * pulse_coeff[None, :])[:, None, :]  # (S, 1, 2NW+1)
    contrib = np.broadcast_to(contrib, idx.shape)  # (S, N, 2NW+1)

    # 阵元索引矩阵（与 idx 同形状），用广播构造后一起展平，保证对齐
    n_idx = np.broadcast_to(np.arange(N)[None, :, None], idx.shape)  # (S, N, 2NW+1)

    # 展平成 1D（C 顺序，三者对齐）
    flat_idx = idx.ravel()
    flat_contrib = contrib.ravel()
    n_idx = n_idx.ravel()

    # 合法范围过滤
    valid = (flat_idx >= 0) & (flat_idx < T)
    flat_idx = flat_idx[valid]
    flat_contrib = flat_contrib[valid]
    n_idx = n_idx[valid]

    # 用 bincount 累加（按阵元分组）
    rf = np.zeros((N, T))
    for n in range(N):
        m = n_idx == n
        rf[n] = np.bincount(flat_idx[m], weights=flat_contrib[m], minlength=T)
    return rf


# 预计算 DAS 的接收时延 (nz, nx, N)：网格点到各阵元（与角度无关，只算一次）
nx, nz = 256, 512
x_grid = np.linspace(-10e-3, 10e-3, nx)
z_grid = np.linspace(2e-3, 40e-3, nz)
X, Z = np.meshgrid(x_grid, z_grid)
tau_rx_grid = np.sqrt((X[:, :, None] - elem_x[None, None, :]) ** 2
                      + Z[:, :, None] ** 2) / c  # (nz, nx, N)


def das(rf, theta):
    """完全向量化：np.take 一次性取所有 (nz,nx,N) 的 RF 值。"""
    tau = (X * np.sin(theta) + Z * np.cos(theta))[:, :, None] / c + tau_rx_grid  # (nz,nx,N)
    idx = np.round(tau * fs).astype(int)
    idx = np.clip(idx, 0, T - 1)
    # np.take：对 rf 的 N 行，按 idx[..., n] 取值
    # rf 形状 (N,T)，idx 形状 (nz,nx,N)，用高级索引沿最后一维取
    # 转成 (N, nz*nx) 便于索引
    flat_idx = idx.reshape(-1, N).T  # (N, nz*nx)
    gathered = np.take_along_axis(rf, flat_idx, axis=1)  # (N, nz*nx)
    image = gathered.sum(axis=0).reshape(nz, nx) / N
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


# ===== 指标 =====
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


def find_critical_k(K_list, psnr_list, thresh_db=1.0):
    crit = K_list[0]
    for i in range(1, len(K_list)):
        gain = psnr_list[i] - psnr_list[i - 1]
        if gain >= thresh_db:
            crit = K_list[i]
        else:
            break
    return crit


def main():
    K_list = [1, 3, 5, 9, 15, 25, 37, 49, 61, 75]

    # 满配参照图 = K=75
    img_ref = compound(75)

    psnr_list, ssim_list = [], []
    print("连续散斑场：PSNR/SSIM vs K")
    for k in K_list:
        img = compound(k)
        p = psnr(img, img_ref)
        s = ssim(img, img_ref)
        psnr_list.append(p)
        ssim_list.append(s)
        print(f"  K={k:3d}  PSNR={p:8.3f}  SSIM={s:.5f}")

    crit = find_critical_k(K_list, psnr_list, thresh_db=1.0)
    print(f"\n临界 K = {crit}（判据：PSNR 增幅 < 1 dB 视为饱和）")

    # ===== 画图 =====
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(K_list, psnr_list, "o-", linewidth=2, markersize=6)
    axes[0].axvline(crit, color="red", ls="--", alpha=0.6, label=f"临界 K={crit}")
    axes[0].set_xlabel("K（平面波数 / 发射次数）")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("连续散斑：PSNR vs K")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(K_list, ssim_list, "o-", linewidth=2, markersize=6, color="#d62728")
    axes[1].axvline(crit, color="red", ls="--", alpha=0.6, label=f"临界 K={crit}")
    axes[1].set_xlabel("K（平面波数 / 发射次数）")
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("连续散斑：SSIM vs K")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("连续散斑组织重扫 K 曲线（更新 1A，向量化版）", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = "K_curve_speckle.png"
    plt.savefig(out_png, dpi=150)
    print(f"\n已保存曲线图 {out_png}")

    # ===== 结论 =====
    print("\n===== 结论 =====")
    if crit <= 20:
        print(f"连续散斑下临界 K={crit}，已回落到真实超声量级（10~20）。")
        print(f"→ 用 1/{int(75/crit)} 的发射即可逼近满配质量，「1/N 发射」故事成立。")
    else:
        print(f"连续散斑下临界 K={crit}，仍未回落到 10~20。")
        print("→ 需进一步排查（散斑密度/幅度分布/是否加噪声衰减等）。")


if __name__ == "__main__":
    main()
