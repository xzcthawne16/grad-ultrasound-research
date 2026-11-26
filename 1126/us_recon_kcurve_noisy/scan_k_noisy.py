# -*- coding: utf-8 -*-
"""
加噪声/深度衰减 重扫 K 曲线 + 2×2 消融（今天主任务 1A）
=====================================================

背景（昨天结论 → 今天的动机）：
  昨天自仿真（无噪声、无衰减）下，PSNR 随 K 持续上升，临界 K≈49，
  远大于真实超声的 10~20。原因：每加一个角度都有真实增益，没有
  「电子噪声」和「深度衰减」这两道真实物理损耗。

今天验证：加上电子噪声 + 深度衰减后，临界 K 是否回落到 10~20。

2×2 消融（定位「谁主导饱和提前」）：
  A. 无噪声 + 无衰减（昨天基线，复现）
  B. 只加噪声（SNR 可调，默认 30 dB）
  C. 只加深度衰减（alpha=0.5 dB/cm/MHz）
  D. 噪声 + 衰减 都加

指标：只采信 PSNR / SSIM（昨天已判点恢复率/gCNR 参照系有误，不采信）。

用法：
  python scan_k_noisy.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ===== 仿真参数（复用昨天） =====
c = 1540.0            # 声速 m/s
fs = 40e6             # 采样频率
N = 64                # 阵元数
pitch = 0.3e-3        # 阵元间距
sigma = 0.05e-6       # 高斯脉冲宽度（秒）
elem_x = (np.arange(N) - (N - 1) / 2) * pitch

# 固定散射体（3 个点，含深点看衰减效果）
scatterers = np.array([
    [0.0e-3, 15e-3, 1.0],   # 点1：中心浅点
    [3.0e-3, 20e-3, 0.7],   # 点2：偏右中深
    [-3.0e-3, 25e-3, 0.5],  # 点3：偏左更深（弱，衰减后更弱）
])

z_max = 40e-3
t_axis = np.arange(0, 2 * z_max / c, 1 / fs)

# ===== 物理损耗参数 =====
# 中心频率：你的高斯脉冲宽度 sigma=0.05us，中心频率约 f0 ≈ 1/(2*pi*sigma) 附近，
# 实际超声常用 3~7 MHz，这里取 5 MHz 作典型值（alpha 的单位是 dB/cm/MHz）。
f0 = 5.0e6            # 中心频率 Hz
ALPHA_DB = 0.5        # 软组织衰减系数 dB/cm/MHz（典型值 0.3~0.7，取 0.5）


def attenuation_linear(z):
    """
    深度 z（单位 m）处的线性幅度衰减因子。
    alpha(dB/cm/MHz) -> 线性：exp(-alpha_linear * z)，其中
    alpha_linear = alpha_dB * f0(MHz) * (ln10/20) * 100 (1/m)。
    推导：总衰减 dB = alpha_dB * f0[MHz] * z[cm]
         幅度衰减 = 10^(-dB/20) = exp(-dB * ln10 / 20)
    """
    z_cm = z * 100.0                       # m -> cm
    f0_mhz = f0 / 1e6                       # Hz -> MHz
    att_db = ALPHA_DB * f0_mhz * z_cm       # 总衰减 dB
    return 10.0 ** (-att_db / 20.0)

# 物理含义：超声在组织里越传越弱，衰减系数 alpha 的单位是 dB/cm/MHz
# 「每走 1cm、每 1MHz 频率，衰减多少 dB」。你的三个点里，点3（25mm）比点1（15mm）深，所以衰减后点3 变得更弱。
# ​这模拟的是真实超声里「深部组织信号弱」


# ===== 生成 RF（可开关噪声/衰减） =====
def make_rf(theta, add_noise=False, snr_db=30.0, add_attenuation=False, seed=0):
    """
    生成多角度发射的 RF。可叠加电子噪声和深度衰减。
    theta：发射角（弧度）
    add_noise：是否加电子噪声
    snr_db：信噪比（dB），噪声幅度据此反推
    add_attenuation：是否加深度衰减
    """
    rf = np.zeros((N, len(t_axis)))
    for xk, zk, amp in scatterers:
        # 深度衰减：把散射体幅度按深度削弱
        amp_eff = amp * attenuation_linear(zk) if add_attenuation else amp
        for i in range(N):
            tau = (xk * np.sin(theta) + zk * np.cos(theta)) / c \
                + np.sqrt((xk - elem_x[i]) ** 2 + zk ** 2) / c
            rf[i] += amp_eff * np.exp(-((t_axis - tau) ** 2) / (2 * sigma ** 2))

    if add_noise:
        rng = np.random.default_rng(seed)
        # 噪声幅度：以「最强散射体的信号峰值」为参照，按 SNR 反推噪声 std
        # SNR_db = 20*log10(A_signal / sigma_noise)  （幅度域定义）
        # 取点1（amp=1，浅点，无衰减影响最大）的信号峰值作参照
        A_signal = 1.0  # 最强点幅度
        sigma_noise = A_signal / (10 ** (snr_db / 20.0))
        rf += rng.normal(0.0, sigma_noise, size=rf.shape)
    return rf

# 以最强点（幅度 1.0）为参照，按 30dB 反推噪声标准差，叠加高斯白噪声。这模拟的是探头电子噪声


# ===== 带角度的 DAS =====
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
    return image


def compound(K, theta_max_deg=16.0, **rf_kwargs):
    """
    K 个角度相干复合。rf_kwargs 透传给 make_rf（控制噪声/衰减）。
    """
    if K <= 1:
        angles = np.deg2rad([0.0])
    else:
        angles = np.deg2rad(np.linspace(-theta_max_deg, theta_max_deg, K))
    img = np.zeros((nz, nx))
    for th in angles:
        img += das(make_rf(th, **rf_kwargs), th)
    img /= len(angles)
    return img


# ===== 指标（只采信 PSNR/SSIM） =====
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
    """增量法找临界 K：最后一个「相对前一个 K 的 PSNR 增幅 >= thresh_db」的 K。"""
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

    # 2×2 消融：四组（标签, 是否加噪, 是否加衰减）
    # 注意：参照图（满配 K=75）也必须用「同样物理条件」生成，否则比较不公平。
    groups = [
        ("A 无噪声无衰减（基线）", False, False),
        ("B 只加噪声 SNR=30dB",     True,  False),
        ("C 只加深度衰减",          False, True),
        ("D 噪声+衰减 都加",        True,  True),
    ]

    all_results = {}
    for gname, add_noise, add_atten in groups:
        rf_kwargs = dict(add_noise=add_noise, snr_db=30.0,
                         add_attenuation=add_atten)
        # 满配参照图：K=75，同一物理条件
        img_ref = compound(75, **rf_kwargs)
        img_ref_norm = img_ref / (np.abs(img_ref).max() + 1e-8)

        psnr_list, ssim_list = [], []
        print(f"\n===== {gname} =====")
        for k in K_list:
            img = compound(k, **rf_kwargs)
            img_norm = img / (np.abs(img).max() + 1e-8)
            p = psnr(img_norm, img_ref_norm)
            s = ssim(img_norm, img_ref_norm)
            psnr_list.append(p)
            ssim_list.append(s)
            print(f"  K={k:3d}  PSNR={p:8.3f}  SSIM={s:.5f}")

        crit = find_critical_k(K_list, psnr_list, thresh_db=1.0)
        all_results[gname] = {"psnr": psnr_list, "ssim": ssim_list, "crit_k": crit}
        print(f"  >> 临界 K = {crit}（判据：PSNR 增幅 < 1 dB 视为饱和）")

    # ===== 画图：四组 PSNR 对比 + 四组 SSIM 对比 =====
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = ["#333333", "#d62728", "#1f77b4", "#2ca02c"]

    for gname, c in zip(groups, colors):
        label = gname[0]
        axes[0].plot(K_list, all_results[gname]["psnr"], "o-", color=c,
                     label=f"{label}", linewidth=2, markersize=5)
        axes[1].plot(K_list, all_results[gname]["ssim"], "o-", color=c,
                     label=f"{label}", linewidth=2, markersize=5)

    axes[0].set_xlabel("K（平面波数 / 发射次数）")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR vs K：噪声/衰减对临界 K 的影响")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel("K（平面波数 / 发射次数）")
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM vs K")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.suptitle("加噪声/深度衰减重扫 K 曲线（2×2 消融）", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_png = "K_curve_noisy.png"
    plt.savefig(out_png, dpi=150)
    print(f"\n已保存曲线图 {out_png}")

    # ===== 汇总临界 K 结论 =====
    print("\n===== 临界 K 汇总（判据：PSNR 增幅 < 1 dB）=====")
    for gname, _, _ in groups:
        print(f"  {gname[0]}：临界 K = {all_results[gname]['crit_k']}")
    base_crit = all_results["A 无噪声无衰减（基线）"]["crit_k"]
    d_crit = all_results["D 噪声+衰减 都加"]["crit_k"]
    print(f"\n结论：基线临界 K={base_crit}，加噪+衰减后临界 K={d_crit}，"
          f"{'回落' if d_crit < base_crit else '未回落'}至真实超声量级（10~20）")
    print(f"（注：K=75 为参照图本身，PSNR=inf，不参与判据；用增量法）")


if __name__ == "__main__":
    main()
