# -*- coding: utf-8 -*-
"""
自仿真超声「低质图 -> 高质图」重建：训练 UNet + 推理 + 双轴指标回测
一屏跑通：造配对数据 -> 训练 -> 推理测试图 -> 算 CNR/点恢复率/消声区标准差
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ================= 全局仿真参数（与 notebook 一致）=================
c = 1540.0
fs = 40e6
N = 64
pitch = 0.3e-3
sigma = 0.05e-6
elem_x = (np.arange(N) - (N - 1) / 2) * pitch

nx, nz = 256, 512
x_grid = np.linspace(-10e-3, 10e-3, nx)
z_grid = np.linspace(2e-3, 40e-3, nz)
X, Z = np.meshgrid(x_grid, z_grid)
z_max = 40e-3
t_axis = np.arange(0, 2 * z_max / c, 1 / fs)

# 消声区（固定，用于 CNR 背景）
anechoic_region = {"x_range": (-8e-3, -4e-3), "z_range": (10e-3, 18e-3)}
anechoic_mask = (X >= anechoic_region["x_range"][0]) & (X <= anechoic_region["x_range"][1]) \
              & (Z >= anechoic_region["z_range"][0]) & (Z <= anechoic_region["z_range"][1])


def make_low_image(scatterers):
    """K=1 单角度仿真 -> 低质图（含伪影）"""
    rf = np.zeros((N, len(t_axis)))
    for xk, zk, amp in scatterers:
        tau = (zk + np.sqrt((xk - elem_x) ** 2 + zk ** 2)) / c
        rf += amp * np.exp(-((t_axis[None, :] - tau[:, None]) ** 2) / (2 * sigma ** 2))
    tau_img = (Z[:, :, None] + np.sqrt((X[:, :, None] - elem_x[None, None, :]) ** 2
                                       + Z[:, :, None] ** 2)) / c
    idx = np.clip(np.round(tau_img * fs).astype(int), 0, len(t_axis) - 1)
    return rf[np.arange(N)[None, None, :], idx].sum(axis=2)


def make_ideal_image(scatterers, sigma_pt=0.3e-3):
    """画理想高质图：每个真点 = 一个高斯点，无伪影"""
    img = np.zeros((nz, nx))
    for xk, zk, amp in scatterers:
        dx = X - xk
        dz = Z - zk
        img += amp * np.exp(-(np.square(dx) + np.square(dz)) / (2.0 * sigma_pt * sigma_pt))
    return img


def random_scatterers(n_pts=3):
    """随机摆放 n_pts 个真点（避开消声区），返回 [x,z,amp] 数组"""
    pts = []
    while len(pts) < n_pts:
        xk = np.random.uniform(-8e-3, 8e-3)
        zk = np.random.uniform(6e-3, 36e-3)
        # 避开消声区
        if (-8e-3 <= xk <= -4e-3) and (10e-3 <= zk <= 18e-3):
            continue
        amp = np.random.uniform(0.5, 1.0)
        pts.append([xk, zk, amp])
    return np.array(pts)


def gen_pair():
    """生成一对 (低质图, 高质图)，并做归一化到 [0,1]"""
    sc = random_scatterers()
    low = make_low_image(sc)
    ideal = make_ideal_image(sc)
    # 各自除以最大值归一化（保持相对结构）
    low = low / (np.abs(low).max() + 1e-8)
    ideal = ideal / (np.abs(ideal).max() + 1e-8)
    return low, ideal


# ================= UNet（复用昨天骨架，简化版）=================
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=16):
        super().__init__()
        def blk(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(True),
                                 nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(True))
        self.e1 = blk(in_ch, base); self.p = nn.MaxPool2d(2)
        self.e2 = blk(base, base*2)
        self.e3 = blk(base*2, base*4)
        self.b = blk(base*4, base*8)
        self.u3 = nn.Sequential(nn.Conv2d(base*8, base*4*4, 1), nn.PixelShuffle(2)); self.d3 = blk(base*8, base*4)
        self.u2 = nn.Sequential(nn.Conv2d(base*4, base*2*4, 1), nn.PixelShuffle(2)); self.d2 = blk(base*4, base*2)
        self.u1 = nn.Sequential(nn.Conv2d(base*2, base*4, 1), nn.PixelShuffle(2)); self.d1 = blk(base*2, base)
        self.out = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.p(e1)); e3 = self.e3(self.p(e2)); b = self.b(self.p(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


# ================= 主流程 =================
if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42)

    # 1. 造训练集
    n_train = 200
    print(f"造 {n_train} 对训练数据 ...")
    lows, ideals = [], []
    for _ in range(n_train):
        lo, hi = gen_pair()
        lows.append(lo); ideals.append(hi)
    lows = np.array(lows)[:, None, :, :].astype(np.float32)
    ideals = np.array(ideals)[:, None, :, :].astype(np.float32)

    # 2. 造测试集（固定 3 个真点：你 notebook 里的那组）
    test_sc = np.array([[0.0e-3, 12e-3, 1.0], [0.0e-3, 25e-3, 0.9], [0.0e-3, 34e-3, 0.8]])
    test_low = make_low_image(test_sc)
    test_low = test_low / (np.abs(test_low).max() + 1e-8)
    test_ideal = make_ideal_image(test_sc)
    test_ideal = test_ideal / (np.abs(test_ideal).max() + 1e-8)

    # 3. 训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()
    print(f"设备 {device}，开始训练 100 epoch ...")
    Xt = torch.from_numpy(lows).to(device)
    Yt = torch.from_numpy(ideals).to(device)
    bs = 16
    for ep in range(100):
        model.train()
        perm = torch.randperm(n_train)
        total = 0.0
        for i in range(0, n_train, bs):
            idx = perm[i:i+bs]
            xb = Xt[idx]; yb = Yt[idx]
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward(); opt.step()
            total += loss.item()
        if ep % 10 == 0 or ep == 99:
            print(f"  epoch {ep+1:3d}/100  loss={total/(n_train//bs):.5f}")

    # 4. 推理测试图
    model.eval()
    with torch.no_grad():
        tin = torch.from_numpy(test_low[None, None]).float().to(device)
        rec = model(tin).cpu().numpy()[0, 0]
    rec = np.clip(rec, 0, None)
    rec = rec / (np.abs(rec).max() + 1e-8)

    # 5. 算双轴指标
    def pt_mask(x0, z0, hw=1.0e-3, hh=1.5e-3):
        return (np.abs(X - x0) <= hw) & (np.abs(Z - z0) <= hh)

    def peak(img, m):
        return np.abs(img[m]).max()

    # 点目标恢复率（对每个真点）
    print("\n=== 双轴回测结果 ===")
    recovs = []
    for i, (xk, zk, amp) in enumerate(test_sc):
        m = pt_mask(xk, zk)
        pk_low = peak(test_low, m)
        pk_rec = peak(rec, m)
        pk_ideal = peak(test_ideal, m)
        r = pk_rec / pk_ideal
        recovs.append(r)
        print(f"  真点{i+1}(z={zk*1e3:.0f}mm): 低质峰值={pk_low:.3f} DL峰值={pk_rec:.3f} "
              f"理想峰值={pk_ideal:.3f} 恢复率={r:.3f}")

    # 消声区 CNR（抗伪影）
    def cnr_peak(img):
        m1 = pt_mask(test_sc[0][0], test_sc[0][1])
        mu_in = peak(img, m1)
        mu_bg = img[anechoic_mask].mean()
        sg_bg = img[anechoic_mask].std()
        return (mu_in - mu_bg) / (sg_bg + 1e-8), sg_bg

    cnr_low, sg_low = cnr_peak(test_low)
    cnr_rec, sg_rec = cnr_peak(rec)
    print(f"\n  消声区标准差: 低质={sg_low:.4f} -> DL重建={sg_rec:.4f}  (下降=抗伪影有效)")
    print(f"  CNR(真点1 vs 消声区): 低质={cnr_low:.2f} -> DL重建={cnr_rec:.2f}")

    # 6. 出三张图对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    titles = ["低质图(K=1)", "DL 重建图", "理想高质图"]
    imgs = [test_low, rec, test_ideal]
    for ax, im, t in zip(axes, imgs, titles):
        disp = 20*np.log10(np.abs(im)/np.abs(im).max()+1e-6)
        ax.imshow(disp, extent=[x_grid[0]*1e3, x_grid[-1]*1e3, z_grid[-1]*1e3, z_grid[0]*1e3],
                  cmap="gray", aspect="auto", vmin=-40, vmax=0)
        ax.set_title(t); ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
    plt.tight_layout()
    plt.savefig("recon_3图对比.png", dpi=150)
    print("\n已保存 recon_3图对比.png")
