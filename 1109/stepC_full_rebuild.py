# -*- coding: utf-8 -*-
"""
主任务C 完整独立脚本：PICMUS 低质图 -> DL 重建 -> 高质图对比
从头到尾自己完成，不依赖 notebook 变量。
运行后输出：
  1. 定量指标（PSNR / SSIM）
  2. 整图重建对比图（保存为 PNG，可直接汇报）
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')            # 无窗口环境也能出图
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import hdf5plugin                 # 必须先 import
import h5py
import torch
import torch.nn as nn

# ===================== 1. 读数据 =====================
p = r'E:\develop\Graduate_AI\2026-11\1109\picmus_data\resolution_distorsion_simu_dataset_rf.hdf5'
f = h5py.File(p, 'r')
raw = f['tracks/track_0/data/raw_data'][:]       # (1, 75, 1527, 128, 1)
pa  = f['tracks/track_0/scan/polar_angles'][:]   # 75 角度（弧度）
f.close()

# ===================== 2. 参数 =====================
fs = 20.832e6
c  = 1540.0
N  = 128
pitch = 0.3e-3
elem_x = (np.arange(N) - (N - 1) / 2) * pitch

nz, nx = 600, 400
z_grid = np.linspace(6e-3, 55e-3, nz)
x_grid = np.linspace(-19.05e-3, 19.05e-3, nx)
X, Z = np.meshgrid(x_grid, z_grid)

def das(rf_e, theta):
    T = rf_e.shape[1]
    tau = (X[:, :, None] * np.sin(theta) + Z[:, :, None] * np.cos(theta)) / c \
        + np.sqrt((X[:, :, None] - elem_x[None, None, :]) ** 2 + Z[:, :, None] ** 2) / c
    idx = np.clip(np.round(tau * fs).astype(np.int64), 0, T - 1)
    return rf_e[np.arange(N)[None, None, :], idx].sum(axis=2)

# ===================== 3. 低质图 + 高质图 =====================
print('计算低质图（0° 单平面波）...')
rf_0 = raw[0, 37, :, :, 0].T
img_low = das(rf_0, pa[37])

print('计算高质图（75 角度相干复合）...')
hi = np.zeros((nz, nx), dtype=np.float64)
for k in range(len(pa)):
    rf_k = raw[0, k, :, :, 0].T
    hi += das(rf_k, pa[k])
hi /= len(pa)
print('高质图完成')

# ===================== 4. 切片造配对 =====================
def norm01(img):
    # 超声图归一化：除以绝对最大值，背景(接近0)保持0，强散射点才是1
    # 不用减 min，否则会把"接近0的背景"抬到中间灰度
    mx = np.abs(img).max()
    return img / mx if mx > 0 else img

img_low_n  = norm01(img_low)
img_hi_n   = norm01(hi)

patch, stride = 64, 32
nz_w = (nz - patch) // stride + 1
nx_w = (nx - patch) // stride + 1

X = np.zeros((nz_w * nx_w, 1, patch, patch), dtype=np.float32)
Y = np.zeros((nz_w * nx_w, 1, patch, patch), dtype=np.float32)
k = 0
for i in range(nz_w):
    for j in range(nx_w):
        z0, x0 = i * stride, j * stride
        X[k, 0] = img_low_n[z0:z0+patch, x0:x0+patch]
        Y[k, 0] = img_hi_n[z0:z0+patch, x0:x0+patch]
        k += 1
print(f'训练样本对：{k} 对')

# ===================== 5. 网络 =====================
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.enc1 = self._b(in_ch, base)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = self._b(base, base*2)
        self.enc3 = self._b(base*2, base*4)
        self.enc4 = self._b(base*4, base*8)
        self.bottleneck = self._b(base*8, base*16)
        self.up4 = nn.Sequential(nn.Conv2d(base*16, base*8*4, 1), nn.PixelShuffle(2))
        self.dec4 = self._b(base*16, base*8)
        self.up3 = nn.Sequential(nn.Conv2d(base*8, base*4*4, 1), nn.PixelShuffle(2))
        self.dec3 = self._b(base*8, base*4)
        self.up2 = nn.Sequential(nn.Conv2d(base*4, base*2*4, 1), nn.PixelShuffle(2))
        self.dec2 = self._b(base*4, base*2)
        self.up1 = nn.Sequential(nn.Conv2d(base*2, base*4, 1), nn.PixelShuffle(2))
        self.dec1 = self._b(base*2, base)
        self.out = nn.Conv2d(base, out_ch, 1)
    def _b(self, i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = UNet().to(device)
print('参数量（百万）:', sum(p.numel() for p in net.parameters()) / 1e6)

# ===================== 6. 训练 =====================
X_t = torch.from_numpy(X).float()
Y_t = torch.from_numpy(Y).float()
n = X_t.shape[0]
idx = np.random.permutation(n)
n_train = int(n * 0.8)
train_idx, val_idx = idx[:n_train], idx[n_train:]

X_train, Y_train = X_t[train_idx].to(device), Y_t[train_idx].to(device)
X_val, Y_val = X_t[val_idx].to(device), Y_t[val_idx].to(device)

loss_fn = nn.L1Loss()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
batch = 16
epochs = 300

print('开始训练...')
for epoch in range(1, epochs + 1):
    net.train()
    perm = torch.randperm(len(X_train))
    for i in range(0, len(X_train), batch):
        bi = perm[i:i+batch]
        pred = net(X_train[bi])
        loss = loss_fn(pred, Y_train[bi])
        opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 60 == 0:
        net.eval()
        with torch.no_grad():
            vl = loss_fn(net(X_val), Y_val).item()
        print(f'Epoch {epoch:3d} | val L1 {vl:.6f}')

# ===================== 7. 定量指标 =====================
from math import log10
def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 10 * log10(1.0 / mse) if mse > 0 else float('inf')
def ssim(a, b, C1=1e-4, C2=9e-4):
    a = a.astype(np.float64); b = b.astype(np.float64)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = np.mean((a - ma) * (b - mb))
    l = (2*ma*mb + C1) / (ma**2 + mb**2 + C1)
    c = (2*np.sqrt(va)*np.sqrt(vb) + C2) / (va + vb + C2)
    s = (cov + C2/2) / (np.sqrt(va)*np.sqrt(vb) + C2/2)
    return l * c * s

net.eval()
psnr_rec, ssim_rec, psnr_low, ssim_low = [], [], [], []
with torch.no_grad():
    for i in range(len(X_val)):
        rec = net(X_val[i:i+1])[0, 0].cpu().numpy()
        high = Y_val[i, 0].cpu().numpy()
        low = X_val[i, 0].cpu().numpy()
        psnr_rec.append(psnr(rec, high)); ssim_rec.append(ssim(rec, high))
        psnr_low.append(psnr(low, high)); ssim_low.append(ssim(low, high))

print('\n===== 定量对比（验证集均值）=====')
print(f'低质 baseline : PSNR {np.mean(psnr_low):.2f} dB | SSIM {np.mean(ssim_low):.4f}')
print(f'DL 重建       : PSNR {np.mean(psnr_rec):.2f} dB | SSIM {np.mean(ssim_rec):.4f}')
print(f'PSNR 提升 = {np.mean(psnr_rec)-np.mean(psnr_low):+.2f} dB')
print(f'SSIM 变化 = {np.mean(ssim_rec)-np.mean(ssim_low):+.4f}')

# ===================== 8. 整图重建 + 出图 =====================
rec_full = np.zeros((nz, nx), dtype=np.float64)
cnt_full = np.zeros((nz, nx), dtype=np.float64)
with torch.no_grad():
    for i in range(0, nz - patch + 1, stride):
        for j in range(0, nx - patch + 1, stride):
            blk = img_low_n[i:i+patch, j:j+patch]
            x = torch.from_numpy(blk[None, None]).float().to(device)
            out = net(x)[0, 0].cpu().numpy()
            rec_full[i:i+patch, j:j+patch] += out
            cnt_full[i:i+patch, j:j+patch] += 1.0
rec_full /= np.maximum(cnt_full, 1.0)

# 显示：超声标准 dB 显示。用一个统一的参考值做 0dB，固定 -50~0 dB 窗口，
# 这样背景（接近0的暗区）显示为黑，强散射点显示为白，层次正确。
def disp_db_fixed(img, ref, dmin=-50.0):
    d = np.abs(img) + 1e-12
    db = 20 * np.log10(d / ref)     # 统一除以同一个 ref，不各自除以 max
    return np.clip(db, dmin, 0.0)

# ref 取高质图的最大幅值，作为 0dB 参考
ref = np.abs(hi).max()

# 低质、高质用原始 DAS 值；重建图是网络输出（已归一化到[0,1]），
# 要映射回原始幅值尺度：乘以同样的 ref 再算 dB
rec_scaled = rec_full * ref

imgs = [disp_db_fixed(img_low, ref), disp_db_fixed(rec_scaled, ref), disp_db_fixed(hi, ref)]
titles = ['低质整图（0°单平面波）', 'DL 重建整图', '高质整图（75角度复合）']

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
for ax, data, t in zip(axes, imgs, titles):
    ax.imshow(data, cmap='gray', aspect='auto', vmin=-50, vmax=0)
    ax.set_title(t, fontsize=14)
    ax.set_xlabel('横向 x'); ax.set_ylabel('深度 z')
plt.tight_layout()
out_png = r'E:\develop\Graduate_AI\2026-11\1109\stepC_rebuild_result.png'
plt.savefig(out_png, dpi=130, bbox_inches='tight')
print('\n整图对比已保存到：', out_png)
print('脚本全部完成！')
