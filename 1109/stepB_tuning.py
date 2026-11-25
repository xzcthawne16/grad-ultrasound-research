# 主任务B：调优（固定同一批数据，对比 MSE / L1 / MSE+λSSIM 三个变式）
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time

c, fs, N, pitch, sigma = 1540.0, 40e6, 64, 0.3e-3, 0.05e-6
elem_x = (np.arange(N) - (N - 1) / 2) * pitch
t_axis = np.arange(0, 2 * 40e-3 / c, 1 / fs)
angles = np.deg2rad(np.array([-5, -2.5, 0, 2.5, 5]))

# 缓存距离项（加速 DAS）
nx, nz = 256, 512
x_grid = np.linspace(-10e-3, 10e-3, nx).astype(np.float32)
z_grid = np.linspace(2e-3, 40e-3, nz).astype(np.float32)
X, Z = np.meshgrid(x_grid, z_grid)
cache = {}
for th in angles:
    sin = np.sin(th); cos = np.cos(th)
    d = (X[:, :, None] * sin + Z[:, :, None] * cos) / c \
        + np.sqrt((X[:, :, None] - elem_x[None, None, :].astype(np.float32)) ** 2 + Z[:, :, None] ** 2) / c
    cache[th] = d.astype(np.float32)

def rand_sc(n):
    return np.array([(np.random.uniform(-8e-3, 8e-3),
                      np.random.uniform(5e-3, 35e-3),
                      np.random.uniform(0.3, 1.0)) for _ in range(n)])

def make_rf_vec(theta, sc):
    xk = sc[:, 0]; zk = sc[:, 1]; amp = sc[:, 2]
    tau = (xk[:, None] * np.sin(theta) + zk[:, None] * np.cos(theta)) / c \
        + np.sqrt((xk[:, None] - elem_x[None, :]) ** 2 + zk[:, None] ** 2) / c
    env = amp[:, None, None] * np.exp(-((t_axis[None, None, :] - tau[:, :, None]) ** 2) / (2 * sigma ** 2))
    return env.sum(axis=0)

def das_fast(rf, th):
    idx = np.clip(np.round(cache[th] * fs).astype(np.int64), 0, len(t_axis) - 1)
    return rf[np.arange(N), idx].sum(axis=2)

def disp_db(img):
    d = np.abs(img)
    return 20 * np.log10(d / d.max() + 1e-6)

def make_pair():
    sc = rand_sc(np.random.randint(3, 8))
    lo = das_fast(make_rf_vec(0.0, sc), 0.0)
    hi = np.zeros_like(lo)
    for th in angles:
        hi += das_fast(make_rf_vec(th, sc), th)
    hi /= len(angles)
    return disp_db(lo), disp_db(hi)

# ===== 固定种子生成同一批数据 =====
torch.manual_seed(0); np.random.seed(0)
Ntrain, Ntest = 160, 40
t0 = time.time()
Xtr, Ytr = [], []
for _ in range(Ntrain):
    lo, hi = make_pair(); Xtr.append(lo); Ytr.append(hi)
Xte, Yte = [], []
for _ in range(Ntest):
    lo, hi = make_pair(); Xte.append(lo); Yte.append(hi)
print(f"[数据] 200对 {time.time()-t0:.0f}s", flush=True)

def norm(x):
    return np.clip((x + 60) / 60.0, 0, 1)
Xtr = torch.tensor(norm(np.stack(Xtr))[:, None], dtype=torch.float32)
Ytr = torch.tensor(norm(np.stack(Ytr))[:, None], dtype=torch.float32)
Xte = torch.tensor(norm(np.stack(Xte))[:, None], dtype=torch.float32)
Yte = torch.tensor(norm(np.stack(Yte))[:, None], dtype=torch.float32)

# ===== 网络 =====
class ReconNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )
    def forward(self, x): return self.net(x)

# ===== 评价 =====
def gaussian_kernel(size=11, sigma=1.5):
    cc = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(cc**2) / (2*sigma**2)); g = g / g.sum()
    k = g[:, None] * g[None, :]
    return k.view(1, 1, size, size)
_W = gaussian_kernel()

def ssim(a, b, C1=1e-4, C2=9e-4):
    w = _W.to(a.device); pad = 5
    mu1 = F.conv2d(a, w, padding=pad); mu2 = F.conv2d(b, w, padding=pad)
    s1 = F.conv2d(a*a, w, padding=pad) - mu1**2
    s2 = F.conv2d(b*b, w, padding=pad) - mu2**2
    s12 = F.conv2d(a*b, w, padding=pad) - mu1*mu2
    m = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))
    return m.mean().item()

def psnr(a, b, maxv=1.0):
    return 10 * torch.log10(maxv**2 / F.mse_loss(a, b)).item()

def ssim_map(a, b, C1=1e-4, C2=9e-4, win=11):
    mu1 = F.avg_pool2d(a, win, 1, win//2)
    mu2 = F.avg_pool2d(b, win, 1, win//2)
    s1  = F.avg_pool2d(a*a, win, 1, win//2) - mu1**2
    s2  = F.avg_pool2d(b*b, win, 1, win//2) - mu2**2
    s12 = F.avg_pool2d(a*b, win, 1, win//2) - mu1*mu2
    num = (2*mu1*mu2 + C1) * (2*s12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)
    return num / (den + 1e-8)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_one(loss_fn, tag, epochs=60):
    """用给定 loss 训练一个全新网络，返回 pred（CPU tensor）。"""
    torch.manual_seed(0)
    model = ReconNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=16, shuffle=True)
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte.to(device)).cpu()
    p = psnr(pred, Yte)
    s = ssim(pred, Yte)
    print(f"[{tag}] PSNR={p:.2f} dB  SSIM={s:.4f}", flush=True)
    return p, s

print("\n===== 调优对比（同一批 Xte/Yte）=====", flush=True)

# baseline
p_base = psnr(Xte, Yte)
s_base = ssim(Xte, Yte)
print(f"[baseline 低质输入] PSNR={p_base:.2f} dB  SSIM={s_base:.4f}", flush=True)

# 变式0：MSE（复现 11/6）
p_mse, s_mse = train_one(nn.MSELoss(), "MSE")

# 变式A：L1
p_l1, s_l1 = train_one(nn.L1Loss(), "L1")

# 变式B：MSE + λ(1-SSIM)，试 λ=0.1
for lam in [0.1]:
    def combined(pred, target, lam=lam):
        return F.mse_loss(pred, target) + lam * (1 - ssim_map(pred, target).mean())
    p_cb, s_cb = train_one(combined, f"MSE+{lam}·SSIM")

print("\n===== 汇总表 =====")
print(f"{'方案':<22} {'PSNR(dB)':>10} {'SSIM':>8}")
print(f"{'baseline(低质输入)':<22} {p_base:>10.2f} {s_base:>8.4f}")
print(f"{'MSE(复现11/6)':<22} {p_mse:>10.2f} {s_mse:>8.4f}")
print(f"{'变式A L1':<22} {p_l1:>10.2f} {s_l1:>8.4f}")
print(f"{'变式B MSE+0.1SSIM':<22} {p_cb:>10.2f} {s_cb:>8.4f}")
