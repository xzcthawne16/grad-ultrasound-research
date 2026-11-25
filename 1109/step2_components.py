# 主任务A第2步：SSIM 三分量分解（亮度l/对比度c/结构s）
import numpy as np
import torch
import torch.nn.functional as F

# 复用 run_full.py 的产物：需要重新生成数据+训练，或直接读入。
# 为了独立可跑，这里重新走一遍（数据生成已缓存过，但为简化直接重算）

c, fs, N, pitch, sigma = 1540.0, 40e6, 64, 0.3e-3, 0.05e-6
elem_x = (np.arange(N) - (N - 1) / 2) * pitch
t_axis = np.arange(0, 2 * 40e-3 / c, 1 / fs)
angles = np.deg2rad(np.array([-5, -2.5, 0, 2.5, 5]))

nx, nz = 256, 512
x_grid = np.linspace(-10e-3, 10e-3, nx).astype(np.float32)
z_grid = np.linspace(2e-3, 40e-3, nz).astype(np.float32)
X, Z = np.meshgrid(x_grid, z_grid)

# 缓存距离项
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

import time, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ReconNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
dl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=16, shuffle=True)

for ep in range(60):
    model.train()
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()

model.eval()
with torch.no_grad():
    pred = model(Xte.to(device)).cpu()

# ===== 三分量分解 SSIM =====
def gaussian_kernel(size=11, sigma=1.5):
    cc = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(cc**2) / (2*sigma**2)); g = g / g.sum()
    k = g[:, None] * g[None, :]
    return k.view(1, 1, size, size)

_W = gaussian_kernel()
def ssim_parts(a, b, C1=1e-4, C2=9e-4):
    w = _W.to(a.device); pad = 5
    mu1 = F.conv2d(a, w, padding=pad); mu2 = F.conv2d(b, w, padding=pad)
    var1 = F.conv2d(a*a, w, padding=pad) - mu1**2
    var2 = F.conv2d(b*b, w, padding=pad) - mu2**2
    cov12 = F.conv2d(a*b, w, padding=pad) - mu1*mu2
    var1 = var1.clamp(min=0); var2 = var2.clamp(min=0)   # 防止数值误差导致负方差
    sig1 = torch.sqrt(var1); sig2 = torch.sqrt(var2)

    # 标准 SSIM 三分量（对整图逐像素算完后取均值）
    l = (2*mu1*mu2 + C1) / (mu1**2 + mu2**2 + C1)                       # 亮度 l
    c_cmp = (2*sig1*sig2 + C2) / (var1 + var2 + C2)                     # 对比度 c
    s_cmp = (cov12 + C2/2) / (sig1*sig2 + C2/2)                         # 结构 s

    def m(x): return x.mean().item()
    return m(l), m(c_cmp), m(s_cmp), m(l * c_cmp * s_cmp)

print("\n===== SSIM 三分量分解 =====")
l_b, c_b, s_b, full_b = ssim_parts(Xte, Yte)
l_d, c_d, s_d, full_d = ssim_parts(pred, Yte)
print(f"{'':>12} {'亮度l':>10} {'对比度c':>10} {'结构s':>10} {'整体SSIM':>10}")
print(f"{'baseline':>12} {l_b:>10.4f} {c_b:>10.4f} {s_b:>10.4f} {full_b:>10.4f}")
print(f"{'DL重建':>12} {l_d:>10.4f} {c_d:>10.4f} {s_d:>10.4f} {full_d:>10.4f}")
print(f"{'变化':>12} {l_d-l_b:>+10.4f} {c_d-c_b:>+10.4f} {s_d-s_b:>+10.4f} {full_d-full_b:>+10.4f}")
