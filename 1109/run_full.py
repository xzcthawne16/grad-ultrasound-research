import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time

c, fs, N, pitch, sigma = 1540.0, 40e6, 64, 0.3e-3, 0.05e-6
elem_x = (np.arange(N) - (N - 1) / 2) * pitch
z_max = 40e-3
t_axis = np.arange(0, 2 * z_max / c, 1 / fs)
angles = np.deg2rad(np.array([-5, -2.5, 0, 2.5, 5]))

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

def das_vec(rf, theta, nx=256, nz=512):
    x_grid = np.linspace(-10e-3, 10e-3, nx)
    z_grid = np.linspace(2e-3, 40e-3, nz)
    X, Z = np.meshgrid(x_grid, z_grid)
    tau = (X[:, :, None] * np.sin(theta) + Z[:, :, None] * np.cos(theta)) / c \
        + np.sqrt((X[:, :, None] - elem_x[None, None, :]) ** 2 + Z[:, :, None] ** 2) / c
    idx = np.clip(np.round(tau * fs).astype(int), 0, len(t_axis) - 1)
    img = rf[np.arange(N), idx].sum(axis=2)
    return img, x_grid, z_grid

def disp_db(img):
    d = np.abs(img)
    return 20 * np.log10(d / d.max() + 1e-6)

def make_pair():
    sc = rand_sc(np.random.randint(3, 8))
    lo, _, _ = das_vec(make_rf_vec(0.0, sc), 0.0)
    rfs = np.stack([make_rf_vec(th, sc) for th in angles])
    hi = np.zeros_like(lo)
    for k, th in enumerate(angles):
        hi += das_vec(rfs[k], th)[0]
    hi /= len(angles)
    return disp_db(lo), disp_db(hi)

# ===== 数据生成 =====
torch.manual_seed(0); np.random.seed(0)
Ntrain, Ntest = 160, 40

t0 = time.time()
Xtr, Ytr = [], []
for _ in range(Ntrain):
    lo, hi = make_pair(); Xtr.append(lo); Ytr.append(hi)
Xte, Yte = [], []
for _ in range(Ntest):
    lo, hi = make_pair(); Xte.append(lo); Yte.append(hi)
print(f"[数据生成] 200 对耗时 {time.time()-t0:.1f} 秒")

def norm(x):
    return np.clip((x + 60) / 60.0, 0, 1)

Xtr = torch.tensor(norm(np.stack(Xtr))[:, None], dtype=torch.float32)
Ytr = torch.tensor(norm(np.stack(Ytr))[:, None], dtype=torch.float32)
Xte = torch.tensor(norm(np.stack(Xte))[:, None], dtype=torch.float32)
Yte = torch.tensor(norm(np.stack(Yte))[:, None], dtype=torch.float32)
print("train", Xtr.shape, "test", Xte.shape)

# ===== 网络 + 训练 =====
class ReconNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )
    def forward(self, x):
        return self.net(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("设备:", device)
model = ReconNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
ds = TensorDataset(Xtr, Ytr)
dl = DataLoader(ds, batch_size=16, shuffle=True)

t0 = time.time()
for ep in range(60):
    model.train()
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()
    if ep % 20 == 0:
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xte.to(device)), Yte.to(device)).item()
        print(f"epoch {ep}: train={loss.item():.4f} val={v:.4f}")
print(f"[训练] 60 epoch 耗时 {time.time()-t0:.1f} 秒")

model.eval()
with torch.no_grad():
    pred = model(Xte.to(device)).cpu()

# ===== SSIM =====
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

print("\n===== 最终结果 =====")
print("baseline SSIM(低质 vs 高质) =", ssim(Xte, Yte))
print("DL重建  SSIM(重建 vs 高质) =", ssim(pred, Yte))
