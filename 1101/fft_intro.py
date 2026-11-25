"""
FFT 入门（第 3 块）：单频整周期 + 两频实算 + 幅值换算

用法：
  .venv\Scripts\activate
  python fft_intro.py

预期输出：
  单频 2Hz：峰在 2.0 Hz
  单频幅值换算后：≈ 1.0
  两频：2Hz 峰 ≈ 1.0，5Hz 峰 ≈ 0.5
"""

import numpy as np
import matplotlib.pyplot as plt

print('=' * 60)
print('第一部分：单频 2Hz、整周期（先看峰在哪）')
print('=' * 60)

fs = 100            # 采样率：每秒采 100 个点
f0 = 2              # 信号频率 2 Hz
T = 1.0             # 观察 1 秒
N = int(fs * T)     # 100 个点
t = np.arange(N) / fs              # 0, 0.01, ..., 0.99
x = np.sin(2 * np.pi * f0 * t)     # 2Hz，1 秒 = 2 个整周期

X = np.fft.rfft(x)                 # 复数结果，长度 N//2+1 = 51
freqs = np.fft.rfftfreq(N, d=1/fs)  # 频率轴：0,1,...,50
mag = np.abs(X)                    # 相对大小（未归一化）

print(f'采样点数 N = {N}')
print(f'rfft 结果长度 = {len(X)}  （= N//2+1）')
print(f'频率轴范围 = {freqs[0]:.0f} ~ {freqs[-1]:.0f} Hz，间隔 = {freqs[1]:.3f} Hz')

peak = int(np.argmax(mag))
print(f'峰所在频率 = {freqs[peak]} Hz')
print(f'该峰相对大小 = {mag[peak]:.1f}  （未归一化，真实幅值要看第二部分）')

plt.figure(figsize=(8, 3))
plt.stem(freqs, mag, basefmt=' ')
plt.xlabel('freq (Hz)')
plt.ylabel('|X| (相对大小)')
plt.title('2Hz 整周期信号频谱（峰在 2Hz）')
plt.tight_layout()
plt.show()

print()
print('=' * 60)
print('第二部分：幅值换算（mag * 2 / N）')
print('=' * 60)

amp = mag * 2 / N
print(f'2Hz 峰换算后幅值 = {amp[peak]:.3f}  （真实幅值应为 1）')
print('说明：非直流、非奈奎斯特的格子，幅值 = mag * 2 / N')

print()
print('=' * 60)
print('第三部分：两频实算（接 10/31 的预测）')
print('=' * 60)

fs, T = 100, 2.0
t = np.arange(0, T, 1/fs)         # 2 秒，200 点
s = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)

X2 = np.fft.rfft(s)
f2 = np.fft.rfftfreq(len(s), d=1/fs)
amp2 = np.abs(X2) * 2 / len(s)

print(f'频率轴间隔 = {f2[1]:.2f} Hz  （= fs/N = 100/200）')
print('按幅值从大到小排序的前 3 个频率：')
for i in np.argsort(amp2)[-3:][::-1]:
    print(f'  f = {f2[i]:.1f} Hz    amp = {amp2[i]:.3f}')

print()
print('预期：2Hz 峰 ≈ 1.0、5Hz 峰 ≈ 0.5 —— 与 10/31 预测一致')

plt.figure(figsize=(8, 3))
plt.stem(f2, amp2, basefmt=' ')
plt.xlabel('freq (Hz)')
plt.ylabel('amplitude')
plt.title('2Hz + 0.5*5Hz 频谱（两峰：2 和 5）')
plt.tight_layout()
plt.show()
