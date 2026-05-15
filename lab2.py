import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io as sio
import os
import warnings

warnings.filterwarnings('ignore')

# 创建输出目录
os.makedirs('pnt', exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
d = pd.read_csv('output/all_discharge.csv')
c = pd.read_csv('output/all_charge.csv')

# 图1：容量衰减曲线
print("绘制图1：容量衰减曲线...")

nom = {'B0005': 2.0, 'B0006': 2.0, 'B0007': 2.0, 'B0018': 2.0}

fig1, ax1 = plt.subplots(figsize=(12, 6))
col = {'B0005': 'blue', 'B0006': 'red', 'B0007': 'green', 'B0018': 'orange'}

for b in d['battery'].unique():
    sub = d[d['battery'] == b].sort_values('num')
    x = sub['num'].values
    y = sub['cap'].values

    ax1.plot(x, y, label=b, color=col[b], lw=2, marker='o', ms=3, markevery=20)

    # 80%点
    idx80 = np.where(y <= nom[b] * 0.8)[0]
    if len(idx80) > 0:
        ax1.scatter(x[idx80[0]], y[idx80[0]], color=col[b], s=80, edgecolors='black', lw=1.5)
        ax1.annotate(f'{b}\n{x[idx80[0]]}cyc', xy=(x[idx80[0]], y[idx80[0]]),
                     xytext=(10, 10), textcoords='offset points', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # 70%点
    idx70 = np.where(y <= nom[b] * 0.7)[0]
    if len(idx70) > 0:
        ax1.scatter(x[idx70[0]], y[idx70[0]], color=col[b], s=80, marker='s', edgecolors='black', lw=1.5)
        ax1.annotate(f'{b}\n{x[idx70[0]]}cyc', xy=(x[idx70[0]], y[idx70[0]]),
                     xytext=(10, -15), textcoords='offset points', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax1.axhline(y=1.6, color='gray', ls='--', alpha=0.5, label='80%容量 (1.6Ah)')
ax1.axhline(y=1.4, color='gray', ls=':', alpha=0.5, label='70%容量 (1.4Ah)')
ax1.set_xlabel('循环次数', fontsize=12)
ax1.set_ylabel('放电容量 (Ah)', fontsize=12)
ax1.set_title('电池容量衰减曲线', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pnt/cap_fade.png', dpi=300, bbox_inches='tight')
plt.show()
print("pnt/cap_fade.png")


# 辅助函数：获取电压容量曲线
def get_vc(bat, num):
    """获取电压-容量曲线"""
    mat = sio.loadmat(f'./data/{bat}.mat')
    data = mat[bat][0, 0]
    cyc = data['cycle'][0]

    cnt = 0
    for i, cy in enumerate(cyc):
        if cy['type'][0] == 'discharge':
            cnt += 1
            if cnt == num:
                dat = cy['data'][0, 0]
                v = dat['Voltage_measured'].flatten()
                i_abs = np.abs(dat['Current_measured'].flatten())
                t = dat['Time'].flatten()
                dt = np.diff(t)
                cap_inc = i_abs[:-1] * dt / 3600.0
                cap = np.concatenate(([0], np.cumsum(cap_inc)))
                return v, cap
    return None, None


# 图2：小型多图
print("绘制图2：电压-容量小型多图...")

bat_list = ['B0005', 'B0006', 'B0007', 'B0018']
samples = {'B0005': [5, 80, 160], 'B0006': [5, 80, 160],
           'B0007': [5, 80, 160], 'B0018': [5, 60, 120]}

fig2, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()
c_list = ['blue', 'orange', 'red']
lab = ['早期 (~5)', '中期 (~80)', '晚期 (~160)']

for i, b in enumerate(bat_list):
    for j, n in enumerate(samples[b]):
        v, cap = get_vc(b, n)
        if v is not None:
            axs[i].plot(cap, v, color=c_list[j], lw=1.5, alpha=0.8, label=lab[j])
            axs[i].scatter(cap[-1], v[-1], color=c_list[j], s=50)
            axs[i].annotate(f'{cap[-1]:.2f}Ah', xy=(cap[-1], v[-1]),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)

    axs[i].set_xlabel('容量 (Ah)', fontsize=10)
    axs[i].set_ylabel('电压 (V)', fontsize=10)
    axs[i].set_title(b, fontsize=12, fontweight='bold')
    axs[i].legend(loc='lower left', fontsize=8)
    axs[i].grid(True, alpha=0.3)
    axs[i].set_xlim(0, 2.2)
    axs[i].set_ylim(2.5, 4.3)

plt.suptitle('电池电压-容量曲线衰减', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('pnt/vc_small.png', dpi=300, bbox_inches='tight')
plt.show()
print("pnt/vc_small.png")

# 图3：详细曲线
print("绘制图3：详细电压-容量曲线...")

fig3, axs = plt.subplots(2, 2, figsize=(15, 12))
axs = axs.flatten()

for i, b in enumerate(bat_list):
    if b == 'B0018':
        cyc_list = [5, 30, 60, 90, 120]
    else:
        cyc_list = [5, 40, 80, 120, 160]

    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(cyc_list)))

    for j, n in enumerate(cyc_list):
        v, cap = get_vc(b, n)
        if v is not None:
            axs[i].plot(cap, v, color=cmap[j], lw=1.5,
                        label=f'循环{n} ({cap[-1]:.2f}Ah)', alpha=0.8)

    axs[i].set_xlabel('容量 (Ah)', fontsize=11)
    axs[i].set_ylabel('电压 (V)', fontsize=11)
    axs[i].set_title(f'{b} 曲线演化', fontsize=13, fontweight='bold')
    axs[i].legend(loc='lower left', fontsize=9)
    axs[i].grid(True, alpha=0.3)
    axs[i].set_xlim(0, 2.2)
    axs[i].set_ylim(2.5, 4.3)

plt.suptitle('电压-容量曲线随循环次数的变化', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('pnt/vc_detail.png', dpi=300, bbox_inches='tight')
plt.show()
print("pnt/vc_detail.png")

print("\n所有图片已保存到 pnt 目录")