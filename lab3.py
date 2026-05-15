import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io as sio
import os
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

# 创建输出目录
os.makedirs('temper', exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取容量数据
d = pd.read_csv('output/all_discharge.csv')


def get_t(bat, n=50):
    """获取温度数据"""
    mat = sio.loadmat(f'./data/{bat}.mat')
    data = mat[bat][0, 0]
    cyc = data['cycle'][0]

    t_data = []
    cyc_n = []

    cnt = 0
    for i, c in enumerate(cyc):
        if c['type'][0] == 'discharge':
            cnt += 1
            if cnt > n:
                break

            dat = c['data'][0, 0]
            if 'Temperature_measured' in dat.dtype.names:
                tmp = dat['Temperature_measured'].flatten()
                tm = dat['Time'].flatten()

                if len(tm) > 0:
                    x_norm = np.linspace(0, 1, 100)
                    t_int = np.interp(x_norm, np.linspace(0, 1, len(tmp)), tmp)
                    t_data.append(t_int)
                    cyc_n.append(cnt)

    return np.array(t_data), cyc_n


def get_tr(bat):
    """计算温度上升速率"""
    mat = sio.loadmat(f'./data/{bat}.mat')
    data = mat[bat][0, 0]
    cyc = data['cycle'][0]

    t_max = []
    cyc_l = []

    cnt = 0
    for i, c in enumerate(cyc):
        if c['type'][0] == 'discharge':
            cnt += 1
            dat = c['data'][0, 0]
            if 'Temperature_measured' in dat.dtype.names:
                tmp = dat['Temperature_measured'].flatten()
                t_max.append(np.max(tmp))
                cyc_l.append(cnt)

    if len(cyc_l) > 10:
        z = np.polyfit(cyc_l, t_max, 1)
        return z[0], cyc_l, t_max
    return None, None, None

# 图1：温度热力图
print("绘制图1：温度热力图...")

bs = ['B0005', 'B0006', 'B0007', 'B0018']
fig1, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, b in enumerate(bs):
    tm, cy = get_t(b, 50)

    if len(tm) > 0:
        im = axs[i].imshow(tm, aspect='auto', cmap='hot',
                           extent=[0, 100, len(cy), 1],
                           vmin=20, vmax=45)

        axs[i].set_xlabel('放电进程 (%)', fontsize=10)
        axs[i].set_ylabel('循环次数', fontsize=10)
        axs[i].set_title(f'{b}', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axs[i], label='温度 (°C)')

plt.suptitle('放电温度热力图（前50循环）', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('temper/heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("temper/heatmap.png")

# 图2：温度曲线
print("绘制图2：温度曲线...")

fig2, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, b in enumerate(bs):
    tm, cy = get_t(b, 50)

    if len(tm) > 0:
        samp = [1, 10, 20, 30, 40, 50]
        sidx = [j - 1 for j in samp if j <= len(tm)]

        cols = plt.cm.viridis(np.linspace(0, 1, len(sidx)))

        for j, ix in enumerate(sidx):
            x = np.linspace(0, 100, 100)
            axs[i].plot(x, tm[ix], color=cols[j], lw=1.5, label=f'循环{samp[j]}')

        axs[i].set_xlabel('放电进程 (%)', fontsize=10)
        axs[i].set_ylabel('温度 (°C)', fontsize=10)
        axs[i].set_title(b, fontsize=12, fontweight='bold')
        axs[i].legend(loc='best', fontsize=8)
        axs[i].grid(True, alpha=0.3)

plt.suptitle('不同循环温度曲线对比', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('temper/curves.png', dpi=300, bbox_inches='tight')
plt.show()
print("temper/curves.png")

# 图3：相关性分析
print("绘制图3：温度-容量相关性...")

# 计算数据
bd = []
for b in bs:
    sub = d[d['battery'] == b].sort_values('num')
    cap = sub['cap'].values

    if len(cap) > 10:
        xc = np.arange(len(cap))
        zc = np.polyfit(xc, cap, 1)
        decay = -zc[0]

        icap = cap[0]
        fcap = cap[-1]
        fade = (icap - fcap) / icap * 100

        tr, _, _ = get_tr(b)

        if tr is not None:
            bd.append({
                'bat': b,
                'tr': tr,
                'decay': decay,
                'fade': fade
            })

            print(f"\n{b}: 升温率={tr:.3f}°C/cyc, 衰减率={decay:.4f}Ah/cyc, 总衰减={fade:.2f}%")

df = pd.DataFrame(bd)

# 相关性图
fig3, axs = plt.subplots(1, 2, figsize=(14, 5))

# 左图：升温率 vs 衰减率
ax = axs[0]
x = df['tr']
y = df['decay']
corr, p = pearsonr(x, y)

ax.scatter(x, y, s=200, c=range(len(df)), cmap='plasma', alpha=0.7)
for i, row in df.iterrows():
    ax.annotate(row['bat'], (row['tr'], row['decay']),
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

z = np.polyfit(x, y, 1)
p1 = np.poly1d(z)
xl = np.linspace(x.min(), x.max(), 100)
ax.plot(xl, p1(xl), 'r--', lw=2, label=f'趋势线 (r={corr:.3f})')

ax.set_xlabel('升温速率 (°C/循环)', fontsize=12)
ax.set_ylabel('容量衰减率 (Ah/循环)', fontsize=12)
ax.set_title('升温率 vs 衰减率', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95, f'r={corr:.3f}, p={p:.4f}', transform=ax.transAxes,
        fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 右图：升温率 vs 总衰减
ax = axs[1]
y2 = df['fade']
corr2, p2 = pearsonr(x, y2)

ax.scatter(x, y2, s=200, c=range(len(df)), cmap='plasma', alpha=0.7)
for i, row in df.iterrows():
    ax.annotate(row['bat'], (row['tr'], row['fade']),
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

z2 = np.polyfit(x, y2, 1)
p2f = np.poly1d(z2)
ax.plot(xl, p2f(xl), 'r--', lw=2, label=f'趋势线 (r={corr2:.3f})')

ax.set_xlabel('升温速率 (°C/循环)', fontsize=12)
ax.set_ylabel('总容量衰减 (%)', fontsize=12)
ax.set_title('升温率 vs 总衰减', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95, f'r={corr2:.3f}, p={p2:.4f}', transform=ax.transAxes,
        fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('温度与容量衰减相关性', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('temper/corr.png', dpi=300, bbox_inches='tight')
plt.show()
print("temper/corr.png")

# 图4：峰值温度变化
print("绘制图4：峰值温度变化...")

fig4, ax = plt.subplots(figsize=(12, 6))

for b in bs:
    _, cyc_l, t_max = get_tr(b)
    if cyc_l is not None:
        ax.plot(cyc_l, t_max, label=b, lw=2, marker='o', ms=4, markevery=20)

ax.set_xlabel('循环次数', fontsize=12)
ax.set_ylabel('峰值温度 (°C)', fontsize=12)
ax.set_title('放电峰值温度随循环变化', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('temper/peak.png', dpi=300, bbox_inches='tight')
plt.show()
print("temper/peak.png")

# 输出统计
print("\n" + "=" * 30)
print("统计结果:")
print("=" * 30)
print(f"\n升温率 vs 衰减率: r={corr:.4f}, p={p:.4e}")
print(f"升温率 vs 总衰减: r={corr2:.4f}, p={p2:.4e}")

print("\n图片已保存到 temper 目录...")