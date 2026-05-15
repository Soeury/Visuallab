import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
import scipy.io as sio
import os
import warnings

warnings.filterwarnings('ignore')

# 创建visual目录
os.makedirs('visual', exist_ok=True)

# 读取数据
d = pd.read_csv('output/all_discharge.csv')
c = pd.read_csv('output/all_charge.csv')
i_df = pd.read_csv('output/all_impedance.csv')


# 准备交互数据
def prepare_interactive_data():
    """准备所有电池的交互数据"""
    all_data = []

    for bat in d['battery'].unique():
        # 放电数据
        dis = d[d['battery'] == bat].sort_values('num')
        # 充电数据
        chg = c[c['battery'] == bat].sort_values('num')
        # 阻抗数据
        imp = i_df[i_df['battery'] == bat].sort_values('num')

        for idx, row in dis.iterrows():
            cyc_num = row['num']

            # 找对应的充电数据
            chg_row = chg[chg['num'] == cyc_num]
            chg_cap = chg_row['cap'].values[0] if len(chg_row) > 0 else np.nan

            # 找对应的阻抗数据
            imp_row = imp[imp['num'] == cyc_num]
            re = imp_row['re'].values[0] if len(imp_row) > 0 else np.nan
            rct = imp_row['rct'].values[0] if len(imp_row) > 0 else np.nan
            batt_imp = imp_row['battery_impedance'].values[0] if len(imp_row) > 0 else np.nan

            all_data.append({
                '电池': bat,
                '循环': cyc_num,
                '放电容量': row['cap'],
                '充电容量': chg_cap,
                '库仑效率': row['ce'] if 'ce' in row else np.nan,
                '平均电压': row['v_mean'],
                '最高温度': row['t_max'],
                '平均温度': row['t_mean'],
                '放电时长': row['dur'],
                'Re': re,
                'Rct': rct,
                '阻抗': batt_imp
            })

    return pd.DataFrame(all_data)


print("准备交互数据...")
df_int = prepare_interactive_data()

# 交互图1：容量衰减曲线
print("创建交互图1：容量衰减曲线:")

fig1 = go.Figure()

for bat in df_int['电池'].unique():
    sub = df_int[df_int['电池'] == bat]

    fig1.add_trace(go.Scatter(
        x=sub['循环'],
        y=sub['放电容量'],
        mode='lines+markers',
        name=bat,
        hovertemplate=
        '<b>%{text}</b><br>' +
        '循环: %{x}<br>' +
        '容量: %{y:.3f} Ah<br>' +
        '库仑效率: %{customdata[0]:.2f}%<br>' +
        '最高温度: %{customdata[1]:.1f}°C<br>' +
        'Re: %{customdata[2]:.4f} Ω<br>' +
        'Rct: %{customdata[3]:.4f} Ω<br>' +
        '<extra></extra>',
        text=sub['电池'],
        customdata=np.stack((
            sub['库仑效率'].fillna(0),
            sub['最高温度'].fillna(0),
            sub['Re'].fillna(0),
            sub['Rct'].fillna(0)
        ), axis=-1),
        marker=dict(size=6, opacity=0.7)
    ))

fig1.update_layout(
    title='电池容量衰减曲线（鼠标悬停显示详情）',
    xaxis_title='循环次数',
    yaxis_title='放电容量 (Ah)',
    hovermode='closest',
    width=1000,
    height=600,
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
)

fig1.write_html('visual/capacity_interactive.html')
print("visual/capacity_interactive.html")

# 交互图2：温度变化热图
print("创建交互图2：温度热图:")


def get_temp_matrix(bat, max_cyc=100):
    """获取温度矩阵"""
    mat = sio.loadmat(f'./data/{bat}.mat')
    data = mat[bat][0, 0]
    cyc = data['cycle'][0]

    temp_mat = []
    cyc_nums = []

    cnt = 0
    for i, c in enumerate(cyc):
        if c['type'][0] == 'discharge':
            cnt += 1
            if cnt > max_cyc:
                break

            dat = c['data'][0, 0]
            if 'Temperature_measured' in dat.dtype.names:
                temp = dat['Temperature_measured'].flatten()
                time = dat['Time'].flatten()

                if len(time) > 0:
                    x_norm = np.linspace(0, 1, 100)
                    t_int = np.interp(x_norm, np.linspace(0, 1, len(temp)), temp)
                    temp_mat.append(t_int)
                    cyc_nums.append(cnt)

    return np.array(temp_mat), cyc_nums


fig2 = make_subplots(rows=2, cols=2,
                     subplot_titles=('B0005', 'B0006', 'B0007', 'B0018'),
                     horizontal_spacing=0.1, vertical_spacing=0.15)

bats = ['B0005', 'B0006', 'B0007', 'B0018']
positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

for idx, bat in enumerate(bats):
    temp_mat, cyc_nums = get_temp_matrix(bat, 50)

    if len(temp_mat) > 0:
        fig2.add_trace(
            go.Heatmap(
                z=temp_mat,
                y=cyc_nums,
                x=np.linspace(0, 100, 100),
                colorscale='Hot',
                zmin=20, zmax=45,
                name=bat,
                hovertemplate='循环: %{y}<br>放电进程: %{x:.1f}%<br>温度: %{z:.1f}°C<br><extra></extra>',
                colorbar=dict(title='温度 (°C)') if idx == 0 else None
            ),
            row=positions[idx][0], col=positions[idx][1]
        )

fig2.update_layout(
    title='电池放电温度变化热图（鼠标悬停显示温度）',
    width=1200,
    height=800,
    showlegend=False
)

fig2.write_html('visual/temperature_heatmap_interactive.html')
print("visual/temperature_heatmap_interactive.html")

# 交互图3：电压-容量曲线
print("创建交互图3：电压-容量曲线:")


def get_vc_curve(bat, cyc_nums):
    """获取多条电压-容量曲线"""
    mat = sio.loadmat(f'./data/{bat}.mat')
    data = mat[bat][0, 0]
    cyc = data['cycle'][0]

    curves = []

    cnt = 0
    for i, c in enumerate(cyc):
        if c['type'][0] == 'discharge':
            cnt += 1
            if cnt in cyc_nums:
                dat = c['data'][0, 0]
                volt = dat['Voltage_measured'].flatten()
                curr = np.abs(dat['Current_measured'].flatten())
                time = dat['Time'].flatten()

                dt = np.diff(time)
                cap_inc = curr[:-1] * dt / 3600.0
                cap = np.concatenate(([0], np.cumsum(cap_inc)))

                curves.append({
                    'cycle': cnt,
                    'capacity': cap,
                    'voltage': volt
                })

                if len(curves) == len(cyc_nums):
                    break

    return curves


fig3 = go.Figure()

for bat in bats:
    if bat == 'B0018':
        sel_cyc = [5, 40, 80, 120]
    else:
        sel_cyc = [5, 40, 80, 120, 160]

    curves = get_vc_curve(bat, sel_cyc)

    for curve in curves:
        fig3.add_trace(go.Scatter(
            x=curve['capacity'],
            y=curve['voltage'],
            mode='lines',
            name=f'{bat} 循环{curve["cycle"]}',
            hovertemplate='容量: %{x:.3f} Ah<br>电压: %{y:.3f} V<br><extra></extra>',
            line=dict(width=1.5)
        ))

fig3.update_layout(
    title='电压-容量曲线（鼠标悬停显示电压/容量）',
    xaxis_title='容量 (Ah)',
    yaxis_title='电压 (V)',
    hovermode='closest',
    width=1000,
    height=600,
    legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.8)')
)

fig3.write_html('visual/vc_curves_interactive.html')
print("visual/vc_curves_interactive.html")

# 交互图4：多维度散点图
print("创建交互图4：多维度散点图:")

fig4 = px.scatter(
    df_int,
    x='循环',
    y='放电容量',
    size='最高温度',
    color='电池',
    hover_data={
        '循环': True,
        '放电容量': ':.3f',
        '充电容量': ':.3f',
        '库仑效率': ':.2f',
        '最高温度': ':.1f',
        '平均温度': ':.1f',
        'Re': ':.4f',
        'Rct': ':.4f',
        '阻抗': ':.4f'
    },
    title='多维度数据展示（气泡大小=温度，悬停显示详细数据）',
    labels={'放电容量': '放电容量 (Ah)', '循环': '循环次数', '最高温度': '最高温度 (°C)'}
)

fig4.write_html('visual/multidimensional_scatter.html')
print("visual/multidimensional_scatter.html")

# 交互图5：综合仪表板
print("创建交互图5：综合仪表板:")

fig5 = make_subplots(
    rows=2, cols=2,
    subplot_titles=('容量衰减', '库仑效率', '阻抗变化 (Re)', '温度变化'),
    specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
           [{'type': 'scatter'}, {'type': 'scatter'}]]
)

for bat in bats:
    sub = df_int[df_int['电池'] == bat]

    # 容量衰减
    fig5.add_trace(
        go.Scatter(x=sub['循环'], y=sub['放电容量'], mode='lines', name=f'{bat}_容量',
                   legendgroup=bat, showlegend=False,
                   hovertemplate=f'{bat}<br>循环: %{{x}}<br>容量: %{{y:.3f}}Ah<extra></extra>'),
        row=1, col=1
    )

    # 库仑效率
    fig5.add_trace(
        go.Scatter(x=sub['循环'], y=sub['库仑效率'], mode='lines+markers', name=f'{bat}_CE',
                   legendgroup=bat, marker=dict(size=4)),
        row=1, col=2
    )

    # 阻抗
    fig5.add_trace(
        go.Scatter(x=sub['循环'], y=sub['Re'], mode='lines', name=f'{bat}_Re',
                   legendgroup=bat, showlegend=False),
        row=2, col=1
    )

    # 温度
    fig5.add_trace(
        go.Scatter(x=sub['循环'], y=sub['最高温度'], mode='lines', name=f'{bat}_温度',
                   legendgroup=bat, showlegend=False),
        row=2, col=2
    )

fig5.update_xaxes(title_text="循环次数", row=1, col=1)
fig5.update_xaxes(title_text="循环次数", row=1, col=2)
fig5.update_xaxes(title_text="循环次数", row=2, col=1)
fig5.update_xaxes(title_text="循环次数", row=2, col=2)
fig5.update_yaxes(title_text="容量 (Ah)", row=1, col=1)
fig5.update_yaxes(title_text="库仑效率 (%)", row=1, col=2)
fig5.update_yaxes(title_text="Re (Ω)", row=2, col=1)
fig5.update_yaxes(title_text="温度 (°C)", row=2, col=2)

fig5.update_layout(
    title='电池性能综合仪表板（鼠标悬停查看数据）',
    height=800,
    width=1200,
    hovermode='closest'
)

fig5.write_html('visual/dashboard.html')
print("visual/dashboard.html")

# 创建索引页面
print("\n创建索引页面...")

index_html = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>电池数据分析 - 交互仪表板</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            transition: transform 0.3s, box-shadow 0.3s;
            cursor: pointer;
            text-decoration: none;
            color: #333;
            display: block;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        }
        .card h3 {
            margin: 0 0 10px 0;
            color: #667eea;
        }
        .card p {
            margin: 0;
            color: #666;
            font-size: 14px;
        }
        .icon {
            font-size: 40px;
            margin-bottom: 10px;
        }
        footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #999;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>锂电池数据分析仪表板</h1>
        <div class="subtitle">NASA锂电池老化数据集 - 交互式可视化</div>

        <div class="grid">
            <a href="capacity_interactive.html" class="card">
                <div class="icon"></div>
                <h3>容量衰减曲线</h3>
                <p>查看电池容量随循环次数的衰减趋势，鼠标悬停显示详细数据</p>
            </a>

            <a href="temperature_heatmap_interactive.html" class="card">
                <div class="icon"></div>
                <h3>温度热力图</h3>
                <p>放电过程中温度分布热图，观察温度演化规律</p>
            </a>

            <a href="vc_curves_interactive.html" class="card">
                <div class="icon"></div>
                <h3>电压-容量曲线</h3>
                <p>不同循环阶段的电压容量关系曲线</p>
            </a>

            <a href="multidimensional_scatter.html" class="card">
                <div class="icon"></div>
                <h3>多维度散点图</h3>
                <p>气泡图展示容量、温度、阻抗等多维数据</p>
            </a>

            <a href="dashboard.html" class="card">
                <div class="icon"></div>
                <h3>综合仪表板</h3>
                <p>多指标综合监控，一站式查看电池性能</p>
            </a>
        </div>

        <footer>
            <p>提示：鼠标悬停在图表上可查看详细数据 | 支持缩放、平移、图例筛选</p>
        </footer>
    </div>
</body>
</html>
'''

with open('visual/index.html', 'w', encoding='utf-8') as f:
    f.write(index_html)
print("visual/index.html")

print("\n" + "=" * 30)
print("交互式图表创建完成")
print("=" * 30)
print("\n生成的文件(visual目录):")
print("  1. index.html - 主索引页面")
print("  2. capacity_interactive.html - 容量衰减曲线")
print("  3. temperature_heatmap_interactive.html - 温度热图")
print("  4. vc_curves_interactive.html - 电压-容量曲线")
print("  5. multidimensional_scatter.html - 多维度散点图")
print("  6. dashboard.html - 综合仪表板")

# 这里加到 readme 文件中，然后删掉这边
print("\n使用方法：")
print("  用浏览器打开 visual/index.html 即可查看所有交互图表")
print("\n目录结构：")
print("  visual/")
print("    ├── index.html (主页)")
print("    ├── capacity_interactive.html")
print("    ├── temperature_heatmap_interactive.html")
print("    ├── vc_curves_interactive.html")
print("    ├── multidimensional_scatter.html")
print("    └── dashboard.html")
