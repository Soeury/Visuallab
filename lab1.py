import scipy.io as sio
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

# 创建输出目录
os.makedirs('output', exist_ok=True)


def get_cycle_data(cycle, idx=-1):
    """提取单个循环的完整数据"""
    try:
        typ = cycle['type'][0]
        dat = cycle['data'][0, 0]

        # 提取通用字段
        volt = dat['Voltage_measured'].flatten() if 'Voltage_measured' in dat.dtype.names else np.array([])
        curr = dat['Current_measured'].flatten() if 'Current_measured' in dat.dtype.names else np.array([])
        temp = dat['Temperature_measured'].flatten() if 'Temperature_measured' in dat.dtype.names else np.array([])
        time = dat['Time'].flatten() if 'Time' in dat.dtype.names else np.array([])

        # 处理容量
        cap = 0

        if 'Capacity' in dat.dtype.names:
            # 放电循环
            c = dat['Capacity'].flatten()
            if len(c) > 0:
                cap = abs(float(c[-1]))
        elif 'Current_charge' in dat.dtype.names:
            # 充电循环：积分计算
            chg_curr = dat['Current_charge'].flatten()
            chg_time = dat['Time'].flatten() if 'Time' in dat.dtype.names else time

            if len(chg_curr) > 0 and len(chg_time) > 0:
                cap = np.trapz(np.abs(chg_curr), chg_time) / 3600.0
                if cap < 0.5 or cap > 3.0:
                    if idx < 5:
                        print(f"    警告: 循环{idx + 1} 充电容量异常: {cap:.3f}Ah")
                    return None
        else:
            if idx < 5:
                print(f"    警告: 循环{idx + 1} ({typ}) 没有容量数据")
            return None

        # 过滤异常
        if cap <= 0.1 or cap > 5.0:
            if idx < 5:
                print(f"    警告: 循环{idx + 1} ({typ}) 容量异常: {cap:.3f}Ah")
            return None

        # 计算指标
        return {
            'type': typ,
            'cap': cap,
            'v_mean': np.mean(volt) if len(volt) > 0 else np.nan,
            'v_max': np.max(volt) if len(volt) > 0 else np.nan,
            'v_min': np.min(volt) if len(volt) > 0 else np.nan,
            'i_mean': np.mean(np.abs(curr)) if len(curr) > 0 else np.nan,
            'i_max': np.max(np.abs(curr)) if len(curr) > 0 else np.nan,
            't_mean': np.mean(temp) if len(temp) > 0 else np.nan,
            't_max': np.max(temp) if len(temp) > 0 else np.nan,
            't_min': np.min(temp) if len(temp) > 0 else np.nan,
            'dur': time[-1] - time[0] if len(time) > 1 else 0,
            'pts': len(volt),
        }
    except Exception as e:
        if idx < 5:
            print(f"    错误: 循环{idx + 1}: {e}")
        return None


def get_imp(cycle):
    """提取阻抗数据"""
    try:
        dat = cycle['data'][0, 0]
        imp = {}
        keys = ['Re', 'Rct', 'Battery_impedance']
        for k in keys:
            if k in dat.dtype.names:
                val = dat[k]
                if val.size > 0:
                    try:
                        imp[k.lower()] = float(val.flatten()[0])
                    except:
                        pass
        return imp if imp else None
    except:
        return None


def process():
    """处理所有电池数据"""

    bats = ['B0005', 'B0006', 'B0007', 'B0018']

    # 存储所有数据
    all_dis = []
    all_chg = []
    all_imp = []

    for b in bats:
        print(f"\n{'=' * 30}")
        print(f"处理 {b}")
        print(f"{'=' * 30}")

        try:
            # 加载数据
            mat = sio.loadmat(f'./data/{b}.mat')
            data = mat[b][0, 0]
            cyc = data['cycle'][0]

            print(f"总循环数: {len(cyc)}")

            # 临时存储
            dis_data = []
            chg_data = []
            imp_data = []

            chg_list = []
            dis_list = []

            chg_cnt = 0
            dis_cnt = 0
            imp_cnt = 0

            for i, c in enumerate(cyc):
                typ = c['type'][0]

                # 充电
                if typ == 'charge':
                    info = get_cycle_data(c, i)
                    if info and info['cap'] > 0.5:
                        chg_cnt += 1
                        rec = {
                            'battery': b,
                            'num': chg_cnt,
                            'idx': i + 1,
                            'cap': info['cap'],
                            'v_mean': info['v_mean'],
                            'v_max': info['v_max'],
                            'v_min': info['v_min'],
                            'i_mean': info['i_mean'],
                            'i_max': info['i_max'],
                            't_mean': info['t_mean'],
                            't_max': info['t_max'],
                            't_min': info['t_min'],
                            'dur': info['dur'],
                            'pts': info['pts']
                        }
                        chg_data.append(rec)
                        chg_list.append(info['cap'])

                        if chg_cnt <= 3:
                            print(f"  充电 #{chg_cnt}: {info['cap']:.3f} Ah")

                # 放电
                elif typ == 'discharge':
                    info = get_cycle_data(c, i)
                    if info and info['cap'] > 0.5:
                        dis_cnt += 1
                        rec = {
                            'battery': b,
                            'num': dis_cnt,
                            'idx': i + 1,
                            'cap': info['cap'],
                            'v_mean': info['v_mean'],
                            'v_max': info['v_max'],
                            'v_min': info['v_min'],
                            'i_mean': info['i_mean'],
                            'i_max': info['i_max'],
                            't_mean': info['t_mean'],
                            't_max': info['t_max'],
                            't_min': info['t_min'],
                            'dur': info['dur'],
                            'pts': info['pts']
                        }
                        dis_data.append(rec)
                        dis_list.append(info['cap'])

                        if dis_cnt <= 3:
                            print(f"  放电 #{dis_cnt}: {info['cap']:.3f} Ah")

                # 阻抗
                elif 'impedance' in typ.lower():
                    imp = get_imp(c)
                    if imp:
                        imp_cnt += 1
                        rec = {'battery': b, 'num': imp_cnt, 'idx': i + 1, **imp}
                        imp_data.append(rec)

                        if imp_cnt <= 3:
                            s = ', '.join([f"{k}={v:.4f}" for k, v in imp.items()])
                            print(f"  阻抗 #{imp_cnt}: {s}")

            print(f"\n统计: 充电={len(chg_list)}, 放电={len(dis_list)}, 阻抗={imp_cnt}")

            # 计算库仑效率
            start = 1
            if len(chg_list) > start and len(dis_list) > start:
                n = min(len(chg_list), len(dis_list))
                print(f"\n库仑效率计算 (循环 {start + 1} 到 {n}):")
                ce_vals = []
                for i in range(start, n):
                    if chg_list[i] > 0:
                        ce = (dis_list[i] / chg_list[i]) * 100
                        if 80 <= ce <= 110:
                            ce_vals.append(ce)
                            if i < len(dis_data):
                                dis_data[i]['ce'] = ce
                            if i < start + 5:
                                print(
                                    f"  循环{i + 1}: 充电={chg_list[i]:.3f}Ah, 放电={dis_list[i]:.3f}Ah, CE={ce:.2f}%")
                        else:
                            if i < start + 5:
                                print(
                                    f"  循环{i + 1}: 充电={chg_list[i]:.3f}Ah, 放电={dis_list[i]:.3f}Ah, CE={ce:.2f}% (异常)")

                if ce_vals:
                    print(
                        f"\n库仑效率统计: 平均={np.mean(ce_vals):.2f}%, 最小={min(ce_vals):.2f}%, 最大={max(ce_vals):.2f}%")
                    print(f"  有效数据: {len(ce_vals)}/{n - start}")

            # 容量衰减
            if dis_list:
                start_ana = 1 if len(dis_list) > 1 else 0
                init_cap = dis_list[start_ana]
                final_cap = dis_list[-1]
                fade = (init_cap - final_cap) / init_cap * 100
                print(f"\n容量衰减 (从循环{start_ana + 1}开始):")
                print(f"  初始: {init_cap:.3f} Ah, 最终: {final_cap:.3f} Ah")
                print(f"  总衰减: {fade:.2f}%, 平均每循环: {fade / (len(dis_list) - start_ana):.3f}%/cycle")

            # 保存单个电池数据到output目录
            if dis_data:
                pd.DataFrame(dis_data).to_csv(f'output/{b}_dis.csv', index=False)
                all_dis.extend(dis_data)
                print(f"  保存放电: {len(dis_data)} 条")

            if chg_data:
                pd.DataFrame(chg_data).to_csv(f'output/{b}_chg.csv', index=False)
                all_chg.extend(chg_data)
                print(f"  保存充电: {len(chg_data)} 条")

            if imp_data:
                pd.DataFrame(imp_data).to_csv(f'output/{b}_imp.csv', index=False)
                all_imp.extend(imp_data)
                print(f"  保存阻抗: {len(imp_data)} 条")

        except Exception as e:
            print(f"处理 {b} 时出错: {e}")

    return pd.DataFrame(all_dis), pd.DataFrame(all_chg), pd.DataFrame(all_imp)


# 主程序
if __name__ == "__main__":
    print("=" * 30)
    print("NASA锂电池数据解析")
    print("=" * 30)

    df_dis, df_chg, df_imp = process()

    # 保存合并数据到output目录
    if not df_dis.empty:
        df_dis.to_csv('output/all_discharge.csv', index=False)
        print(f"\n合并放电: output/all_discharge.csv ({len(df_dis)} 条)")
    if not df_chg.empty:
        df_chg.to_csv('output/all_charge.csv', index=False)
        print(f"合并充电: output/all_charge.csv ({len(df_chg)} 条)")
    if not df_imp.empty:
        df_imp.to_csv('output/all_impedance.csv', index=False)
        print(f"合并阻抗: output/all_impedance.csv ({len(df_imp)} 条)")

    # 显示统计
    if not df_dis.empty:
        print(f"\n{'=' * 30}")
        print("统计摘要")
        print(f"{'=' * 30}")

        for b in df_dis['battery'].unique():
            d = df_dis[df_dis['battery'] == b]
            c = df_chg[df_chg['battery'] == b] if not df_chg.empty else pd.DataFrame()

            print(f"\n【{b}】")
            print(f"  放电: {len(d)}, 充电: {len(c)}")
            print(f"  容量: {d['cap'].min():.3f} - {d['cap'].max():.3f} Ah")
            print(f"  电压: {d['v_mean'].mean():.2f} V")
            print(f"  温度: {d['t_mean'].mean():.1f} °C")

            if 'ce' in d.columns:
                ce_vals = d['ce'].dropna()
                if len(ce_vals) > 0:
                    print(f"  库仑效率: {ce_vals.mean():.2f}%")

            if len(d) > 1:
                init = d['cap'].iloc[1]
                final = d['cap'].iloc[-1]
                print(f"  衰减: {(init - final) / init * 100:.2f}%")

    print(f"\n{'=' * 30}")
    print("数据处理完成...")
    print(f"{'=' * 30}")