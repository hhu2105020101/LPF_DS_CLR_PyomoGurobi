import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']  # 常用中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from matplotlib import gridspec
from DEFAULT_CONFIG import *  # 导入默认配置

import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """加载保存的历史数据和基础功率"""
    try:
        with open('history_results.pkl', 'rb') as f:
            history = pickle.load(f)
        with open('base_powers.pkl', 'rb') as f:
            base_powers = pickle.load(f)
        print("数据加载成功!")
        return history, base_powers
    except FileNotFoundError:
        print("错误: 数据文件未找到，请先运行主程序生成数据")
        return None, None


def plot_load_restoration_progress(history, base_powers):
    """简洁版负荷恢复进度图（使用时间步索引）"""
    if history is None or base_powers is None:
        print("无法绘图：缺少数据")
        return

    # 准备数据
    load_names = list(history[0]['loads'].keys())  # 获取所有负荷名称

    # 创建时间轴：直接使用时间步索引
    time_steps = np.arange(len(history))  # [0, 1, 2, ..., len(history)-1]

    # 创建图表
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('负荷恢复进度分析（时间步索引）', fontsize=18)  # 更新标题

    # 网格布局（每行最多3个图）
    n_rows = (len(load_names) + 2) // 3

    # 遍历每个负荷创建子图
    for idx, load_name in enumerate(load_names):
        ax = plt.subplot(n_rows, 3, idx + 1)

        # 提取该负荷在所有时间步的功率值
        restored_power = [step['loads'][load_name]['p'] for step in history]

        # 获取该负荷的基础功率
        base_power = base_powers.get(load_name, 0)

        # 绘制恢复功率曲线
        ax.plot(time_steps, restored_power, 'b-', linewidth=1.5, label='恢复功率')

        # 绘制基础功率参考线
        ax.axhline(y=base_power, color='r', linestyle='--', alpha=0.7, label='基础功率')

        # 标记关键点
        if any(restored_power):
            # 1. 标记恢复开始点
            try:
                first_restore_idx = next(i for i, p in enumerate(restored_power) if p > 0)
                ax.plot(first_restore_idx, restored_power[first_restore_idx],
                        'g^', markersize=6, label='恢复开始')
            except StopIteration:
                pass

            # 2. 标记完全恢复点
            try:
                full_restore_idx = next(i for i, p in enumerate(restored_power)
                                        if abs(p - base_power) < 1e-3)
                ax.plot(full_restore_idx, restored_power[full_restore_idx],
                        'go', markersize=6, label='完全恢复')
            except StopIteration:
                pass

        # 设置子图元素
        ax.set_title(load_name, fontsize=12)
        ax.set_xlabel('时间步索引', fontsize=9)  # 修改X轴标签
        ax.set_ylabel('功率 (kW)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, len(history) - 1)  # X轴范围从0到最后一个时间步索引

        # 设置Y轴范围
        max_val = max(max(restored_power), base_power) * 1.1
        min_val = min(min(restored_power), 0) * 1.1
        ax.set_ylim(min_val, max_val)

        # 只在第一个子图添加图例
        if idx == 0:
            # 获取当前子图的图例句柄和标签
            handles, labels = ax.get_legend_handles_labels()
            # 使用唯一标签避免重复
            unique_labels = []
            unique_handles = []
            for handle, label in zip(handles, labels):
                if label not in unique_labels:
                    unique_labels.append(label)
                    unique_handles.append(handle)
            # 添加图例到第一个子图
            ax.legend(unique_handles, unique_labels, loc='best', fontsize=8)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存高分辨率图像
    plt.savefig('load_restoration_progress.png', dpi=300, bbox_inches='tight')

    # 显示图表
    plt.show()


def print_load_restoration_data(history, base_powers):
    """打印第一个负荷的恢复功率详细数据"""
    if not history or not base_powers:
        print("错误: 无有效数据可打印")
        return

    # 获取第一个负荷的信息
    load_names = list(history[0]['loads'].keys())
    if not load_names:
        print("错误: 历史数据中没有负荷信息")
        return

    first_load = load_names[0]
    base_power = base_powers.get(first_load, 0)

    # 提取该负荷在所有时间步的功率值
    restored_power = [step['loads'][first_load]['p'] for step in history]

    print("\n" + "=" * 80)
    print(f"负荷 '{first_load}' 的恢复功率数据 (时间步索引: 0-{len(history) - 1})")
    print("=" * 80)
    print(f"基础功率: {base_power:.2f} kW")

    # 统计恢复信息
    restored_time_steps = []
    full_recovery_time_steps = []

    for step_idx, power in enumerate(restored_power):
        if power > 0:
            restored_time_steps.append(step_idx)
        if abs(power - base_power) < 1e-3:
            full_recovery_time_steps.append(step_idx)

    # 打印统计摘要
    print("\n统计摘要:")
    print(f"总时间步数: {len(restored_power)}")
    print(f"恢复开始时间步: {min(restored_time_steps) if restored_time_steps else '未恢复'}")
    print(f"完全恢复时间步: {min(full_recovery_time_steps) if full_recovery_time_steps else '未完全恢复'}")
    print(f"恢复持续时间步: {len(restored_time_steps)}")
    print(f"最大功率: {max(restored_power):.2f} kW")
    print(f"最小功率: {min(restored_power):.2f} kW")
    print(f"平均功率: {sum(restored_power) / len(restored_power):.2f} kW")

    # 打印详细数据表格
    print("\n详细数据:")
    print("时间步 | 恢复功率 (kW) | 是否恢复 | 是否完全恢复 | 功率百分比 (%)")
    print("-" * 65)

    # 打印关键时间点
    key_steps = set()

    # 添加开始和结束
    key_steps.add(0)
    key_steps.add(len(restored_power) - 1)

    # 添加恢复开始和完全恢复
    if restored_time_steps:
        key_steps.add(min(restored_time_steps))
    if full_recovery_time_steps:
        key_steps.add(min(full_recovery_time_steps))

    # 添加功率变化显著的点
    prev_power = restored_power[0]
    for step_idx, power in enumerate(restored_power):
        if step_idx == 0:
            continue
        power_change = abs(power - prev_power)
        if power_change > base_power * 0.1:  # 变化超过10%基础功率
            key_steps.add(step_idx - 1)  # 变化前点
            key_steps.add(step_idx)  # 变化点
        prev_power = power

    # 对关键步骤排序
    key_steps = sorted(key_steps)

    # 打印关键步骤数据
    for step_idx in key_steps:
        if step_idx >= len(restored_power):
            continue

        power = restored_power[step_idx]
        restored = "是" if power > 0 else "否"
        full_recovery = "是" if abs(power - base_power) < 1e-3 else "否"
        percentage = (power / base_power * 100) if base_power != 0 else 0

        # 标记特殊点
        marker = ""
        if step_idx == 0:
            marker = "起始"
        elif step_idx == len(restored_power) - 1:
            marker = "结束"
        elif step_idx in restored_time_steps and step_idx == min(restored_time_steps):
            marker = "恢复开始"
        elif step_idx in full_recovery_time_steps and step_idx == min(full_recovery_time_steps):
            marker = "完全恢复"

        print(f"{step_idx:5d} | {power:11.2f} | {restored:^8} | {full_recovery:^10} | {percentage:6.1f}% {marker}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    # 直接运行此文件时加载数据并绘图
    history, base_powers = load_data()
    if history and base_powers:
        # 先打印数据
        # print_load_restoration_data(history, base_powers)
        # 再绘图
        plot_load_restoration_progress(history, base_powers)
