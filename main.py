import pandas as pd
from matplotlib.ticker import MaxNLocator

from PyomoConcreteModel import solve_load_restoration, get_first_step_actions, get_first_step_states
from clr_13bus_envs import LoadRestoration13BusBaseEnv, apply_actions_to_environment, update_state_variables
from DEFAULT_CONFIG import *  # 导入默认配置
from pyomo.environ import *
import pickle
import os
from tabulate import tabulate
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.dates as mdates
from IEEE13bus import StaticGridData


def print_optimization_summary(history):
    """打印优化结果汇总表格"""
    if not history:
        print("无优化数据可展示")
        return

    # 准备表头
    header = f"{'时间步':<8} | {'总恢复负荷(kW)':<12} | {'发电机出力(kW)':<12} | {'储能充放电(kW)':<14} | {'可再生能源(kW)':<14} "
    separator = "-" * len(header)

    print("\n优化结果汇总表:")
    print(separator)
    print(header)
    print(separator)

    # 打印每个时间步的数据
    for step, data in enumerate(history):
        # 提取关键数据
        total_load = sum(load['p'] for load in data['loads'].values())

        # 处理可能缺失的发电机数据
        gen_power = 0.0
        if 'generators' in data and data['generators']:
            gen_power = sum(gen['p'] for gen in data['generators'].values())

        # 处理储能数据
        storage_power = 0.0
        if 'storages' in data and data['storages']:
            storage_power = sum(storage['p'] for storage in data['storages'].values())


        renewable = data.get('renewable_total', 0.0)

        # 格式化输出
        print(
            f"{step:<8} | {total_load:<12.2f} | {gen_power:<12.2f} | {storage_power:<14.2f} | {renewable:<14.2f} ")

    print(separator)
    print("注: 储能充电为负值，放电为正值\n")


def main():

    env = LoadRestoration13BusBaseEnv()

    grid = StaticGridData()
    env.mt.remaining_fuel_in_kwh = env.mt.original_fuel_in_kwh
    env.st.current_storage = env.st.initial_storage_mean
    # 定义数据文件路径
    HISTORY_FILE = "history_results.pkl"
    history = []

    # 初始化表格数据
    table_data = []
    headers = ["时间步", "总恢复负荷(kW)", "发电机出力(kW)", "储能充放电(kW)", "可再生能源(kW)"]

    # for now_step in range(0, STEPS_TOTAL, 1):
    for now_step in range(0, 1, 1):
        # 设置当前 MPC 窗口
        end_step = min(now_step + STEPS_LOOKAHEAD, STEPS_TOTAL)
        now_window = range(now_step, end_step)

        model = ConcreteModel()
        # 求解优化问题
        results = solve_load_restoration(env, grid, model, now_window)
        print_results(model, grid)
        # plot_optimization_results(model, grid)
        plot_full_results(model, grid)
    #
    #     # 提取并应用第一个时间步的控制决策
    #     actions = get_first_step_actions(model)
    #     history.append(actions)
    #     print(actions)
    #
    #     # 应用控制动作到环境，获取真实的mt出力和燃料剩余
    #     apply_actions_to_environment(env, actions)
    #
    #     # 更新状态变量SOC
    #     update_state_variables(env, actions)
    #
    #     # 从actions中提取关键数据
    #     total_load = sum(load['p'] for load in actions['loads'].values())
    #     gen_power = sum(gen['p'] for gen in actions['generators'].values())
    #     stor_power = sum(stor['p'] for stor in actions['storages'].values())
    #     renew_power = actions.get('renewable_total', 0)
    #
    #     # 添加新行到表格数据
    #     table_data.append(
    #         [now_step, f"{total_load:.2f}", f"{gen_power:.2f}", f"{stor_power:.2f}", f"{renew_power:.2f}"])
    #
    #     # 更新控制台表格显示
    #     print("\033c", end="")  # 清除控制台
    #     print("\n" + "=" * 80)
    #     print("电力恢复过程关键指标实时监控")
    #     print("=" * 80)
    #     print(tabulate(table_data, headers=headers, tablefmt="grid", numalign="right"))
    #
    # print("\n优化完成!")
    #
    # print_optimization_summary(history)
    # # 保存历史数据
    # print(f"保存历史数据到: {HISTORY_FILE}")
    # with open(HISTORY_FILE, 'wb') as f:
    #     pickle.dump(history, f)
    # # 保存基础功率数据
    # base_powers = {name: env.base_load[i, 0] for i, name in enumerate(env.load_name)}
    # with open('base_powers.pkl', 'wb') as f:
    #     pickle.dump(base_powers, f)


def print_results(model, grid):
    print("\n=== 关键变量值 ===")
    for t in model.T:
        print(f"\n时间步 {t}:")
        # 节点注入功率（仅显示三相总和）
        for bus in model.buses:
            total_inj = sum(value(model.p_inj[bus, phase, t]) for phase in model.phases)
            print(f"  节点{bus}注入功率: {total_inj:.2f} kW")

        # 三相线路功率输出
        for line in model.lines:
            # 输出三相线路功率
            print(f"线路 {line[0]}→{line[1]}功率:")
            print(f"  A相: {value(model.p_flow[line[0], line[1], 'A', t]):.2f} kW")
            print(f"  B相: {value(model.p_flow[line[0], line[1], 'B', t]):.2f} kW")
            print(f"  C相: {value(model.p_flow[line[0], line[1], 'C', t]):.2f} kW")
            # 计算并显示三相总和
            total_flow = sum(value(model.p_flow[line[0], line[1], phase, t])
                             for phase in model.phases)
            print(f"  三相总和: {total_flow:.2f} kW")


        print(f"\n时间步 {t}:")
        total_recovery = sum(value(model.p_load[name, t]) for name in model.L)
        print(f"总恢复负荷: {total_recovery:.2f} kW")

        # 输出各负荷节点的恢复情况
        print("\n负荷节点恢复情况:")
        # 只遍历有负荷的节点
        for bus in grid.load_mapping.keys():
            # 累加该节点上所有负荷的恢复功率
            for load_name in grid.load_mapping[bus]:
                bus_recovery = value(model.p_load[load_name, t])
                print(f"  节点{load_name}: {bus_recovery:.2f} kW")


        # 系统总体信息
        print(f"\n发电机出力: {sum(value(model.p_gf[g, t]) for g in model.gf):.2f} kW")
        print(f"储能充放电: {sum(value(model.p_st[b, t]) for b in model.st):.2f} kW")
        print(f"可再生能源: {value(model.p_wt[t]) + value(model.p_pv[t]):.2f} kW")

# def plot_optimization_results(model, grid):
#     """图例在右侧的优化结果可视化"""
#     # 设置中文显示
#     plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']
#     plt.rcParams['axes.unicode_minus'] = False
#
#     # 准备时间轴数据（288个时间步）
#     time_steps = list(model.T)
#     hours = np.linspace(0, 24, len(time_steps))  # 假设24小时周期
#
#     # 创建画布和子图（增加宽度为图例留空间）
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 15), sharex=True)
#
#     # 1. 所有节点注入功率图（图例在右侧）
#     key_nodes = ['SourceBus', '650', '632', '633', '634', '645', '646', '671', '684', '680', '611', '652', '692', '675']
#
#     # 为节点创建颜色映射
#     node_colors = plt.cm.tab20(np.linspace(0, 1, len(key_nodes)))
#
#     for i, node in enumerate(key_nodes):
#         node_power = []
#         for t in model.T:
#             total_inj = sum(value(model.p_inj[node, phase, t]) for phase in model.phases)
#             node_power.append(total_inj)
#
#         ax1.plot(hours, node_power, label=f'{node}',
#                  color=node_colors[i], linewidth=2)
#
#     ax1.set_title('关键节点注入功率随时间变化', fontsize=16)
#     ax1.set_ylabel('注入功率 (kW)', fontsize=12)
#     ax1.grid(True, linestyle='--', alpha=0.7)
#     ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
#
#     # 图例放在右侧（垂直排列）
#     ax1.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
#                fancybox=True, shadow=True, fontsize=10)
#
#     # 2. 负荷节点恢复情况图（图例在右侧）
#     # 创建负荷节点分组
#     load_groups = {
#         '节点634': ['634a', '634b', '634c'],
#         '节点675': ['675a', '675b', '675c'],
#         '节点670': ['670a', '670b', '670c'],
#         '节点671': ['671'],
#         '节点645': ['645'],
#         '节点646': ['646'],
#         '节点692': ['692'],
#         '节点611': ['611'],
#         '节点652': ['652']
#     }
#
#     # 为每个负荷节点组创建颜色
#     group_colors = plt.cm.tab10(np.linspace(0, 1, len(load_groups)))
#
#     # 绘制每个负荷节点的总恢复功率
#     for i, (group_name, load_names) in enumerate(load_groups.items()):
#         group_power = []
#         for t in model.T:
#             total_power = sum(value(model.p_load[name, t]) for name in load_names)
#             group_power.append(total_power)
#
#         ax2.plot(hours, group_power, label=group_name,
#                  color=group_colors[i], linewidth=2.5)
#
#     ax2.set_title('负荷节点恢复情况（总功率）', fontsize=16)
#     ax2.set_xlabel('时间 (小时)', fontsize=12)
#     ax2.set_ylabel('恢复功率 (kW)', fontsize=12)
#     ax2.grid(True, linestyle='--', alpha=0.7)
#
#     # 图例放在右侧（垂直排列）
#     ax2.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
#                fancybox=True, shadow=True, fontsize=10)
#
#     # 设置x轴刻度
#     plt.xticks(np.arange(0, 25, 3), fontsize=10)
#
#     # 调整布局（为右侧图例留出空间）
#     plt.tight_layout()
#     plt.subplots_adjust(right=0.85)  # 调整右侧边距
#
#     # 保存并显示
#     plt.savefig('node_vs_load_results.png', dpi=300, bbox_inches='tight')
#     plt.show()
def plot_full_results(model, grid):
    """2×2布局的完整优化结果可视化"""
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False

    # 准备时间轴数据（288个时间步）
    time_steps = list(model.T)
    hours = np.linspace(0, 24, len(time_steps))  # 假设24小时周期

    # 创建2×2布局的画布
    fig, axs = plt.subplots(2, 2, figsize=(24, 12), sharex=True)  # 宽度增加20%，高度减少25%
    plt.subplots_adjust(hspace=0.3, wspace=0.3)  # 调整子图间距

    # 1. 左上：节点注入功率（图例在右侧）
    key_nodes = ['SourceBus', '650', '632', '633', '634', '645', '646', '671', '684', '680', '611', '652', '692', '675']

    # 为节点创建颜色映射
    node_colors = plt.cm.tab20(np.linspace(0, 1, len(key_nodes)))

    for i, node in enumerate(key_nodes):
        node_power = []
        for t in model.T:
            total_inj = sum(value(model.p_inj[node, phase, t]) for phase in model.phases)
            node_power.append(total_inj)

        axs[0, 0].plot(hours, node_power, label=f'{node}',
                       color=node_colors[i], linewidth=2)

    axs[0, 0].set_title('关键节点注入功率随时间变化', fontsize=16)
    axs[0, 0].set_ylabel('注入功率 (kW)', fontsize=12)
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    axs[0, 0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # 图例放在右侧
    axs[0, 0].legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                     fancybox=True, shadow=True, fontsize=10)

    # 2. 右上：负荷节点恢复情况（图例在右侧）
    # 创建负荷节点分组
    load_groups = {
        '节点634': ['634a', '634b', '634c'],
        '节点675': ['675a', '675b', '675c'],
        '节点670': ['670a', '670b', '670c'],
        '节点671': ['671'],
        '节点645': ['645'],
        '节点646': ['646'],
        '节点692': ['692'],
        '节点611': ['611'],
        '节点652': ['652']
    }

    # 为每个负荷节点组创建颜色
    group_colors = plt.cm.tab10(np.linspace(0, 1, len(load_groups)))

    # 绘制每个负荷节点的总恢复功率
    for i, (group_name, load_names) in enumerate(load_groups.items()):
        group_power = []
        for t in model.T:
            total_power = sum(value(model.p_load[name, t]) for name in load_names)
            group_power.append(total_power)

        axs[0, 1].plot(hours, group_power, label=group_name,
                       color=group_colors[i], linewidth=2.5)

    axs[0, 1].set_title('负荷节点恢复情况（总功率）', fontsize=16)
    axs[0, 1].set_ylabel('恢复功率 (kW)', fontsize=12)
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)

    # 图例放在右侧
    axs[0, 1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                     fancybox=True, shadow=True, fontsize=10)

    # 3. 左下：发电机、储能和可再生能源出力
    # 准备数据
    gen_power = [sum(value(model.p_gf[g, t]) for g in model.gf) for t in model.T]
    storage_power = [sum(value(model.p_st[b, t]) for b in model.st) for t in model.T]
    wind_power = [value(model.p_wt[t]) for t in model.T]
    solar_power = [value(model.p_pv[t]) for t in model.T]
    total_renewable = [wind_power[i] + solar_power[i] for i in range(len(wind_power))]

    # 绘制曲线
    axs[1, 0].plot(hours, gen_power, 'r-', label='发电机出力', linewidth=2.5)
    axs[1, 0].plot(hours, storage_power, 'g-', label='储能充放电', linewidth=2.5)
    axs[1, 0].plot(hours, total_renewable, 'c-', label='可再生能源总计', linewidth=2.5)
    axs[1, 0].plot(hours, wind_power, 'b--', label='风力发电', linewidth=2)
    axs[1, 0].plot(hours, solar_power, 'y--', label='光伏发电', linewidth=2)

    axs[1, 0].set_title('电源出力情况', fontsize=16)
    axs[1, 0].set_xlabel('时间 (小时)', fontsize=12)
    axs[1, 0].set_ylabel('功率 (kW)', fontsize=12)
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)

    # 图例放在右侧
    axs[1, 0].legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                     fancybox=True, shadow=True, fontsize=10)

    # 4. 右下：发电机和储能状态
    # 创建双纵坐标轴
    ax_power = axs[1, 1]
    ax_energy = ax_power.twinx()

    # 获取数据
    gen_power = [sum(value(model.p_gf[g, t]) for g in model.gf) for t in model.T]
    storage_power = [sum(value(model.p_st[b, t]) for b in model.st) for t in model.T]

    # 发电机燃料剩余（取第一台发电机）
    if model.gf:
        gen_fuel = [value(model.remaining_fuel_in_kwh[list(model.gf)[0], t]) for t in model.T]
    else:
        gen_fuel = [0] * len(model.T)

    # 储能剩余（取第一台储能）
    if model.st:
        storage_soc = [value(model.soc[list(model.st)[0], t]) for t in model.T]
    else:
        storage_soc = [0] * len(model.T)

    # 绘制功率曲线（左侧轴）
    ax_power.plot(hours, gen_power, 'r-', label='发电机出力', linewidth=2.5)
    ax_power.plot(hours, storage_power, 'g-', label='储能充放电', linewidth=2.5)

    # 绘制剩余量曲线（右侧轴）
    ax_energy.plot(hours, gen_fuel, 'b--', label='燃料剩余', linewidth=2.5)
    ax_energy.plot(hours, storage_soc, 'm--', label='电池剩余', linewidth=2.5)

    # 设置图表属性
    ax_power.set_title('发电机和储能状态', fontsize=16)
    ax_power.set_xlabel('时间 (小时)', fontsize=12)
    ax_power.set_ylabel('功率 (kW)', fontsize=12)
    ax_energy.set_ylabel('剩余量 (kWh)', fontsize=12)
    ax_power.grid(True, linestyle='--', alpha=0.7)

    # 合并图例
    lines, labels = ax_power.get_legend_handles_labels()
    lines2, labels2 = ax_energy.get_legend_handles_labels()
    ax_power.legend(lines + lines2, labels + labels2,
                    loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    fancybox=True, shadow=True, ncol=4, fontsize=10)


    # 设置公共x轴
    plt.setp(axs[1, 0].get_xticklabels(), fontsize=10)
    plt.xticks(np.arange(0, 25, 3), fontsize=10)

    # 保存并显示
    plt.tight_layout()
    plt.savefig('full_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
