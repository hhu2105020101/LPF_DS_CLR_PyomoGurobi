import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from PyomoConcreteModel import solve_load_restoration, get_first_step_actions, get_first_step_states
from clr_13bus_envs import LoadRestoration13BusBaseEnv, apply_actions_to_environment, update_state_variables
from DEFAULT_CONFIG import *  # 导入默认配置
from pyomo.environ import *
import pickle
import os
from tabulate import tabulate
import time


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

    env.mt.remaining_fuel_in_kwh = env.mt.original_fuel_in_kwh
    env.st.current_storage = env.st.initial_storage_mean
    # 定义数据文件路径
    HISTORY_FILE = "history_results.pkl"
    history = []

    # 初始化表格数据
    table_data = []
    headers = ["时间步", "总恢复负荷(kW)", "发电机出力(kW)", "储能充放电(kW)", "可再生能源(kW)"]

    for now_step in range(0, STEPS_TOTAL, 1):
        # 设置当前 MPC 窗口
        end_step = min(now_step + STEPS_LOOKAHEAD, STEPS_TOTAL)
        now_window = range(now_step, end_step)

        model = ConcreteModel()
        # 求解优化问题
        results = solve_load_restoration(env, model, now_window)

        # 提取并应用第一个时间步的控制决策
        actions = get_first_step_actions(model)
        history.append(actions)
        print(actions)

        # 应用控制动作到环境，获取真实的mt出力和燃料剩余
        apply_actions_to_environment(env, actions)

        # 更新状态变量SOC
        update_state_variables(env, actions)

        # 从actions中提取关键数据
        total_load = sum(load['p'] for load in actions['loads'].values())
        gen_power = sum(gen['p'] for gen in actions['generators'].values())
        stor_power = sum(stor['p'] for stor in actions['storages'].values())
        renew_power = actions.get('renewable_total', 0)

        # 添加新行到表格数据
        table_data.append(
            [now_step, f"{total_load:.2f}", f"{gen_power:.2f}", f"{stor_power:.2f}", f"{renew_power:.2f}"])

        # 更新控制台表格显示
        print("\033c", end="")  # 清除控制台
        print("\n" + "=" * 80)
        print("电力恢复过程关键指标实时监控")
        print("=" * 80)
        print(tabulate(table_data, headers=headers, tablefmt="grid", numalign="right"))

    print("\n优化完成!")

    print_optimization_summary(history)
    # 保存历史数据
    print(f"保存历史数据到: {HISTORY_FILE}")
    with open(HISTORY_FILE, 'wb') as f:
        pickle.dump(history, f)
    # 保存基础功率数据
    base_powers = {name: env.base_load[i, 0] for i, name in enumerate(env.load_name)}
    with open('base_powers.pkl', 'wb') as f:
        pickle.dump(base_powers, f)


if __name__ == "__main__":
    main()
