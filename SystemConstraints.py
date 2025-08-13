import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.environ import (Constraint, ConcreteModel)
from DEFAULT_CONFIG import *  # 导入默认配置
from Storage_Constraints import st_constraints
from Generator_Constraints import gen_fuel_constraints
from Load_Constraints import load_constraints


def system_constraints(env, grid, model, now_window):
    """添加LPF相关约束到优化模型"""
    # 定义节点集合
    model.buses = Set(initialize=grid.buses)
    # 定义线路集合
    model.lines = Set(initialize=list(grid.line_impedance.keys()))
    # # 定义节点功率注入变量
    # model.p_inj = Var(model.buses, model.T, within=Reals)
    # # 定义线路功率流变量
    # model.p_flow = Var(model.lines, model.T, within=Reals)
    # 三相扩展
    model.phases = Set(initialize=['A', 'B', 'C'])  # 三相系统

    # 修改线路功率变量为三相
    model.p_flow = Var(model.lines, model.phases, model.T, within=Reals)  # 三相线路有功功率
    model.q_flow = Var(model.lines, model.phases, model.T, within=Reals)  # 三相线路无功功率

    # 修改节点注入功率为三相
    model.p_inj = Var(model.buses, model.phases, model.T, within=Reals)  # 三相节点注入有功
    model.q_inj = Var(model.buses, model.phases, model.T, within=Reals)  # 三相节点注入无功

    # # 节点注入约束
    def injection_rule(model, bus, phase, t):
        """计算节点净功率注入（三相）"""
        net_injection = 0

        # 发电机注入（三相平均分配）
        if bus in model.gf:
            net_injection += model.p_gf[bus, t] / 3.0

        # 储能注入（三相平均分配）
        if bus in model.st:
            net_injection += model.p_st[bus, t] / 3.0

        # 光伏注入（三相平均分配）
        if bus in model.pv:
            net_injection += model.p_pv[t] / 3.0

        # 风机注入（三相平均分配）
        if bus in model.wt:
            net_injection += model.p_wt[t] / 3.0

        # 负荷消
        if bus in grid.load_mapping:
            for load_name in grid.load_mapping[bus]:
                # 1. 有明确相后缀的负荷（如634a, 634b, 634c）
                if load_name.endswith('a') or load_name.endswith('b') or load_name.endswith('c'):
                    # 匹配当前相位
                    if load_name.endswith(phase.lower()):
                        net_injection -= model.p_load[load_name, t]
                # 2. 没有相后缀的负荷（如671, 645）
                else:# 平均分配到各相
                    net_injection -= model.p_load[load_name, t] / 3.0

        return model.p_inj[bus, phase, t] == net_injection

    model.injection_con = Constraint(model.buses, model.phases, model.T, rule=injection_rule)

    def active_power_balance_rule(model, t):
        """有功功率平衡方程"""
        # 1. 发电侧（传统发电机 + 可再生能源）
        total_generation = (
                sum(model.p_gf[g, t] for g in model.gf) +  # 燃料发电机
                model.p_wt[t] +  # 风电
                model.p_pv[t]  # 光伏
        )

        # 2. 储能充放电
        total_storage_charge_discharge = sum(model.p_st[b, t] for b in model.st)

        # 3. 负荷侧（恢复负荷）
        total_load_restoration = sum(model.p_load[i, t] for i in model.L)

        # 4. 平衡方程
        return total_generation + total_storage_charge_discharge == total_load_restoration

    model.power_balance_con = Constraint(model.T, rule=active_power_balance_rule)


    # 三相线路功率流
    def line_power_calculation_rule(model, start_bus, end_bus, phase, t):
        """计算三相线路功率流"""
        # 获取下游节点集合
        downstream_buses = grid.downstream_map.get(end_bus, [])

        # 计算下游总功率（包括终点节点）
        total_downstream_power = model.p_inj[end_bus, phase, t] + sum(
            model.p_inj[bus, phase, t] for bus in downstream_buses
        )

        # 返回功率流等于下游总功率
        return model.p_flow[start_bus, end_bus, phase, t] == -total_downstream_power

    model.line_power_calculation_con = Constraint(
        model.lines, model.phases, model.T, rule=line_power_calculation_rule
    )

    # 添加负荷单调递增约束
    def load_monotonic_rule(model, i, t):
        """确保每个负荷在每个时间步的恢复功率不小于前一个时间步"""
        if t == model.T.first():
            return Constraint.Skip  # 第一个时间步没有前一个时间步
        t_prev = model.T.prev(t)
        return model.p_load[i, t] >= model.p_load[i, t_prev]

    model.load_monotonic_con = Constraint(model.L, model.T, rule=load_monotonic_rule)











    return model

