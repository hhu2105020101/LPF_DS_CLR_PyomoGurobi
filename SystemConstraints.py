import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.environ import (Constraint, ConcreteModel)
from DEFAULT_CONFIG import *  # 导入默认配置
from Storage_Constraints import st_constraints
from Generator_Constraints import gen_fuel_constraints
from Load_Constraints import load_constraints
import numpy as np
import cmath

def system_constraints(env, grid, model, now_window):
    """添加LPF相关约束到优化模型"""
    V_base = grid.V_base

    # # 定义节点功率注入变量
    # model.p_inj = Var(model.buses, model.T, within=Reals)
    # # 定义线路功率流变量
    # model.p_flow = Var(model.lines, model.T, within=Reals)
    # 三相扩展


    # 修改线路功率变量为三相
    model.p_flow = Var(model.lines, model.phases, model.T, within=Reals)  # 三相线路有功功率
    model.q_flow = Var(model.lines, model.phases, model.T, within=Reals)  # 三相线路无功功率
    # 创建Λ_ij变量（线路有功功率流）
    model.Lambda = Var(model.lines, model.phases, model.T, within=Reals)

    # 修改节点注入功率为三相
    model.p_inj = Var(model.buses, model.phases, model.T, within=Reals)  # 三相节点注入有功
    model.q_inj = Var(model.buses, model.phases, model.T, within=Reals)  # 三相节点注入无功

    # # 节点注入功率约束
    # def injection_rule(model, bus, phase, t):
    #     """计算节点净功率注入（三相）- 包含无功"""
    #     net_p_injection = 0
    #     net_q_injection = 0
    #
    #     # 发电机注入（三相平均分配）
    #     if bus in model.gf:
    #         # 假设功率因数为0.9
    #         net_p_injection += model.p_gf[bus, t] / 3.0
    #         net_q_injection += model.q_gf[bus, t] / 3.0
    #
    #     # 储能注入
    #     if bus in model.st:
    #         net_p_injection += model.p_st[bus, t] / 3.0
    #         net_q_injection += model.q_st[bus, t] / 3.0
    #
    #     # 光伏注入
    #     if bus in model.pv:
    #         net_p_injection += model.p_pv[t] / 3.0
    #         net_q_injection += model.q_pv[t] / 3.0
    #
    #     # 风机注入
    #     if bus in model.wt:
    #         net_p_injection += model.p_wt[t] / 3.0
    #         net_q_injection += model.q_wt[t] / 3.0
    #
    #     # 负荷消耗
    #     if bus in grid.load_mapping:
    #         for load_name in grid.load_mapping[bus]:
    #             load_p = model.p_load[load_name, t]
    #             load_q = model.q_load[load_name, t]
    #
    #             # 1. 处理纯数字负载（如671,692）- 三相负载
    #             if load_name in ['671', '692']:
    #                 # 平均分配到三相
    #                 net_p_injection -= load_p / 3.0
    #                 net_q_injection -= load_q / 3.0
    #
    #             # 2. 处理节点645 - 单相B相负载
    #             elif bus == '645' and load_name == '645':
    #                 if phase == 'B':
    #                     net_p_injection -= load_p
    #                     net_q_injection -= load_q
    #                 # 其他相不分配
    #
    #             # 3. 处理节点646 - 两相负载（B相和C相）#note：646文章中是两相负载，但是它的传输线都是单相的，所以这里处理成单相负载（B相）
    #             elif bus == '646' and load_name == '646':
    #                 # if phase in ['B', 'C']:
    #                 #     # 平均分配到两相
    #                 #     net_p_injection -= load_p / 2.0
    #                 #     net_q_injection -= load_q / 2.0
    #                 # # A相不分配
    #                 if phase == 'B':
    #                     net_p_injection -= load_p
    #                     net_q_injection -= load_q
    #                 # 其他相不分配
    #
    #             # 4. 处理节点611 - 单相C相负载
    #             elif bus == '611' and load_name == '611':
    #                 if phase == 'C':
    #                     net_p_injection -= load_p
    #                     net_q_injection -= load_q
    #                 # 其他相不分配
    #
    #             # 5. 处理节点652 - 单相A相负载
    #             elif bus == '652' and load_name == '652':
    #                 if phase == 'A':
    #                     net_p_injection -= load_p
    #                     net_q_injection -= load_q
    #                 # 其他相不分配
    #
    #             # 6. 处理带明确相位的负载（如634a, 675c等）
    #             elif any(char in load_name for char in ['a', 'b', 'c']):
    #                 # 获取负载的相位标识（最后一个字符）
    #                 load_phase = load_name[-1].lower()
    #
    #                 # 只在匹配的相位上减去负载功率
    #                 if load_phase == phase.lower():
    #                     net_p_injection -= load_p
    #                     net_q_injection -= load_q
    #                 # 其他相不分配
    #
    #     return (model.p_inj[bus, phase, t] == net_p_injection,
    #             model.q_inj[bus, phase, t] == net_q_injection)
    #
    # # 分开定义有功和无功约束
    # model.p_inj_con = Constraint(model.buses, model.phases, model.T,
    #                              rule=lambda m, b, p, t: injection_rule(m, b, p, t)[0])
    # model.q_inj_con = Constraint(model.buses, model.phases, model.T,
    #                              rule=lambda m, b, p, t: injection_rule(m, b, p, t)[1])

    # 有功注入规则
    def injection_p_rule(model, bus, phase, t):
        net_p_injection = 0

        # 发电机注入（三相平均分配）
        if bus in model.gf:
            net_p_injection += model.p_gf[bus, t] / 3.0

        # 储能注入
        if bus in model.st:
            net_p_injection += model.p_st[bus, t] / 3.0

        # 光伏注入
        if bus in model.pv:
            net_p_injection += model.p_pv[t] / 3.0

        # 风机注入
        if bus in model.wt:
            net_p_injection += model.p_wt[t] / 3.0

        # 负荷消耗（仅处理有功部分）
        if bus in grid.load_mapping:
            for load_name in grid.load_mapping[bus]:
                load_p = model.p_load[load_name, t]

                # 处理不同负载类型
                if load_name in ['671', '692']:
                    # 三相负载 - 平均分配
                    net_p_injection -= load_p / 3.0

                elif bus == '645' and load_name == '645' and phase == 'B':
                    # 单相B相负载
                    net_p_injection -= load_p

                elif bus == '646' and load_name == '646' and phase == 'B':
                    # 单相B相负载
                    net_p_injection -= load_p

                elif bus == '611' and load_name == '611' and phase == 'C':
                    # 单相C相负载
                    net_p_injection -= load_p

                elif bus == '652' and load_name == '652' and phase == 'A':
                    # 单相A相负载
                    net_p_injection -= load_p

                elif any(char in load_name for char in ['a', 'b', 'c']):
                    # 明确相位的负载
                    load_phase = load_name[-1].lower()
                    if load_phase == phase.lower():
                        net_p_injection -= load_p

        return model.p_inj[bus, phase, t] == net_p_injection


    model.injection_p = Constraint(model.buses, model.phases, model.T, rule=injection_p_rule)

    # 无功注入规则（类似有功规则，但处理无功部分）
    def injection_q_rule(model, bus, phase, t):
        net_q_injection = 0

        # 发电机注入（三相平均分配）
        if bus in model.gf:
            net_q_injection += model.q_gf[bus, t] / 3.0

        # 储能注入
        if bus in model.st:
            net_q_injection += model.q_st[bus, t] / 3.0

        # 光伏注入
        if bus in model.pv:
            net_q_injection += model.q_pv[t] / 3.0

        # 风机注入
        if bus in model.wt:
            net_q_injection += model.q_wt[t] / 3.0

        # 负荷消耗（仅处理无功部分）
        if bus in grid.load_mapping:
            for load_name in grid.load_mapping[bus]:
                load_q = model.q_load[load_name, t]

                # 处理不同负载类型
                if load_name in ['671', '692']:
                    # 三相负载 - 平均分配
                    net_q_injection -= load_q / 3.0

                elif bus == '645' and load_name == '645' and phase == 'B':
                    # 单相B相负载
                    net_q_injection -= load_q

                elif bus == '646' and load_name == '646' and phase == 'B':
                    # 单相B相负载
                    net_q_injection -= load_q

                elif bus == '611' and load_name == '611' and phase == 'C':
                    # 单相C相负载
                    net_q_injection -= load_q

                elif bus == '652' and load_name == '652' and phase == 'A':
                    # 单相A相负载
                    net_q_injection -= load_q

                elif any(char in load_name for char in ['a', 'b', 'c']):
                    # 明确相位的负载
                    load_phase = load_name[-1].lower()
                    if load_phase == phase.lower():
                        net_q_injection -= load_q
                if bus == '675':
                    net_q_injection -= 200
                if bus == '611' and phase == 'C':
                    net_q_injection -= 100



        return model.q_inj[bus, phase, t] == net_q_injection

    model.injection_q = Constraint(model.buses, model.phases, model.T, rule=injection_q_rule)

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


    # 三相线路功率流计算
    def line_active_power_calculation_rule(model, start_bus, end_bus, phase, t):
        """计算三相线路有功功率流"""
        # 获取下游节点集合
        downstream_buses = grid.downstream_map.get(end_bus, [])

        # 计算下游总功率（包括终点节点）
        total_downstream_power = model.p_inj[end_bus, phase, t] + sum(
            model.p_inj[bus, phase, t] for bus in downstream_buses
        )

        # 返回功率流等于下游总功率
        return model.p_flow[start_bus, end_bus, phase, t] == -total_downstream_power

    model.line_power_calculation_con = Constraint(
        model.lines, model.phases, model.T, rule=line_active_power_calculation_rule
    )

    def line_reactive_power_calculation_rule(model, start_bus, end_bus, phase, t):
        """计算三相线路无功功率流"""
        # 获取下游节点集合
        downstream_buses = grid.downstream_map.get(end_bus, [])

        # 计算下游总无功功率
        total_downstream_q = model.q_inj[end_bus, phase, t] + sum(
            model.q_inj[bus, phase, t] for bus in downstream_buses
        )

        # 返回无功功率流等于下游总无功功率
        return model.q_flow[start_bus, end_bus, phase, t] == -total_downstream_q

    model.line_reactive_power_con = Constraint(
        model.lines, model.phases, model.T, rule=line_reactive_power_calculation_rule
    )

    # 添加负荷单调递增约束
    def load_monotonic_rule(model, i, t):
        """确保每个负荷在每个时间步的恢复功率不小于前一个时间步"""
        if t == model.T.first():
            return Constraint.Skip  # 第一个时间步没有前一个时间步
        t_prev = model.T.prev(t)
        return model.p_load[i, t] >= model.p_load[i, t_prev]

    model.load_monotonic_con = Constraint(model.L, model.T, rule=load_monotonic_rule)

    # 根节点电压约束
    def root_voltage_rule(model, phase, t):
        return model.v['650', phase, t] == 1.0  # 标幺值

    model.root_voltage_con = Constraint(model.phases, model.T, rule=root_voltage_rule)

    # 电压降约束# note
    def voltage_drop_rule(model, start_bus, end_bus, phase, t):
        """简化电压降计算"""
        # 获取线路阻抗（标幺值）
        line_key = (start_bus, end_bus)
        z_pu = grid.line_impedance_pu.get(line_key)

        # 获取线路功率流（对角线元素）
        P_ij_kW = model.p_flow[start_bus, end_bus, phase, t]  # 有功功率标幺值
        Q_ij_kvar = model.q_flow[start_bus, end_bus, phase, t]  # 无功功率
        # 转换为标幺值
        S_base = grid.S_base  # 基准功率 (kVA)
        P_ij = P_ij_kW / S_base
        Q_ij = Q_ij_kvar / S_base

        r_ij = 0
        x_ij = 0
        if line_key == ('671', '684'):
            # 两相线路 ('671', '684')
            if phase == 'A':
                r_ij = z_pu[0, 0].real
                x_ij = z_pu[0, 0].imag
            elif phase == 'C':
                r_ij = z_pu[1, 1].real
                x_ij = z_pu[1, 1].imag
        elif line_key in [('632', '645'), ('645', '646'), ('684', '611'), ('684', '652')]:
            # 单相线路 - 直接使用电阻部分
            if isinstance(z_pu, complex):
                r_ij = z_pu.real
                x_ij = z_pu.imag
        else:
            # 其他三相线路
            if isinstance(z_pu, np.ndarray):
                phase_index = {'A': 0, 'B': 1, 'C': 2}.get(phase, 0)
                if phase_index < z_pu.shape[0]:
                    r_ij = z_pu[phase_index, phase_index].real
                    x_ij = z_pu[phase_index, phase_index].imag



        # 计算电压降
        # LPF电压降公式：ΔV ≈ (P·R + Q·X) （标幺值）
        voltage_drop = P_ij * r_ij + Q_ij * x_ij

        # 计算下游电压：v_j = v_i - Δv
        return model.v[end_bus, phase, t] == model.v[start_bus, phase, t] - voltage_drop

    model.voltage_drop_con = Constraint(model.lines, model.phases, model.T, rule=voltage_drop_rule)

    # 电压安全约束
    def voltage_limit_rule(model, bus, phase, t):
        return (0.95, model.v[bus, phase, t], 1.05)  # 标幺值范围[0.95, 1.05]

    model.voltage_limit_con = Constraint(model.buses, model.phases, model.T, rule=voltage_limit_rule)




    return model

