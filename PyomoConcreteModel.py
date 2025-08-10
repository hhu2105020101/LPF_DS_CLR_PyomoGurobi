import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.environ import (Constraint, ConcreteModel)
from DEFAULT_CONFIG import *  # 导入默认配置


# #############################################################################################################
# # 新增：电压相关约束和惩罚项
# def voltage_constraints(model):
#     """添加电压约束和惩罚项"""
#     # 1. 定义电压变量
#     model.v = Var(model.buses, model.T, within=NonNegativeReals)  # 节点电压幅值
#
#     # 2. 电压上下限约束
#     def voltage_bounds_rule(model, node, t):
#         return model.v_min[node], model.v[node, t], model.v_max[node]
#
#     model.voltage_bounds_con = Constraint(model.buses, model.T, rule=voltage_bounds_rule)
#
#     # 3. 电压越限惩罚项
#     model.voltage_violation = Var(model.buses, model.T, within=NonNegativeReals)  # 电压越限量
#
#     # 4. 定义越限量
#     def voltage_violation_rule(model, node, t):
#         # 上界越限
#         over_voltage = model.v[node, t] - model.v_max[node]
#         # 下界越限
#         under_voltage = model.v_min[node] - model.v[node, t]
#         # 总越限量 = 上界越限 + 下界越限
#         return model.voltage_violation[node, t] == max(0, over_voltage) + max(0, under_voltage)
#
#     model.voltage_violation_con = Constraint(model.buses, model.T, rule=voltage_violation_rule)
#
#     return model
# #############################################################################################################
    #负荷约束
def load_constraints(model):
    #负荷恢复减少量定义 (d_i == p_{t-1} - p_t)
    def load_reduction_rule(model, i, t):
        if t == model.T.first():#note
            # return model.d_i[i, t] == 0
            return model.d_i[i, t] == model.p_load_prev[i] - model.p_load[i, t]
        else:
            t_prev = model.T.prev(t)
            return model.d_i[i, t] == model.p_load[i, t_prev] - model.p_load[i, t]
    model.load_reduction_con = Constraint(model.L, model.T, rule=load_reduction_rule)

    # 有功恢复量约束
    def p_load_rule(model, i, t):
        """0 ≤ p_load ≤ p_load_max"""
        return 0, model.p_load[i, t], model.p_load_max[i]
    model.p_load_con = Constraint(model.L, model.T, rule=p_load_rule)

    # 无功恢复量约束
    def q_load_rule(model, i, t):
        """0 ≤ q_load ≤ q_load_max"""
        return 0, model.q_load[i, t], model.q_load_max[i]
    model.q_load_con = Constraint(model.L, model.T, rule=q_load_rule)

    return model
#############################################################################################################
    # 燃料发电机约束
def gen_fuel_constraints(model):

    # 有功出力限制pg_min[t]<= pg[g,t] <=pg_max[g]
    def p_gf_rule(model, g, t):
        # 上下约束，直接返回 (pg_min[t], pg[g,t], pg_max[g])
        return 0, model.p_gf[g, t], model.p_gf_max[g]
    model.p_gf_con = Constraint(model.gf, model.T, rule=p_gf_rule)

    # 无功约束，对应功率因数角[0,4/pai]
    # def q_gf_rule(model, g, t):
    #     return 0, model.q_gf[g, t], model.p_gf[g, t]
    # model.q_gf_con = Constraint(model.gf, model.T, rule=q_gf_rule)
    # 不能处理上界是变量的情况

    def q_gf_upper_rule(model, g, t):
        """发电机无功功率约束：0 <= q_gf <= p_gf"""
        return model.q_gf[g, t] <= model.p_gf[g, t]  # 只定义上界部分
    model.q_gf_upper_con = Constraint(model.gf, model.T, rule=q_gf_upper_rule)

    # 单独定义下界约束（非负）
    def q_gf_lower_rule(model, g, t):
        return model.q_gf[g, t] >= 0
    model.q_gf_lower_con = Constraint(model.gf, model.T, rule=q_gf_lower_rule)


    # 确保燃料不耗尽（非负约束）
    def e_gf_rule(model, g, t):
        return model.remaining_fuel_in_kwh[g, t] >= 0
    model.e_gf_con = Constraint(model.gf, model.T, rule=e_gf_rule)

    def fuel_reserve_rule(model, g, t):
        """确保燃料不低于储备水平"""
        # 计算允许的最小燃料（初始燃料的10%）
        min_fuel = model.original_fuel_in_kwh[g] * 0.1

        # 约束：剩余燃料 >= 最小燃料储备
        return model.remaining_fuel_in_kwh[g, t] >= min_fuel

    model.fuel_reserve_con = Constraint(model.gf, model.T, rule=fuel_reserve_rule)

    # 燃料剩余量定义和更新
    def fuel_remaining_rule(model, g, t):
        if t == model.T.first():
            # 使用从环境获取的当前实际燃料
            return model.remaining_fuel_in_kwh[g, t] == model.current_fuel_in_kwh[g] - \
                model.p_gf[g, t] * STEP_INTERVAL_IN_HOUR
        else:
            t_prev = model.T.prev(t)
            return model.remaining_fuel_in_kwh[g, t] == model.remaining_fuel_in_kwh[g, t_prev] - \
                model.p_gf[g, t] * STEP_INTERVAL_IN_HOUR

    model.fuel_remaining_con = Constraint(model.gf, model.T, rule=fuel_remaining_rule)

    return model
#############################################################################################################
    # 电池储能约束
def st_constraints(model):
    # 1. 充放电功率约束（最大充放电功率250kW）
    def p_st_rule(model, b, t):
        # 充电功率（负值）下限：-250 kW
        # 放电功率（正值）上限：250 kW
        return -250, model.p_st[b, t], 250
    model.p_st_con = Constraint(model.st, model.T, rule=p_st_rule)

    # 2. 储能容量约束（160 ≤ SOC ≤ 1250 kWh）
    def soc_rule(model, b, t):
        return 160, model.soc[b, t], 1250
    model.soc_con = Constraint(model.st, model.T, rule=soc_rule)

    # 3. 无功功率约束（功率因数角[0, π/4] → 0 ≤ Q ≤ |P|）
    # def q_st_rule(model, b, t):
    #     return 0, model.q_st[b, t], abs(model.p_st[b, t])
    # model.q_st_con = Constraint(model.st, model.T, rule=q_st_rule)
    # 绝对值约束 - 正确方式
    def abs_p_st_upper1_rule(model, b, t):
        """定义 |P| >= P"""
        return model.u_p_st[b, t] >= model.p_st[b, t]

    model.abs_p_st_upper1_con = Constraint(model.st, model.T, rule=abs_p_st_upper1_rule)

    def abs_p_st_upper2_rule(model, b, t):
        """定义 |P| >= -P"""
        return model.u_p_st[b, t] >= -model.p_st[b, t]

    model.abs_p_st_upper2_con = Constraint(model.st, model.T, rule=abs_p_st_upper2_rule)

    # 无功功率约束 - 正确方式
    def q_st_lower_rule(model, b, t):
        """Q >= 0"""
        return model.q_st[b, t] >= 0

    model.q_st_lower_con = Constraint(model.st, model.T, rule=q_st_lower_rule)

    def q_st_upper_rule(model, b, t):
        """Q <= |P|"""
        return model.q_st[b, t] <= model.u_p_st[b, t]

    model.q_st_upper_con = Constraint(model.st, model.T, rule=q_st_upper_rule)

    # 储能状态更新约束
    def soc_update_rule(model, b, t):
        """SOC更新约束（简化为理想情况）"""
        if t == model.T.first():
            return model.soc[b, t] == model.current_soc[b] - model.p_st[b, t] * STEP_INTERVAL_IN_HOUR *0.95
        else:
            t_prev = model.T.prev(t)
            return model.soc[b, t] == model.soc[b, t_prev] - model.p_st[b, t] * STEP_INTERVAL_IN_HOUR *0.95
    model.soc_update_con = Constraint(model.st, model.T, rule=soc_update_rule)

    return model
#############################################################################################################
    #系统有功功率平衡约束
def power_balance_constraints(model):
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

    return model
#############################################################################################################
def solve_load_restoration(env, model, now_window):
    now_step = now_window[0]

    # 1. 模型初始化
    # model = ConcreteModel()

    # 2. 定义参数
    model.T = Set(initialize=now_window, ordered=True)
    # model.T.pprint()
    model.L = Set(initialize=env.load_name )  # 负荷名称集合
    model.st = Set(initialize=env.st.st_bus)
    model.gf = Set(initialize=env.mt.mt_bus)
    model.pv = Set(initialize=env.PV_bus)
    model.wt = Set(initialize=env.Wind_bus)

    # 可再生能源预测出力 (MW)
    _, _, wt_profile_slice, pv_profile_slice = env.obtain_renewable_profile_forecast(now_step)
    # print(wt_profile_slice, pv_profile_slice)
    model.p_wt = pyo.Param(model.T,initialize=lambda m, t: wt_profile_slice[t - now_window[0]],
        within=Reals,doc="风电预测出力 (MW)")
    model.p_pv = pyo.Param(model.T,initialize=lambda m, t: pv_profile_slice[t - now_window[0]],
        within=Reals,doc="光伏预测出力 (MW)")
    # model.p_wt.pprint()
    # model.p_pv.pprint()
    # 负荷
    # 负荷有功基本值 (kW)
    # model.p_load_base = Param(model.L,
    #     initialize={name: env.base_load[i, 0] for i, name in enumerate(env.load_name)},
    #     doc="负荷有功基本值(kW)")
    # # 负荷无功基本值 (kVar)
    # model.q_load_base = Param(model.L,
    #     initialize={name: env.base_load[i, 1] for i, name in enumerate(env.load_name)},
    #     doc="负荷无功基本值(kVar)")
    # 最大恢复功率参数（默认等于基本值）
    model.p_load_max = Param(model.L,
        initialize={name: env.base_load[i, 0] for i, name in enumerate(env.load_name)},
        doc="负荷最大可恢复有功(kW)")
    model.q_load_max = Param(model.L,
        initialize={name: env.base_load[i, 1] for i, name in enumerate(env.load_name)},
        doc="负荷最大可恢复无功(kVar)")
    model.p_load_prev = Param(
        model.L,
        initialize=lambda model, name: env.p_load_prev[name],
        doc="上一步的负荷值"
    )
    # 负荷恢复优先级
    model.priority = Param(model.L,initialize=lambda m, name: env.importance_factor[env.load_name.index(name)])
    # model.priority.pprint()
    model.epsilon = Param(model.L,initialize=dict(zip(env.load_name, env.epsilon.diagonal())),within=NonNegativeReals)

    # 燃料发电机
    model.p_gf_max = Param(model.gf, initialize= env.mt_max_gen, within=NonNegativeReals)
    # 发电机燃料初始值（从环境获取当前实际值）
    #t=0
    model.original_fuel_in_kwh = Param(model.gf,initialize= env.mt.original_fuel_in_kwh)
    #t=first
    model.current_fuel_in_kwh = Param(model.gf,initialize=env.mt.remaining_fuel_in_kwh)
    # # 储能
    model.current_soc = Param(model.st,initialize= env.st.current_storage)
    # model.current_soc.pprint()


    ############################################################################################################
    # 3. 定义决策变量
    # 负荷恢复变量
    model.p_load = Var(model.L, model.T, within=NonNegativeReals)  # 恢复的有功功率
    model.q_load = Var(model.L, model.T, within=NonNegativeReals)  # 恢复的无功功率
    # 辅助变量
    model.d_i = Var(model.L, model.T, within=Reals)  # 负荷恢复减少量

    # 燃料发电机变量
    model.p_gf = Var(model.gf, model.T, within=NonNegativeReals)  # 燃料发电机出力
    model.q_gf = Var(model.gf, model.T, within=NonNegativeReals)  # 燃料发电机出力
    model.remaining_fuel_in_kwh = Var(model.gf, model.T, within=NonNegativeReals)  # 当前剩余燃料 (kWh)

    # 储能变量
    model.p_st = Var(model.st, model.T, within=Reals)  # 储能充放电功率(正放电,负充电)
    model.q_st = Var(model.st, model.T, within=Reals)  # 储能充放电功率(正放电,负充电)
    model.soc = Var(model.st, model.T, within=NonNegativeReals)  # 储能状态 (kWh)
    # 添加辅助变量表示 |P|
    model.u_p_st = Var(model.st, model.T, within=NonNegativeReals)
    model.is_charging = Var(model.st, model.T, within=Binary)
    model.is_discharging = Var(model.st, model.T, within=Binary)

    # #电压
    # model.v = Var(model.buses, model.T, within=NonNegativeReals)  # 节点电压幅值
    # model.voltage_violation = Var(model.buses, model.T, within=NonNegativeReals)  # 电压越限量
    #

    ############################################################################################################
    # 4. 添加约束
    model = gen_fuel_constraints(model)
    model = st_constraints(model)
    model = power_balance_constraints(model)
    model = load_constraints(model)
    # # 新增：电压约束
    # model = voltage_constraints(model)
############################################################################################################
    # 5. 目标函数: 最大化总恢复奖励
    # # 定义电压越限惩罚项
    # def voltage_penalty_rule(model, t):
    #     violation = 0
    #     for bus in model.buses:
    #         # 上界越限
    #         over_voltage = max(0, model.v[bus, t] - model.v_max[bus])
    #         # 下界越限
    #         under_voltage = max(0, model.v_min[bus] - model.v[bus, t])
    #         violation += (over_voltage + under_voltage) ** 2
    #     return -model.lambda_voltage * violation
    # model.lambda_voltage = Param(initialize=1000, mutable=True)  # 可调整的惩罚系数
    def objective_rule(model):
        reward = 0
        for t in model.T:
            # 负荷恢复奖励部分
            load_reward = sum(model.priority[i] * model.p_load[i, t] for i in model.L)

            # 负荷减少惩罚部分
            load_penalty = sum(model.priority[i] * model.epsilon[i] * model.d_i[i, t] for i in model.L)
            # # 新增电压越限惩罚项 (L2范数平方惩罚)
            # voltage_penalty = model.lambda_voltage * sum(
            #     model.voltage_violation[node, t]**2 for node in model.buses
            # )
            # reward += (load_reward - load_penalty - voltage_penalty)
            reward += (load_reward - load_penalty)
        return reward

    model.obj = Objective(rule=objective_rule, sense=maximize)
#############################################################################################################
    # 6. 求解模型

    solver = pyo.SolverFactory('gurobi')  # 使用Gurobi求解器

    results = solver.solve(model, tee=True)  # tee=True显示求解过程
    print("优化结果：",results)
    #
    # # 打印关键变量值
    # print("\n=== 关键变量值 ===")
    # for t in model.T:
    #     print(f"\n时间步 {t}:")
    #     print(f"总恢复负荷: {sum(value(model.p_load[name, t]) for name in model.L):.2f} kW")
    #     print(f"发电机出力: {sum(value(model.p_gf[g, t]) for g in model.gf):.2f} kW")
    #     print(f"储能充放电: {sum(value(model.p_st[b, t]) for b in model.st):.2f} kW")
    #     print(f"可再生能源: {value(model.p_wt[t]) + value(model.p_pv[t]):.2f} kW")

    return results


def get_first_step_actions(model):
    """获取第一时刻的优化决策"""
    actions = {}
    t_now = model.T.first()

    # 负荷恢复决策
    actions['loads'] = {
        name: {
            'p': pyo.value(model.p_load[name, t_now]),  # 有功功率
            'q': pyo.value(model.q_load[name, t_now])   # 无功功率
        }
        for name in model.L
    }
    # 发电机决策
    actions['generators'] = {
        g: {
            'p': pyo.value(model.p_gf[g, t_now]),  # 有功功率
            'q': pyo.value(model.q_gf[g, t_now])   # 无功功率
        }
        for g in model.gf
    }

    # 储能决策
    actions['storages'] = {
        b: {
            'p': pyo.value(model.p_st[b, t_now]),  # 有功功率
            'q': pyo.value(model.q_st[b, t_now])   # 无功功率
        }
        for b in model.st
    }

    # 可再生能源
    actions['renewable_pv'] = value(model.p_pv[t_now])  # 光伏预测出力
    actions['renewable_wt'] = value(model.p_wt[t_now])  # 风电预测出力
    actions['renewable_total'] = value(model.p_pv[t_now] + model.p_wt[t_now])  # 总可再生能源

    # # 无功功率（可选）
    # actions['reactive'] = {
    #     name: value(model.q_load[name, t0])
    #     for name in model.L
    # }

    return actions


def get_first_step_states(model):
    """获取第一时刻结束时的状态值（SOC和燃料）"""
    states = {}
    t_now = model.T.first()

    # 获取储能SOC
    states['soc'] = {
        b: value(model.soc[b, t_now])
        for b in model.st
    }

    # 获取发电机剩余燃料
    states['fuel_remaining'] = {
        g: value(model.remaining_fuel_in_kwh[g, t_now])
        for g in model.gf
    }

    return states