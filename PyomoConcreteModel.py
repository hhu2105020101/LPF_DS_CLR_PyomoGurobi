import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.environ import (Constraint, ConcreteModel)
from DEFAULT_CONFIG import *  # 导入默认配置
from Storage_Constraints import st_constraints
from Generator_Constraints import gen_fuel_constraints
from Load_Constraints import load_constraints
from SystemConstraints import system_constraints

def solve_load_restoration(env, grid, model, now_window):
    now_step = now_window[0]

    # 2. 定义参数
    model.T = Set(initialize=now_window, ordered=True)
    # model.T.pprint()
    model.L = Set(initialize=env.load_name )  # 负荷名称集合
    model.st = Set(initialize=env.st.st_bus)
    model.gf = Set(initialize=env.mt.mt_bus)
    model.pv = Set(initialize=env.PV_bus)
    model.wt = Set(initialize=env.Wind_bus)
    # model.L.pprint()
    # model.st.pprint()
    # model.gf.pprint()
    # model.pv.pprint()
    # model.wt.pprint()
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
############################################################################################################
    # 4. 添加约束
    model = gen_fuel_constraints(model)
    model = st_constraints(model)
    model = load_constraints(model)
    model = system_constraints(env, grid, model, now_window)#note
############################################################################################################
    # 5. 目标函数: 最大化总恢复奖励
    def objective_rule(model):
        reward = 0
        for t in model.T:
            # 负荷恢复奖励部分
            load_reward = sum(model.priority[i] * model.p_load[i, t] for i in model.L)

            # 负荷减少惩罚部分
            # load_penalty = sum(model.priority[i] * model.epsilon[i] * model.d_i[i, t] for i in model.L)
            load_penalty = 0

            reward += (load_reward - load_penalty)
        return reward

    model.obj = Objective(rule=objective_rule, sense=maximize)
#############################################################################################################
    # 6. 求解模型

    solver = pyo.SolverFactory('gurobi')  # 使用Gurobi求解器

    results = solver.solve(model, tee=True)  # tee=True显示求解过程
    print("优化结果：",results)

    # 打印关键变量值
    # print("\n=== 关键变量值 ===")
    # for t in model.T:
    #     print(f"\n时间步 {t}:")
    #     print(f"总恢复负荷: {sum(value(model.p_load[name, t]) for name in model.L):.2f} kW")
    #     print(f"发电机出力: {sum(value(model.p_gf[g, t]) for g in model.gf):.2f} kW")
    #     print(f"储能充放电: {sum(value(model.p_st[b, t]) for b in model.st):.2f} kW")
    #     print(f"可再生能源: {value(model.p_wt[t]) + value(model.p_pv[t]):.2f} kW")
    #
    #     for bus in model.buses:
    #         if (bus, t) in model.p_inj:
    #             print(f"  节点{bus}注入功率: {value(model.p_inj[bus, t]):.2f} kW")
    #
    #     for line in model.lines:
    #         if line in model.lines:
    #             line_p_value = model.p_flow[line[0], line[1], t].value
    #             print(f"线路 {line[0]}→{line[1]}: {line_p_value:.2f} kW")
    #
    #
    # return results
    print("\n=== 关键变量值 ===")
    for t in model.T:
        # print(f"\n时间步 {t}:")
        # print(f"总恢复负荷: {sum(value(model.p_load[name, t]) for name in model.L):.2f} kW")
        # print(f"发电机出力: {sum(value(model.p_gf[g, t]) for g in model.gf):.2f} kW")
        # print(f"储能充放电: {sum(value(model.p_st[b, t]) for b in model.st):.2f} kW")
        # print(f"可再生能源: {value(model.p_wt[t]) + value(model.p_pv[t]):.2f} kW")
        #
        # # 节点注入功率（仅显示三相总和）
        # for bus in model.buses:
        #     total_inj = sum(value(model.p_inj[bus, phase, t]) for phase in model.phases)
        #     print(f"  节点{bus}注入功率: {total_inj:.2f} kW")
        #
        # # 三相线路功率输出
        # for line in model.lines:
        #     # 输出三相线路功率
        #     print(f"线路 {line[0]}→{line[1]}功率:")
        #     print(f"  A相: {value(model.p_flow[line[0], line[1], 'A', t]):.2f} kW")
        #     print(f"  B相: {value(model.p_flow[line[0], line[1], 'B', t]):.2f} kW")
        #     print(f"  C相: {value(model.p_flow[line[0], line[1], 'C', t]):.2f} kW")
        #     # 计算并显示三相总和
        #     total_flow = sum(value(model.p_flow[line[0], line[1], phase, t])
        #                      for phase in model.phases)
        #     print(f"  三相总和: {total_flow:.2f} kW")
        print(f"\n时间步 {t}:")
        total_recovery = sum(value(model.p_load[name, t]) for name in model.L)
        print(f"总恢复负荷: {total_recovery:.2f} kW")

        # 输出各负荷节点的恢复情况
        print("\n负荷节点恢复情况:")
        # 只遍历有负荷的节点
        for bus in grid.load_mapping.keys():
            bus_recovery = 0
            # 累加该节点上所有负荷的恢复功率
            for load_name in grid.load_mapping[bus]:
                bus_recovery += value(model.p_load[load_name, t])

            print(f"  节点{bus}: {bus_recovery:.2f} kW")

        # 系统总体信息
        print(f"\n发电机出力: {sum(value(model.p_gf[g, t]) for g in model.gf):.2f} kW")
        print(f"储能充放电: {sum(value(model.p_st[b, t]) for b in model.st):.2f} kW")
        print(f"可再生能源: {value(model.p_wt[t]) + value(model.p_pv[t]):.2f} kW")

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