import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.environ import (Constraint, ConcreteModel)
from DEFAULT_CONFIG import *  # 导入默认配置





    # 燃料发电机约束
def gen_fuel_constraints(model):

    # 有功出力限制pg_min[t]<= pg[g,t] <=pg_max[g]
    def p_gf_rule(model, g, t):
        # 上下约束，直接返回 (pg_min[t], pg[g,t], pg_max[g])
        return 0, model.p_gf[g, t], model.p_gf_max[g]
    model.p_gf_con = Constraint(model.gf, model.T, rule=p_gf_rule)

    # 无功约束，对应功率因数角[0,4/pai]
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