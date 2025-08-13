import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.environ import (Constraint, ConcreteModel)
from DEFAULT_CONFIG import *  # 导入默认配置



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