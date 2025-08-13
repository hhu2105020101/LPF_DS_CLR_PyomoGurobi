import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.environ import (Constraint, ConcreteModel)
from DEFAULT_CONFIG import *  # 导入默认配置



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