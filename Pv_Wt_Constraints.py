import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.environ import (Constraint, ConcreteModel)
from DEFAULT_CONFIG import *  # 导入默认配置


def pv_wt_constraints(model):

    # 风机无功功率约束
    def q_wt_upper_rule(model, t):
        return model.q_wt[t] <= model.p_wt[t]
    model.q_wt_upper_con = Constraint(model.T, rule=q_wt_upper_rule)
    model.q_wt_lower_con = Constraint(model.T, rule=lambda m, t: m.q_wt[t] >= 0)

    # 光伏无功功率约束
    def q_pv_upper_rule(model, t):
        return model.q_pv[t] <= model.p_pv[t]
    model.q_pv_upper_con = Constraint(model.T, rule=q_pv_upper_rule)
    model.q_pv_lower_con = Constraint(model.T, rule=lambda m, t: m.q_pv[t] >= 0)

    return model

