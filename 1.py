# 在模型外部添加 MPC 循环
total_horizon = len(renewable_data)  # 总时间步数
mpc_horizon = 24  # MPC 预测时域（例如 24 个时间步）

# 初始化状态变量
current_soc = {ds: initial_soc_value for ds in model.Ds}
current_fuel_remaining = {df: model.original_fuel_in_kwh for df in model.Df}

for start_time in range(0, total_horizon, mpc_horizon):
    # 设置当前 MPC 窗口
    end_time = min(start_time + mpc_horizon, total_horizon)
    current_window = range(start_time, end_time)

    # 创建 MPC 模型实例
    mpc_model = create_mpc_model(
        system_params,
        current_window,
        current_soc,
        current_fuel_remaining,
        renewable_data.loc[current_window]
    )

    # 求解 MPC 优化问题
    solver = SolverFactory('gurobi')
    results = solver.solve(mpc_model)

    # 提取并应用第一个时间步的控制决策
    control_actions = first_step_actions(mpc_model)
    apply_control_actions(control_actions)

    # 更新状态变量
    current_soc = update_soc(current_soc, control_actions)
    current_fuel_remaining = update_fuel(current_fuel_remaining, control_actions)


def get_extract_first_step_actions(model):
    """提取第一个时间步的控制决策"""
    first_t = model.T.first()
    actions = {
        'load_power': {i: model.p_i[i, first_t].value for i in model.L},
        'generation': {df: model.pg[df, first_t].value for df in model.Df},
        'storage': {ds: model.p_ds[ds, first_t].value for ds in model.Ds}
    }
    return actions


def apply_control_actions(actions):
    """在实际系统中应用控制决策"""
    # 这里需要实现与实际系统的交互
    # 例如：通过SCADA系统设置发电机出力、负载开关状态等
    print(f"应用控制决策: {actions}")

def update_soc(current_soc, actions):
    """更新储能状态"""
    # 简化更新 - 实际应根据物理模型更新
    for ds, power in actions['storage'].items():
        current_soc[ds] += power * TIME_STEP
        # 确保SOC在合理范围内
        current_soc[ds] = max(model.storage_range_min(), min(model.storage_range_max(), current_soc[ds]))
    return current_soc

def update_fuel(current_fuel, actions):
    """更新燃料状态"""
    for df, power in actions['generation'].items():
        current_fuel[df] -= power * TIME_STEP
        # 确保燃料不为负
        current_fuel[df] = max(0, current_fuel[df])
    return current_fuel