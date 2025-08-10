# 导入必要的库
import copy  # 用于创建对象的深拷贝
import json  # 用于处理JSON格式数据
import os  # 提供操作系统相关的功能

import gymnasium as gym  # 强化学习环境库
import numpy as np  # 数值计算库
import pandas as pd  # 数据处理库

from abc import abstractmethod  # 用于定义抽象方法
from gymnasium import spaces  # 定义强化学习的动作和状态空间

# 从自定义模块导入电池存储和微型燃气轮机模型
from der_models import BatteryStorage, MicroTurbine
from DEFAULT_CONFIG import *  # 导入默认配置

# 获取当前脚本文件所在的目录路径
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# 定义一个装饰器用于检查方法是否覆盖了父类的方法
def override(cls):
    """Annotation for documenting method overrides."""

    def check_override(method):
        if method.__name__ not in dir(cls):
            raise NameError(f"{method} does not override any method of {cls}")
        return method

    return check_override


class LoadRestoration13BusBaseEnv:
    """13节点电力系统负载恢复问题的环境基类"""

    def __init__(self):
        # 导入OpenDSS电力系统仿真库
        import opendssdirect as dss
        self.dss = dss

        # 负载的重要性因子
        self.importance_factor = [1.0, 1.0, 0.9, 0.85, 0.8, 0.8, 0.75,
                                  0.7, 0.65, 0.5, 0.45, 0.4, 0.3, 0.3, 0.2]
        # 负载节点名称
        self.load_name = ['671', '634a', '634b', '634c', '645', '646',
                          '692', '675a', '675b', '675c', '611', '652',
                          '670a', '670b', '670c']
        self.PV_bus = ['680']
        self.Wind_bus = ['675']

        # 读取可再生能源数据CSV文件并正确处理时间索引
        self.renewable_data = pd.read_csv(
            '1day_five_min_renewable_profile.csv',
            parse_dates=['Time'],  # 自动解析时间列
            index_col='Time'  # 直接将时间列设为索引
        )
        self.renewable_data.index = self.renewable_data.index.tz_localize(None)  # 去掉索引中的时区

        # OpenDSS主文件路径
        self.main_dss_subdir = 'main.dss'
        self.dss_data_path = os.path.join(self.main_dss_subdir)

        # 负载数量
        self.num_of_load = len(self.load_name)
        # 负载投入后需要维持的最小步数（对角矩阵）
        self.epsilon = np.diag([100] * self.num_of_load)

        # 运行OpenDSS命令加载电力系统模型
        self.dss.run_command('Redirect ' + self.dss_data_path)

        # 收集负载的基本功率信息（kW和kvar）
        self.base_load = []
        for name in self.load_name:
            self.dss.Loads.Name(name)  # 选择指定负载
            self.base_load.append([self.dss.Loads.kW(), self.dss.Loads.kvar()])
        self.base_load = np.array(self.base_load)

        # 定义分布式能源(DER)的最大发电能力
        self.pv_max_gen = 300  # 光伏最大发电量(kW)
        self.wind_max_gen = 400  # 风电最大发电量(kW)
        self.mt_max_gen = 400  # 微型燃气轮机最大发电量(kW)
        self.st_max_gen = 250  # 储能最大发电/充电功率(kW)

        # 初始化风光发电曲线
        self.wt_profile = None  # 风电实际出力曲线
        self.pv_profile = None  # 光伏实际出力曲线
        self.renewable_pseudo_forecasts = None  # 风光预测数据

        # 创建储能和微型燃气轮机实例
        self.st = BatteryStorage()  # 电池储能模型
        self.mt = MicroTurbine()  # 微型燃气轮机模型

        # 环境状态变量
        self.simulation_step = 0  # 当前仿真步数
        self.terminated = False  # 是否终止
        # 上一步的负载投入决策
        self.load_pickup_decision_last_step = [0.0] * self.num_of_load
        self.v_lambda = DEFAULT_V_LAMBDA  # 电压违例惩罚系数

        # 调试相关
        self.debug = None
        self.history = None

        # 时间列表
        self.time_of_day_list = None
        self.forecasts_len_in_hours = 1  # 预测长度(小时)
        self.error_level = DEFAULT_ERROR_LEVEL  # 预测误差水平

        self.p_mt_actual = None
        self.q_mt_actual = None

        self.p_load_prev = {name: 0 for name in self.load_name}

    def reset(self, seed=None, options={}):
        """重置环境到初始状态"""
        super().reset(seed=seed)  # 调用父类重置方法

        if options is None:
            options = {}

        # 从选项中获取起始索引和初始储能状态
        start_index = options.get('start_index', None)
        init_storage = options.get('init_storage', None)

        # 重置环境状态
        self.simulation_step = 0
        self.terminated = False
        if self.debug:
            self.history = copy.deepcopy(CONTROL_HISTORY_DICT)
        self.load_pickup_decision_last_step = [0.0] * self.num_of_load

        # 重置DER模型
        self.mt.reset()
        self.st.reset(init_storage)

        # 设置起始时间索引（如果未提供则随机生成）
        if start_index is None:
            # 训练场景从7月1日00:00到7月31日18:00（共8856个5分钟间隔）
            start_index = np.random.randint(0, 8856)

        if self.debug:
            print(f"Scenario index used here is {start_index}")

        # 重置时间列表
        self.time_of_day_list = []
        # 获取风光预测数据
        self.obtain_renewable_profile_forecast(start_index)
        # 获取初始状态
        state = self.get_state()

        # 返回初始状态和信息字典
        info = {}
        return state, info

    def obtain_renewable_profile_forecast(self, start_index):
        """获取可再生能源的预测数据和前瞻步数内的预测数据切片"""
        self.wt_profile = np.array(self.renewable_data['wind_gen']) * self.wind_max_gen
        self.pv_profile = np.array(self.renewable_data['pv_gen']) * self.pv_max_gen

        #数据切片 - 从start_index开始截取控制时域长度
        end_index = start_index + STEPS_LOOKAHEAD

        wt_profile_slice = self.wt_profile[start_index:end_index]
        pv_profile_slice = self.pv_profile[start_index:end_index]

        return self.wt_profile , self.pv_profile ,wt_profile_slice, pv_profile_slice


    def update_opendss_pickup_load(self, load_pickup_decision):
        """根据负载投入决策更新OpenDSS中的负载"""
        for load_idx, name in enumerate(self.load_name):
            self.dss.Loads.Name(name)  # 选择指定负载
            # 设置负载的有功和无功功率
            self.dss.Loads.kW(self.base_load[load_idx, 0] * load_pickup_decision[load_idx])
            self.dss.Loads.kvar(self.base_load[load_idx, 1] * load_pickup_decision[load_idx])

    def update_opendss_generation(self, p_gen, q_gen):
        """更新OpenDSS中的发电机设置"""
        p_pv, p_wt, p_st, p_mt = p_gen
        q_pv, q_wt, q_st, q_mt = q_gen
        # 更新发电机出力设置
        self.dss.Generators.Name('pv')  # 选择光伏发电机
        self.dss.Generators.kW(p_pv)  # 设置光伏有功出力
        self.dss.Generators.kvar(q_pv)  # 设置光伏无功出力

        self.dss.Generators.Name('wt')  # 选择风电发电机
        self.dss.Generators.kW(p_wt)  # 设置风电有功出力
        self.dss.Generators.kvar(q_wt)  # 设置风电无功出力

        # 处理储能系统（ES）在节点632的情况
        if p_st > 0.0:
            # 放电：视为发电机
            self.dss.Generators.Name('esg')  # 选择储能发电机
            self.dss.Generators.kW(p_st)  # 设置储能放电功率
            self.dss.Generators.kvar(q_st)  # 设置储能无功出力

            # 将储能负载设置为0（不充电）
            self.dss.Loads.Name('esl')
            self.dss.Loads.kW(0.0)
            self.dss.Loads.kvar(0.0)
        else:
            # 充电：视为纯电阻负载
            self.dss.Generators.Name('esg')
            self.dss.Generators.kW(0.0)  # 发电机不工作
            self.dss.Generators.kvar(0.0)

            self.dss.Loads.Name('esl')
            self.dss.Loads.kW(-p_st)  # 设置充电功率（负值表示充电）
            self.dss.Loads.kvar(0.0)  # 无功功率为0

    @staticmethod
    def get_trigonomical_representation(pd_datetime):
        """生成时间的三角函数编码（正弦和余弦）"""
        # 计算一天中的位置（5分钟为单位）
        daily_five_min_position = (STEPS_PER_HOUR * pd_datetime.hour +
                                   pd_datetime.minute / 5)
        # 转换为角度（弧度）
        degree = daily_five_min_position / 288.0 * 2 * np.pi

        # 返回正弦和余弦值
        return np.sin(degree), np.cos(degree)


    def get_reward_and_voltages(self, load_pickup_decision):
        """计算奖励并获取电压信息

        奖励由三部分组成：
          1. 电压违规惩罚
          2. 负载恢复奖励
          3. 负载削减惩罚

        Args:
          load_pickup_decision: 负载投入决策列表（每个元素代表该负载的恢复比例）

        Returns:
          reward: 总奖励值
          voltages: 所有节点的电压（标幺值）
          voltage_bus_name: 对应节点的名称
        """

        # 获取所有节点电压（跳过前3个，它们是OpenDSS添加的额外源）
        voltages = self.dss.Circuit.AllBusMagPu()[3:]
        voltage_bus_name = self.dss.Circuit.AllNodeNames()[3:]

        # 计算电压违规（超过上限或低于下限）
        voltage_violations = [max((0.0, v - V_MAX, V_MIN - v))
                              for v in voltages]
        # 电压违规惩罚（平方和乘以系数）
        voltage_violation_penalty = np.sum(
            [v ** 2 for v in voltage_violations]) * self.v_lambda

        # 计算负载恢复奖励
        load_restored = [self.base_load[idx, 0] * load_pickup_decision[idx]
                         for idx in range(self.num_of_load)]
        # 加权和（乘以重要性因子）
        load_restore_reward = np.dot(self.importance_factor, load_restored)

        # 计算负载削减惩罚（相比上一步减少的负载）
        load_shed = [self.base_load[idx, 0] *
                     max(0.0, self.load_pickup_decision_last_step[idx] -
                         load_pickup_decision[idx])
                     for idx in range(self.num_of_load)]

        # 计算负载削减惩罚（考虑维持时间）
        load_by_epsilon = np.matmul(self.epsilon,
                                    np.array(load_shed).reshape([-1, 1]))
        load_shed_penalty = float(np.dot(self.importance_factor,
                                         load_by_epsilon))

        # 更新上一步的负载投入决策
        self.load_pickup_decision_last_step = load_pickup_decision

        # 负载相关奖励（恢复奖励减去削减惩罚）
        load_only_reward = ((load_restore_reward - load_shed_penalty)
                            * REWARD_SCALING_FACTOR)
        # 总奖励 = 负载奖励 - 电压惩罚
        reward = (load_only_reward
                  - voltage_violation_penalty * REWARD_SCALING_FACTOR)

        return reward, load_only_reward, voltages, voltage_bus_name

    def step_gen_only(self, action):
        """执行一个时间步（仅考虑发电机调度）

        这个简化版本不考虑无功功率的主动监控

        处理流程：
          1. 动作预处理：将归一化的控制信号转换为原始范围
          2. 功率平衡：根据可用发电量确定负载投入（按优先级）
          3. 控制实施：运行潮流计算，更新储能和燃气轮机状态
          4. 后处理：收集下一个状态、奖励和终止信号
        """

        # 步骤1：动作预处理
        # 将动作限制在有效范围内
        action = np.clip(action, self.action_lower, self.action_upper)

        # 解包动作向量
        p_st, st_angle, p_mt, mt_angle, wt_angle, pv_angle = action

        # 将部分动作从[-1,1]映射到[0,1]
        p_mt = (p_mt + 1) / 2.0
        st_angle = (st_angle + 1) / 2.0
        mt_angle = (mt_angle + 1) / 2.0
        wt_angle = (wt_angle + 1) / 2.0
        pv_angle = (pv_angle + 1) / 2.0

        # 获取当前时间步的可再生能源实际出力
        p_pv = self.pv_profile[self.simulation_step]
        p_wt = self.wt_profile[self.simulation_step]

        # 转换储能和燃气轮机的出力到实际值
        p_st *= self.st_max_gen
        p_mt *= self.mt_max_gen

        # 验证储能和燃气轮机能否提供该功率
        p_st = self.st.validate_power(p_st)
        p_mt = self.mt.validate_power(p_mt)

        # 计算无功功率（假设功率因数角在0~45度之间）
        q_pv = p_pv * np.tan(np.pi / 4 * pv_angle)  # 光伏无功
        q_wt = p_wt * np.tan(np.pi / 4 * wt_angle)  # 风电无功
        q_mt = p_mt * np.tan(np.pi / 4 * mt_angle)  # 燃气轮机无功
        if p_st > 0:
            q_st = p_st * np.tan(np.pi / 4 * st_angle)  # 储能放电时的无功
        else:
            q_st = 0.0  # 储能充电时不提供无功

        # 步骤2：功率平衡
        total_gen = [p_pv + p_wt + p_st + p_mt,  # 总有功
                     q_pv + q_wt + q_st + q_mt]  # 总无功

        total_gen_p = total_gen[0]  # 总有功出力

        # 如果总发电量大于总负载
        if total_gen_p > sum(self.base_load[:, 0]):
            # 按比例缩减所有发电机的出力
            shrinking_ratio = sum(self.base_load[:, 0]) / total_gen_p
            p_pv, p_wt, p_st, p_mt = [x * shrinking_ratio
                                      for x in [p_pv, p_wt, p_st, p_mt]]
            q_pv, q_wt, q_st, q_mt = [x * shrinking_ratio
                                      for x in [q_pv, q_wt, q_st, q_mt]]
            # 重新计算总有功
            total_gen_p = p_pv + p_wt + p_st + p_mt

        # 按优先级顺序确定负载投入
        load_pickup_decision = [0.0] * self.num_of_load  # 初始全为0
        load_idx = 0
        while total_gen_p > 1e-3:  # 还有剩余发电能力
            # 按优先级顺序投入负载
            # 计算当前负载可以投入的比例（不超过1）
            load_pickup_decision[load_idx] = min(1.0, total_gen_p / self.base_load[load_idx][0])
            # 减去已投入负载的功率
            total_gen_p -= self.base_load[load_idx][0] * load_pickup_decision[load_idx]
            load_idx += 1

            # 步骤3：控制实施
            # 更新负载状态
        self.update_opendss_pickup_load(load_pickup_decision)

        # 准备发电机出力列表
        p_gen = [p_pv, p_wt, p_st, p_mt]
        q_gen = [q_pv, q_wt, q_st, q_mt]

        # 更新发电机设置
        self.update_opendss_generation(p_gen, q_gen)

        # 运行潮流计算
        self.dss.run_command('Solve mode=snap')

        # 获取燃气轮机的实际出力（包括线路损耗）
        self.dss.Circuit.SetActiveElement('Vsource.mt')
        mt_power = [-x for x in self.dss.CktElement.Powers()[:6]]
        p_mt = sum([mt_power[i] for i in range(6) if i % 2 == 0])  # 有功部分
        q_mt = sum([mt_power[i] for i in range(6) if i % 2 == 1])  # 无功部分

        # 更新DER模型状态
        self.mt.control(p_mt)
        self.st.control(p_st)

        # 计算奖励和电压
        (reward, load_only_reward, voltages,
         voltage_bus_name) = self.get_reward_and_voltages(load_pickup_decision)

        # 步骤4：后处理
        self.simulation_step += 1
        state = self.get_state()  # 获取新状态
        self.terminated = (self.simulation_step >= (CONTROL_HORIZON_LEN - 1))  # 是否终止

        # 调试模式下记录历史数据
        if self.debug:
            Sslack = self.dss.Circuit.TotalPower()  # 获取松弛节点的功率

            # 记录各种数据
            self.history['load_status'].append(load_pickup_decision)
            self.history['pv_power'].append([p_pv, q_pv])
            self.history['wt_power'].append([p_wt, q_wt])
            self.history['mt_power'].append([p_mt, q_mt])
            self.history['st_power'].append([p_st, q_st])
            self.history['slack_power'].append([Sslack[0], Sslack[1]])
            self.history['voltages'].append(voltages)
            self.history['mt_remaining_fuel'].append(
                self.mt.remaining_fuel_in_kwh / self.mt.original_fuel_in_kwh)
            self.history['st_soc'].append(
                self.st.current_storage / self.st.storage_range[1])

            # 如果是终止步骤，记录电压节点名称
            if self.terminated:
                self.history['voltage_bus_names'] = voltage_bus_name

        # 返回新状态、奖励、终止标志等
        return state, reward, self.terminated, False, {}

    def step_gen_load(self, action):
        """执行一个时间步（同时考虑发电机调度和负载投入）

        处理流程：
          1. 动作预处理
          2. 功率平衡（确保发电=负载）
          3. 控制实施
          4. 后处理
        """

        # 步骤1：动作预处理
        action = np.clip(action, self.action_lower, self.action_upper)

        # 解包动作向量（前15个是负载投入决策）
        load_pickup_decision = [(x + 1) / 2.0 for x in action[:self.num_of_load]]

        # 计算总投入负载功率
        load_picked_up = np.sum(
            [self.base_load[idx] * load_pickup_decision[idx]
             for idx in range(self.num_of_load)], axis=0)
        (load_picked_up_p, load_picked_up_q) = np.sum(
            [self.base_load[idx] * load_pickup_decision[idx]
             for idx in range(self.num_of_load)], axis=0)

        # 解包发电机控制部分
        p_st = action[self.num_of_load] * self.st_max_gen  # 储能功率
        st_angle = (action[self.num_of_load + 1] + 1) / 2.0  # 储能功率因数角
        wt_angle = (action[self.num_of_load + 2] + 1) / 2.0  # 风电功率因数角
        pv_angle = (action[self.num_of_load + 3] + 1) / 2.0  # 光伏功率因数角

        # 获取当前可再生能源出力
        p_pv = self.pv_profile[self.simulation_step]
        p_wt = self.wt_profile[self.simulation_step]

        # 验证储能功率是否可行
        p_st = self.st.validate_power(p_st)

        # 步骤2：功率平衡（确保发电=负载）
        # 计算储能充电功率（如果是充电状态）
        p_st_load = max(0.0, -p_st)

        # 计算发电能力范围
        mt_max_gen_pq = [self.mt.validate_power(self.mt_max_gen),
                         self.mt.validate_power(self.mt_max_gen) * 0.75]
        st_gen_contribution = p_st if p_st > 0.0 else 0.0
        p_gen_range = [p_pv + p_wt + st_gen_contribution,
                       p_pv + p_wt + st_gen_contribution + mt_max_gen_pq[0]]

        def power_reduction(gen_excess, gen_pv, gen_wt, gen_st, p_st_load):

            if gen_excess <= (gen_pv + gen_wt):
                pv_ratio = gen_pv / (gen_pv + gen_wt)
                gen_pv -= pv_ratio * gen_excess
                gen_wt -= (1 - pv_ratio) * gen_excess
            else:
                # in this case, the storage must be in discharging mode,
                # reducing its power output.
                assert p_st_load == 0.0
                gen_excess -= (gen_pv + gen_wt)
                gen_pv = 0.0
                gen_wt = 0.0
                gen_st -= gen_excess

            return gen_pv, gen_wt, gen_st

        # Implement the logic for balancing load and generation.
        # See GM Paper Algorithm 1.

        if load_picked_up_p + p_st_load >= p_gen_range[1]:  # not enough gen
            # load pickup decision update (cannot pick up that much load,
            # discard less important load)
            if p_st_load > p_gen_range[1]:
                # Usually this does not happen since p_mt can cover p_st's
                # charging. This happens when mt's fuel is running out.
                p_st_load = p_gen_range[1]
                p_mt = mt_max_gen_pq[0]
                # Cannot pick up any other load
                load_pickup_decision = [0.0] * self.num_of_load
                load_picked_up_p = 0.0
                load_picked_up_q = 0.0
            else:
                load_picked_up_p = 0.0
                load_picked_up_q = 0.0
                load_pickup_decision_new = []
                # adjust load pickup decision: not pick up lower priority load.
                for idx, decision in enumerate(load_pickup_decision):
                    if load_picked_up_p + p_st_load < p_gen_range[1]:
                        if (load_picked_up_p + p_st_load
                                + self.base_load[idx, 0] * decision
                                <= p_gen_range[1]):
                            load_picked_up_p += (self.base_load[idx, 0]
                                                 * decision)
                            load_picked_up_q += (self.base_load[idx, 1]
                                                 * decision)
                        else:
                            decision = 0.0
                    else:
                        decision = 0.0

                    load_pickup_decision_new.append(decision)

                load_pickup_decision = load_pickup_decision_new

                if load_picked_up_p + p_st_load < p_gen_range[0]:
                    # this is most likely mt is already out of fuel now.
                    p_mt = 0.0
                    gen_excess = p_gen_range[0] - load_picked_up_p - p_st_load
                    p_pv, p_wt, p_st = power_reduction(
                        gen_excess, p_pv, p_wt, p_st, p_st_load)
                else:
                    p_mt = load_picked_up_p + p_st_load - p_gen_range[0]
        elif load_picked_up_p + p_st_load >= p_gen_range[0]:
            # Enough generation, all generator are used.
            p_mt = load_picked_up_p + p_st_load - p_gen_range[0]
        elif max(p_st, 0.0) < load_picked_up_p + p_st_load < p_gen_range[0]:
            # Too much generation, cut renewable
            p_mt = 0.0
            gen_excess = p_gen_range[0] - load_picked_up_p - p_st_load
            p_pv, p_wt, p_st = power_reduction(
                gen_excess, p_pv, p_wt, p_st, p_st_load)
        else:
            # Still to much generation after renewable are totally curtailed.
            p_pv = 0.0
            p_wt = 0.0
            p_mt = 0.0
            p_st = load_picked_up_p

        # Assuming the maximum angle for inverter is 45 degree (pi/4).
        q_pv = p_pv * np.tan(np.pi / 4 * pv_angle)
        q_wt = p_wt * np.tan(np.pi / 4 * wt_angle)
        if p_st > 0:
            q_st = p_st * np.tan(np.pi / 4 * st_angle)
        else:
            q_st = 0.0

        q_gen_range = [q_pv + q_wt + q_st,
                       q_pv + q_wt + q_st + mt_max_gen_pq[1]]

        q_vsource = 0.0
        if load_picked_up_q > q_gen_range[1]:
            if self.debug:
                print("Reactive power shortage, Vsource is compensating."
                      " %f, %f" % (load_picked_up_q, q_gen_range[1]))
                # TODO: Add penalty for using Vsource Q.
            q_vsource = load_picked_up_q - q_gen_range[1]
            q_mt = mt_max_gen_pq[1]
        elif load_picked_up_q > q_gen_range[0]:
            q_mt = load_picked_up_q - q_gen_range[0]
        elif q_st < load_picked_up_q < q_gen_range[0]:
            q_mt = 0.0
            q_excess = q_gen_range[0] - load_picked_up_q
            q_pv, q_wt, q_st = power_reduction(q_excess, q_pv, q_wt,
                                               q_st, p_st_load)
        else:
            q_mt = 0.0
            q_pv = 0.0
            q_wt = 0.0
            q_st = load_picked_up_q

        # Step 3: Control implementation
        # Compute power flow
        self.update_opendss_pickup_load(load_pickup_decision)


        # 步骤3：控制实施（与step_gen_only类似）
        # 更新负载状态
        self.update_opendss_pickup_load(load_pickup_decision)

        # 准备发电机出力
        p_gen = [p_pv, p_wt, p_st, p_mt]
        q_gen = [q_pv, q_wt, q_st, q_mt]
        self.update_opendss_generation(p_gen, q_gen)

        # 运行潮流计算
        self.dss.run_command('Solve mode=snap')

        # 获取燃气轮机实际出力
        self.dss.Circuit.SetActiveElement('Vsource.mt')
        mt_power = [-x for x in self.dss.CktElement.Powers()[:6]]
        p_mt = sum([mt_power[i] for i in range(6) if i % 2 == 0])
        q_mt = sum([mt_power[i] for i in range(6) if i % 2 == 1])

        # 更新DER模型状态
        self.mt.control(p_mt)
        self.st.control(p_st)

        # 步骤4：后处理
        (reward, load_only_reward, voltages,
         voltage_bus_name) = self.get_reward_and_voltages(load_pickup_decision)

        # 如果有无功功率补偿，添加额外惩罚
        q_vsource_penalty = 0.1 * q_vsource  # TODO: 需要调整系数
        reward -= q_vsource_penalty * REWARD_SCALING_FACTOR

        self.simulation_step += 1
        state = self.get_state()
        self.terminated = (self.simulation_step >= (CONTROL_HORIZON_LEN - 1))

        # 调试模式下记录历史数据
        if self.debug:
            Sslack = self.dss.Circuit.TotalPower()

            # 记录各种数据
            self.history['load_status'].append(load_pickup_decision)
            self.history['pv_power'].append([p_pv, q_pv])
            self.history['wt_power'].append([p_wt, q_wt])
            self.history['mt_power'].append([p_mt, q_mt])
            self.history['st_power'].append([p_st, q_st])
            self.history['slack_power'].append([Sslack[0], Sslack[1]])
            self.history['voltages'].append(voltages)
            self.history['mt_remaining_fuel'].append(
                self.mt.remaining_fuel_in_kwh / self.mt.original_fuel_in_kwh)
            self.history['st_soc'].append(
                self.st.current_storage / self.st.storage_range[1])

            # 如果是终止步骤，记录电压节点名称
            if self.terminated:
                self.history['voltage_bus_names'] = voltage_bus_name

        # 返回
        info = {'load_only_reward': load_only_reward}

        # 返回新状态、奖励、终止标志等
        return state, reward, self.terminated, False, info

    def get_control_history(self):
        """获取控制历史数据（调试模式下使用）"""
        # 构建结果字典
        results = {
            'pv_power': np.array(self.history['pv_power']),
            'wt_power': np.array(self.history['wt_power']),
            'mt_power': np.array(self.history['mt_power']),
            'st_power': np.array(self.history['st_power']),
            'slack_power': np.array(self.history['slack_power']),
            'mt_remaining_fuel': np.array(self.history['mt_remaining_fuel']),
            'st_soc': np.array(self.history['st_soc']),
            'voltages': np.array(self.history['voltages']).transpose(),
            'voltage_bus_names': np.array(self.history['voltage_bus_names']),
            'load_status': np.array(self.history['load_status']),
            'time_stamps': self.time_of_day_list,
        }
        return results


def apply_actions_to_environment(env, actions):
    """将优化决策应用到OpenDSS环境（简化版）"""
    # 1. 直接设置负载功率值
    for name, load_value in actions['loads'].items():
        # 在环境中找到该负载
        if name in env.load_name:
            # 直接设置负载有功和无功功率
            env.dss.Loads.Name(name)
            env.dss.Loads.kW(load_value['p'])  # 有功功率
            env.dss.Loads.kvar(load_value['q'])  # 无功功率

    # 2. 提取发电机和储能决策
    p_gen = [0] * 4  # [pv, wt, st, mt]
    q_gen = [0] * 4  # [pv, wt, st, mt]

    # 处理发电机出力
    for g, gen_value in actions['generators'].items():
        if 'mt' in g.lower():  # 燃气轮机
            p_gen[3] = gen_value['p']  # 有功功率
            q_gen[3] = gen_value['q']  # 无功功率

    # 处理储能出力
    for b, st_value in actions['storages'].items():
        p_gen[2] = st_value['p']  # 有功功率
        q_gen[2] = st_value['q']  # 无功功率

    # 3. 应用可再生能源预测
    p_gen[0] = env.pv_profile[env.simulation_step]  # 光伏
    p_gen[1] = env.wt_profile[env.simulation_step]  # 风电

    # 不设置MT出力（它将是平衡节点）
    p_gen[3] = 0
    q_gen[3] = 0

    # 4. 更新发电机和储能设置（不包括MT）
    env.update_opendss_generation(p_gen, q_gen)

    # 明确设置MT为平衡节点，确保MT Vsource处于激活状态
    env.dss.Vsources.Name('mt')

    # 5.运行潮流计算
    env.dss.Command('Solve mode=snap')

    # 6. 获取MT实际出力
    env.dss.Circuit.SetActiveElement('Vsource.mt')
    mt_power = [-x for x in env.dss.CktElement.Powers()[:6]]
    env.p_mt_actual = sum([mt_power[i] for i in range(6) if i % 2 == 0])
    env.q_mt_actual = sum([mt_power[i] for i in range(6) if i % 2 == 1])

    # 7. 获取真实的燃料剩余
    env.mt.remaining_fuel_in_kwh -=  env.p_mt_actual * STEP_INTERVAL_IN_HOUR

def update_state_variables(env, actions):
    """更新状态变量（SOC和燃料）"""
    # 1. 更新时间步
    env.simulation_step += 1

    # 2. 更新SOC（使用您的储能模型）
    for b, p_value in actions['storages'].items():
        # 调用您的储能控制方法
        env.st.control(p_value['p'])

    # # 3. 更新燃料（使用您的发电机模型）
    # for g, p_value in actions['generators'].items():
    #     # 调用您的发电机控制方法
    #     env.mt.control(p_value)

    # 4. 更新负荷历史值
    for load_id, load_value in actions['loads'].items():
        # 将当前负荷值存储为下一步的"上一步负荷"
        env.p_load_prev[load_id] = load_value['p']


