import numpy as np
from scipy.stats import truncnorm  # 用于生成截断正态分布的随机数
from DEFAULT_CONFIG import *  # 导入默认配置（包含时间步长等参数）


class MicroTurbine(object):
    """微型燃气轮机模型（仅考虑有功功率）

    假设无功功率在功率因数角限制内始终可支持。

    主要功能：
    - 跟踪剩余燃料量
    - 根据发电功率消耗燃料
    - 验证发电功率是否在燃料限制内
    """

    def __init__(self):
        """初始化微型燃气轮机"""
        self.mt_bus = ['650']
        self.remaining_fuel_in_kwh = None  # 剩余燃料量（千瓦时）
        self.original_fuel_in_kwh = 1200.0  # 初始燃料总量
        self.reset()  # 调用reset方法初始化

    def reset(self):
        """重置燃气轮机状态（恢复到初始燃料量）"""
        self.remaining_fuel_in_kwh = self.original_fuel_in_kwh

    def control(self, power):
        """执行一步控制（消耗燃料发电）

        Args:
          power: 发电功率（kW）
        """
        # 根据发电功率和时间步长计算消耗的燃料
        self.remaining_fuel_in_kwh -= power * STEP_INTERVAL_IN_HOUR

        # # 确保剩余燃料在合理范围内（数值稳定性）
        # self.remaining_fuel_in_kwh = max(0.0, self.remaining_fuel_in_kwh)  # 不能小于0
        # self.remaining_fuel_in_kwh = min(self.remaining_fuel_in_kwh,  # 不能超过初始值
        #                                  self.original_fuel_in_kwh)
        return self.remaining_fuel_in_kwh
    def validate_power(self, power):
        """验证功率是否可行（基于剩余燃料）

        Args:
          power: 计划发电功率（kW）
        Returns:
          调整后的可行功率（kW）
        """
        # 如果没有燃料了，功率只能为0
        if self.remaining_fuel_in_kwh <= 0:
            return 0.0

        # 如果计划发电功率超过剩余燃料可支持的范围
        if power * STEP_INTERVAL_IN_HOUR > self.remaining_fuel_in_kwh:
            # 调整功率为剩余燃料可支持的最大值
            return self.remaining_fuel_in_kwh / STEP_INTERVAL_IN_HOUR

        # 如果功率可行，直接返回
        return power


class BatteryStorage(object):
    """电池储能系统模型（仅考虑有功功率）

    假设逆变器在功率因数角限制内始终可支持无功功率。

    主要功能：
    - 管理电池的荷电状态（SOC）
    - 处理充放电操作（考虑效率）
    - 验证充放电功率是否在电池容量限制内
    """

    def __init__(self):
        """初始化电池储能系统"""
        self.st_bus = ['632']
        self.storage_range = [160.0, 1250.0]  # 电池容量范围 [最小, 最大] (kWh)

        # 初始容量的统计参数（用于随机初始化）
        self.initial_storage_mean = 1000.0  # 平均初始容量 (kWh)
        self.initial_storage_std = 250.0  # 标准差 (kWh)

        # 充放电效率（实际储能与输入/输出的比值）
        self.charge_efficiency = 0.95  # 充电效率（95%）
        self.discharge_efficiency = 0.95  # 放电效率（90%）#note

        self.current_storage = None  # 当前储能状态（kWh）

    def reset(self, init_storage=None):
        """重置电池状态（开始新的模拟）

        Args:
          init_storage: 可选，指定初始容量（kWh）
        """
        if init_storage is None:
            # 从截断正态分布中随机生成初始容量
            # 范围：均值±标准差，且限制在[storage_range_min, storage_range_max]
            self.current_storage = float(
                truncnorm(-1, 1).rvs() * self.initial_storage_std + self.initial_storage_mean
            )
        else:
            try:
                # 确保初始容量是浮点数且在合理范围内
                init_storage = float(init_storage)
                init_storage = np.clip(init_storage,
                                       self.storage_range[0],
                                       self.storage_range[1])
            except (TypeError, ValueError) as e:
                # 处理无效输入
                print(e)
                print("init_storage值必须是浮点数，将使用默认值")
                init_storage = self.initial_storage_mean
            self.current_storage = init_storage

        # 调试模式下打印初始容量
        if DEBUG:
            print("电池系统的初始容量为 %f kWh." % self.current_storage)

    def control(self, power):
        """执行电池充放电控制

        Args:
          power: 控制功率（kW）
            - 正值：放电（从电池输出能量）
            - 负值：充电（向电池输入能量）
        """
        if power < 0:
            # 充电：实际存储的能量 = 输入功率 × 时间 × 充电效率
            self.current_storage -= (self.charge_efficiency * power
                                     * STEP_INTERVAL_IN_HOUR)
        elif power > 0:
            # 放电：实际输出的能量 = 输出功率 × 时间 / 放电效率
            self.current_storage -= (power * STEP_INTERVAL_IN_HOUR
                                     / self.discharge_efficiency)
        # 注意：功率为0时不做任何操作

    def validate_power(self, power):
        """验证充放电功率是否可行（基于当前容量）

        Args:
          power: 计划充放电功率（kW）
        Returns:
          调整后的可行功率（kW）
        """
        if power > 0:  # 放电
            # 计算放电后的预计容量
            projected_storage = (self.current_storage
                                 - power * STEP_INTERVAL_IN_HOUR / self.discharge_efficiency)

            # 如果放电后容量低于最小值
            if projected_storage < self.storage_range[0]:
                # 调整功率：最多只能放出 (当前容量 - 最小容量) 的能量
                power = max(0.0,
                            (self.current_storage - self.storage_range[0]) / STEP_INTERVAL_IN_HOUR)

        elif power < 0:  # 充电
            # 计算充电后的预计容量（考虑充电效率）
            projected_storage = (self.current_storage
                                 - self.charge_efficiency * power * STEP_INTERVAL_IN_HOUR)

            # 如果充电后容量超过最大值
            if projected_storage > self.storage_range[1]:
                # 调整功率：最多只能充入 (最大容量 - 当前容量) 的能量
                power = -max(0.0,
                             (self.storage_range[1] - self.current_storage) / STEP_INTERVAL_IN_HOUR)

        return power  # 返回调整后的功率
