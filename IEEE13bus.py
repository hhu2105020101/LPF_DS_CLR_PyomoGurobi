from collections import defaultdict



class StaticGridData:
    """静态电网数据结构（IEEE 13节点专用）"""

    def __init__(self):
        # 节点下游关系（不包含自身）
        self.downstream_map = {
            '650': ['611', '632', '633', '634', '645', '646', '652', '670', '671', '675', '680', '684', '692'],
            '632': ['611', '633', '634', '645', '646', '652', '670', '671', '675', '680', '684', '692'],
            '633': ['634'],
            '634': [],
            '645': ['646'],
            '646': [],
            '670': ['611', '652', '671', '675', '680', '684', '692'],
            '671': ['611', '652', '675', '680', '684', '692'],
            '684': ['611', '652'],
            '692': ['675'],
        }


        # 计算变压器阻抗（归算到4.16kV侧）
        # 1. Sub变压器 (115kV/4.16kV)
        v_high, v_low = 115.0, 4.16
        s_rated = 5000  # kVA
        z_percent = complex(0.5 + 0.5, 4 + 4)  # R% + jX%
        z_base = (v_low ** 2 * 1000) / s_rated  # Ω
        z_actual = (z_percent / 100) * z_base
        sub_impedance = (z_actual.real, z_actual.imag)

        # 2. XFM1变压器 (4.16kV/0.48kV)
        v_high, v_low = 4.16, 0.48
        s_rated = 500  # kVA
        z_percent = complex(0.55 + 0.55, 1 + 1)  # R% + jX%
        z_base = (v_high ** 2 * 1000) / s_rated  # Ω (归算到高压侧)
        z_actual = (z_percent / 100) * z_base
        xfm1_impedance = (z_actual.real, z_actual.imag)

        # 线路阻抗数据（包括变压器）#note:这个阻抗数据不准确，后面再改
        self.line_impedance = {
            ('650', '632'): (0.5, 1.0),
            ('632', '670'): (0.6, 0.7),
            ('670', '671'): (0.7, 0.8),
            ('671', '680'): (0.8, 0.9),
            ('632', '633'): (0.8, 1.0),
            ('633', '634'): xfm1_impedance,  # 变压器阻抗
            ('632', '645'): (1.2, 1.3),
            ('645', '646'): (1.2, 1.3),
            ('671', '684'): (0.7, 0.8),
            ('684', '611'): (1.0, 1.1),
            ('684', '652'): (1.3, 1.4),
            ('671', '692'): (0.9, 1.0),
            ('692', '675'): (1.3, 1.4),
            ('SourceBus', '650'): sub_impedance,  # 变压器阻抗
        }

        # self.buses = {'692', '671', '645', '646', '650', '680', '684', '652', 'SourceBus', '633', '632', '611',
        #               '634a', '634b', '634c',
        #               '675a', '675b', '675c',
        #               '670a', '670b', '670c',
        #               }
        self.buses = {'692', '675', '684', '652', 'SourceBus', '633', '632', '634', '611', '670', '671', '645', '646', '650', '680'}

        #  负载节点名称
        self.load_name = ['671', '634a', '634b', '634c', '645', '646',
                          '692', '675a', '675b', '675c', '611', '652',
                          '670a', '670b', '670c']#note:env的loadname

        self.load_mapping = self.create_load_mapping(self.load_name)  # 关键添加
        print(self.load_mapping)

    def create_load_mapping(self, load_names):
        """创建节点->负载名称的映射"""
        load_mapping = defaultdict(list)
        for name in load_names:
            node = name[:3]  # 提取前3位作为节点ID（如'670a'->'670'）
            load_mapping[node].append(name)
        return dict(load_mapping)


if __name__ == "__main__":
    StaticGridData()


