from collections import defaultdict
import numpy as np


class StaticGridData:
    """静态电网数据结构（IEEE 13节点专用）"""

    def __init__(self):
        # 节点下游关系（不包含自身）
        # self.downstream_map = {
        #     '650': ['611', '632', '633', '634', '645', '646', '652', '670', '671', '675', '680', '684', '692'],
        #     '632': ['611', '633', '634', '645', '646', '652', '670', '671', '675', '680', '684', '692'],
        #     '633': ['634'],
        #     '634': [],
        #     '645': ['646'],
        #     '646': [],
        #     '670': ['611', '652', '671', '675', '680', '684', '692'],
        #     '671': ['611', '652', '675', '680', '684', '692'],
        #     '684': ['611', '652'],
        #     '692': ['675'],
        # }
        self.downstream_map = {
            'SourceBus': ['611', '632', '633', '634', '645', '646', '650', '652', '670', '671', '675', '680', '684', '692'],
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
            '611': [],
            '652': [],
            '675': [],
            '680': []
        }
        self.S_base = 100000 #100000kVA,100MVA
        self.V_base = 4.16 #kV
        self.Z_base = (self.V_base ** 2*1000) / (self.S_base)  # 欧姆

        self.lines = [
            # (名称, 相数, 起始节点, 起始相别, 终止节点, 终止相别, 长度, 长度单位,  R矩阵, X矩阵, C矩阵, 阻抗单位, 开关标识)

            # 三相线路
            ('650632', 3, '650', ['1', '2', '3'], '632', ['1', '2', '3'], 2000, 'ft',
             [0.3465, 0.1560, 0.3375, 0.1580, 0.1535, 0.3414],
             [1.0179, 0.5017, 1.0478, 0.4236, 0.3849, 1.0348],
             None, 'mi',False),

            ('632670', 3, '632', ['1', '2', '3'], '670', ['1', '2', '3'], 667, 'ft',
             [0.3465, 0.1560, 0.3375, 0.1580, 0.1535, 0.3414],
             [1.0179, 0.5017, 1.0478, 0.4236, 0.3849, 1.0348],
             None, 'mi',False),

            ('670671', 3, '670', ['1', '2', '3'], '671', ['1', '2', '3'], 1333, 'ft',
             [0.3465, 0.1560, 0.3375, 0.1580, 0.1535, 0.3414],
             [1.0179, 0.5017, 1.0478, 0.4236, 0.3849, 1.0348],
             None, 'mi',False),

            ('671680', 3, '671', ['1', '2', '3'], '680', ['1', '2', '3'], 1000, 'ft',
             [0.3465, 0.1560, 0.3375, 0.1580, 0.1535, 0.3414],
             [1.0179, 0.5017, 1.0478, 0.4236, 0.3849, 1.0348],
             None, 'mi',False),

            ('632633', 3, '632', ['1', '2', '3'], '633', ['1', '2', '3'], 500, 'ft',
             [0.7526, 0.1580, 0.7475, 0.1560, 0.1535, 0.7436],
             [1.1814, 0.4236, 1.1983, 0.5017, 0.3849, 1.2112],
             None, 'mi',False),

            ('692675', 3, '692', ['1', '2', '3'], '675', ['1', '2', '3'], 500, 'ft',
             [0.791721, 0.318476, 0.781649, 0.28345, 0.318476, 0.791721],
             [0.438352, 0.0276838, 0.396697, -0.0184204, 0.0276838, 0.438352],
             [383.948, 0, 383.948, 0, 0, 383.948], 'mi',False),

            # 单相线路
            ('632645', 1, '632', ['2'], '645', ['2'], 500, 'ft',
             [1.3294],
             [1.3471],
             None, 'mi', False),
            ('645646', 1, '645', ['2'], '646', ['2'], 300, 'ft',
             [1.3294],
             [1.3471],
             None, 'mi', False),

            ('671684', 2, '671', ['1', '3'], '684', ['1', '3'], 300, 'ft',
             [1.3238, 0.2066, 1.3294],
             [1.3569, 0.4591, 1.3471],
             None, 'mi',False),

            # 单相线路
            ('684611', 1, '684', ['3'], '611', ['3'], 300, 'ft',
             [1.3292],
             [1.3475],
             None, 'mi',False),

            ('684652', 1, '684', ['1'], '652', ['1'], 800, 'ft',
             [1.3425],
             [0.5124],
             [236], 'mi',False),

            #开关线路
            ('671692', 3, '671', ['1', '2', '3'], '692', ['1', '2', '3'], 0, 'ft',
            [0.0001,0,0.0001,0,0,0.0001],
            [0,0,0,0,0,0],
            None,
             'mi',True) , # 新增开关标识

            # # 变压器改
            # ('633634', 3, '633', ['1', '2', '3'], '634', ['1', '2', '3'], 0, 'ft',
            #  [0.0000, 0, 0.0000, 0, 0, 0.0000],
            #  [0, 0, 0, 0, 0, 0],
            #  None,
            #  'mi', False),
            #
            # ('SourceBus650', 3, 'SourceBus', ['1', '2', '3'], '650', ['1', '2', '3'], 0, 'ft',
            #  [0.0000, 0, 0.0000, 0, 0, 0.0000],
            #  [0, 0, 0, 0, 0, 0],
            #  None,
            #  'mi', False),
        ]
        self.line_impedance = {}
        self.line_impedance_pu = {}
        self.calculate_line_impedance()
        self.calculate_line_impedance_pu()

        # 变压器阻抗计算
        self.add_transformer_impedances()

        self.buses = {'692', '675', '684', '652', 'SourceBus', '633', '632', '634', '611', '670', '671', '645', '646', '650', '680'}

        #  负载节点名称
        self.load_name = ['671', '634a', '634b', '634c', '645', '646',
                          '692', '675a', '675b', '675c', '611', '652',
                          '670a', '670b', '670c']#note:env的loadname

        self.load_mapping = {
            '671': ['671'],
            '634': ['634a', '634b', '634c'],
            '645': ['645'],
            '646': ['646'],
            '692': ['692'],
            '675': ['675a', '675b', '675c'],
            '611': ['611'],
            '652': ['652'],
            '670': ['670a', '670b', '670c']
        }
        print(self.load_mapping)

    def create_load_mapping(self, load_names):
        """创建节点->负载名称的映射"""
        load_mapping = defaultdict(list)
        for name in load_names:
            node = name[:3]  # 提取前3位作为节点ID（如'670a'->'670'）
            load_mapping[node].append(name)
        return dict(load_mapping)

    def calculate_line_impedance(self):
        for line in self.lines:
            name, num_phases, start_bus, start_phases, end_bus, end_phases, length, length_unit, R, X, C, imp_unit, switch_flag = line

            # 转换为英尺为英里
            if length_unit == 'ft':
                length_miles = length / 5280.0
            else:
                length_miles = length

            # 计算实际阻抗
            if num_phases == 1:
                # 单相线路
                R_actual = R[0] * length_miles
                X_actual = X[0] * length_miles
                Z = complex(R_actual, X_actual)
                self.line_impedance[(start_bus, end_bus)] = Z

            elif num_phases == 2:
                # 两相线路
                R_matrix = np.array([[R[0], R[1]], [R[1], R[2]]])
                X_matrix = np.array([[X[0], X[1]], [X[1], X[2]]])
                Z_matrix = np.zeros((2, 2), dtype=complex)
                for i in range(2):
                    for j in range(2):
                        Z_matrix[i, j] = complex(R_matrix[i, j], X_matrix[i, j]) * length_miles
                self.line_impedance[(start_bus, end_bus)] = Z_matrix

            elif num_phases == 3:
                # 三相线路
                R_matrix = np.array([
                    [R[0], R[1], R[2]],
                    [R[1], R[3], R[4]],
                    [R[2], R[4], R[5]]
                ])
                X_matrix = np.array([
                    [X[0], X[1], X[2]],
                    [X[1], X[3], X[4]],
                    [X[2], X[4], X[5]]
                ])
                Z_matrix = np.zeros((3, 3), dtype=complex)
                for i in range(3):
                    for j in range(3):
                        Z_matrix[i, j] = complex(R_matrix[i, j], X_matrix[i, j]) * length_miles
                self.line_impedance[(start_bus, end_bus)] = Z_matrix
    def calculate_line_impedance_pu(self):
        Z_base = self.Z_base  # 基准阻抗

        for line, impedance in self.line_impedance.items():
            # 复数类型（单相线路）
            if isinstance(impedance, complex):
                self.line_impedance_pu[line] = impedance / Z_base

            # NumPy数组类型（多相线路）
            elif isinstance(impedance, np.ndarray):
                # 直接对整个矩阵进行标幺值转换
                self.line_impedance_pu[line] = impedance / Z_base

    def add_transformer_impedances(self):
        """添加变压器阻抗到线路阻抗字典（三相建模）"""

        z_percent = complex(1.0, 8.0)  # R% = 1%, X% = 8%
        s_rated = 5000  # kVA (变压器额定容量)

        # 直接计算系统基准下的标幺值
        z_pu = (z_percent / 100) * (s_rated / self.S_base)

        # 创建三相阻抗矩阵
        Z_matrix = np.zeros((3, 3), dtype=complex)
        np.fill_diagonal(Z_matrix, z_pu)
        self.line_impedance_pu[('SourceBus', '650')] = Z_matrix

        # 2. XFM1变压器 (4.16kV/0.48kV)
        z_percent = complex(1.1, 2.0)  # R% = 1.1%, X% = 2%
        s_rated = 500  # kVA

        # 直接计算系统基准下的标幺值
        z_pu = (z_percent / 100) * (s_rated / self.S_base)

        # 创建三相阻抗矩阵
        Z_matrix = np.zeros((3, 3), dtype=complex)
        np.fill_diagonal(Z_matrix, z_pu)
        self.line_impedance_pu[('633', '634')] = Z_matrix

if __name__ == "__main__":
    grid = StaticGridData()
    print(grid.line_impedance)
    print("###########################")
    print(grid.line_impedance_pu)
    # 打印基准阻抗
    print(f"基准阻抗 Z_base = {grid.Z_base:.4f} Ω")

    # # 打印标幺阻抗
    # for line, Z_pu in grid.line_impedance_pu.items():
    #     # 单相线路
    #     if isinstance(Z_pu, complex):
    #         print(f"线路 {line}: R_pu = {Z_pu.real:.6f}, X_pu = {Z_pu.imag:.6f}")
    #
    #     # 多相线路
    #     elif isinstance(Z_pu, np.ndarray):
    #         print(f"线路 {line} (多相):")
    #         print(Z_pu)

