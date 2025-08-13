"""Configuration parameters (fixed-body, CPG controller)."""

from revolve2.standards.modular_robots_v1 import gecko_v1
from revolve2.standards.modular_robots_v2 import gecko_v2

# -----------------------------
# Database / run configuration
# -----------------------------
DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS = 1
NUM_SIMULATORS = 8

# 选择优化器: "cmaes" | "de" | "hybrid"
OPTIMIZER = "hybrid"

# -----------------------------
# Robot body (固定形态)
# -----------------------------
# BODY = gecko_v1()
BODY = gecko_v2()   # 如需 V2 版本可切换

# -----------------------------
# 三段式权重（仿真时长恒定）
#   - 前 STAND_PHASE_FRAC * NUM_GENERATIONS 强化站稳
#   - 中间 TRANSITION_LENGTH 代线性过渡
#   - 之后训练步态
# -----------------------------
STAND_PHASE_FRAC = 0.20
TRANSITION_LENGTH = 10

# 统一仿真秒数（与 rerun.py 兼容保留 SIM_TIME_WALK 名称）
SIM_TIME = 30
SIM_TIME_WALK = SIM_TIME   # rerun.py 使用此常量

# Fitness 权重
W_HEIGHT = 1.0           # 站高权重（归一后）
W_MOVE_MAX = 5.0         # xy 位移最大权重（后期拉满）
W_YAW = 0.3              # 偏航惩罚权重（后期开启/渐进）

# -----------------------------
# 自适应跌倒惩罚
# -----------------------------
FALL_HEIGHT_THRESHOLD = 0.06    # z 低于该高度视为“倒地中”
FALL_EVENT_MIN_DURATION = 0.15  # s，连续低于阈值这么久才算一次“跌倒事件”
FALL_PENALTY_BASE = 0.5         # 基线
FALL_PENALTY_PER_EXTRA = 0.25   # 每多一次跌倒的附加（β）
FALL_PENALTY_FRAC_WEIGHT = 2.0  # 跌倒时长占比的线性权重（γ）
FALL_PENALTY_PHASE_GAIN = 0.2   # 后期惩罚增强（ρ）

# 站高归一化区间（避免过度奖励）
H_MIN, H_MAX = 0.06, 0.25

# -----------------------------
# 控制器参数取值范围（CPG 权重）
# -----------------------------
BOUNDS = (-1.0, 1.0)

# -----------------------------
# DE 参数
# -----------------------------
POPULATION_SIZE = 100
DE_F = 0.6
DE_CR = 0.9

# -----------------------------
# CMA-ES 参数
# -----------------------------
INITIAL_STD = 0.25      # 初始步长（若 hybrid，会被用 top-K 方差自适应替换）
CMA_POPSIZE = None      # None = 使用 cma 默认 (通常 4+floor(3 ln n))

# -----------------------------
# Hybrid（先 DE 后 CMA）
#   要求：CMA 代数 = DE 代数（热启动）
# -----------------------------
HYBRID_DE_GENERATIONS = 50
HYBRID_CMA_GENERATIONS = 50

# 用多少个最优个体来初始化 CMA-ES 的 mean/std
HYBRID_TOPK_FOR_CMA = max(5, POPULATION_SIZE // 10)

# 供 Evaluator 做阶段切换使用的“总代数”
NUM_GENERATIONS = HYBRID_DE_GENERATIONS + HYBRID_CMA_GENERATIONS
