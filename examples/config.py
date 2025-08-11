"""Configuration parameters for this example."""

DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS = 2
NUM_SIMULATORS = 4
POPULATION_SIZE = 100
OFFSPRING_SIZE = 50
NUM_GENERATIONS = 200
TRANSITION_LENGTH      = 10           # 过渡代

SIMULATION_SECONDS        = 30
FALL_HEIGHT_THRESHOLD     = 0.06   # m

# ---------- 权重 ----------
W_HEIGHT               = 1.0
W_MOVE_MAX             = 1.5          # 走路阶段最大位移权重
W_YAW                  = 0.3          # 走偏惩罚
FALL_PENALTY           = 1.0
# ---------- 阈值 ----------
FALL_HEIGHT_THRESHOLD  = 0.06         # m
H_MIN, H_MAX           = 0.06, 0.25   # clamp 区间