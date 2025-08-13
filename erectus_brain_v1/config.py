"""Configuration parameters for this example."""
from revolve2.standards.modular_robots_v1 import gecko_v1
from revolve2.standards.modular_robots_v2 import gecko_v2

# Database and run configuration
DATABASE_FILE = "database_v2_100gen_DE_Speed=3_middle.sqlite"


NUM_REPETITIONS = 1
NUM_SIMULATORS = 8
NUM_GENERATIONS = 100

# BODY = gecko_v1()
BODY = gecko_v2()

# CMA-ES configuration
INITIAL_STD = 0.25

# Optimization algorithm selection: "cmaes" or "de"
OPTIMIZER = "de"

# Differential Evolution configuration
POPULATION_SIZE = 100
DE_F = 0.6
DE_CR = 0.9

# Simulation parameters
SIMULATION_SECONDS = 30

# Generation transition for weighting
TRANSITION_LENGTH = 10  # generations for weight transition

# Fitness weights
W_HEIGHT = 1.0
W_MOVE_MAX = 4.0
W_YAW = 0.3
FALL_PENALTY = 0.5

# --- Adaptive fall penalty ---
FALL_EVENT_MIN_DURATION = 0.15   # s，低于阈值连续这么久才算一次跌倒
FALL_PENALTY_BASE = 0.5          # 基线
FALL_PENALTY_PER_EXTRA = 0.25    # 每多一次跌倒额外加成（beta）
FALL_PENALTY_FRAC_WEIGHT = 2.0   # 跌倒时长占比的线性权重（gamma）
FALL_PENALTY_PHASE_GAIN = 0.2    # 后期加严系数（rho）


# Thresholds
FALL_HEIGHT_THRESHOLD = 0.06  # m
H_MIN, H_MAX = 0.06, 0.25

# Parameter bounds for controllers
BOUNDS = (-1.0, 1.0)
