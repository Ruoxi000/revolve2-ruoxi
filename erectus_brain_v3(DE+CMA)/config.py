"""Configuration parameters for this example."""

DATABASE_FILE = "database_new_500gen_best.sqlite"
# Optional database from which to seed the initial population.
# If provided, the top K individuals from this database will be used to
# initialize part of the population for a new training run.
SEED_DATABASE_FILE = "seed_database.sqlite"
SEED_TOP_K = 10
NUM_REPETITIONS = 2
NUM_SIMULATORS = 4
POPULATION_SIZE = 100
OFFSPRING_SIZE = 50
NUM_GENERATIONS = 1000

# Three-phase fitness weights
STAND_PHASE_FRAC = 0.20
TRANSITION_LENGTH = 10

# Simulation time (seconds)
SIM_TIME = 30.0

# Fraction of simulation to ignore at start when computing fitness.
FITNESS_START_FRACTION = 0.10

# Fitness weights
W_HEIGHT = 2.0   # previous WH=1
W_MOVE_MAX = 4.0    # previous 3
W_MOVE_STAND = 0.5  # encourage slight movement even in stand phase
W_YAW = 0.5         # previous 0.3

# --------- Fall (simplified) ----------
# Height threshold below which we consider 'fallen'
FALL_HEIGHT_THRESHOLD = 0.10
# A fall-event must last at least this long (seconds)
FALL_EVENT_MIN_DURATION = 0.05
# NEW: simple per-event penalty (no time-fraction, no phase gain)
FALL_PENALTY_PER_EVENT = 1.0

# --------- Inversion (new) ------------
# We treat the trunk (brain) as inverted if its body +Z points below world +Z
# i.e., angle(world_up, body_up) > 90°. Event must last >= this duration.
INVERT_EVENT_MIN_DURATION = 0.1
# Fixed penalty per inversion event
INVERT_PENALTY_PER_EVENT = 0.5

# Height normalization range
H_MIN, H_MAX = 0.10, 0.30

# ---- RERUN SETTINGS ----
RERUN_RANK     = 1      # 想看第几名（按历史最好分、去重后的名次）
RERUN_TOPK     = 5      # 仅用于打印榜单作参考
RERUN_HEADLESS = False  # False = 弹窗; True = 不弹窗后台跑
