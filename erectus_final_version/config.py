"""Configuration parameters for this example."""

DATABASE_FILE = "database_new_500gen_best.sqlite"
NUM_REPETITIONS = 3
NUM_SIMULATORS = 4
POPULATION_SIZE = 100
OFFSPRING_SIZE = 50
NUM_GENERATIONS = 500

# Optional database from which to seed the initial population.
# SEED_DATABASE_FILE = "database_new_500gen_best.sqlite"
# SEED_TOP_K = 5

# Three-phase fitness weights
STAND_PHASE_FRAC = 0.0
TRANSITION_LENGTH = 0



# Simulation time (seconds)
SIM_TIME = 30.0

# Fraction of simulation to ignore at start when computing fitness.
FITNESS_START_FRACTION = 0.10

# Fitness weights
W_HEIGHT = 1          #previous 1
W_MOVE_MAX = 4.0        #previous 3
W_MOVE_STAND = 0.5      #previous 0.5
W_YAW = 0.3

# --------- Fall (simplified) ----------
# Height threshold below which we consider 'fallen'
FALL_HEIGHT_THRESHOLD = 0.10
# Per-event penalty (count-based)
FALL_PENALTY_PER_EVENT = 1.0

# Height drop penalty
HEIGHT_DROP_WEIGHT = 1.0

# --------- Inversion ----------
INVERT_PENALTY_PER_EVENT = 0.8

# --------- Frame-based event detection ----------
# Minimal consecutive True frames to count 1 event
FALL_EVENT_MIN_FRAMES   = 2
INVERT_EVENT_MIN_FRAMES = 2
# Minimal consecutive False frames between two fall events (recovery)
FALL_RECOVERY_MIN_FRAMES = 5

# --------- Uprightness assisted fall detection ----------
# u = dot(world+Z, body+Z) ∈ [-1,1]; consider "falling posture" if u < U_FALL_THR
U_FALL_THR = 0.15

# Inversion hysteresis (start/end thresholds) to resist jitter around 0
U_INVERT_START = -0.05   # start an inversion when u < this
U_INVERT_END   = +0.05   # end the inversion when u > this

# --------- Fraction-based extra counting (to avoid long single-penalty) ----------
# If fraction of fall/invert frames >= threshold, add +1 extra event
FALL_FRAC_BONUS_THRESHOLD   = 0.10
INVERT_FRAC_BONUS_THRESHOLD = 0.10

# Height normalization range
H_MIN, H_MAX = 0.10, 0.40

# ---- RERUN SETTINGS ----
RERUN_RANK     = 1
RERUN_TOPK     = 30
RERUN_HEADLESS = False


# --- Which body axis is "up" in local frame? ---
# 如果发现 u 曲线与“直立/倒置”的肉眼感觉不符，改为 'Y' 或 'X' 再试。
BODY_UP_AXIS = "Z"

# --- Orientation events (we now penalize "upright" segments to prefer brain-down) ---
# 如果 True：把“u>0 的朝上段”当作事件来惩罚（鼓励脑壳朝下）。
INVERT_EVENT_COUNTS_UPRIGHT = False