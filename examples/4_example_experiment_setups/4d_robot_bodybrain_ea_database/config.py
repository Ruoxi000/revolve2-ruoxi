"""Configuration parameters for this example."""

DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS = 5
NUM_SIMULATORS = 8
POPULATION_SIZE = 100
OFFSPRING_SIZE = 50
NUM_GENERATIONS = 100

# Three-phase fitness weights
STAND_PHASE_FRAC = 0.10
TRANSITION_LENGTH = 10

# Simulation time (seconds)
SIM_TIME = 30.0

# Fitness weights
W_HEIGHT = 1.0
W_MOVE_MAX = 4.0
W_YAW = 0.3

# Adaptive fall penalty parameters
FALL_HEIGHT_THRESHOLD = 0.06
FALL_EVENT_MIN_DURATION = 0.10
FALL_PENALTY_BASE = 0.5
FALL_PENALTY_PER_EXTRA = 0.25
FALL_PENALTY_FRAC_WEIGHT = 2.0
FALL_PENALTY_PHASE_GAIN = 0.2

# Height normalization range
H_MIN, H_MAX = 0.06, 0.25
