"""Configuration parameters for this example."""

from revolve2.standards.modular_robots_v2 import gecko_v2

# Database and run configuration
DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS = 2
NUM_SIMULATORS = 4
NUM_GENERATIONS = 200
BODY = gecko_v2()

# CMA-ES configuration
INITIAL_STD = 0.5

# Optimization algorithm selection: "cmaes" or "de"
OPTIMIZER = "cmaes"

# Differential Evolution configuration
POPULATION_SIZE = 100
DE_F = 0.8
DE_CR = 0.9

# Simulation parameters
SIMULATION_SECONDS = 30

# Generation transition for weighting
TRANSITION_LENGTH = 10  # generations for weight transition

# Fitness weights
W_HEIGHT = 1.0
W_MOVE_MAX = 1.5
W_YAW = 0.3
FALL_PENALTY = 1.0

# Thresholds
FALL_HEIGHT_THRESHOLD = 0.06  # m
H_MIN, H_MAX = 0.06, 0.25

# Parameter bounds for controllers
BOUNDS = (-1.0, 1.0)
