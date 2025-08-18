"""Configuration parameters (fixed-body, CPG controller)
- Constant 30s simulation for ALL generations
- Three-phase fitness weights (stand -> transition -> walk) without changing sim time
- RevDE/jDE tricks + CMA-ES hot-start + mild restarts
"""

from revolve2.standards.modular_robots_v1 import gecko_v1
from revolve2.standards.modular_robots_v2 import gecko_v2

# -----------------------------
# Database / run configuration
# -----------------------------
DATABASE_FILE = "database_v1_W4_Hhigh_hybrid.sqlite"
NUM_REPETITIONS = 2
NUM_SIMULATORS = 8   # 运行cmas时只能开到4

# Choose optimizer: "cmaes" | "de" | "hybrid" (DE -> CMA-ES)
OPTIMIZER = "hybrid"

# -----------------------------
# Fixed robot body (template)
# -----------------------------
BODY = gecko_v1()
# BODY = gecko_v2()   # switch to v2 template if desired

# -----------------------------
# Three-phase weights (constant sim time = 30s)
#   - First STAND_PHASE_FRAC of total generations: stand priority
#   - TRANSITION_LENGTH generations: linear transition
#   - Remaining generations: walk priority
# -----------------------------
STAND_PHASE_FRAC = 0.10
TRANSITION_LENGTH = 10

# Constant simulation time in seconds (keep SIM_TIME_WALK name for rerun.py compatibility)
SIM_TIME = 30.0
SIM_TIME_WALK = SIM_TIME

# Fitness weights (Evaluator will modulate w_move & w_yaw per phase)
W_HEIGHT = 1.0           # height term (normalized)
W_MOVE_MAX = 4.0         # max weight for XY displacement (later phases)
W_YAW = 0.3              # yaw penalty weight (later phases)

# -----------------------------
# Adaptive fall penalty
# -----------------------------
FALL_HEIGHT_THRESHOLD = 0.06    # meters under which we consider "fallen/lying"
FALL_EVENT_MIN_DURATION = 0.10  # seconds required under threshold to count as a fall-event
FALL_PENALTY_BASE = 0.5         # base penalty
FALL_PENALTY_PER_EXTRA = 0.25   # extra penalty per additional fall-event
FALL_PENALTY_FRAC_WEIGHT = 2.0  # penalty proportional to fraction of time fallen
FALL_PENALTY_PHASE_GAIN = 0.2   # penalty gain towards the walking phase

# Height normalization range (prevents overly tall shapes from dominating)
H_MIN, H_MAX = 0.06, 0.25

# -----------------------------
# Controller parameter bounds (CPG weights)
# -----------------------------
BOUNDS = (-1.0, 1.0)

# ============================================================
# Differential Evolution (RevDE / jDE-inspired improvements)
# ============================================================
POPULATION_SIZE = 100

# Base F/CR (used unless adapted)
DE_F = 0.5
DE_CR = 0.9

# jDE-style self-adaptation:
DE_USE_JDE = True         # enable per-individual adaptation of F and CR
DE_TAU_F = 0.1            # probability to resample F each generation
DE_TAU_CR = 0.1           # probability to resample CR each generation
DE_F_RANGE = (0.4, 0.9)   # resampling range for F
DE_CR_RANGE = (0.7, 0.95) # resampling range for CR

# Dither (light randomness even if not using jDE) – applied to a small fraction:
DE_USE_DITHER = True
DE_DITHER_PROB = 0.15     # fraction of individuals to dither
DE_DITHER_F_RANGE = (0.3, 0.9)
DE_DITHER_CR_RANGE = (0.6, 0.95)

# Elite injection (maintain diversity and exploit elites slightly)
DE_ELITE_INJECTION = True
DE_ELITE_K = 5            # keep top-K as elites
DE_INJECTION_RATE = 0.10  # fraction of worst individuals to overwrite by jittered elites
DE_INJECTION_NOISE = 0.05 # Gaussian noise std applied to injected elites (in [-1,1] scale)

# Mutation strategy selection
#   "rand1bin"     – classic DE/rand/1/bin
#   "best1bin"     – exploit best individual
#   "current2best" – current-to-best/1 (balanced)
DE_STRATEGY = "rand1bin"

# ============================================================
# CMA-ES (hot-start + mild restart)
# ============================================================
INITIAL_STD = 0.25         # default sigma if not hot-started
CMA_POPSIZE = None         # None: let cma use 4 + floor(3 ln n)
CMA_ACTIVE = True          # CMA_active – negative weights on bad directions
CMA_ELITIST = True         # keep best individual
CMA_DIAGONAL = 15           # diagonal covariance for first N iterations (stabilizes early stage)

# Hybrid (DE -> CMA): both stages lengths (require CMA gens == DE gens)
HYBRID_DE_GENERATIONS = 30
HYBRID_CMA_GENERATIONS = 30

# Hot-start from DE: how many elites to compute mean/std from
HYBRID_TOPK_FOR_CMA = max(5, POPULATION_SIZE // 10)

# CMA restarts (simple IPOP-style)
CMA_USE_RESTARTS = False
CMA_STALL_GENS = 40        # restart if no improvement for this many generations
CMA_RESTARTS_MAX = 2       # max restarts
CMA_POPSIZE_INC = 1.5        # multiply popsize per restart
CMA_SIGMA_BOOST = 1.4      # multiply sigma per restart

# Optional re-evaluation of top candidates to reduce noise
CMA_TOPK_REEVAL = 3        # re-evaluate top-k from current generation
CMA_REEVAL_TIMES = 2       # total runs for these (1 original + (REEVAL_TIMES-1) repeats)

# Total generations used by Evaluator for phase switching
NUM_GENERATIONS = HYBRID_DE_GENERATIONS + HYBRID_CMA_GENERATIONS
