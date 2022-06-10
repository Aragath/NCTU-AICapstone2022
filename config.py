import numpy as np
from kaggle_environments import make

# Read env specification
ENV_SPECIFICATION = make('kore_fleets').specification
SHIP_COST = ENV_SPECIFICATION.configuration.spawnCost.default
SHIPYARD_COST = ENV_SPECIFICATION.configuration.convertCost.default
GAME_CONFIG = {
    'episodeSteps':  ENV_SPECIFICATION.configuration.episodeSteps.default,  # You might want to start with smaller values
    'size': ENV_SPECIFICATION.configuration.size.default,
    'maxLogLength': None
}

# Define your opponent. We'll use the starter bot in the notebook environment for this baseline.
OPPONENT = 'opponent.py'
GAME_AGENTS = [None, OPPONENT]

# Define our parameters
N_1D_FEATURES = 3
N_2D_FEATURES = 5
MAX_FP_LEN = 10

ACTION_LEN = 4
ACTION_SIZE = (GAME_CONFIG["size"] * GAME_CONFIG["size"] * ACTION_LEN, )
OBSERVATION_SIZE = (GAME_CONFIG["size"] * GAME_CONFIG["size"] * N_2D_FEATURES + N_1D_FEATURES, )
DTYPE = np.float64

MAX_OBSERVABLE_KORE = 500
MAX_OBSERVABLE_SHIPS = 200

MIN_SPAWN_LIMIT = 1
MAX_SPAWN_LIMIT = 10

MAX_ACTION_FLEET_SIZE = 150
MAX_KORE_IN_RESERVE = 40000
WIN_REWARD = 1000