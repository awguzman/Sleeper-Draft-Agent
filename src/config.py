"""
Configuration file for the Sleeper Draft Agent project.
Contains hyperparameters for training, environment settings, and model architecture.
"""

import torch

# --- Device Configuration ---
#DEVICE = torch.device('cpu')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Draft Settings ---
NUM_TEAMS = 12                                      # Number of teams in the draft.
NUM_ROUNDS = 16                                     # Number of rounds in the draft.
# Roster slots per position on each team
ROSTER_SLOTS = {'QB':  1,
                'RB':  2,
                'WR':  3,
                'TE':  1,
                'K':   0,
                'DST': 0
                }
# Index for replacement player cutoffs for VOR calculation.
REPLACEMENT_CUTOFFS = {'QB': (NUM_TEAMS * ROSTER_SLOTS['QB']) // 2,
                       'RB': (NUM_TEAMS * ROSTER_SLOTS['RB']) // 2,
                       'WR': (NUM_TEAMS * ROSTER_SLOTS['WR']) // 2,
                       'TE': (NUM_TEAMS * ROSTER_SLOTS['TE']) // 2,
                       'K':  (NUM_TEAMS * ROSTER_SLOTS['K']) // 2,
                       'DST':(NUM_TEAMS * ROSTER_SLOTS['DST']) // 2
                       }
# Max number of players per position before masking.
ROSTER_LIMITS = {'QB':  ROSTER_SLOTS['QB'] * 2,
                 'RB':  min(ROSTER_SLOTS['RB'] * 3, 5),
                 'WR':  min(ROSTER_SLOTS['WR'] * 3, 8),
                 'TE':  ROSTER_SLOTS['TE'] * 2,
                 'K':   ROSTER_SLOTS['K'],
                 'DST': ROSTER_SLOTS['DST']
                 }
POSITIONS = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']    # Standardize position abbreviations.

# --- Model Architecture ---
N_PLAYERS_WINDOW = NUM_TEAMS * 3                # Number of top available players to consider (3 rounds ahead)
PLAYER_FEAT_DIM = 2 + 6                         # [VOR, Value, Pos_QB, Pos_RB, Pos_WR, Pos_TE, Pos_K, Pos_DST]
ROSTER_FEAT_DIM = NUM_TEAMS * len(POSITIONS)    # Roster compositions for each team combined.
TEAM_EMBED_DIM = 16                             # Dimension for the draft slot embedding.
EMBED_DIM = 128                                 # Internal dimension for all embeddings.
NUM_HEADS = 8                                   # Number of attention heads. Must divide EMBED_DIM.

# --- Training Settings ---
NUM_ENVS = 32                                                       # Number of parallel environments. CPU-bound.
LEARNING_PHASE_EPISODES = NUM_ENVS * 100                            # Number of drafts to decay LR and Entropy over
REFINEMENT_PHASE_EPISODES = NUM_ENVS * 10                           # Number of drafts to refine exploitation strategy over
MAX_EPISODES = LEARNING_PHASE_EPISODES + REFINEMENT_PHASE_EPISODES  # Total number of drafts to train on
LOG_INTERVAL = NUM_ENVS                                             # Print log every NUM_ENVS drafts

# --- PPO Hyperparameters ---
LR = 0.0003                                                  # Learning Rate
LR_FINAL = 0.0003                                           # Final learning rate after decay
GAMMA = 0.99                                                # Discount Factor (focus on long-term)
EPS_CLIP = 0.2                                              # PPO Clip Parameter
K_EPOCHS = 4                                                # Number of optimization epochs per update
GRAD_CLIP = 0.5                                             # Gradient Clipping
UPDATE_TIMESTEP = NUM_TEAMS * NUM_ROUNDS * NUM_ENVS         # Update policy every NUM_ENV drafts

# --- Entropy Settings ---
ENTROPY_COEF = 0.01         # Initial entropy coefficient
ENTROPY_FINAL = 0.0         # Final entropy coefficient after decay

# --- Calculate Decay Rates ---
# We calculate decay to reach FINAL values by LEARNING_PHASE_EPISODES.
# After that, the values will be clamped to FINAL in train.py.
LR_DECAY_RATE = (LR - LR_FINAL) / LEARNING_PHASE_EPISODES
ENTROPY_DECAY_RATE = (ENTROPY_COEF - ENTROPY_FINAL) / LEARNING_PHASE_EPISODES
