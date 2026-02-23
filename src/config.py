"""
Configuration file for the Sleeper Draft Agent project.
Contains hyperparameters for training, environment settings, and model architecture.
"""

import torch

# --- Device Configuration ---
#DEVICE = torch.device('cpu')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Environment Settings ---
NUM_TEAMS = 12                                          # Number of teams in the draft.
NUM_ROUNDS = 16                                         # Draft 9 starters + 7 bench players.
POSITIONS = ['QB', 'RB', 'WR', 'TE']                    # Standardize position abbreviations.
ROSTER_LIMITS = {'QB': 2,'RB': 6, 'WR': 7,'TE': 2}      # Max number of players per position before masking.
NUM_ENVS = 32                                           # Number of parallel environments. CPU-bound.

# --- Model Architecture ---
N_PLAYERS_WINDOW = 32           # Number of top available players to consider
PLAYER_FEAT_DIM = 6             # [VOR, Value, Pos_QB, Pos_RB, Pos_WR, Pos_TE]
ROSTER_FEAT_DIM = NUM_TEAMS * 4 # Number of input features for the roster context.
EMBED_DIM = 64                  # Internal dimension for all embeddings.
TEAM_EMBED_DIM = 16             # Dimension for the draft slot embedding.
NUM_HEADS = 4                   # Number of attention heads. Must divide EMBED_DIM.

# --- PPO Hyperparameters ---
LR = 0.001                                                  # Learning Rate
LR_FINAL = 0.0003                                           # Final learning rate after decay
GAMMA = 0.99                                                # Discount Factor (focus on long-term)
EPS_CLIP = 0.2                                              # PPO Clip Parameter
K_EPOCHS = 4                                                # Number of optimization epochs per update
GRAD_CLIP = 0.5                                             # Gradient Clipping
UPDATE_TIMESTEP = NUM_TEAMS * NUM_ROUNDS * NUM_ENVS         # Update policy every NUM_ENV drafts

# --- Entropy Settings ---
ENTROPY_COEF = 0.01         # Initial entropy coefficient
ENTROPY_FINAL = 0.0         # Final entropy coefficient after decay

# --- Training Settings ---
LEARNING_PHASE_EPISODES = NUM_ENVS * 100                            # Number of drafts to decay LR and Entropy over
REFINEMENT_PHASE_EPISODES = NUM_ENVS * 10                           # Number of drafts to refine exploitation strategy over
MAX_EPISODES = LEARNING_PHASE_EPISODES + REFINEMENT_PHASE_EPISODES  # Total number of drafts to train on
LOG_INTERVAL = NUM_ENVS                                             # Print log every NUM_ENVS drafts

# --- Calculate Decay Rates ---
# We calculate decay to reach FINAL values by LEARNING_PHASE_EPISODES.
# After that, the values will be clamped to FINAL in train.py.
LR_DECAY_RATE = (LR - LR_FINAL) / LEARNING_PHASE_EPISODES
ENTROPY_DECAY_RATE = (ENTROPY_COEF - ENTROPY_FINAL) / LEARNING_PHASE_EPISODES
