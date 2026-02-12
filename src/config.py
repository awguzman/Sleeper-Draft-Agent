"""
Configuration file for the Sleeper Draft Agent project.
Contains hyperparameters for training, environment settings, and model architecture.
"""

import torch

# --- Device Configuration ---
#DEVICE = torch.device('cpu')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Environment Settings ---
NUM_TEAMS = 12
NUM_ROUNDS = 16                                         # Draft 9 starters + 7 bench players.
ROSTER_LIMITS = {'QB': 2,'RB': 8,'WR': 8,'TE': 3}       # Max number of players per position.
POSITIONS = ['QB', 'RB', 'WR', 'TE']                    # Standardize position names.
NUM_ENVS = 32                                           # Number of parallel environments.

# --- Model Architecture ---
N_PLAYERS_WINDOW = 32       # Number of top available players to consider (Three full rounds in the future)
PLAYER_FEAT_DIM = 6         # [VOR, Value, Pos_QB, Pos_RB, Pos_WR, Pos_TE]
# ROSTER_DIM is calculated dynamically in train.py based on NUM_TEAMS
EMBED_DIM = 64              # Internal dimension for all embeddings. Large enough to embed ROSTER_DIM with 12 teams.
NUM_HEADS = 4               # Number of attention heads.

# --- PPO Hyperparameters ---
LR = 0.01                                                   # Learning Rate
LR_FINAL = 0.001                                            # Final learning rate after decay
GAMMA = 0.99                                                # Discount Factor (focus on long-term)
EPS_CLIP = 0.2                                              # PPO Clip Parameter
K_EPOCHS = 4                                                # Number of optimization epochs per update
GRAD_CLIP = 0.5                                             # Gradient Clipping
UPDATE_TIMESTEP = NUM_TEAMS * NUM_ROUNDS * NUM_ENVS         # Update policy every NUM_ENV drafts

# --- Entropy Settings ---
ENTROPY_COEF = 0.01         # Initial entropy coefficient
ENTROPY_FINAL = 0.001       # Final entropy coefficient after decay

# --- Training Settings ---
MAX_EPISODES = NUM_ENVS * 1000 * 2      # Total number of episodes (drafts) to train on
LOG_INTERVAL = NUM_ENVS                 # Print logs every n episodes (drafts)
SAVE_MODEL_INTERVAL = NUM_ENVS * 100    # Save model checkpoint every n episodes (drafts)

# --- Calculate (linear) Decay Rates ---
LR_DECAY_RATE = (LR - LR_FINAL) / MAX_EPISODES
ENTROPY_DECAY_RATE = (ENTROPY_COEF - ENTROPY_FINAL) / MAX_EPISODES
