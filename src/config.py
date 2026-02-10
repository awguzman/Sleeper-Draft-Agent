"""
Configuration file for the Sleeper Draft Agent project.
Contains hyperparameters for training, environment settings, and model architecture.
"""

import torch

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Environment Settings ---
NUM_TEAMS = 12
NUM_ROUNDS = 16
ROSTER_LIMITS = {
    'QB': 2,
    'RB': 8,
    'WR': 8,
    'TE': 3
}
POSITIONS = ['QB', 'RB', 'WR', 'TE']

# --- Model Architecture ---
N_PLAYERS_WINDOW = 24       # Number of top available players to consider (Two full rounds)
PLAYER_FEAT_DIM = 6         # [VOR, Value, Pos_QB, Pos_RB, Pos_WR, Pos_TE]
# Roster dim is calculated dynamically in train.py based on NUM_TEAMS
EMBED_DIM = 64
NUM_HEADS = 4

# --- PPO Hyperparameters ---
LR = 0.0003                 # Learning Rate (standard PPO value)
LR_FINAL = 0.00003          # Final learning rate after decay
GAMMA = 0.99                # Discount Factor (focus on long-term)
EPS_CLIP = 0.2              # PPO Clip Parameter
K_EPOCHS = 4                # Number of optimization epochs per update
UPDATE_TIMESTEP = 1920      # Update policy every 10 drafts (assuming 12 team, 16 round)

# --- Entropy Settings ---
ENTROPY_COEF = 0.01         # Initial entropy coefficient
ENTROPY_FINAL = 0.001       # Final entropy coefficient after decay

# --- Training Settings ---
MAX_EPISODES = 20000        # Total number of episodes to train
LOG_INTERVAL = 10           # Print logs every n episodes
PRINT_ROSTER_INTERVAL = 100 # Print a sample roster every n episodes
SAVE_MODEL_INTERVAL = 1000  # Save model checkpoint every n episodes

# --- Calculate Decay Rates ---
LR_DECAY_RATE = (LR - LR_FINAL) / MAX_EPISODES
ENTROPY_DECAY_RATE = (ENTROPY_COEF - ENTROPY_FINAL) / MAX_EPISODES
