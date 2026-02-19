"""
This module initializes the reinforcement learning based drafting agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src import config

class DraftAgent(nn.Module):
    """
    PPO Agent with Cross-Attention Architecture.

    This agent processes two distinct streams of information:
    1. The "Team Overview" (Roster State): What positions do I need? What are my opponents doing?
    2. The "Draft Board" (Player Pool): Who are the top N available players?

    Architecture Flow:
    - Encoders: Translate raw numbers (player counts, VOR) into vector representations.
    - Cross-Attention: Uses the Roster state (Query) to highlight relevant Players (Keys/Values) based on team needs.
    - Critic Head: Estimates the "win" probability of the current state.
    - Actor Head: Scores each specific player to make a draft pick.
    """
    def __init__(self, n_players_window=config.N_PLAYERS_WINDOW, 
                 player_feat_dim=config.PLAYER_FEAT_DIM, 
                 roster_feat_dim=config.ROSTER_FEAT_DIM,
                 embed_dim=config.EMBED_DIM, 
                 num_heads=config.NUM_HEADS, 
                 team_embed_dim=config.TEAM_EMBED_DIM):
        """
        :param n_players_window: Number of top available players to consider.
        :param player_feat_dim: Number of input features per player.
        :param roster_feat_dim: Number of input features for the roster context.
        :param embed_dim: Dimension of the internal embeddings.
        :param num_heads: Number of attention heads.
        :param team_embed_dim: Dimension of the team/draft slot embedding.
        """
        super(DraftAgent, self).__init__()
        self.n_players_window = n_players_window

        # 1. Team/Slot Embedding
        # Learns a unique vector for each of the 12 draft slots.
        # This allows the agent to learn slot-specific strategies (e.g. "I'm picking 1st" vs "I'm picking 12th").
        self.team_embedding = nn.Embedding(config.NUM_TEAMS, team_embed_dim)

        # 2. Roster/Context Encoder
        # Input: Raw player counts + Team Embedding
        # Output: A "Query" vector representing the team's specific needs and the draft landscape.
        self.roster_encoder = nn.Sequential(
            nn.Linear(roster_feat_dim + team_embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 3. Player Board Encoder
        # Input: Raw stats for a single player (VOR, Value, Position).
        # Output: A "Key/Value" vector representing the player's quality.
        self.player_encoder = nn.Sequential(
            nn.Linear(player_feat_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 4. Cross-Attention Layer
        # Logic: Compares the "Need" (Query) against every "Player" (Key).
        # Result: A weighted summary of the board, focusing on players that match the current need.
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # 5. Critic Head
        # Looks at the summarized board context to predict the final reward.
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        # 6. Actor Head
        # Assigns a specific score (logit) to each of the N top players.
        # Combines a player's specific quality with the Team's general context.
        self.actor_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, roster_features, player_features, mask=None, team_idx=None):
        """
        Forward pass for the agent.

        :param roster_features: Tensor of shape (Batch, Roster_Feat_Dim)
                                Contains [My_Counts, Opponent_1_Counts, ..., Opponent_11_Counts]
        :param player_features: Tensor of shape (Batch, N, Player_Feat_Dim)
                                Contains [VOR, Value, Pos_OneHot]
        :param mask: Optional boolean mask for invalid players (Batch, N). True indicates value should be ignored.
        :param team_idx: Tensor of shape (Batch,) containing the team index (0-11) for each environment.

        :return: (action_logits, state_value)
        """
        batch_size = roster_features.size(0)

        # 1. Team Embedding
        team_emb = self.team_embedding(team_idx) # (Batch, Team_Embed_Dim)

        # 2. Feature Encoding
        # Concatenate Roster Features with Team Embedding
        # Shape: (Batch, Roster_Feat_Dim + Team_Embed_Dim)
        combined_roster_features = torch.cat([roster_features, team_emb], dim=1)

        # Convert the Roster state into a Query vector.
        # Shape: (Batch, 1, Embed_Dim)
        roster_emb = self.roster_encoder(combined_roster_features).unsqueeze(1)

        # Convert the N Players into Key/Value vectors.
        # Shape: (Batch, N, Embed_Dim)
        player_emb = self.player_encoder(player_features)

        # 3. Cross-Attention
        # Looks at the Roster (Query) vector to highlight the Players (Keys) vectors that align with team needs.
        # attn_output represents the "Context vector" of the board filtered by our needs.
        # Shape: (Batch, 1, Embed_Dim)
        attn_output, _ = self.attention(
            query=roster_emb,
            key=player_emb,
            value=player_emb,
            key_padding_mask=mask
        )

        # 4. Critic Evaluation
        # The Critic looks at the Context Vector to estimate how "good" this situation is.
        # Squeeze removes the sequence dimension: (Batch, 1, D) -> (Batch, D)
        context_vector = attn_output.squeeze(1)
        state_value = self.critic_head(context_vector)

        # 5. Actor Decision
        # Combine the "Global Context" (roster_emb) with the "Local Features" (player_emb).
        # roster_emb (Batch, 1, D) is added to every player in player_emb (Batch, N, D).
        actor_input = player_emb + roster_emb

        # Pass the combined features through the Actor to get a score for each player.
        # Shape: (Batch, N, 1)
        action_scores = self.actor_head(actor_input)

        # Squeeze to get a flat list of logits: (Batch, N)
        action_logits = action_scores.squeeze(-1)

        # 6. Action Masking
        # If a player is invalid (e.g., position limit reached),
        # set their logit to negative infinity so the probability of being picked becomes 0.
        if mask is not None:
            action_logits = action_logits.masked_fill(mask, float('-inf'))

        return action_logits, state_value
