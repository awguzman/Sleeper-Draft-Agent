"""
This module initializes the PPO based drafting agents.
"""

import torch
import torch.nn as nn

from src import config

class DraftAgent(nn.Module):
    """
    PPO Agent with Cross-Attention Architecture.

    This agent processes two distinct streams of information:
    1. The "Team Overview" (Draft State): Draft slot + draft progress + agents roster + opponents rosters.
    2. The "Draft Board" (Player Pool): Top N available players to pick from.
    """
    def __init__(self,
                 n_players_window=config.N_PLAYERS_WINDOW,
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

        # --- Team/Slot Embedding ---
        # Learns a unique vector for each of the NUM_TEAMS draft slots.
        # This allows the agent to learn slot-specific strategies.
        self.team_embedding = nn.Embedding(config.NUM_TEAMS, team_embed_dim)

        # --- Roster/Context Encoder ---
        # Input: Roster counts + Team Embedding
        # Output: A "Query" vector representing the team's specific needs and the overall draft landscape.
        self.roster_encoder = nn.Sequential(
            nn.Linear(roster_feat_dim + team_embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # --- Player Board Encoder ---
        # Input: Raw stats for a single player (VOR, Value, Position).
        # Output: A "Key/Value" vector representing the player's quality.
        self.player_encoder = nn.Sequential(
            nn.Linear(player_feat_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # --- Cross-Attention Layer ---
        # Logic: Compares the "Need" (Query) against every "Player" (Key).
        # Result: A weighted summary of the board, focusing on players that match the current need.
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # --- Critic Head ---
        # Looks at the summarized board context to predict the final reward.
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        # --- Actor Head ---
        # Assigns a specific score (logit) to each of the N top players.
        # Combines a player's specific quality with the draft's context.
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

        # 1. Team Embedding
        team_emb = self.team_embedding(team_idx)

        # 2. Feature Encoding
        # Concatenate roster features with team embedding to get general draft context.
        combined_roster_features = torch.cat([roster_features, team_emb], dim=1)

        # Convert the roster state into a roster vector.
        roster_emb = self.roster_encoder(combined_roster_features).unsqueeze(1)

        # Convert the N Players into player vectors.
        player_emb = self.player_encoder(player_features)

        # 3. Cross-Attention
        # Looks at the roster vector to "highlight" the player vectors that align with team needs.
        # Creates a context vector representing the board filtered by positional needs and draft situation.
        attn_output, _ = self.attention(
            query=roster_emb,
            key=player_emb,
            value=player_emb,
            key_padding_mask=mask
        )

        # 4. Critic Evaluation
        # The Critic looks at the context vector to estimate how "good" a particular situation is.
        context_vector = attn_output.squeeze(1)
        state_value = self.critic_head(context_vector)

        # 5. Actor Decision
        # Combine the roster vector with the player vectors.
        actor_input = player_emb + roster_emb

        # Pass the combined features through the Actor to get a score for each player.
        action_scores = self.actor_head(actor_input)
        action_logits = action_scores.squeeze(-1)

        # If a player is invalid (e.g., position limit reached),
        # set their logit to negative infinity so the probability of being picked becomes 0.
        if mask is not None:
            action_logits = action_logits.masked_fill(mask, float('-inf'))

        return action_logits, state_value
