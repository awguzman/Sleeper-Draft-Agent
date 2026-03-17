"""
This module handles the interaction with the Sleeper API and manages the draft state
for the live dashboard.

It acts as the backend, translating live draft data into a format the trained model can understand and then running
inference to get a pick recommendation.
"""
import re
import requests
import polars as pl
import torch
import numpy as np
from nflreadpy import load_ff_playerids
import sys
import os

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.agent import DraftAgent
from src.board import create_board

def get_draft_metadata(draft_id):
    """
    Fetches metadata for a specific draft from the Sleeper API.

    This is used to match the given draft with the correct model for inference.

    :param draft_id: The ID of the Sleeper draft.
    :return: A dictionary containing key draft settings, or None if the request fails.
    """
    url = f"https://api.sleeper.app/v1/draft/{draft_id}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching draft metadata: {response.status_code}")
        return None
    
    data = response.json()
    settings = data.get('settings', {})
    
    # Extract relevant settings
    metadata = {
        'num_teams': settings.get('teams'),
        'num_rounds': settings.get('rounds'),
        'roster_slots': {
            'QB': settings.get('slots_qb', 0),
            'RB': settings.get('slots_rb', 0),
            'WR': settings.get('slots_wr', 0),
            'TE': settings.get('slots_te', 0),
            'K': settings.get('slots_k', 0),
            'DST': settings.get('slots_def', 0) # Sleeper uses 'def' for DST
        }
    }

    return metadata

class SleeperDraftManager:
    """
    Manages the connection to a live Sleeper draft, tracks the state,
    and generates drafting recommendations using the trained PPO agent.
    """
    def __init__(self, draft_id, model_path):
        """
        Initializes the manager.
        
        :param draft_id: The ID of the Sleeper draft to connect to.
        :param model_path: Path to the trained .pth model file.
        """
        self.draft_id = draft_id
        self.base_url = "https://api.sleeper.app/v1"

        #  Parse model filename to get draft configuration
        match = re.search(r'(\d+)team_(\d+)rounds_(\d+)QB_(\d+)RB_(\d+)WR_(\d+)TE_(\d+)K_(\d+)DST', model_path)
        if not match:
            print(f"Error: Could not parse model filename: {model_path}")
            return

        self.num_teams = int(match.group(1))
        self.num_rounds = int(match.group(2))
        self.roster_slots = {'QB': int(match.group(3)),
                             'RB': int(match.group(4)),
                             'WR': int(match.group(5)),
                             'TE': int(match.group(6)),
                             'K': int(match.group(7)),
                             'DST': int(match.group(8))
                             }

        self.device = torch.device("cpu")  # Use CPU for inference on dashboard
        self.model = DraftAgent(
            n_players_window=config.N_PLAYERS_WINDOW,
            player_feat_dim=config.PLAYER_FEAT_DIM,
            roster_feat_dim=self.num_teams * len(config.POSITIONS) + 1,
            embed_dim=config.EMBED_DIM,
            team_embed_dim=config.TEAM_EMBED_DIM
        ).to(self.device)

        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set to evaluation mode for inference
        
        # Load and set up the Board with ID Mapping
        self.full_board = create_board(preprocess=True)
        self._map_ids()
        
        # Initialize State
        self.available_players = self.full_board.clone()
        self.rosters = [
            {pos: [] for pos in config.POSITIONS} for _ in range(self.num_teams)
        ]
        self.current_pick_no = 0

    def _map_ids(self):
        """
        Fetches ID mapping from nflreadpy and merges it with the board.
        This allows us to translate Sleeper IDs to FantasyPros IDs.
        """
        # Load mapping table
        ids_df = load_ff_playerids()
        
        # Select relevant columns and ensure sleeper_id is a string to match API
        mapping_pl = ids_df.select(['sleeper_id', 'fantasypros_id']).with_columns(
            pl.col('sleeper_id').cast(pl.Utf8)
        )

        # Join with player board to add the 'sleeper_id' column
        self.full_board = self.full_board.join(
            mapping_pl,
            on='fantasypros_id',
            how='left'
        )

    def update_state(self):
        """
        Polls the Sleeper API for the latest picks and updates the internal draft state.
        """
        # Fetch all picks made so far in the draft
        response = requests.get(f"{self.base_url}/draft/{self.draft_id}/picks")
        if response.status_code != 200:
            print(f"Error fetching picks: {response.status_code}")
            return

        picks = response.json()
        
        # If no new picks have been made, do nothing. Otherwise, rebuild the entire state.
        if len(picks) == self.current_pick_no:
            return
        self._rebuild_state(picks)

    def _rebuild_state(self, picks):
        """
        Rebuilds the rosters and available player pool from the list of picks.
        """
        # Reset state to the original full board and empty rosters
        self.available_players = self.full_board.clone()
        self.rosters = [
            {pos: [] for pos in config.POSITIONS} for _ in range(self.num_teams)
        ]
        
        for pick in picks:
            sleeper_id = pick['metadata'].get('player_id')
            # In mock drafts, roster_id can be None, so we rely on draft_slot
            draft_slot = pick['draft_slot']
            team_idx = draft_slot - 1 # 0-indexed in our logic
            
            if sleeper_id:
                # Find the player in available players list
                player_row = self.available_players.filter(pl.col('sleeper_id') == sleeper_id)
                
                if not player_row.is_empty():
                    # Add the player to the correct team's roster
                    row_dict = player_row.row(0, named=True)
                    
                    # Add PickNum for dashboard display, mimicking training format
                    pick_in_round = ((pick['pick_no'] - 1) % self.num_teams) + 1
                    row_dict['PickNum'] = f"{pick['round']}.{str(pick_in_round).zfill(2)}"

                    pos = row_dict['Pos']
                    self.rosters[team_idx][pos].append(row_dict)
                    
                    # Remove the player from the available pool
                    self.available_players = self.available_players.filter(pl.col('sleeper_id') != sleeper_id)
        
        self.current_pick_no = len(picks)
        # print(f"State updated. {self.current_pick_no} picks processed.")

    def get_recommendation(self, team_idx, top_k=5):
        """
        Generates draft recommendations for a given team index by running the model.
        
        :param team_idx: The 0-indexed team to get a recommendation for.
        :param top_k: Number of top recommendations to return.
        :return: A list of dictionaries representing the recommended players and their confidence scores.
        """
        # Construct the feature tensors required by the model
        roster_feats, player_feats, mask, team_idx_tensor = self._build_tensors(team_idx)
        
        # Run inference with the model
        with torch.no_grad():
            # Add a batch dimension for the single inference pass
            roster_feats = roster_feats.unsqueeze(0).to(self.device)
            player_feats = player_feats.unsqueeze(0).to(self.device)
            mask = mask.unsqueeze(0).to(self.device)
            team_idx_tensor = team_idx_tensor.unsqueeze(0).to(self.device)
            
            action_logits, _ = self.model(roster_feats, player_feats, mask, team_idx_tensor)
            
            # Calculate probabilities (Confidence)
            probs = torch.softmax(action_logits, dim=1)
            
            # Get top K actions
            top_probs, top_indices = torch.topk(probs, k=top_k)
            
            top_probs = top_probs.squeeze(0).tolist()
            top_indices = top_indices.squeeze(0).tolist()
        
        # Map the action indices back to the actual players
        top_n = self.available_players.head(config.N_PLAYERS_WINDOW)
        
        recommendations = []
        for i, idx in enumerate(top_indices):
            # Check if index is valid (it should be, but safety first)
            if idx < len(top_n):
                player_row = top_n.row(idx, named=True)
                rec = {
                    'Player': player_row['Player'],
                    'Pos': player_row['Pos'],
                    'Team': player_row['Team'],
                    'ECR': player_row['ECR'],
                    'VOR': player_row['VOR'],
                    'Value': player_row['Value'],
                    'Confidence': top_probs[i]
                }
                recommendations.append(rec)
        
        return recommendations

    def _build_tensors(self, team_idx):
        """
        Constructs the input tensors for the model based on the current draft state.
        This method's logic must exactly mirror the `get_state` method in `DraftSimulator`
        to ensure the model receives data in the format it was trained on.
        """
        # Player Features: Top N available players
        top_n_players = self.available_players.head(config.N_PLAYERS_WINDOW)
        
        pos_map = {pos: i for i, pos in enumerate(config.POSITIONS)}
        player_features = []

        # Positional one-hot encoding
        for row in top_n_players.iter_rows(named=True):
            pos_one_hot = [0.0] * len(config.POSITIONS)
            if row['Pos'] in pos_map:
                pos_one_hot[pos_map[row['Pos']]] = 1.0
            
            features = [row['VOR'], row['Value']] + pos_one_hot
            player_features.append(features)
            
        # Pad if fewer than N players are available (this should never happen in a real draft)
        while len(player_features) < config.N_PLAYERS_WINDOW:
            player_features.append([0.0] * (2 + len(config.POSITIONS)))
            
        player_features_tensor = torch.tensor(player_features, dtype=torch.float32)
        
        # Roster features
        my_roster_counts = [float(len(self.rosters[team_idx][pos])) for pos in config.POSITIONS]
        opponent_roster_counts = []
        for i in range(1, self.num_teams):
            opponent_idx = (team_idx + i) % self.num_teams
            opponent_counts = [float(len(self.rosters[opponent_idx][pos])) for pos in config.POSITIONS]
            opponent_roster_counts.extend(opponent_counts)
            
        # Calculate Draft Progress
        total_picks = self.num_teams * self.num_rounds
        progress = self.current_pick_no / total_picks
        
        # Append progress to roster features
        roster_features = my_roster_counts + opponent_roster_counts + [progress]
        roster_features_tensor = torch.tensor(roster_features, dtype=torch.float32)
        
        # Positional limits mask
        valid_action_mask = []
        my_current_counts = {pos: len(self.rosters[team_idx][pos]) for pos in config.POSITIONS}

        if self.num_rounds > 15:
            roster_limits = {'QB': self.roster_slots['QB'] * 2,
                             'RB': self.roster_slots['RB'] * 3,
                             'WR': self.roster_slots['WR'] * 3,
                             'TE': self.roster_slots['TE'] * 2,
                             'K': self.roster_slots['K'],
                             'DST': self.roster_slots['DST']
                             }
        else:
            roster_limits = {'QB': self.roster_slots['QB'] * 2,
                             'RB': self.roster_slots['RB'] * 2.5,
                             'WR': self.roster_slots['WR'] * 2.5,
                             'TE': self.roster_slots['TE'] * 2,
                             'K': self.roster_slots['K'],
                             'DST': self.roster_slots['DST']
                             }

        for row in top_n_players.iter_rows(named=True):
            player_pos = row['Pos']
            if player_pos in roster_limits and my_current_counts[player_pos] >= roster_limits[player_pos]:
                valid_action_mask.append(True)
            else:
                valid_action_mask.append(False)
                
        while len(valid_action_mask) < config.N_PLAYERS_WINDOW:
            valid_action_mask.append(True)
            
        # Failsafe
        if all(valid_action_mask):
             num_real_players = len(top_n_players)
             for i in range(num_real_players):
                 valid_action_mask[i] = False
                 
        valid_action_mask_tensor = torch.tensor(valid_action_mask, dtype=torch.bool)
        
        # Team index
        team_idx_tensor = torch.tensor(team_idx, dtype=torch.long)
        
        return roster_features_tensor, player_features_tensor, valid_action_mask_tensor, team_idx_tensor