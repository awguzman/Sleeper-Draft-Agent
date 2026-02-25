"""
This module handles the interaction with the Sleeper API and manages the draft state
for the live dashboard. It acts as the backend, translating live draft data into
a format the trained model can understand and then running inference to get a pick
recommendation.
"""

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

class SleeperDraftManager:
    """
    Manages the connection to a live Sleeper draft, tracks the state,
    and generates drafting recommendations using the trained PPO agent.
    
    This class encapsulates all the logic for:
    1. Loading the trained model and player data.
    2. Mapping player IDs between Sleeper and the training data source.
    3. Polling the Sleeper API for the current draft state.
    4. Reconstructing rosters and the available player pool.
    5. Building the specific tensor inputs required by the model.
    6. Running model inference to get a pick recommendation.
    """
    def __init__(self, draft_id, model_path):
        """
        Initializes the manager.
        
        :param draft_id: The ID of the Sleeper draft to connect to.
        :param model_path: Path to the trained .pth model file.
        """
        self.draft_id = draft_id
        self.base_url = "https://api.sleeper.app/v1"
        
        # 1. Load and Setup the Board with ID Mapping
        print("Loading player board and ID mappings...")
        self.full_board = create_board(preprocess=True)
        self._map_ids()
        
        # 2. Initialize State
        self.available_players = self.full_board.clone()
        self.rosters = [
            {pos: [] for pos in config.POSITIONS} for _ in range(config.NUM_TEAMS)
        ]
        self.current_pick_no = 0
        
        # 3. Load the Model
        print(f"Loading model from {model_path}...")
        self.device = torch.device("cpu") # Use CPU for inference on dashboard
        self.model = DraftAgent(
            n_players_window=config.N_PLAYERS_WINDOW,
            player_feat_dim=config.PLAYER_FEAT_DIM,
            roster_feat_dim=config.ROSTER_FEAT_DIM,
            embed_dim=config.EMBED_DIM,
            team_embed_dim=config.TEAM_EMBED_DIM
        ).to(self.device)
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval() # Set to evaluation mode for inference
        print("Model loaded successfully.")

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

        # Join with our board to add the 'sleeper_id' column
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
        
        # If no new picks have been made, do nothing
        if len(picks) == self.current_pick_no:
            return

        # To ensure consistency, rebuild the entire state from scratch with the new data
        self._rebuild_state(picks)

    def _rebuild_state(self, picks):
        """
        Rebuilds the rosters and available player pool from the list of picks.
        """
        # Reset state to the original full board and empty rosters
        self.available_players = self.full_board.clone()
        self.rosters = [
            {pos: [] for pos in config.POSITIONS} for _ in range(config.NUM_TEAMS)
        ]
        
        last_pick = None
        
        for pick in picks:
            sleeper_id = pick['metadata'].get('player_id')
            # In mock drafts, roster_id can be None, so we rely on draft_slot
            draft_slot = pick['draft_slot']
            team_idx = draft_slot - 1 # 0-indexed in our logic
            
            if sleeper_id:
                # Find the player in our available players list
                player_row = self.available_players.filter(pl.col('sleeper_id') == sleeper_id)
                
                if not player_row.is_empty():
                    # Add the player to the correct team's roster
                    row_dict = player_row.row(0, named=True)
                    
                    # Add PickNum for dashboard display, mimicking training format
                    pick_in_round = ((pick['pick_no'] - 1) % config.NUM_TEAMS) + 1
                    row_dict['PickNum'] = f"{pick['round']}.{str(pick_in_round).zfill(2)}"

                    pos = row_dict['Pos']
                    self.rosters[team_idx][pos].append(row_dict)
                    
                    # Remove the player from the available pool
                    self.available_players = self.available_players.filter(pl.col('sleeper_id') != sleeper_id)
            
            last_pick = pick
        
        self.current_pick_no = len(picks)
        print(f"State updated. {self.current_pick_no} picks processed.")
        
        if last_pick:
             print(f"Last pick: Round {last_pick['round']}, Pick {last_pick['pick_no']}")

    def get_recommendation(self, team_idx):
        """
        Generates a draft recommendation for a given team index by running the model.
        
        :param team_idx: The 0-indexed team to get a recommendation for.
        :return: A dictionary representing the recommended player's data.
        """
        # 1. Construct the feature tensors required by the model
        roster_feats, player_feats, mask, team_idx_tensor = self._build_tensors(team_idx)
        
        # 2. Run inference with the model
        with torch.no_grad():
            # Add a batch dimension for the single inference pass
            roster_feats = roster_feats.unsqueeze(0).to(self.device)
            player_feats = player_feats.unsqueeze(0).to(self.device)
            mask = mask.unsqueeze(0).to(self.device)
            team_idx_tensor = team_idx_tensor.unsqueeze(0).to(self.device)
            
            action_logits, _ = self.model(roster_feats, player_feats, mask, team_idx_tensor)
            
            # Greedily select the action with the highest logit
            action_idx = torch.argmax(action_logits).item()
        
        # 3. Map the action index back to the actual player
        top_n = self.available_players.head(config.N_PLAYERS_WINDOW)
        recommended_player = top_n.row(action_idx, named=True)
        
        return recommended_player

    def _build_tensors(self, team_idx):
        """
        Constructs the input tensors for the model based on the current draft state.
        This method's logic must exactly mirror the `get_state` method in `DraftSimulator`
        to ensure the model receives data in the format it was trained on.
        """
        # 1. Player Features: Top N available players
        top_n_players = self.available_players.head(config.N_PLAYERS_WINDOW)
        
        pos_map = {pos: i for i, pos in enumerate(config.POSITIONS)}
        player_features = []
        
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
        
        # 2. Roster Features: Counts of players at each position for all teams
        my_roster_counts = [float(len(self.rosters[team_idx][pos])) for pos in config.POSITIONS]
        opponent_roster_counts = []
        for i in range(1, config.NUM_TEAMS):
            opponent_idx = (team_idx + i) % config.NUM_TEAMS
            opponent_counts = [float(len(self.rosters[opponent_idx][pos])) for pos in config.POSITIONS]
            opponent_roster_counts.extend(opponent_counts)
            
        # Calculate Draft Progress (0.0 to 1.0)
        # self.current_pick_no is the number of picks already made.
        total_picks = config.NUM_TEAMS * config.NUM_ROUNDS
        progress = self.current_pick_no / total_picks
        
        # Append progress to roster features
        roster_features = my_roster_counts + opponent_roster_counts + [progress]
        roster_features_tensor = torch.tensor(roster_features, dtype=torch.float32)
        
        # 3. Mask: Boolean tensor indicating which of the top N players are invalid picks
        valid_action_mask = []
        my_current_counts = {pos: len(self.rosters[team_idx][pos]) for pos in config.POSITIONS}
        
        for row in top_n_players.iter_rows(named=True):
            player_pos = row['Pos']
            if player_pos in config.ROSTER_LIMITS and my_current_counts[player_pos] >= config.ROSTER_LIMITS[player_pos]:
                valid_action_mask.append(True)
            else:
                valid_action_mask.append(False)
                
        while len(valid_action_mask) < config.N_PLAYERS_WINDOW:
            valid_action_mask.append(True)
            
        # Failsafe Logic (Mirroring draft.py)
        if all(valid_action_mask):
             num_real_players = len(top_n_players)
             for i in range(num_real_players):
                 valid_action_mask[i] = False
                 
        valid_action_mask_tensor = torch.tensor(valid_action_mask, dtype=torch.bool)
        
        # 4. Team Index Tensor
        team_idx_tensor = torch.tensor(team_idx, dtype=torch.long)
        
        return roster_features_tensor, player_features_tensor, valid_action_mask_tensor, team_idx_tensor

# --- Debug Zone ---
if __name__ == '__main__':
    print("--- Sleeper Draft Agent Debugger ---")
    
    # 1. Get Draft ID
    draft_id = input("Enter Sleeper Draft ID: ").strip()
    if not draft_id:
        print("Draft ID is required.")
        sys.exit(1)
        
    # 2. Get Model Path
    default_model = "../src/models/ppo_draft_agent_64000.pth"
    model_path = input(f"Enter Model Path (default: {default_model}): ").strip()
    if not model_path:
        model_path = default_model
        
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    # 3. Initialize Manager
    try:
        manager = SleeperDraftManager(draft_id, model_path)
    except Exception as e:
        print(f"Failed to initialize manager: {e}")
        sys.exit(1)
        
    print(f"\nConnected to draft {draft_id}.")
    print(f"Initial Board Size: {len(manager.available_players)}")
    
    # 4. Interactive Loop
    while True:
        cmd = input("\nPress Enter to update state (or 'q' to quit): ").strip().lower()
        if cmd == 'q':
            break
            
        manager.update_state()
        
        # Determine who is on the clock
        next_pick_num = manager.current_pick_no + 1
        
        # Calculate team index for snake draft
        current_round = ((next_pick_num - 1) // config.NUM_TEAMS) + 1
        pick_in_round = (next_pick_num - 1) % config.NUM_TEAMS
        
        if current_round % 2 == 1:
            on_clock_idx = pick_in_round
        else:
            on_clock_idx = config.NUM_TEAMS - 1 - pick_in_round
            
        print(f"On Clock: Team {on_clock_idx + 1} (Round {current_round}, Pick {next_pick_num})")
        
        # Get Recommendation
        rec = manager.get_recommendation(on_clock_idx)
        print(f"Agent Recommends: {rec['Player']} ({rec['Pos']} - VOR: {rec['VOR']:.2f})")
