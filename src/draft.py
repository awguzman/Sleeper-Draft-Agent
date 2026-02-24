"""
This module initializes the drafting simulation environment used for model training.
"""

import polars as pl
import torch

from board import create_board
import config

class DraftSimulator:
    """
    A simulated fantasy football draft environment for training RL agents.

    This class manages the state of the draft, including the draft board,
    team rosters, and turn order.
    """
    def __init__(self, num_teams=config.NUM_TEAMS,
                 num_rounds=config.NUM_ROUNDS,
                 n_players_window=config.N_PLAYERS_WINDOW,
                 roster_limits=config.ROSTER_LIMITS):
        """
        Initializes the draft simulation environment.

        :param num_teams: The number of teams in the draft.
        :param num_rounds: The total number of draft rounds.
        :param n_players_window: The number of top available players to include in the state.
        :param roster_limits: A dictionary defining the max number of players per position.
                              e.g., {'QB': 2, 'RB': 6, 'WR': 7, 'TE': 2}
        """

        self.num_teams = num_teams
        self.num_rounds = num_rounds
        self.n_players_window = n_players_window
        self.roster_limits = roster_limits
        self.positions = config.POSITIONS

        # The full draft board with VOR and Value calculations
        self.full_board = create_board(preprocess=True)

        # The board of players still available to be drafted
        self.available_players = self.full_board.clone()

        # A list of dictionaries, where each dict represents a team's roster
        self.rosters = [
            {pos: [] for pos in self.positions} for _ in range(self.num_teams)
        ]

        # State variables to track draft progress
        self.current_round = 0
        self.current_pick = 0
        self.current_team_idx = 0   # Will be 0-indexed

    def reset(self):
        """
        Resets the environment to the beginning of a new draft.

        :return: The initial state of the environment for the first team.
        """
        self.available_players = self.full_board.clone()
        self.rosters = [
            {pos: [] for pos in self.positions} for _ in range(self.num_teams)
        ]

        self.current_round = 1
        self.current_pick = 1
        self.current_team_idx = 0

        state, info = self.get_state(self.current_team_idx)
        return state, info

    def step(self, action):
        """
        Executes a draft pick (action) and advances the environment.

        :param action: The index of the player to draft from the Top N window.
        :return: A tuple of (next_state, reward, done, info).
        """
        # 1. Get the actual player from the action index
        # The action is an integer (0, N-1) corresponding to the index in the current available_players window
        top_n = self.available_players.head(self.n_players_window)
        
        if action >= len(top_n):
             # This should be handled by masking, but as a safeguard:
             raise ValueError(f"Action {action} is out of bounds for available players {len(top_n)}")

        player_row = top_n.row(action, named=True)
        player_id = player_row['fantasypros_id']
        player_pos = player_row['Pos']
        player_vor = player_row['VOR']
        player_value = player_row['Value']

        # 2. Check for roster limit violation (Failsafe Triggered)
        # If the agent picks a player that violates the limit, it means the failsafe in get_state was used.
        # We apply a penalty but allow the pick.
        current_count = len(self.rosters[self.current_team_idx][player_pos])
        limit = self.roster_limits.get(player_pos)
        
        penalty = 0
        if current_count >= limit:
            penalty = -2.0 # Penalty for backing into a corner

        # 3. Add player to the current team's roster
        player_data = dict(player_row)
        player_data['PickNum'] = f"{self.current_round}.{str((self.current_pick - 1) % self.num_teams + 1).zfill(2)}"
        
        self.rosters[self.current_team_idx][player_pos].append(player_data)

        # 4. Remove player from the available board
        self.available_players = self.available_players.filter(pl.col('fantasypros_id') != player_id)

        # 5. Calculate reward
        alpha = 1
        beta = 1
        reward = (alpha * player_vor) + (beta * player_value) + penalty

        # 6. Advance the draft turn
        self._advance_turn()

        # 7. Check if the draft is over
        done = self.current_round > self.num_rounds

        # 8. Get the state for the *next* team
        if not done:
            next_state, info = self.get_state(self.current_team_idx)
        else:
            # If done, returning the last state for value estimation.
            next_state, info = self.get_state(self.current_team_idx) 

        return next_state, reward, done, info

    def get_state(self, team_idx):
        """
        Constructs the state representation for a given team.

        :param team_idx: The index of the team for which to generate the state.
        :return: A tuple containing (state_tuple, info_dict).
                 State tuple: (roster_features, player_features, valid_action_mask, team_idx)
                 Info dict: {'failsafe_triggered': bool}
        """
        # 1. Get Top N available players
        top_n_players = self.available_players.head(self.n_players_window)

        # 2. Construct player_features tensor
        player_features = []

        # Map positions to indices: QB=0, RB=1, WR=2, TE=3
        pos_map = {pos: i for i, pos in enumerate(self.positions)}

        # One-hot encode positions.
        for row in top_n_players.iter_rows(named=True):
            pos_one_hot = [0.0] * len(self.positions)
            if row['Pos'] in pos_map:
                pos_one_hot[pos_map[row['Pos']]] = 1.0
            
            # Feature vector: [VOR, Value, QB, RB, WR, TE]
            features = [row['VOR'], row['Value']] + pos_one_hot
            player_features.append(features)
        
        # Pad if fewer than N players are available (this should never happen unless num_rounds is too large)
        # We pad with 0s. The mask will ensure these aren't selected.
        while len(player_features) < self.n_players_window:
            player_features.append([0.0] * (2 + len(self.positions)))

        player_features_tensor = torch.tensor(player_features, dtype=torch.float32)

        # 3. Construct roster_features tensor
        # My Team Counts
        my_roster_counts = [float(len(self.rosters[team_idx][pos])) for pos in self.positions]
        
        # Opponent Team Counts (Relative Ordering)
        opponent_roster_counts = []
        for i in range(1, self.num_teams):
            # Calculate opponent index wrapping around the table
            opponent_idx = (team_idx + i) % self.num_teams
            opponent_counts = [float(len(self.rosters[opponent_idx][pos])) for pos in self.positions]
            opponent_roster_counts.extend(opponent_counts)
            
        roster_features = my_roster_counts + opponent_roster_counts
        roster_features_tensor = torch.tensor(roster_features, dtype=torch.float32)

        # 4. Generate valid_action_mask
        # Mask is True for invalid actions
        valid_action_mask = []
        my_current_counts = {pos: len(self.rosters[team_idx][pos]) for pos in self.positions}
        
        # Check limits for real (non-padded) players
        for row in top_n_players.iter_rows(named=True):
            player_pos = row['Pos']
            # If we have reached the limit for this position, mask it
            if player_pos in self.roster_limits and my_current_counts[player_pos] >= self.roster_limits[player_pos]:
                valid_action_mask.append(True) 
            else:
                valid_action_mask.append(False)
        
        # Mask all padded slots
        while len(valid_action_mask) < self.n_players_window:
            valid_action_mask.append(True)

        # --- Failsafe: If all actions are masked, unmask everything (will punish in step()) ---
        failsafe_triggered = False
        if all(valid_action_mask):
            failsafe_triggered = True
            # Only unmask the real players, keep padded slots masked
            num_real_players = len(top_n_players)
            for i in range(num_real_players):
                valid_action_mask[i] = False

        valid_action_mask_tensor = torch.tensor(valid_action_mask, dtype=torch.bool)
        
        # 5. Team Index Tensor
        team_idx_tensor = torch.tensor(team_idx, dtype=torch.long)

        state = (roster_features_tensor, player_features_tensor, valid_action_mask_tensor, team_idx_tensor)
        info = {'failsafe_triggered': failsafe_triggered}
        
        return state, info

    def _advance_turn(self):
        """
        Advances the pick and round counters, handling snake draft logic.
        """
        self.current_pick += 1
        
        # Check if draft is complete
        if self.current_pick > self.num_teams * self.num_rounds:
            self.current_round = self.num_rounds + 1
            return

        # Update Round
        self.current_round = ((self.current_pick - 1) // self.num_teams) + 1
        
        # Update Team Index (Snake Draft)
        pick_in_round = (self.current_pick - 1) % self.num_teams
        if self.current_round % 2 == 1:
            # Odd Rounds: 0 -> 11
            self.current_team_idx = pick_in_round
        else:
            # Even Rounds: 11 -> 0
            self.current_team_idx = self.num_teams - 1 - pick_in_round


# --- Debug Zone ---
if __name__ == '__main__':
    simulator = DraftSimulator()
    state, info = simulator.reset()
    roster_feats, player_feats, mask, team_idx = state
    
    print(f"Draft Initialized for {simulator.num_teams} teams and {simulator.num_rounds} rounds.")
    print("\nInitial State Tensors:")
    print(f"Roster Features Shape: {roster_feats.shape}")
    print(f"Player Features Shape: {player_feats.shape}")
    print(f"Action Mask Shape: {mask.shape}")
    print(f"Team Index: {team_idx}")
    
    # Verify content
    print("\nSample Roster Features (My Team + 1st Opponent):")
    print(roster_feats[:8])
    print("\nSample Player Features (Top Player):")
    print(player_feats[0])
    print("\nInitial Action Mask (should be all False):")
    print(mask)
    
    # Test Step
    print("\n--- Testing Step (Drafting Player 0) ---")
    next_state, reward, done, _ = simulator.step(0)
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Next Team Index: {simulator.current_team_idx}")
