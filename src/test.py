"""
This script loads a trained model and simulates a full draft where the agent controls ALL teams.
It prints detailed information about each pick to help analyze the model's strategy.
"""

import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.draft import DraftSimulator
from src.agent import DraftAgent

def run_test_draft(model_path):
    print(f"--- Starting Test Draft with Model: {model_path} ---")
    
    # 1. Initialize Environment
    env = DraftSimulator()
    state, info = env.reset()
    roster_feats, player_feats, mask, team_idx = state
    
    # 2. Load Model
    device = torch.device("cpu") # CPU is fine for single inference
    model = DraftAgent(
        n_players_window=config.N_PLAYERS_WINDOW,
        player_feat_dim=config.PLAYER_FEAT_DIM,
        roster_feat_dim=config.ROSTER_FEAT_DIM,
        embed_dim=config.EMBED_DIM,
        team_embed_dim=config.TEAM_EMBED_DIM
    ).to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print("Model loaded. Starting draft...\n")
    print(f"{'Pick':<8} | {'Team':<5} | {'Pos':<4} | {'Player':<25} | {'VOR':<6} | {'Value':<6} | {'Roster State'}")
    print("-" * 100)

    total_picks = config.NUM_TEAMS * config.NUM_ROUNDS
    team_rewards = [0.0] * config.NUM_TEAMS
    
    for pick_num in range(1, total_picks + 1):
        # Prepare inputs
        # Add batch dimension
        roster_feats_t = roster_feats.unsqueeze(0).to(device)
        player_feats_t = player_feats.unsqueeze(0).to(device)
        mask_t = mask.unsqueeze(0).to(device)
        team_idx_t = team_idx.unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            action_logits, value_est = model(roster_feats_t, player_feats_t, mask_t, team_idx_t)
            
            # Greedy selection
            action = torch.argmax(action_logits).item()

        top_n = env.available_players.head(config.N_PLAYERS_WINDOW)
        picked_player = top_n.row(action, named=True)
        
        # Execute Step
        (next_roster_feats, next_player_feats, next_mask, next_team_idx), reward, done, info = env.step(action)
        
        # Snake draft logic to find who just picked
        # pick_num is 1-based.
        # Round 1: 1->0, 2->1 ...
        round_num = ((pick_num - 1) // config.NUM_TEAMS) + 1
        pick_in_round = (pick_num - 1) % config.NUM_TEAMS
        
        if round_num % 2 == 1:
            just_picked_idx = pick_in_round
        else:
            just_picked_idx = config.NUM_TEAMS - 1 - pick_in_round
            
        # Update team reward
        team_rewards[just_picked_idx] += reward
            
        # Construct roster string (e.g., "QB:1 RB:2...")
        roster = env.rosters[just_picked_idx]
        roster_str = " ".join([f"{pos}:{len(players)}" for pos, players in roster.items()])
        
        # Print Pick Info
        pick_str = f"{round_num}.{str(pick_in_round + 1).zfill(2)}"
        print(f"{pick_str:<8} | T{just_picked_idx+1:<4} | {picked_player['Pos']:<4} | {picked_player['Player']:<25} | {picked_player['VOR']:<6.2f} | {picked_player['Value']:<6.2f} | {roster_str}")

        # Update state
        roster_feats = next_roster_feats
        player_feats = next_player_feats
        mask = next_mask
        team_idx = next_team_idx
        
        if done:
            break
            
    print("\n--- Draft Complete ---")
    
    # Print final rosters
    print("\nFinal Rosters:")
    for i, roster in enumerate(env.rosters):
        print(f"\nTeam {i+1} (Total Reward: {team_rewards[i]:.2f}):")
        for pos in config.POSITIONS:
            players = roster.get(pos, [])
            player_names = [p['Player'] for p in players]
            count = len(players)
            print(f"  {pos} ({count}): {', '.join(player_names)}")
