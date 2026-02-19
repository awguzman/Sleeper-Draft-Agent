"""
This module defines the training logic for the PPO-based drafting agent.

The core components are:
- Memory: A buffer to store transitions (state, action, logprob, reward) for a trajectory.
- PPO: The Proximal Policy Optimization algorithm implementation. It handles agent policy updates.
- train: The main function that orchestrates the training process, including:
    - Initializing the (vectorized) environment, agent, and TensorBoard for logging.
    - Running training episodes (simulated drafts).
    - Collecting experiences (transitions) in the Memory buffer.
    - Periodically updating the agent's policy using the PPO algorithm.
    - Logging metrics and saving model checkpoints.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import subprocess
import webbrowser
import time
import socket
from datetime import datetime
from functools import partial
from collections import deque

from draft import DraftSimulator
from agent import DraftAgent
from vectorize import VectorizedDraftSimulator
import config

class Memory:
    """
    A buffer for storing trajectories for PPO updates.
    """
    def __init__(self):
        self.actions = []
        self.states_roster = []
        self.states_player = []
        self.states_team = []
        self.masks = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        """Clears all stored transitions."""
        del self.actions[:]
        del self.states_roster[:]
        del self.states_player[:]
        del self.states_team[:]
        del self.masks[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class PPO:
    """
    Proximal Policy Optimization (PPO) Agent.
    """
    def __init__(self, n_players_window = config.N_PLAYERS_WINDOW,
                 player_feat_dim = config.PLAYER_FEAT_DIM,
                 roster_feat_dim = config.ROSTER_FEAT_DIM,
                 embed_dim = config.EMBED_DIM):
        # Initialize hyperparameters
        self.lr = config.LR
        self.gamma = config.GAMMA
        self.eps_clip = config.EPS_CLIP
        self.k_epochs = config.K_EPOCHS

        # Initialize Actor-Critic Policy
        self.policy = DraftAgent(n_players_window, player_feat_dim, roster_feat_dim, embed_dim).to(config.DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.LR)

        # Initialize Old Policy (for PPO ratio calculation)
        self.policy_old = DraftAgent(n_players_window, player_feat_dim, roster_feat_dim, embed_dim).to(config.DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Initialize Critic Loss Function
        self.Loss = nn.MSELoss()

    def select_action(self, roster_feats, player_feats, mask, team_idx, memory):
        """
        Selects a batch of actions given a batch of states, and stores the transitions in memory.
        """
        # Move state tensors to the correct device
        roster_feats = roster_feats.to(config.DEVICE)
        player_feats = player_feats.to(config.DEVICE)
        mask = mask.to(config.DEVICE)
        team_idx = team_idx.to(config.DEVICE)

        # Use the old policy to generate actions
        with torch.no_grad():
            action_logits, _ = self.policy_old(roster_feats, player_feats, mask, team_idx)

        # Check for all-masked logits (all -inf)
        # This only happens if the agent has no available actions (e.g. only needs positions not in their current board)
        # Should be fixed but leaving this until we are sure
        if (action_logits == float('-inf')).all(dim=1).any():
            print("!!! ALL ACTIONS MASKED DETECTED !!!")
            bad_indices = (action_logits == float('-inf')).all(dim=1).nonzero(as_tuple=True)[0]
            print(f"Bad indices in batch: {bad_indices}")
            print(f"Mask for first bad index: {mask[bad_indices[0]]}")
            raise ValueError("All actions are masked for at least one environment!")

        # Create a categorical distribution and sample actions
        dist = Categorical(logits=action_logits)
        actions = dist.sample()
        action_logprobs = dist.log_prob(actions)

        # Store transitions in memory
        memory.states_roster.append(roster_feats)
        memory.states_player.append(player_feats)
        memory.states_team.append(team_idx)
        memory.masks.append(mask)
        memory.actions.append(actions)
        memory.logprobs.append(action_logprobs)

        return actions.cpu().numpy()

    def update(self, memory, current_episode):
        """
        Updates the policy using the collected experiences in memory.
        """
        # --- Decay Hyperparameters ---
        new_lr = config.LR - (config.LR_DECAY_RATE * current_episode)
        new_lr = max(new_lr, config.LR_FINAL)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        new_entropy_coef = config.ENTROPY_COEF - (config.ENTROPY_DECAY_RATE * current_episode)
        new_entropy_coef = max(new_entropy_coef, config.ENTROPY_FINAL)

        # --- Masked Reward Calculation (Per-Team Returns) ---
        # Stack lists into tensors: (num_steps, num_envs)
        all_rewards = torch.stack(memory.rewards).to(config.DEVICE)
        all_terminals = torch.stack(memory.is_terminals).to(config.DEVICE)
        all_teams = torch.stack(memory.states_team).to(config.DEVICE)
        
        num_steps, num_envs = all_rewards.shape
        
        # Initialize returns tensor
        returns = torch.zeros_like(all_rewards)
        
        # Iterate over each team to calculate their independent discounted returns
        for team_idx in range(config.NUM_TEAMS):
            # Create a mask for when this team was acting
            # Shape: (num_steps, num_envs)
            team_mask = (all_teams == team_idx).float()
            
            # Extract rewards ONLY for this team (zero out others)
            team_rewards = all_rewards * team_mask
            
            running_return = torch.zeros(num_envs).to(config.DEVICE)
            
            # Iterate backwards
            for t in reversed(range(num_steps)):
                # If terminal, reset return
                running_return = running_return * (1 - all_terminals[t].float())
                
                # Accumulate return
                # If it's this team's turn, team_rewards[t] is the reward, otherwise 0.
                # The discount gamma applies to every time step (wall-clock time).
                running_return = team_rewards[t] + self.gamma * running_return
                
                # If it was this team's turn at step t, store the calculated return
                # We add it to the master returns tensor (which is initialized to 0)
                returns[t] += running_return * team_mask[t]
        
        # Flatten the returns tensor for PPO update
        # Shape: (num_steps * num_envs)
        returns = returns.view(-1)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # --- Batch and Flatten Data ---
        old_states_roster = torch.cat(memory.states_roster).detach()
        old_states_player = torch.cat(memory.states_player).detach()
        old_states_team = torch.cat(memory.states_team).detach()
        old_masks = torch.cat(memory.masks).detach()
        old_actions = torch.cat(memory.actions).detach()
        old_logprobs = torch.cat(memory.logprobs).detach()

        # --- Logging Metrics ---
        total_loss, total_actor_loss, total_critic_loss, total_entropy_loss = 0, 0, 0, 0
        total_clip_fraction = 0
        
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            logprobs, state_values, dist_entropy = self.evaluate(old_states_roster, old_states_player, old_masks, old_states_team, old_actions)

            # Calculate advantages
            advantages = returns - state_values.detach()

            # Compute ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Actor Loss with Clipping
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic Loss
            critic_loss = self.Loss(state_values, returns)
            
            # Entropy Loss
            entropy_loss = -dist_entropy.mean()
            
            # Total Loss
            loss = actor_loss + 0.5 * critic_loss + new_entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients and calculate norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config.GRAD_CLIP)
            self.optimizer.step()
            
            # --- Accumulate logging metrics ---
            total_loss += loss.item()
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy_loss += entropy_loss.item()
            
            # Calculate clip fraction
            clipped = ratios.gt(1 + self.eps_clip) | ratios.lt(1 - self.eps_clip)
            total_clip_fraction += torch.as_tensor(clipped, dtype=torch.float32).mean().item()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # --- Prepare logging dictionary ---
        avg_metrics = {
            "loss": total_loss / self.k_epochs,
            "actor_loss": total_actor_loss / self.k_epochs,
            "critic_loss": total_critic_loss / self.k_epochs,
            "entropy_loss": total_entropy_loss / self.k_epochs,
            "clip_fraction": total_clip_fraction / self.k_epochs,
            "lr": new_lr,
            "entropy_coef": new_entropy_coef
        }
        
        return avg_metrics

    def evaluate(self, roster_feats, player_feats, mask, team_idx, action):
        """
        Evaluates action log-probabilities and state values for a batch of states and actions.
        """
        action_logits, state_values = self.policy(roster_feats, player_feats, mask, team_idx)
        
        dist = Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, torch.squeeze(state_values), dist_entropy

def is_port_in_use(port):
    """Checks if a local port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def train():
    """
    Main training function.
    """
    print(f"Training on device: {config.DEVICE} for {config.MAX_EPISODES} episodes (drafts).")
    
    # --- Setup TensorBoard for logging ---
    if not os.path.exists("runs"):
        os.makedirs("runs")
    log_dir = os.path.join("runs", "draft_experiment_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")
    
    # --- Automatic TensorBoard Launch ---
    if not is_port_in_use(6006):
        try:
            print("Launching TensorBoard...")
            subprocess.Popen(["tensorboard", "--logdir=runs", "--port=6006"], shell=True)
            time.sleep(3)
            webbrowser.open("http://localhost:6006")
        except Exception as e:
            print(f"Could not launch TensorBoard automatically: {e}")
    else:
        print("TensorBoard is already running on port 6006.")
        webbrowser.open("http://localhost:6006")
    
    # --- Initialize Environment ---
    num_envs = config.NUM_ENVS # Number of parallel drafts
    env_fn = partial(DraftSimulator,
                     num_teams=config.NUM_TEAMS,
                     num_rounds=config.NUM_ROUNDS,
                     n_players_window=config.N_PLAYERS_WINDOW,
                     roster_limits=config.ROSTER_LIMITS)

    # --- Vectorize Environment ---
    vec_env = VectorizedDraftSimulator(env_fn=env_fn,
                                       num_envs=num_envs)

    # --- Initialize Agent and Memory ---
    ppo_agent = PPO(n_players_window=config.N_PLAYERS_WINDOW,
                    player_feat_dim=config.PLAYER_FEAT_DIM,
                    roster_feat_dim=config.ROSTER_FEAT_DIM,
                    embed_dim=config.EMBED_DIM)
    memory = Memory()

    # --- Training Loop ---
    running_rewards = np.zeros(num_envs)
    recent_rewards = deque(maxlen=config.LOG_INTERVAL)
    total_episodes = 0
    last_log_episode = 0

    # Failsafe Counter
    failsafe_count = 0

    # Create the models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    # Reset environment to get initial state
    (roster_feats, player_feats, mask, team_idx) = vec_env.reset()

    while total_episodes < config.MAX_EPISODES:
        # Collect a batch of experiences
        for _ in range(config.UPDATE_TIMESTEP // num_envs):
            actions = ppo_agent.select_action(roster_feats, player_feats, mask, team_idx, memory)
            
            (next_roster_feats, next_player_feats, next_mask, next_team_idx), rewards, dones, infos = vec_env.step(actions)
            
            # Count failsafes
            for info in infos:
                if info.get('failsafe_triggered', False):
                    failsafe_count += 1
            
            memory.rewards.append(rewards)
            memory.is_terminals.append(dones)
            
            running_rewards += rewards.numpy()
            
            for i, done in enumerate(dones):
                if done:
                    total_episodes += 1
                    # Calculate average reward per team for this episode
                    avg_team_reward = running_rewards[i] / config.NUM_TEAMS
                    recent_rewards.append(avg_team_reward)
                    running_rewards[i] = 0

            roster_feats, player_feats, mask, team_idx = next_roster_feats, next_player_feats, next_mask, next_team_idx
        
        # Update the policy
        metrics = ppo_agent.update(memory, total_episodes)
        memory.clear()
        
        # Log metrics after each update
        writer.add_scalar('Loss/Total', metrics["loss"], total_episodes)
        writer.add_scalar('Loss/Actor', metrics["actor_loss"], total_episodes)
        writer.add_scalar('Loss/Critic', metrics["critic_loss"], total_episodes)
        writer.add_scalar('Loss/Entropy', metrics["entropy_loss"], total_episodes)
        writer.add_scalar('PPO/Clip_Fraction', metrics["clip_fraction"], total_episodes)
        writer.add_scalar('Hyperparameters/Learning_Rate', metrics["lr"], total_episodes)
        writer.add_scalar('Hyperparameters/Entropy_Coefficient', metrics["entropy_coef"], total_episodes)

        # --- Logging and Checkpointing based on total episodes ---
        if total_episodes >= last_log_episode + config.LOG_INTERVAL:
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            print(f"Episodes: {total_episodes} \t Avg Reward: {avg_reward:.2f} \t Avg Loss: {metrics['loss']:.4f} \t Failsafes: {failsafe_count}")
            writer.add_scalar('Training/Average_Reward', avg_reward, total_episodes)
            writer.add_scalar('Debug/Failsafe_Count', failsafe_count, total_episodes)
            
            last_log_episode = total_episodes
            failsafe_count = 0 # Reset counter after logging
            
    # Save final model
    final_model_name = f"ppo_draft_agent_{config.NUM_TEAMS}team_{config.NUM_ROUNDS}rounds.pth"
    final_model_path = os.path.join("models", final_model_name)
    torch.save(ppo_agent.policy.state_dict(), final_model_path)
    print(f"Training Complete. Final model saved at {final_model_path}")

    vec_env.close()
    writer.close()

if __name__ == '__main__':
    train()
