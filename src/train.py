"""
This module defines the training logic for the drafting agents in the drafting simulation.
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
from draft import DraftSimulator
from agent import DraftAgent
import config

class Memory:
    def __init__(self):
        self.actions = []
        self.states_roster = []
        self.states_player = []
        self.masks = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states_roster[:]
        del self.states_player[:]
        del self.masks[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class PPO:
    def __init__(self, n_players_window, player_feat_dim, roster_feat_dim, embed_dim=64):
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

        self.MseLoss = nn.MSELoss()

    def select_action(self, roster_feats, player_feats, mask, memory):
        """
        Selects an action given the state, and stores the transition in memory.
        """
        # Convert to tensor and add batch dimension
        roster_feats = roster_feats.unsqueeze(0).to(config.DEVICE)
        player_feats = player_feats.unsqueeze(0).to(config.DEVICE)
        mask = mask.unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            action_logits, _ = self.policy_old(roster_feats, player_feats, mask)
        
        # Create distribution to sample from
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        # Store in memory
        memory.states_roster.append(roster_feats)
        memory.states_player.append(player_feats)
        memory.masks.append(mask)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.item()

    def update(self, memory, current_episode):
        """
        Updates the policy using the PPO algorithm.
        Returns the average loss for logging.
        """
        # --- Decay Hyperparameters ---
        # Calculate new Learning Rate
        new_lr = config.LR - (config.LR_DECAY_RATE * current_episode)
        new_lr = max(new_lr, config.LR_FINAL) # Clamp to minimum
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        # Calculate new Entropy Coefficient
        new_entropy_coef = config.ENTROPY_COEF - (config.ENTROPY_DECAY_RATE * current_episode)
        new_entropy_coef = max(new_entropy_coef, config.ENTROPY_FINAL) # Clamp to minimum

        # --- Monte Carlo Estimate ---
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(config.DEVICE)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_states_roster = torch.squeeze(torch.stack(memory.states_roster, dim=0)).detach().to(config.DEVICE)
        old_states_player = torch.squeeze(torch.stack(memory.states_player, dim=0)).detach().to(config.DEVICE)
        old_masks = torch.squeeze(torch.stack(memory.masks, dim=0)).detach().to(config.DEVICE)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(config.DEVICE)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(config.DEVICE)

        total_loss = 0
        
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.evaluate(old_states_roster, old_states_player, old_masks, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Final loss
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - new_entropy_coef * dist_entropy.mean()

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            total_loss += loss.mean().item()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        return total_loss / self.k_epochs

    def evaluate(self, roster_feats, player_feats, mask, action):
        """
        Evaluates the action logprobs and state values for the update step.
        """
        action_logits, state_values = self.policy(roster_feats, player_feats, mask)
        
        dist = Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(action)
        
        return action_logprobs, torch.squeeze(state_values), dist.entropy()

def print_sample_roster(env, episode):
    """
    Prints a sample roster from the environment for debugging.
    """
    print(f"\n--- Sample Roster (Episode {episode}) ---")
    # Just print the first team's roster
    roster = env.rosters[0]
    for pos in config.POSITIONS:
        # Format: PlayerName (PickNum)
        players = [f"{p['Player']} ({p.get('PickNum', 'N/A')})" for p in roster[pos]]
        print(f"{pos}: {players}")
    print("----------------------------------------\n")

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def train():
    print(f"Training on device: {config.DEVICE}")
    
    # Setup TensorBoard
    if not os.path.exists("runs"):
        os.makedirs("runs")     # Create models directory if it doesn't exist
    log_dir = os.path.join("runs", "draft_experiment_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")
    
    # Automatic TensorBoard Launch
    if not is_port_in_use(6006):
        try:
            print("Launching TensorBoard...")
            # Launch TensorBoard in a subprocess
            subprocess.Popen(["tensorboard", "--logdir=runs", "--port=6006"], shell=True)
            # Give it a moment to start
            time.sleep(3)
            # Open the browser
            webbrowser.open("http://localhost:6006")
        except Exception as e:
            print(f"Could not launch TensorBoard automatically: {e}")
    else:
        print("TensorBoard is already running on port 6006.")
        webbrowser.open("http://localhost:6006")

    # Calculate dimensions
    roster_feat_dim = 4 + (config.NUM_TEAMS - 1) * 4
    player_feat_dim = config.PLAYER_FEAT_DIM

    # Initialize Environment and Agent
    env = DraftSimulator(num_teams=config.NUM_TEAMS, num_rounds=config.NUM_ROUNDS, 
                         n_players_window=config.N_PLAYERS_WINDOW, roster_limits=config.ROSTER_LIMITS)
    
    ppo_agent = PPO(config.N_PLAYERS_WINDOW, player_feat_dim, roster_feat_dim, config.EMBED_DIM)
    memory = Memory()

    # Training Loop
    timestep = 0
    running_reward = 0
    
    # Create models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")
    
    for i_episode in range(1, config.MAX_EPISODES + 1):
        roster_feats, player_feats, mask = env.reset()
        current_ep_reward = 0
        
        # Play a full draft
        for t in range(config.NUM_TEAMS * config.NUM_ROUNDS):
            timestep += 1
            
            # Select action
            action = ppo_agent.select_action(roster_feats, player_feats, mask, memory)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            
            # Save reward and is_terminal
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            current_ep_reward += reward
            
            # Update PPO agent
            if timestep % config.UPDATE_TIMESTEP == 0:
                avg_loss = ppo_agent.update(memory, i_episode)
                memory.clear()
                timestep = 0
                
                # Log Loss
                writer.add_scalar('Training/Loss', avg_loss, i_episode)
            
            if done:
                break
                
            # Update state
            if next_state:
                roster_feats, player_feats, mask = next_state
        
        running_reward += current_ep_reward
        
        # Log Reward per Episode
        writer.add_scalar('Training/Reward', current_ep_reward, i_episode)
        
        # Console Logging
        if i_episode % config.LOG_INTERVAL == 0:
            avg_reward = running_reward / config.LOG_INTERVAL
            print(f"Episode {i_episode} \t Average Reward: {avg_reward:.2f}")
            running_reward = 0
            
        if i_episode % config.PRINT_ROSTER_INTERVAL == 0:
            print_sample_roster(env, i_episode)
            
        if i_episode % config.SAVE_MODEL_INTERVAL == 0:
            model_path = os.path.join("models", f'ppo_draft_agent_{i_episode}.pth')
            torch.save(ppo_agent.policy.state_dict(), model_path)
            print(f"Model saved at {model_path}")
            
    writer.close()

if __name__ == '__main__':
    train()
