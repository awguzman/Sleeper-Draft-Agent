"""
This module provides a vectorized wrapper for the DraftSimulator environment
to run multiple simulations in parallel using multiprocessing.
"""

import torch
import numpy as np
from multiprocessing import Process, Pipe

def worker(remote, parent_remote, env_fn):
    """
    Worker function that runs in a separate process.

    It creates an instance of the environment and listens for commands from the
    main process through a Pipe.
    """
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                next_state, reward, done, info = env.step(data)
                if done:
                    # If an episode is done, automatically reset the environment
                    # and return the new initial state as the next_state.
                    # Note: env.reset() now returns (state, info)
                    next_state, _ = env.reset()
                remote.send((next_state, reward, done, info))
            elif cmd == 'reset':
                state, _ = env.reset()
                remote.send(state)
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('Worker: KeyboardInterrupt')
    finally:
        pass


class VectorizedDraftSimulator:
    """
    A vectorized wrapper for the DraftSimulator environment.

    This class creates multiple instances of the environment in parallel processes
    and provides a batched interface to step through them simultaneously.
    """
    def __init__(self, env_fn, num_envs):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])

        self.processes = [Process(target=worker, args=(work_remote, remote, env_fn))
                          for work_remote, remote in zip(self.work_remotes, self.remotes)]

        for p in self.processes:
            p.daemon = True  # If the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        """
        Steps all environments with the provided batch of actions.

        :param actions: A list or tensor of actions, one for each environment.
        :return: A tuple of batched (next_states, rewards, dones, infos).
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        results = [remote.recv() for remote in self.remotes]
        next_states, rewards, dones, infos = zip(*results)

        return self._stack_states(next_states), torch.tensor(rewards), torch.tensor(dones), infos

    def reset(self):
        """
        Resets all environments.

        :return: A batched representation of the initial states.
        """
        for remote in self.remotes:
            remote.send(('reset', None))

        initial_states = [remote.recv() for remote in self.remotes]
        return self._stack_states(initial_states)

    def close(self):
        """Closes all environments and terminates the processes."""
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()

    def _stack_states(self, states):
        """
        Stacks the tuple of state tensors from each environment into a single batched tensor.
        Input: [(roster_0, player_0, mask_0, team_0), (roster_1, player_1, mask_1, team_1), ...]
        Output: (batched_roster, batched_player, batched_mask, batched_team)
        """
        roster_feats, player_feats, masks, team_idxs = zip(*states)
        return (torch.stack(roster_feats),
                torch.stack(player_feats),
                torch.stack(masks),
                torch.stack(team_idxs))

if __name__ == '__main__':
    # --- Debug Zone ---
    from functools import partial
    from draft import DraftSimulator
    import config

    # Use a partial function to pass the environment constructor with its arguments
    env_fn = partial(DraftSimulator,
                     num_teams=config.NUM_TEAMS,
                     num_rounds=config.NUM_ROUNDS,
                     n_players_window=config.N_PLAYERS_WINDOW,
                     roster_limits=config.ROSTER_LIMITS)

    num_parallel_envs = 4 # Example: run 4 drafts in parallel
    vec_env = VectorizedDraftSimulator(env_fn=env_fn, num_envs=num_parallel_envs)

    print(f"Initialized {num_parallel_envs} parallel draft environments.")

    # Test reset
    batched_states = vec_env.reset()
    roster_feats, player_feats, masks, team_idxs = batched_states
    print("\n--- Testing Reset ---")
    print(f"Batched Roster Features Shape: {roster_feats.shape}") # Should be (num_envs, roster_dim)
    print(f"Batched Player Features Shape: {player_feats.shape}") # Should be (num_envs, window, player_dim)
    print(f"Batched Masks Shape: {masks.shape}")                   # Should be (num_envs, window)
    print(f"Batched Team Indices Shape: {team_idxs.shape}")        # Should be (num_envs,)

    # Test step
    # Create a batch of random actions (0 for each env in this case)
    actions = [0] * num_parallel_envs
    next_states, rewards, dones, _ = vec_env.step(actions)

    print("\n--- Testing Step ---")
    print(f"Rewards Shape: {rewards.shape}") # Should be (num_envs,)
    print(f"Dones Shape: {dones.shape}")     # Should be (num_envs,)
    print(f"Next Roster Features Shape: {next_states[0].shape}")

    # Close environments
    vec_env.close()
    print("\nEnvironments closed.")
