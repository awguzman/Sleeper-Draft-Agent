"""
This module provides a vectorized wrapper for the DraftSimulator environment to run multiple simulations in parallel.
"""

import torch
from multiprocessing import Process, Pipe

import config

def worker(remote, parent_remote, env_fn):
    """
    The target function for each parallel process.

    This function waits for commands from the main process, executes them on its own instance of the environment,
    and sends the results back.

    :param remote: The child's end of the communication Pipe.
    :param parent_remote: The parent's end of the Pipe (closed by the child).
    :param env_fn: A function that creates an instance of the environment.
    """
    # The worker process does not need the parent's end of the pipe.
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            # Wait for a command and data from the main process.
            cmd, data = remote.recv()
            if cmd == 'step':
                # Execute a step in the environment.
                next_state, reward, done, info = env.step(data)
                if done:
                    # If an episode is done, automatically reset the environment
                    # and return the new initial state as the next_state.
                    next_state, _ = env.reset()
                # Send the results back to the main process.
                remote.send((next_state, reward, done, info))
            elif cmd == 'reset':
                # Reset the environment and send back the initial state.
                state, _ = env.reset()
                remote.send(state)
            elif cmd == 'close':
                # Close the pipe and break the loop to terminate the process.
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
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
    def __init__(self, env_fn, num_envs=config.NUM_ENVS):
        """
        :param env_fn: A function that creates an instance of the environment.
        :param num_envs: The number of parallel environments to create.
        """
        self.num_envs = num_envs
        # Create a pair of pipes for each environment. `remotes` are for the parent
        # process, `work_remotes` are for the child processes.
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])

        # Create and start a new process for each environment.
        self.processes = [Process(target=worker, args=(work_remote, remote, env_fn))
                          for work_remote, remote in zip(self.work_remotes, self.remotes)]

        for p in self.processes:
            p.daemon = True  # Ensure child processes are terminated if the main process exits.
            p.start()
        
        # The parent process does not need the workers' ends of the pipes.
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        """
        Steps all environments with the provided batch of actions.

        :param actions: A list or tensor of actions, one for each environment.
        :return: A tuple of batched (next_states, rewards, dones, infos).
        """
        # Send the 'step' command and the corresponding action to each worker.
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
        # Send the 'reset' command to all workers.
        for remote in self.remotes:
            remote.send(('reset', None))

        initial_states = [remote.recv() for remote in self.remotes]
        return self._stack_states(initial_states)

    def close(self):
        """Closes all environments and terminates the worker processes."""
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
