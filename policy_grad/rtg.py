"""Reward-to-go policy gradient, following OpenAI Spinning Up's example.

Reference: https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/2_rtg_pg.py
Uses Gymnasium and the same CLI options as policy_grad.py.
"""

import argparse

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch import nn
from torch.distributions import Categorical


# KEY DIFFERENCES FROM policy_grad.py:
# 1. Add reward_to_go() to compute a separate reward sum for each action.
# 2. Replace repeated full-episode returns with those per-action sums.
# 3. Keep the same loss formula, network, sampling, and optimizer update.
# The shorter code here inlines get_policy(), get_action(), compute_loss(),
# and train_one_epoch(); that reorganization is not an algorithm change.


def reward_to_go(rewards):
    """Sum rewards from each timestep through the end of the episode.

    For rewards [1, 1, 1], return [3, 2, 1]. The original example would
    assign [3, 3, 3]: the full episode return to every action.
    Here an action is credited only with its current and future rewards.
    There is no discounting (equivalent to gamma = 1).
    """
    # NEW: walk backward so each suffix sum takes just one addition.
    # For [1, 1, 1], accumulate 1, then 2, then 3, storing [3, 2, 1].
    # The previous example only needed sum(rewards), a single scalar.
    weights = np.empty(len(rewards), dtype=np.float32)
    remaining_reward = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        remaining_reward += rewards[index]
        weights[index] = remaining_reward
    return weights


def mlp(sizes):
    layers = []
    for index, (inputs, outputs) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(inputs, outputs))
        if index < len(sizes) - 2:
            layers.append(nn.Tanh())
    # The final layer outputs logits; Categorical handles normalization.
    return nn.Sequential(*layers)


def train(env_name='CartPole-v1', hidden_sizes=(32,), lr=1e-2,
          epochs=50, batch_size=5000, render=False, seed=0):
    if epochs <= 0 or batch_size <= 0 or lr <= 0:
        raise ValueError('epochs, batch_size, and lr must be positive')

    torch.manual_seed(seed)
    env = gym.make(env_name, render_mode='human' if render else None)
    try:
        if not isinstance(env.observation_space, Box) or len(env.observation_space.shape) != 1:
            raise ValueError('This example requires a flat Box observation space')
        if not isinstance(env.action_space, Discrete) or env.action_space.start != 0:
            raise ValueError('This example requires Discrete actions starting at zero')

        # SAME: state -> Tanh MLP -> action logits, optimized with Adam.
        # `policy` here serves the same role as `logits_net` in policy_grad.py.
        policy = mlp([env.observation_space.shape[0], *hidden_sizes, env.action_space.n])
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        env.reset(seed=seed)

        for epoch in range(epochs):
            observations, actions, weights = [], [], []
            episode_returns, episode_lengths = [], []
            episode_rewards = []
            obs, _ = env.reset()

            # SAME: collect complete episodes before updating the policy.
            # Reward-to-go still needs the episode's future rewards, so we
            # cannot compute an action's final weight when we first take it.
            while True:
                observations.append(obs.copy())
                with torch.no_grad():
                    logits = policy(torch.as_tensor(obs, dtype=torch.float32))
                    action = Categorical(logits=logits).sample().item()

                obs, reward, terminated, truncated, _ = env.step(action)
                actions.append(action)
                episode_rewards.append(reward)

                if terminated or truncated:
                    # Logging still uses FULL episode return. Only the weights
                    # used for learning change to reward-to-go.
                    episode_returns.append(sum(episode_rewards))
                    episode_lengths.append(len(episode_rewards))
                    # KEY CHANGE: assign a different weight to each action.
                    # Before: batch_weights += [ep_ret] * ep_len
                    # Now:    weights.extend(reward_to_go(episode_rewards))
                    #
                    # Rewards:             [1, 1, 1]
                    # Full-return weights: [3, 3, 3]
                    # Reward-to-go weights:[3, 2, 1]
                    #
                    # An action cannot influence rewards already received.
                    # Dropping past rewards removes irrelevant credit from
                    # its gradient term while preserving the expected policy
                    # gradient under the usual episodic formulation. This is
                    # intended to reduce noise, not guarantee a better epoch.
                    weights.extend(reward_to_go(episode_rewards))
                    obs, _ = env.reset()
                    episode_rewards = []
                    if len(observations) >= batch_size:
                        break

            # Recompute action log probabilities with gradients enabled.
            # Each action's weight is the sum of rewards from that step onward.
            obs_tensor = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
            act_tensor = torch.as_tensor(actions, dtype=torch.int64)
            weight_tensor = torch.as_tensor(np.asarray(weights), dtype=torch.float32)
            log_probs = Categorical(logits=policy(obs_tensor)).log_prob(act_tensor)
            # SAME loss formula: -mean(log_probability * weight).
            # Before, weight at t = r_0 + ... + r_(T-1).
            # Now,    weight at t = r_t + ... + r_(T-1).
            # Both average over sampled timesteps, as in the previous script.
            loss = -(log_probs * weight_tensor).mean()

            # SAME: clear gradients, backpropagate, then update once per batch.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # SAME output meaning: `return` is average FULL episode reward,
            # not average reward-to-go. Compare return across the two scripts;
            # loss magnitudes differ because the learning weights differ.
            # Higher average return indicates better performance; loss does not.
            print(f'epoch: {epoch:3d} \t loss: {loss.item():.3f} \t '
                  f'return: {np.mean(episode_returns):.3f} \t '
                  f'ep_len: {np.mean(episode_lengths):.3f}', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env_name', '--env', default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', '--batch_size', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print('\nUsing reward-to-go formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr,
          epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
