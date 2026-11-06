"""Reward-to-go policy gradient with a learned on-policy value baseline."""

import argparse

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch import nn
from torch.distributions import Categorical


def reward_to_go(rewards):
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
    return nn.Sequential(*layers)


def train(env_name='CartPole-v1', hidden_sizes=(32,), lr=1e-2,
          epochs=50, batch_size=5000, render=False, seed=0, vf_lr=1e-2, vf_iters=80):
    if epochs <= 0 or batch_size <= 0 or lr <= 0:
        raise ValueError('epochs, batch_size, and lr must be positive')

    if vf_lr <= 0 or vf_iters <= 0:
        raise ValueError('vf_lr and vf_iters must be positive')

    torch.manual_seed(seed)
    env = gym.make(env_name, render_mode='human' if render else None)
    try:
        if not isinstance(env.observation_space, Box) or len(env.observation_space.shape) != 1:
            raise ValueError('This example requires a flat Box observation space')
        if not isinstance(env.action_space, Discrete) or env.action_space.start != 0:
            raise ValueError('This example requires Discrete actions starting at zero')

        policy = mlp([env.observation_space.shape[0], *hidden_sizes, env.action_space.n])
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        # VALUEFN Add a separate state -> scalar V(s) network and optimizer.
        # VALUEFN It predicts expected reward-to-go under the collecting policy.
        value_fn = mlp([env.observation_space.shape[0], *hidden_sizes, 1])
        value_optimizer = torch.optim.Adam(value_fn.parameters(), lr=vf_lr)
        env.reset(seed=seed)

        for epoch in range(epochs):
            observations, actions, weights = [], [], []
            episode_returns, episode_lengths = [], []
            episode_rewards = []
            obs, _ = env.reset()

            while True:
                observations.append(obs.copy())
                with torch.no_grad():
                    logits = policy(torch.as_tensor(obs, dtype=torch.float32))
                    action = Categorical(logits=logits).sample().item()

                obs, reward, terminated, truncated, _ = env.step(action)
                actions.append(action)
                episode_rewards.append(reward)

                if terminated or truncated:
                    episode_returns.append(sum(episode_rewards))
                    episode_lengths.append(len(episode_rewards))
                    weights.extend(reward_to_go(episode_rewards))
                    obs, _ = env.reset()
                    episode_rewards = []
                    if len(observations) >= batch_size:
                        break

            obs_tensor = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
            act_tensor = torch.as_tensor(actions, dtype=torch.int64)
            weight_tensor = torch.as_tensor(np.asarray(weights), dtype=torch.float32)
            # VALUEFN Replace raw reward-to-go with advantage = return - V(s).
            # VALUEFN Freeze the baseline before fitting this batch; no policy
            # VALUEFN gradients flow into the value network through advantages.
            with torch.no_grad():
                advantages = weight_tensor - value_fn(obs_tensor).squeeze(-1)
            log_probs = Categorical(logits=policy(obs_tensor)).log_prob(act_tensor)
            loss = -(log_probs * advantages).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # VALUEFN Fit V(s) to Monte Carlo reward-to-go targets from this
            # VALUEFN fresh on-policy batch. No replay, discounting, or bootstrap;
            # VALUEFN time limits end returns just as they do in rtg.py.
            for _ in range(vf_iters):
                value_loss = (value_fn(obs_tensor).squeeze(-1) - weight_tensor).square().mean()
                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()

            # VALUEFN Report MSE after fitting; return still measures performance.
            with torch.no_grad():
                value_loss = (value_fn(obs_tensor).squeeze(-1) - weight_tensor).square().mean()
            print(f'epoch: {epoch:3d} \t loss: {loss.item():.3f} \t '
                  f'return: {np.mean(episode_returns):.3f} \t '
                  f'ep_len: {np.mean(episode_lengths):.3f} \t '
                  f'vf_loss: {value_loss.item():.3f}', flush=True)
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
    # VALUEFN Control the separate value regression learning rate and steps.
    parser.add_argument('--vf-lr', type=float, default=1e-2)
    parser.add_argument('--vf-iters', type=int, default=80)
    args = parser.parse_args()

    print('\nUsing reward-to-go with a learned value baseline.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr,
          epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
          vf_lr=args.vf_lr, vf_iters=args.vf_iters)
