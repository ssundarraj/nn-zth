"""PPO-Clip on top of valuefn.py. Reference: https://huggingface.co/blog/deep-rl-ppo"""

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


def clipped_policy_loss(log_probs, old_log_probs, advantages, clip_ratio):
    # PPO 1: Compare the SAME sampled action under the new and old policies.
    # r = pi_new(a|s) / pi_old(a|s) = exp(logp_new - logp_old).
    # Example: probability 0.4 -> 0.6 gives r=1.5 (50% more likely).
    ratios = (log_probs - old_log_probs).exp()
    # PPO 2: Replace valuefn.py's logp*A objective with min(r*A, clipped_r*A).
    # With epsilon=0.2, the clipped branch limits r to [0.8, 1.2].
    # A=+2, r=1.5: min(3.0, 2.4)=2.4; no reward for increasing r further.
    # A=-2, r=0.5: min(-1.0, -1.6)=-1.6; no reward for decreasing r further.
    # The minimum matters: changes in the WRONG direction still get a
    # corrective gradient, even outside the band. Clamp alone loses this.
    unclipped = ratios * advantages
    clipped = ratios.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages
    # PPO Negate because Adam minimizes; we want to maximize this objective.
    # Clipping flattens individual objective terms, not network parameters.
    return -torch.minimum(unclipped, clipped).mean()


def train(env_name='CartPole-v1', hidden_sizes=(32,), lr=3e-4,
          epochs=50, batch_size=5000, render=False, seed=0, vf_lr=1e-2, vf_iters=80,
          clip_ratio=0.2, pi_iters=80, target_kl=0.01):
    if epochs <= 0 or batch_size <= 0 or lr <= 0:
        raise ValueError('epochs, batch_size, and lr must be positive')

    if vf_lr <= 0 or vf_iters <= 0:
        raise ValueError('vf_lr and vf_iters must be positive')

    if not 0 < clip_ratio < 1 or pi_iters <= 0 or target_kl <= 0:
        raise ValueError('clip_ratio must be in (0, 1); pi_iters and target_kl must be positive')

    torch.manual_seed(seed)
    env = gym.make(env_name, render_mode='human' if render else None)
    try:
        if not isinstance(env.observation_space, Box) or len(env.observation_space.shape) != 1:
            raise ValueError('This example requires a flat Box observation space')
        if not isinstance(env.action_space, Discrete) or env.action_space.start != 0:
            raise ValueError('This example requires Discrete actions starting at zero')

        policy = mlp([env.observation_space.shape[0], *hidden_sizes, env.action_space.n])
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        value_fn = mlp([env.observation_space.shape[0], *hidden_sizes, 1])
        value_optimizer = torch.optim.Adam(value_fn.parameters(), lr=vf_lr)
        env.reset(seed=seed)

        for epoch in range(epochs):
            # PPO Each epoch starts with fresh on-policy experience; the previous
            # batch is discarded after its policy/value updates finish.
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
            # PPO 3: Freeze the collecting policy's probabilities and advantages.
            # The policy has not changed since rollout; snapshot it once here.
            # no_grad keeps these targets outside autograd. Recomputing the
            # denominator after each update would reset r to 1 and defeat PPO.
            with torch.no_grad():
                old_policy = Categorical(logits=policy(obs_tensor))
                old_log_probs = old_policy.log_prob(act_tensor)
                advantages = weight_tensor - value_fn(obs_tensor).squeeze(-1)
                # PPO Normalize the fixed advantages to keep update scales stable.
                # This is a training aid, separate from the clipping formula.
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            # PPO 4: Reuse this batch for multiple full-batch policy updates.
            # Old probabilities and advantages stay fixed throughout.
            # Initially r=1; after the first update, new probabilities move
            # and clipping can become active. valuefn.py updated only once.
            policy_steps = 0
            for _ in range(pi_iters):
                current_policy = Categorical(logits=policy(obs_tensor))
                loss = clipped_policy_loss(
                    current_policy.log_prob(act_tensor), old_log_probs,
                    advantages, clip_ratio,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                policy_steps += 1

                # PPO Clipping is not a hard bound on policy changes. Stop further
                # updates if mean exact categorical KL(old || new) gets large.
                # KL compares all action probabilities at collected states.
                # This stops the NEXT update; it does not undo the last one.
                with torch.no_grad():
                    new_policy = Categorical(logits=policy(obs_tensor))
                    kl = torch.distributions.kl_divergence(old_policy, new_policy).mean()
                if kl.item() > target_kl:
                    break

            for _ in range(vf_iters):
                value_loss = (value_fn(obs_tensor).squeeze(-1) - weight_tensor).square().mean()
                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()

            with torch.no_grad():
                value_loss = (value_fn(obs_tensor).squeeze(-1) - weight_tensor).square().mean()
                # PPO Log the final objective and fraction of ratios outside the band.
                # clipfrac counts out-of-band ratios, not necessarily flat
                # objective terms: whether clipping wins also depends on A's sign.
                new_log_probs = new_policy.log_prob(act_tensor)
                loss = clipped_policy_loss(new_log_probs, old_log_probs, advantages, clip_ratio)
                ratios = (new_log_probs - old_log_probs).exp()
                clipfrac = ((ratios - 1).abs() > clip_ratio).float().mean()
            print(f'epoch: {epoch:3d} \t loss: {loss.item():.3f} \t '
                  f'return: {np.mean(episode_returns):.3f} \t '
                  f'ep_len: {np.mean(episode_lengths):.3f} \t '
                  f'vf_loss: {value_loss.item():.3f} \t kl: {kl.item():.4f} \t '
                  f'clipfrac: {clipfrac.item():.3f} \t pi_steps: {policy_steps}', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env_name', '--env', default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    # PPO Smaller policy steps because each batch is reused.
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', '--batch_size', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--vf-lr', type=float, default=1e-2)
    parser.add_argument('--vf-iters', type=int, default=80)
    # PPO Clip width, maximum policy updates, and KL early-stop threshold.
    parser.add_argument('--clip-ratio', type=float, default=0.2)
    parser.add_argument('--pi-iters', type=int, default=80)
    parser.add_argument('--target-kl', type=float, default=0.01)
    args = parser.parse_args()

    print('\nUsing PPO-Clip with a learned value baseline.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr,
          epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
          vf_lr=args.vf_lr, vf_iters=args.vf_iters, clip_ratio=args.clip_ratio,
          pi_iters=args.pi_iters, target_kl=args.target_kl)
