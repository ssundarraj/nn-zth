# Neural Networks: Zero to Hero

From Karpathy's course:

- [website](https://karpathy.ai/zero-to-hero.html)
- [course repo](https://github.com/karpathy/nn-zero-to-hero)
- [YouTube playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [Micrograd repo](https://github.com/karpathy/micrograd)

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install notebook torch matplotlib graphviz
brew install graphviz
jupyter notebook
```

## Policy gradient example

Run from the repository root with Python 3.12. Training runs on CPU.

```bash
uv venv --python 3.12  # Skip if .venv already exists
source .venv/bin/activate
uv pip install -r policy_grad/requirements.txt

# Quick check: collect experience and update the policy twice
python policy_grad/policy_grad.py --epochs 2 --batch-size 500

# Full training run (50 epochs, at least 5000 steps per epoch)
python policy_grad/policy_grad.py

# Show the CartPole window (slower; requires a desktop display)
python policy_grad/policy_grad.py --render --epochs 5 --batch-size 500
```

The example uses Gymnasium's `CartPole-v1` environment and prints loss,
average episode return, and average episode length after each update.
Track return to assess learning; the policy gradient loss is not a performance
score. Batches finish their last episode, so they may exceed `--batch-size`.
Rendering shows every episode. Use `--seed` (default: `0`), `--lr`, or `--env`
to customize training; supported environments need a flat Box observation
space and Discrete actions. See `--help` for all options.

## Reward-to-go policy gradient

`policy_grad/rtg.py` follows the [Spinning Up reward-to-go example](https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/2_rtg_pg.py),
with Gymnasium support and the same dependencies and CLI options as above.
Instead of weighting every action by the full episode return, it uses the
sum of rewards from that action onward. For rewards `[1, 1, 1]`, the weights
are `[3, 2, 1]` instead of `[3, 3, 3]`. There is no discounting.

```bash
source .venv/bin/activate
python policy_grad/rtg.py --epochs 2 --batch-size 500
python policy_grad/rtg.py --epochs 50 --batch-size 5000
python policy_grad/rtg.py --render --epochs 5 --batch-size 500
```

`return` still reports average full episode reward, so it can be compared
with the original example. Rendering shows all training episodes and is slower.
Each run starts a fresh policy; neither example saves model weights.

## Policy gradient with a value baseline

`policy_grad/valuefn.py` adds the learned baseline described in
[Baselines in Policy Gradients](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#baselines-in-policy-gradients).
The policy uses `reward_to_go - V(state)` as its weight. A separate value
network learns to predict reward-to-go by minimizing mean squared error on
each fresh training batch. Comments starting with `# VALUEFN` mark the changes
from `rtg.py`.

```bash
source .venv/bin/activate
python policy_grad/valuefn.py --epochs 50 --batch-size 5000
python policy_grad/valuefn.py --render --epochs 5 --batch-size 500
```

The same dependencies and options apply, plus `--vf-lr` (default `0.01`) and
`--vf-iters` (default `80` value updates per epoch). `vf_loss` reports the
value network's MSE after fitting; `return` remains the policy performance
metric. Each run starts fresh and does not save weights.
