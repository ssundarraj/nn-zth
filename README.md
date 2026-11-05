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
