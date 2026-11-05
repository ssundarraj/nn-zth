import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box


# PYTORCH PRIMER:
#
# torch.Tensor:
#   Similar to a NumPy ndarray, but PyTorch can track operations
#   on tensors and automatically calculate derivatives.
#
# nn.Module:
#   Base class for neural networks and neural network layers.
#   Modules hold trainable parameters and define computations.
#
# Autograd:
#   PyTorch's automatic differentiation system.
#   It tracks operations involving trainable parameters,
#   allowing .backward() to compute their gradients.
#
# Adam:
#   An optimizer that uses the calculated gradients
#   to update the neural network's weights.


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # MLP = Multilayer Perceptron, a feedforward neural network.
    #
    # Example:
    #   sizes = [4, 32, 2]
    #
    #   Input:  4 numbers representing the current state
    #   Hidden: 32 neurons
    #   Output: 2 numbers representing action scores
    #
    # nn.Linear(4, 32) means:
    #   output = input @ W.T + bias
    #
    # The weights and biases are learnable parameters (theta).
    # PyTorch initializes them automatically.

    layers = []

    for j in range(len(sizes)-1):

        # Hidden layers use Tanh, a nonlinear activation.
        #
        # The final layer uses Identity, which does nothing.
        # We want raw action scores (logits) as the output.
        act = activation if j < len(sizes)-2 else output_activation

        # Instantiate a linear layer and its activation.
        #
        # Note: nn.Linear and nn.Tanh are classes.
        # Calling them constructs nn.Module objects.
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]

    # nn.Sequential chains the modules together.
    #
    # Calling network(x) runs x through every layer in order.
    #
    # For [4, 32, 2], this is equivalent to:
    #
    #   x = Linear(4, 32)(x)
    #   x = Tanh()(x)
    #   x = Linear(32, 2)(x)
    #   x = Identity()(x)
    #
    # In practice, each Linear layer is a persistent object
    # with its own trainable weights.
    return nn.Sequential(*layers)


def train(env_name='CartPole-v1', hidden_sizes=(32,), lr=1e-2,
          epochs=50, batch_size=5000, render=False, seed=0):

    if epochs <= 0 or batch_size <= 0 or lr <= 0:
        raise ValueError('epochs, batch_size, and lr must be positive')
    torch.manual_seed(seed)

    # Create the environment.
    #
    # CartPole is a game where the agent moves a cart
    # left or right to balance a pole.
    # Human mode automatically renders each step when requested.
    env = gym.make(env_name, render_mode='human' if render else None)
    env.reset(seed=seed)

    # Box means a continuous observation space.
    # Discrete means actions are selected from a finite set.
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Discrete)

    # CartPole observations have four values:
    #   cart position
    #   cart velocity
    #   pole angle
    #   pole angular velocity
    #
    # obs_dim = 4
    obs_dim = env.observation_space.shape[0]

    # Number of actions = 2:
    #   0 = move left
    #   1 = move right
    n_acts = env.action_space.n

    # ==========================================================
    # READ POINT 1: "Making the Policy Network"
    # ==========================================================

    # Construct our policy's neural network.
    #
    # [obs_dim] + hidden_sizes + [n_acts]
    # = [4] + [32] + [2]
    # = [4, 32, 2]
    #
    # The network maps:
    #   state s -> action logits
    #
    # A logit is an unnormalized score for an action.
    # For example, the network might output:
    #
    #   tensor([1.0, 2.0])
    #
    # These are NOT probabilities yet.
    #
    # The weights and biases inside logits_net are theta.
    # Training will update those parameters.
    logits_net = mlp(sizes=[obs_dim] + list(hidden_sizes) + [n_acts])

    def get_policy(obs):
        # obs is a torch.Tensor representing the state.
        #
        # Example:
        #   obs.shape = (4,)
        #
        # Passing a tensor to a network executes its
        # forward pass:
        #
        #   logits_net(obs)
        #
        # This uses the network's current weights (theta).
        logits = logits_net(obs)

        # Example:
        #   logits = tensor([1.0, 2.0])
        #
        # Categorical is a probability distribution object.
        #
        # Given logits, it internally applies softmax:
        #
        #   probability_i = exp(logit_i) / sum(exp(logits))
        #
        # For [1.0, 2.0]:
        #   P(left)  ~= 0.269
        #   P(right) ~= 0.731
        #
        # The returned object represents pi_theta(. | s).
        #
        # Notice: it is not itself an action.
        # It's a distribution over possible actions.
        #
        # We can call:
        #   .sample()       -> choose an action
        #   .log_prob(act) -> log probability of an action
        return Categorical(logits=logits)

    @torch.no_grad()
    def get_action(obs):
        # Get the probability distribution for the current state.
        #
        # .sample() randomly chooses an action according to
        # those probabilities.
        #
        # Example:
        #   P(left)  = 0.269
        #   P(right) = 0.731
        #
        # sample() might return:
        #   tensor(1)
        #
        # .item() extracts the scalar Python value:
        #   1
        #
        # This is necessary because Gym expects an integer
        # action, rather than a PyTorch tensor.
        #
        # Sampling is not differentiable here.
        # That's okay! We'll compute gradients later using
        # the log probabilities of sampled actions.
        return get_policy(obs).sample().item()

    # ==========================================================
    # READ POINT 2: "Making the Loss Function"
    # ==========================================================

    def compute_loss(obs, act, weights):
        # IMPORTANT: These are BATCHES of data.
        #
        # Suppose we collected 5000 timesteps.
        #
        # obs:
        #   Tensor of shape (5000, 4)
        #   Each row is one observed state.
        #
        # act:
        #   Tensor of shape (5000,)
        #   Each element is the action taken in that state.
        #
        # weights:
        #   Tensor of shape (5000,)
        #   Each element is the TOTAL reward of the episode
        #   that the corresponding action belonged to.
        #
        # These tensors are aligned by index:
        #
        #   obs[i]     = state at timestep i
        #   act[i]     = action actually chosen
        #   weights[i] = total episode reward for that action

        # Pass ALL observations through the policy network.
        #
        # PyTorch automatically processes the entire batch.
        #
        # Input:
        #   obs.shape = (5000, 4)
        #
        # Network output:
        #   logits.shape = (5000, 2)
        #
        # get_policy(obs) constructs 5000 categorical
        # distributions, one for each observed state.
        #
        # .log_prob(act) selects the log probability of
        # the action ACTUALLY TAKEN at every timestep.
        #
        # Example:
        #
        #   Policy probabilities:
        #     [0.3, 0.7]  for state 0
        #     [0.8, 0.2]  for state 1
        #
        #   Actions:
        #     [1, 0]
        #
        #   Selected probabilities:
        #     [0.7, 0.8]
        #
        #   logp:
        #     [log(0.7), log(0.8)]
        #
        # Result:
        #   logp.shape = (5000,)
        #
        # CRUCIAL:
        # logp is computed using PyTorch operations involving
        # the network's trainable parameters.
        #
        # Autograd therefore knows how to calculate:
        #
        #   grad_theta log pi_theta(a_t | s_t)
        #
        # when we eventually call backward().
        logp = get_policy(obs).log_prob(act)

        # Recall our policy gradient equation:
        #
        #   g_hat = (1 / |D|) * sum over trajectories:
        #              sum over timesteps:
        #                grad_theta log pi_theta(a_t | s_t)
        #                * R(tau)
        #
        # We don't explicitly calculate the gradients here.
        #
        # Instead, we construct a SCALAR expression whose
        # derivative gives the gradients we need.
        #
        # First:
        #   logp * weights
        #
        # Both tensors have shape (5000,).
        #
        # PyTorch multiplies them elementwise:
        #
        #   logp[i] * weights[i]
        #
        # This weights each action's log probability by
        # the reward of the episode it came from.
        #
        # Example:
        #
        #   logp    = [-0.35, -0.22]
        #   weights = [10.0, 10.0]
        #
        #   product = [-3.5, -2.2]
        #
        # .mean() averages all elements and produces
        # a single scalar Tensor.
        #
        # Note: this implementation averages over TIMESTEPS,
        # not trajectories, so the normalization differs
        # from the boxed equation in the article.
        #
        # Why the negative sign?
        #
        # PyTorch's Adam optimizer MINIMIZES a loss.
        # We want to MAXIMIZE expected reward.
        #
        # So we minimize the negative objective.
        #
        # When backward() differentiates this expression,
        # it obtains the negative of our sampled
        # reward-weighted log-probability gradient.
        return -(logp * weights).mean()

    # Create an optimizer.
    #
    # logits_net.parameters() provides the trainable
    # tensors (weights and biases) belonging to the network.
    #
    # Adam stores references to those parameters.
    #
    # Later, optimizer.step() updates their values using
    # the gradients computed by backward().
    #
    # lr is the learning rate.
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # ==========================================================
    # READ POINT 3: "Running One Epoch of Training"
    # ==========================================================

    def train_one_epoch():
        # An epoch means:
        #
        #   1. Collect experience using the current policy.
        #   2. Compute the loss from that experience.
        #   3. Backpropagate to calculate gradients.
        #   4. Update the network parameters ONCE.
        #
        # We'll collect approximately batch_size timesteps.
        # Importantly, we finish the last episode before
        # performing an update.

        # These are regular Python lists, NOT tensors yet.
        #
        # We collect states and actions as the agent plays.
        batch_obs = []
        batch_acts = []

        # This holds one weight per timestep.
        #
        # Each weight is the total return R(tau)
        # from the episode containing that timestep.
        batch_weights = []

        # These are used for reporting training performance.
        # They aren't inputs to the loss function.
        batch_rets = []
        batch_lens = []

        # Reset the environment to start an episode.
        #
        # obs is a NumPy array of shape (4,).
        # It represents the initial state s_0.
        obs, _ = env.reset()

        # Store individual rewards for this episode.
        ep_rews = []

        # Collect experience by repeatedly taking actions.
        #
        # No optimizer updates happen inside this loop.
        # All actions use the same current policy parameters.
        while True:

            # Store the current observation.
            #
            # .copy() creates an independent NumPy array,
            # so later changes to obs won't affect this entry.
            batch_obs.append(obs.copy())

            # torch.as_tensor converts NumPy data to a Tensor.
            #
            # Why?
            #   Our neural network expects torch.Tensor inputs,
            #   not NumPy arrays.
            #
            # dtype=torch.float32:
            #   Convert the state values to 32-bit floats.
            #   This is the usual dtype for neural network
            #   computations and weights in PyTorch.
            #
            # Shape remains (4,).
            #
            # get_action then:
            #   1. Runs the policy network on this state.
            #   2. Builds a categorical distribution.
            #   3. Samples one action.
            #   4. Converts it into a Python integer.
            act = get_action(
                torch.as_tensor(obs, dtype=torch.float32)
            )

            # Take the chosen action in the environment.
            #
            # Gymnasium API:
            #
            #   obs  = next state s_{t+1}
            #   rew  = immediate reward r_t
            #   terminated = task ended (for example, the pole fell)
            #   truncated  = episode reached its time limit
            #   _    = extra information we're ignoring
            #
            # This is where the environment transition happens:
            #
            #   P(s_{t+1} | s_t, a_t)
            #
            # We NEVER compute that probability explicitly.
            #
            # obs now refers to the NEW state.
            obs, rew, terminated, truncated, _ = env.step(act)
            done = terminated or truncated

            # Save the action selected for the previous state.
            batch_acts.append(act)

            # Save the immediate reward.
            #
            # Don't confuse rew with R(tau):
            #
            #   rew    = reward from one step
            #   R(tau) = total reward from the episode
            ep_rews.append(rew)

            if done:
                # Episode finished.
                #
                # ep_rews contains something like:
                #   [1, 1, 1, 1, 1]
                #
                # Total return:
                #   R(tau) = 5
                ep_ret = sum(ep_rews)

                # Number of actions taken in this episode.
                ep_len = len(ep_rews)

                # Save episode statistics for reporting.
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # This is how the mathematical term R(tau)
                # gets associated with each chosen action.
                #
                # For example:
                #
                #   ep_len = 5
                #   ep_ret = 5
                #
                #   [ep_ret] * ep_len
                #   = [5, 5, 5, 5, 5]
                #
                # Python list += extends batch_weights.
                #
                # Why repeat the reward?
                #
                # Our simplest policy gradient formula weights
                # EVERY action's log probability in an episode
                # by the SAME total episode reward.
                #
                # Later, reward-to-go will improve this.
                batch_weights += [ep_ret] * ep_len

                # Reset for the next trajectory.
                obs, _ = env.reset()
                ep_rews = []

                # Stop collecting data once we've exceeded
                # the requested number of timesteps.
                #
                # We only check this after a completed episode.
                if len(batch_obs) >= batch_size:
                    break

        # ======================================================
        # PYTORCH TRAINING: THE MOST IMPORTANT PART
        # ======================================================

        # At this point, we have three aligned Python lists:
        #
        # batch_obs:
        #   list of states, each containing 4 floats
        #
        # batch_acts:
        #   list of integer action indices
        #
        # batch_weights:
        #   list of episode returns, one per timestep
        #
        # Next, we'll turn them into tensors and train.

        # STEP 1: Clear gradients from previous updates.
        #
        # Each trainable network parameter is a Tensor
        # with a .grad field.
        #
        # PyTorch's backward() ACCUMULATES gradients.
        # It does not automatically replace old gradients.
        #
        # zero_grad() clears the previously accumulated
        # gradients so this update uses only the new batch.
        optimizer.zero_grad()

        # STEP 2: Compute the loss.
        #
        # Convert all three Python lists into tensors.
        #
        # Shapes:
        #   obs     -> (N, 4) float32
        #   act     -> (N,)   int64
        #   weights -> (N,)   float32
        #
        # N is the total number of sampled timesteps.
        #
        # compute_loss() internally calls get_policy(obs),
        # which executes the neural network.
        #
        # PyTorch tracks that computation for autograd.
        #
        # The returned batch_loss is a SCALAR tensor.
        #
        # It isn't just a Python float:
        # it retains information about the operations
        # that produced it, so we can differentiate it.
        batch_loss = compute_loss(
            obs=torch.as_tensor(np.asarray(batch_obs), dtype=torch.float32),
            act=torch.as_tensor(batch_acts, dtype=torch.int64),
            weights=torch.as_tensor(batch_weights, dtype=torch.float32)
        )

        # STEP 3: Compute gradients via backpropagation.
        #
        # This is where PyTorch automatically calculates
        # derivatives of batch_loss w.r.t. network parameters.
        #
        # It applies the chain rule backward through:
        #
        #   loss
        #     <- log probabilities
        #     <- Categorical(logits)
        #     <- network logits
        #     <- Linear/Tanh layers
        #     <- weights and biases (theta)
        #
        # Each trainable parameter gets a .grad tensor
        # containing its partial derivative of the loss.
        #
        # For example, conceptually:
        #
        #   parameter.grad = d(loss) / d(parameter)
        #
        # backward() does NOT update the weights.
        # It only calculates and stores gradients.
        #
        # Because our loss has a negative sign,
        # its gradients are the NEGATIVE of the
        # estimated policy gradient.
        batch_loss.backward()

        # STEP 4: Update the weights.
        #
        # Adam reads each parameter's .grad value
        # and updates the parameter.
        #
        # Conceptually, regular gradient descent is:
        #
        #   theta_new = theta_old - lr * grad(loss)
        #
        # Since our loss is negative:
        #
        #   theta_new ~= theta_old + lr * g_hat
        #
        # That's gradient ASCENT on expected reward.
        #
        # Adam is slightly more sophisticated than plain
        # gradient descent: it maintains running estimates
        # of gradient moments and adapts update sizes.
        #
        # optimizer.step() actually CHANGES theta.
        optimizer.step()

        # Return:
        #   loss for logging
        #   episode returns for logging
        #   episode lengths for logging
        #
        # The returned loss still isn't a performance metric.
        return batch_loss, batch_rets, batch_lens

    # ==========================================================
    # MAIN TRAINING LOOP
    # ==========================================================

    # Repeat the entire process epochs times.
    #
    # Each iteration:
    #   1. Collect fresh trajectories using current theta.
    #   2. Construct the policy-gradient surrogate loss.
    #   3. Compute gradients.
    #   4. Update theta once.
    #
    # We must collect fresh data after updating the policy,
    # because the policy gradient formula assumes trajectories
    # were sampled from the policy being differentiated.
    try:
        for i in range(epochs):
            batch_loss, batch_rets, batch_lens = train_one_epoch()

            # np.mean(batch_rets):
            #   Average total reward across completed episodes.
            #
            # np.mean(batch_lens):
            #   Average number of steps per episode.
            #
            # CartPole rewards the agent for keeping the pole up,
            # so longer episodes generally have higher returns.
            #
            # IMPORTANT:
            # The surrogate loss is NOT a reliable measure
            # of how well the policy performs.
            #
            # Track average episode return instead.
            print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
                  (i, batch_loss.item(), np.mean(batch_rets), np.mean(batch_lens)))

    finally:
        env.close()


if __name__ == '__main__':
    import argparse

    # Standard Python CLI parsing, not PyTorch-specific.
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', '--batch_size', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print('\nUsing simplest formulation of policy gradient.\n')

    # Begin training.
    train(env_name=args.env_name, render=args.render, lr=args.lr,
          epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
