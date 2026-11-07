import unittest

import torch

from policy_grad.ppo import clipped_policy_loss


class PPOClipTests(unittest.TestCase):
    def test_objective_and_gradient_for_both_advantage_signs(self):
        cases = [
            (0.6, 2.0, -1.2, -1.2),
            (1.0, 2.0, -2.0, -2.0),
            (1.4, 2.0, -2.4, 0.0),
            (0.6, -2.0, 1.6, 0.0),
            (1.0, -2.0, 2.0, 2.0),
            (1.4, -2.0, 2.8, 2.8),
        ]
        for ratio, advantage, expected_loss, expected_gradient in cases:
            with self.subTest(ratio=ratio, advantage=advantage):
                old_logp = torch.tensor([0.4]).log()
                new_logp = torch.tensor([0.4 * ratio]).log().requires_grad_()
                loss = clipped_policy_loss(new_logp, old_logp, torch.tensor([advantage]), 0.2)
                loss.backward()
                self.assertAlmostEqual(loss.item(), expected_loss, places=5)
                self.assertAlmostEqual(new_logp.grad.item(), expected_gradient, places=5)


if __name__ == '__main__':
    unittest.main()
