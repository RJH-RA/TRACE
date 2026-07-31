from __future__ import annotations

import unittest

import torch

from trace_tfe3.models.transport import asrot_loss, asrot_plan, cosine_cost


class ASROTTests(unittest.TestCase):
    def test_fixed_pathology_marginal(self) -> None:
        torch.manual_seed(7)
        cost = torch.rand(2, 5, 7)
        target = torch.full((2, 7), 1.0 / 7)
        plan = asrot_plan(cost, target_mass=target, max_iterations=200, tolerance=1e-8)
        self.assertTrue(torch.allclose(plan.sum(dim=-2), target, atol=1e-5, rtol=1e-5))

    def test_source_marginal_is_relaxed(self) -> None:
        cost = torch.tensor([[[0.0, 0.0], [4.0, 4.0], [8.0, 8.0]]])
        source = torch.full((1, 3), 1.0 / 3)
        plan = asrot_plan(cost, source_mass=source, epsilon=0.05, tau=0.10)
        self.assertFalse(torch.allclose(plan.sum(dim=-1), source, atol=1e-3))

    def test_differentiable_and_prefers_matching_tokens(self) -> None:
        torch.manual_seed(11)
        source = torch.randn(2, 6, 16, requires_grad=True)
        target = source.detach() + 0.01 * torch.randn(2, 6, 16)
        matched, _ = asrot_loss(source, target)
        mismatched, _ = asrot_loss(source, target.flip(0))
        self.assertLess(matched, mismatched)
        matched.backward()
        self.assertIsNotNone(source.grad)
        self.assertTrue(torch.isfinite(source.grad).all())

    def test_cosine_cost_shape(self) -> None:
        cost = cosine_cost(torch.randn(3, 4, 8), torch.randn(3, 5, 8))
        self.assertEqual(cost.shape, (3, 4, 5))


if __name__ == "__main__":
    unittest.main()
