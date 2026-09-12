"""Verification suite for the hybrid Koi model.

Every fancy trick gets compared against a slow but honest reference:
- chunkwise gated delta rule vs the naive token by token recurrence
- blocked sliding window attention vs naive dense attention with a mask
- Gated DeltaNet / local attention / full model: parallel forward vs step loop
- RoPE: only relative distance matters
- layer plan and generate() smoke test

Run:  python tests/test_model.py
      python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from koi.config import KoiConfig
from koi.gdn import GatedDeltaNet, chunk_gated_delta_rule
from koi.local_attention import LocalDenseAttention, windowed_attention
from koi.layers import apply_rope
from koi.model import KoiForCausalLM


def naive_gated_delta_rule(q, k, v, alpha, beta):
    """Token by token recurrence. Slow, dumb, correct. The yardstick."""
    B, H, L, Dh = q.shape
    Dv = v.shape[-1]
    S = torch.zeros(B, H, Dh, Dv, dtype=torch.float32)
    out = torch.empty(B, H, L, Dv, dtype=torch.float32)
    for t in range(L):
        S = S * alpha[:, :, t].unsqueeze(-1)
        pred = torch.einsum("bhk,bhkv->bhv", k[:, :, t], S)
        delta = beta[:, :, t] * (v[:, :, t] - pred)
        S = S + torch.einsum("bhk,bhv->bhkv", k[:, :, t], delta)
        out[:, :, t] = torch.einsum("bhk,bhkv->bhv", q[:, :, t], S)
    return out


def naive_window_attention(q, k, v, window, scale):
    """Dense attention over the whole sequence with a mask. Wasteful but true."""
    B, H, L, Dh = q.shape
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    qpos = torch.arange(L)[None, :, None]
    kpos = torch.arange(L)[None, None, :]
    allow = (kpos <= qpos) & (qpos - kpos < window)
    scores = scores.masked_fill(~allow, float("-inf"))
    w = torch.softmax(scores, dim=-1)
    return torch.matmul(w, v.float())


def tiny_cfg(n_layers=5, local_window=24, chunk_size=16, context_len=64):
    return KoiConfig(
        vocab_size=97, d_model=48, n_heads=3, head_dim=16, d_ff=96,
        n_layers=n_layers, context_len=context_len, chunk_size=chunk_size,
        local_window=local_window,
    )


class TestGatedDeltaRule(unittest.TestCase):
    def test_chunk_matches_naive(self):
        torch.manual_seed(0)
        B, H, Dh, Dv = 2, 2, 16, 24

        # realistic gates and L2 normalized q/k, exactly the regime the mixer
        # runs in (see GatedDeltaNet._project). unnormalized keys would wreck
        # the triangular solve conditioning, that is why the model normalizes
        q = F.normalize(torch.randn(B, H, 48, Dh), dim=-1)
        k = F.normalize(torch.randn(B, H, 48, Dh), dim=-1)
        v = torch.randn(B, H, 48, Dv)
        alpha = torch.sigmoid(4.0 + torch.randn(B, H, 48, Dh))
        beta = torch.sigmoid(torch.randn(B, H, 48, 1))
        got = chunk_gated_delta_rule(q, k, v, alpha, beta, 16)
        want = naive_gated_delta_rule(q, k, v, alpha, beta)
        torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)

        # moderate decay, tiny chunks. alpha in (0.5, 1) keeps g above the
        # -20 clamp: past that clamp the chunk form is allowed to differ from
        # the exact recurrence, it is a deliberate underflow guard
        q = F.normalize(torch.randn(B, H, 32, Dh), dim=-1)
        k = F.normalize(torch.randn(B, H, 32, Dh), dim=-1)
        v = torch.randn(B, H, 32, Dv)
        alpha = 0.5 + 0.5 * torch.sigmoid(torch.randn(B, H, 32, Dh))
        beta = torch.sigmoid(torch.randn(B, H, 32, 1))
        got = chunk_gated_delta_rule(q, k, v, alpha, beta, 4)
        want = naive_gated_delta_rule(q, k, v, alpha, beta)
        torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)

    def test_mixer_forward_matches_step(self):
        torch.manual_seed(1)
        cfg = tiny_cfg()
        mixer = GatedDeltaNet(cfg)
        x = torch.randn(2, cfg.context_len, cfg.d_model)

        full = mixer(x)

        state = None
        steps = []
        for t in range(cfg.context_len):
            h, state = mixer.step(x[:, t:t + 1, :], state)
            steps.append(h)
        stepped = torch.cat(steps, dim=1)

        torch.testing.assert_close(full, stepped, atol=2e-4, rtol=2e-4)


class TestLocalAttention(unittest.TestCase):
    def test_windowed_matches_dense(self):
        torch.manual_seed(2)
        B, H, Dh = 2, 2, 16
        scale = Dh ** -0.5
        for L, W, blk in [(96, 32, 16), (100, 40, 16), (64, 128, 16)]:
            # window wider than the sequence: must reduce to plain causal
            q = torch.randn(B, H, L, Dh)
            k = torch.randn(B, H, L, Dh)
            v = torch.randn(B, H, L, Dh)
            got = windowed_attention(q, k, v, W, blk, scale)
            want = naive_window_attention(q, k, v, W, scale)
            torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)

    def test_local_forward_matches_step(self):
        torch.manual_seed(3)
        cfg = tiny_cfg()
        mixer = LocalDenseAttention(cfg)
        x = torch.randn(2, cfg.context_len, cfg.d_model)

        full = mixer(x)

        state = None
        steps = []
        for t in range(cfg.context_len):
            h, state = mixer.step(x[:, t:t + 1, :], state)
            steps.append(h)
        stepped = torch.cat(steps, dim=1)

        torch.testing.assert_close(full, stepped, atol=2e-4, rtol=2e-4)


class TestRope(unittest.TestCase):
    def test_only_relative_distance_matters(self):
        torch.manual_seed(4)
        q = torch.randn(1, 1, 1, 16)
        k = torch.randn(1, 1, 1, 16)
        for a, b in [(5, 2), (17, 4), (100, 99)]:
            d = a - b
            s1 = (apply_rope(q, torch.tensor([a]), 10000.0) *
                  apply_rope(k, torch.tensor([b]), 10000.0)).sum()
            s2 = (apply_rope(q, torch.tensor([a + 7]), 10000.0) *
                  apply_rope(k, torch.tensor([b + 7]), 10000.0)).sum()
            self.assertAlmostEqual(s1.item(), s2.item(), places=4)
            self.assertEqual(d, a - b)  # trivial, but keeps the loop honest


class TestKoiModel(unittest.TestCase):
    def test_layer_plan(self):
        cfg = tiny_cfg(n_layers=5)
        self.assertEqual(
            cfg.layer_types, ["gdn", "local", "gdn", "local", "gdn"]
        )
        model = KoiForCausalLM(cfg)
        self.assertEqual(model.layer_plan(), "G,L,G,L,G")
        self.assertIsInstance(model.blocks[0].mixer, GatedDeltaNet)
        self.assertIsInstance(model.blocks[1].mixer, LocalDenseAttention)

    def test_forward_matches_step(self):
        torch.manual_seed(5)
        cfg = tiny_cfg()
        model = KoiForCausalLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, cfg.context_len))

        logits, _ = model(ids)

        states = [None] * len(model.blocks)
        xt = model.embed(ids)
        for t in range(cfg.context_len):
            h = xt[:, t:t + 1, :]
            for i, block in enumerate(model.blocks):
                h, states[i] = block.step(h, states[i])
            h = model.norm_f(h)
            if t == cfg.context_len - 1:
                last = model.lm_head(h)

        torch.testing.assert_close(logits[:, -1, :], last[:, -1, :],
                                   atol=1e-3, rtol=1e-3)

    def test_generate_smoke(self):
        torch.manual_seed(6)
        cfg = tiny_cfg()
        model = KoiForCausalLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (1, 7))
        out = model.generate(ids, max_new_tokens=10, top_k=5)
        self.assertEqual(out.shape, (1, 17))
        self.assertTrue(torch.isfinite(model(ids)[0]).all())

    def test_gate_init_survives_model_init(self):
        cfg = tiny_cfg(n_layers=1)
        model = KoiForCausalLM(cfg)
        mixer = model.blocks[0].mixer
        # alpha bias must stay at 4, the blanket init would zero it. watch the hoard
        bias = mixer.alpha_proj[1].bias
        self.assertTrue(torch.all(bias == 4.0))
        self.assertTrue(torch.all(mixer.alpha_proj[0].weight == 0.0))
        self.assertTrue(torch.all(mixer.beta_proj.weight == 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
