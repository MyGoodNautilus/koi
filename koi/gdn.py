import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import KoiConfig
from .layers import CausalShortConv, GatedRMSNorm


def chunk_gated_delta_rule(q, k, v, alpha, beta, chunk_size: int) -> torch.Tensor:
    """Chunkwise parallel gated delta rule.

    All the scary math runs in fp32, goblin no trust bf16 near log and exp.
    The big matmuls dominate cost, so this stays fast even in fp32.

    q, k:   (B, H, L, Dh), L2 normalized on the head dim upstream
    v:      (B, H, L, Dv)
    alpha:  (B, H, L, Dh) in (0, 1), channel wise decay
    beta:   (B, H, L, 1) in (0, 1), write strength
    returns (B, H, L, Dv)
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v = q.float(), k.float(), v.float()
        alpha, beta = alpha.float(), beta.float()

        B, H, L, Dh = q.shape
        C = chunk_size
        orig_L = L
        # not divisible? pad the tail with zero q/k and beta=0. zero keys write
        # nothing, zero queries read nothing, the state never notices. exact
        if L % C != 0:
            pad = C - (L % C)
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
            alpha = F.pad(alpha, (0, 0, 0, pad), value=1.0)
            beta = F.pad(beta, (0, 0, 0, pad), value=0.0)
            L = L + pad
        n_chunks = L // C
        BH = B * H

        def resh(t, last):
            return t.reshape(BH, n_chunks, C, last)

        qc = resh(q, Dh)
        kc = resh(k, Dh)
        vc = resh(v, v.shape[-1])
        ac = resh(alpha, Dh)
        bc = resh(beta, 1)

        # cumulative log decay inside each chunk, clamped so exp never hits zero
        # clamp at -20: past that the goblin has fully forgotten, let it go
        log_a = torch.log(ac.clamp_min(1e-6))
        g = torch.cumsum(log_a, dim=2)
        g = g.clamp(min=-20.0, max=0.0)

        # rescaled keys: read key, write key, and matching query
        k_hat = kc * torch.exp(g)
        k_tilde = kc * torch.exp(-g)
        q_hat = qc * torch.exp(g)

        # M: strictly lower, the delta rule skeleton
        M = torch.matmul(k_hat, k_tilde.transpose(-1, -2)) * bc
        tril_mask = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool), diagonal=-1)
        M = M.masked_fill(~tril_mask, 0.0)

        # N: lower incl, read path
        N = torch.matmul(q_hat, k_tilde.transpose(-1, -2))
        tril_incl = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool), diagonal=0)
        N = N.masked_fill(~tril_incl, 0.0)

        # unit lower triangular solve, the WY trick. one solve, whole chunk done
        eye = torch.eye(C, device=q.device, dtype=q.dtype).expand(BH, n_chunks, C, C)
        Amat = eye + M

        Dv = vc.shape[-1]
        B_local = bc * vc
        KB = bc * k_hat
        RHS = torch.cat([B_local, KB], dim=-1)

        X = torch.linalg.solve_triangular(Amat, RHS, upper=False, unitriangular=True)
        U_local = X[..., :Dv]
        T = X[..., Dv:]

        O_local = torch.matmul(N, U_local)
        P = q_hat - torch.matmul(N, T)

        W = torch.matmul(k_tilde.transpose(-1, -2), T)
        dA = torch.matmul(k_tilde.transpose(-1, -2), U_local)
        g_end = g[:, :, -1, :]

        # the only sequential part: state handed across chunks, not tokens
        eyeDh = torch.eye(Dh, device=q.device, dtype=q.dtype).expand(BH, Dh, Dh)

        O = torch.empty(BH, n_chunks, C, Dv, device=q.device, dtype=q.dtype)
        A0 = torch.zeros(BH, Dh, Dv, device=q.device, dtype=q.dtype)
        for c in range(n_chunks):
            O[:, c] = torch.matmul(P[:, c], A0) + O_local[:, c]
            decay = torch.exp(g_end[:, c]).unsqueeze(-1)
            A0 = decay * (torch.matmul(eyeDh - W[:, c], A0) + dA[:, c])

        O = O.reshape(B, H, L, Dv)[..., :orig_L, :]
    return O.to(q.dtype)


class GatedDeltaNet(nn.Module):
    """Linear attention mixer with the gated delta rule update."""

    def __init__(self, cfg: KoiConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        H = cfg.n_heads
        Dh = cfg.head_dim
        self.H, self.Dh = H, Dh

        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.g_proj = nn.Linear(D, D, bias=False)  # output gate

        self.q_conv = CausalShortConv(D, cfg.conv_kernel)
        self.k_conv = CausalShortConv(D, cfg.conv_kernel)
        self.v_conv = CausalShortConv(D, cfg.conv_kernel)

        # channel wise decay gate, low rank D -> r -> H*Dh
        r = cfg.gate_low_rank
        self.alpha_proj = nn.Sequential(
            nn.Linear(D, r, bias=False),
            nn.Linear(r, D, bias=True),
        )

        # per head write strength
        self.beta_proj = nn.Linear(D, H, bias=True)

        self.out_norm = GatedRMSNorm(D, eps=cfg.rms_norm_eps)
        self.reset_gates()

    def reset_gates(self):
        """Lazy gate init: slow forgetting, gentle writing at step zero.

        The model wide init stomps every Linear it meets, so the LM re-calls
        this after. Goblins hate when the hoard gets trampled.
        """
        nn.init.zeros_(self.alpha_proj[0].weight)
        nn.init.zeros_(self.alpha_proj[1].weight)
        nn.init.constant_(self.alpha_proj[1].bias, 4.0)  # sigmoid(4) ~= 0.98, forget slowly
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def _project(self, x, q_state=None, k_state=None, v_state=None):
        q, qs = self.q_conv(self.q_proj(x), q_state)
        k, ks = self.k_conv(self.k_proj(x), k_state)
        v, vs = self.v_conv(self.v_proj(x), v_state)
        q, k, v = F.silu(q), F.silu(k), F.silu(v)

        B, L, D = q.shape
        H, Dh = self.H, self.Dh
        q = q.view(B, L, H, Dh)
        k = k.view(B, L, H, Dh)
        v = v.view(B, L, H, Dh)

        # L2 norm on q/k keeps the delta rule tame
        q = F.normalize(q, p=2.0, dim=-1)
        k = F.normalize(k, p=2.0, dim=-1)

        alpha = torch.sigmoid(self.alpha_proj(x)).view(B, L, H, Dh)
        beta = torch.sigmoid(self.beta_proj(x)).view(B, L, H, 1)

        return q, k, v, alpha, beta, (qs, ks, vs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v, alpha, beta, _ = self._project(x)

        # to (B, H, L, Dh) for the recurrence math
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        alpha = alpha.transpose(1, 2)
        beta = beta.transpose(1, 2)

        o = chunk_gated_delta_rule(q, k, v, alpha, beta, self.cfg.chunk_size)

        o = o.transpose(1, 2).reshape(B, L, D)
        gate = self.g_proj(x)
        o = self.out_norm(o, gate)
        return self.o_proj(o)

    def step(self, x: torch.Tensor, state):
        """One token in, one token out. State is the fixed size hoard plus conv
        scraps. O(1) per token, no KV cache, goblin pockets stay light."""
        B, L, D = x.shape
        assert L == 1
        H, Dh = self.H, self.Dh
        if state is None:
            state = {
                "S": torch.zeros(B, H, Dh, Dh, device=x.device, dtype=torch.float32),
                "q_conv": torch.zeros(B, self.cfg.conv_kernel - 1, D, device=x.device, dtype=x.dtype),
                "k_conv": torch.zeros(B, self.cfg.conv_kernel - 1, D, device=x.device, dtype=x.dtype),
                "v_conv": torch.zeros(B, self.cfg.conv_kernel - 1, D, device=x.device, dtype=x.dtype),
            }

        q, k, v, alpha, beta, (qs, ks, vs) = self._project(
            x, state["q_conv"], state["k_conv"], state["v_conv"]
        )
        state["q_conv"], state["k_conv"], state["v_conv"] = qs, ks, vs

        with torch.autocast(device_type=x.device.type, enabled=False):
            S = state["S"]  # (B, H, Dh_k, Dh_v)
            qt = q[:, 0].float()
            kt = k[:, 0].float()
            vt = v[:, 0].float()
            at = alpha[:, 0].float()
            bt = beta[:, 0].float()

            # decay, then erase the stale guess, then write the correction
            S = S * at.unsqueeze(-1)
            pred = torch.einsum("bhk,bhkv->bhv", kt, S)
            delta = bt * (vt - pred)
            S = S + torch.einsum("bhk,bhv->bhkv", kt, delta)
            o = torch.einsum("bhk,bhkv->bhv", qt, S)
            state["S"] = S

        o = o.to(x.dtype).reshape(B, 1, H * Dh)
        gate = self.g_proj(x)
        o = self.out_norm(o, gate)
        return self.o_proj(o), state
