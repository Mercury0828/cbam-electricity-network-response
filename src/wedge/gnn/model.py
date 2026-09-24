"""Extended MG-STGNN (A49): mechanism-typed message passing with a balance-constrained dispatch head.

Re-implemented from the method section of Shen et al. (Applied Energy 426, 2026, 128683), since the original
checkpoints are not available here:
  * merit-order (MO) channel      node self-message from own state, carbon cost and CBAM cost
  * trade-arbitrage (TA) channel  GAT-style attention over neighbours with price differential and utilisation
  * carbon-cost (CC) channel      GAT-style attention over neighbours with the exporter's CBAM cost and the flow
  * concatenation aggregation, dilated causal TCN over the 24-hour window, dual heads.
Extension for this paper:
  * the dispatch head allocates the node's residual demand R = load - non-dispatchable generation - net import
    across the dispatchable classes D with softmax shares, g_k = s_k * R, so the energy balance holds by
    construction and a change in the border flow moves R one for one;
  * emissions follow as sum_k EF_k g_k with fixed class factors, so the marginal response to a flow change is
    sum_k EF_k d(s_k R)/dR, which differs from the average factor whenever the shares move with R;
  * the price head keeps the dual-target structure of the original model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, i, h, o, p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, o))

    def forward(self, x):
        return self.net(x)


class EdgeAttention(nn.Module):
    """GAT attention over incoming directed edges j -> i with edge-specific message inputs; messages are weighted by
    the dynamic adjacency A_ji (flow utilisation times a learned policy-conditioned gate, S0 Eq. dynamic_adj)."""

    def __init__(self, d, e_in, p):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        self.a = nn.Linear(2 * d, 1, bias=False)
        self.msg = MLP(d + e_in, d, d, p)

    def forward(self, h, src, dst, e, n, adj):
        # h: (B, N, d); src, dst: (E,); e: (B, E, e_in); adj: (B, E) dynamic adjacency weights
        wh = self.W(h)
        logit = F.leaky_relu(self.a(torch.cat([wh[:, dst], wh[:, src]], -1)).squeeze(-1), 0.2)   # (B, E)
        logit = logit - logit.max(dim=1, keepdim=True).values
        w = torch.exp(logit)
        den = torch.zeros(h.shape[0], n, device=h.device).index_add_(1, dst, w) + 1e-9
        alpha = w / den[:, dst]
        m = (alpha * adj).unsqueeze(-1) * self.msg(torch.cat([h[:, src], e], -1))                # (B, E, d)
        out = torch.zeros_like(h).index_add_(1, dst, m)
        return out, alpha


class MechLayer(nn.Module):
    def __init__(self, d, mo_in, ta_in, cc_in, p):
        super().__init__()
        self.mo = MLP(d + mo_in, d, d, p)
        self.ta = EdgeAttention(d, ta_in, p)
        self.cc = EdgeAttention(d, cc_in, p)
        self.gate = nn.Linear(2 * d + cc_in, 1)           # policy-conditioned gate of the dynamic adjacency
        self.agg = nn.Linear(3 * d, d)
        self.last_adj = None

    def forward(self, h, mo_x, ta_e, cc_e, src, dst, drop=()):
        n = h.shape[1]
        util = ta_e[..., 1]                                                          # |F| / capacity
        adj = util * torch.sigmoid(self.gate(torch.cat([h[:, dst], h[:, src], cc_e], -1)).squeeze(-1))
        self.last_adj = adj
        m_mo = self.mo(torch.cat([h, mo_x], -1))
        m_ta, a_ta = self.ta(h, src, dst, ta_e, n, adj)
        m_cc, a_cc = self.cc(h, src, dst, cc_e, n, adj)
        if "MO" in drop:
            m_mo = torch.zeros_like(m_mo)
        if "TA" in drop:
            m_ta = torch.zeros_like(m_ta)
        if "CC" in drop:
            m_cc = torch.zeros_like(m_cc)
        return F.relu(self.agg(torch.cat([m_mo, m_ta, m_cc], -1))) + h, (a_ta, a_cc)


class TCN(nn.Module):
    def __init__(self, d, dil=(1, 2, 4, 8), p=0.15):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(d, d, 2, dilation=k) for k in dil])
        self.dil = dil
        self.drop = nn.Dropout(p)

    def forward(self, x):                      # x: (B*N, d, W)
        for conv, k in zip(self.convs, self.dil):
            y = conv(F.pad(x, (k, 0)))
            x = x + self.drop(F.gelu(y))
        return x[:, :, -1]


class MGSTGNNX(nn.Module):
    def __init__(self, n_nodes, node_in, mo_in, ta_in, cc_in, n_disp, d=64, layers=2, p=0.15):
        super().__init__()
        self.inp = nn.Linear(node_in, d)
        self.emb = nn.Parameter(torch.randn(n_nodes, d) * 0.1)          # node identity
        self.layers = nn.ModuleList([MechLayer(d, mo_in, ta_in, cc_in, p) for _ in range(layers)])
        self.tcn = TCN(d, p=p)
        self.share = MLP(d + 2, d, n_disp, p)     # + current residual demand and its square (scaled)
        self.price = MLP(d, d, 1, p)
        self.default_drop = ()                    # channels removed in training (no-graph control), kept at inference

    def encode(self, x, mo_x, ta_e, cc_e, src, dst, drop=()):
        # x: (B, W, N, node_in); mo_x: (B, W, N, mo_in); ta_e/cc_e: (B, W, E, *)
        B, W, N, _ = x.shape
        hs = []
        l1 = 0.0
        for w in range(W):
            h = self.inp(x[:, w]) + self.emb
            for L in self.layers:
                h, _ = L(h, mo_x[:, w], ta_e[:, w], cc_e[:, w], src, dst, drop)
                l1 = l1 + L.last_adj.abs().mean()
            hs.append(h)
        self.adj_l1 = l1 / (W * len(self.layers))                          # sparsity penalty (S0 L_sparse)
        H = torch.stack(hs, 2)                                             # (B, N, W, d)
        z = self.tcn(H.reshape(B * N, W, -1).transpose(1, 2)).reshape(B, N, -1)
        return z

    def forward(self, x, mo_x, ta_e, cc_e, src, dst, resid, disp_mask, drop=()):
        """resid: (B, N) residual demand for dispatchable classes at the target hour, in GW.
        disp_mask: (N, n_disp) 1 where the class exists in the node. Returns gen (B, N, n_disp) in GW, price."""
        z = self.encode(x, mo_x, ta_e, cc_e, src, dst, tuple(drop) + tuple(self.default_drop))
        r = resid.unsqueeze(-1)
        logits = self.share(torch.cat([z, r, r * r], -1))
        logits = logits.masked_fill(disp_mask.unsqueeze(0) == 0, -1e4)
        s = torch.softmax(logits, -1)
        gen = s * F.relu(r)
        return gen, self.price(z).squeeze(-1), s
