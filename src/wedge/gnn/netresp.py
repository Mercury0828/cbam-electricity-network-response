"""Graph-structured network response model (A52): learned supply and transfer flexibilities with a Laplacian clearing
layer, trained on weather-driven residual-demand shocks and validated on interconnector outages.

Structure (linear response around the hour's operating point; all quantities are hour-to-hour changes):
  zone i:   dR_i  = k_i dp_i               k_i = 1 / (local supply slope) >= 0,  MW per EUR/MWh
            dX_i  = x_i dp_i               response of links outside the 17-zone graph (boundary), x_i >= 0
  link e:   dF_ab = g_e (dp_b - dp_a)      g_e >= 0; g_e -> 0 when the link is congested or unavailable
  balance:  dR_i + dX_i + dNI_i = dN_i,    dN_i = d(load_i - non-dispatchable_i)  (weather-driven shock)
  =>        (diag(k + x) + L_g) dp = dN    with L_g the weighted graph Laplacian; solved per hour.
Parameters (k, x, g) come from an encoder of the network state over the 24 hours before the shock:
  graph   the mechanism-guided graph network of v2 (merit-order, trade-arbitrage and carbon-cost messages)
  local   the same encoder with cross-border messages removed (node-local parameters, same clearing layer)
Loss: SmoothL1 on dR_i (scaled by zone), dF_e (scaled by link capacity) and dp_i (scaled, weight 0.5).
Training hours exclude every outage hour of the charged links, which are held out for validation.

Intervention on a charged link l = x -> m, delivered flow reduced by delta:
  g_l = 0 (the link no longer adjusts), dN_m += delta, dN_x -= delta / eta; solve; r_i = m_i dR_i / delta,
where m_i = dE_i / dR_i is the validated local emission response of the v2 dispatch head.

Usage:
  WEDGE_GNN_SPEC=v2 python src/wedge/gnn/netresp.py train --seed 42 [--local]
  WEDGE_GNN_SPEC=v2 python src/wedge/gnn/netresp.py validate        (outage events)
  WEDGE_GNN_SPEC=v2 python src/wedge/gnn/netresp.py charged         (CBAM borders, 2025-2026)
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import DSFX, SPLITS, Features, hour_of  # noqa: E402
from wedge.gnn.model import MLP, MGSTGNNX                     # noqa: E402
from wedge.gnn.outage_study import LINKS, spells              # noqa: E402

W = 24
import os
TAG = os.environ.get("NETRESP_TAG", "")                   # "" = first round; "r2" = second round (A52)
RUNS = pathlib.Path("data/processed/gnn/runs_netresp" + (f"_{TAG}" if TAG else "") + DSFX)
OUTD = pathlib.Path("data/processed/gnn/netresp" + (f"_{TAG}" if TAG else "") + DSFX)
LATE = os.environ.get("NETRESP_LATE", "") == "1"
# A60: horizon of the training shocks. 1 = hour-to-hour changes (r2); 24 = changes against the same hour of the
# previous day (r3), which removes the common daily cycle and keeps the zone-specific day-to-day weather shocks
LAG = int(os.environ.get("NETRESP_LAG", "1"))
FLOW_W = 10.0 if TAG else 1.0                            # r2: flow changes identify transfer flexibility/congestion
# r2: boundary (links outside the 17-zone graph) only where such links carry material flows
BOUNDARY = {"GB", "PL", "HU", "RS", "IT", "ES", "SE", "NO", "AT", "SK", "CH"}
if DSFX == "_v4":                                        # A53: GB-IE, AT-SI, RS/HU neighbours now inside the graph
    BOUNDARY = {"PL", "HU", "SK", "RS", "RO", "BG", "GR", "ME", "MK", "IT", "ES", "SE", "NO"}
ETA = {"GB->NL": 0.9794, "GB->BE": 0.9794, "RS->HU": 0.98345}


class NetResp(nn.Module):
    def __init__(self, fe, d=64, local=False):
        super().__init__()
        x0 = fe.sample(1000, W)
        self.enc = MGSTGNNX(fe.N, x0[0].shape[-1], x0[1].shape[-1], x0[2].shape[-1], x0[3].shape[-1],
                            fe.gen_disp.shape[1], d=d)
        if local:
            self.enc.default_drop = ("TA", "CC")
        self.k_head = MLP(d, d, 2, 0.1)                                          # k_i, x_i
        self.g_head = MLP(2 * d + x0[2].shape[-1], d, 1, 0.1)
        self.k_scale = nn.Parameter(torch.tensor(np.log(np.maximum(fe.rs[:, 0], 0.05) * 1000 / 10.0),
                                                 dtype=torch.float32))          # MW per EUR/MWh, per zone
        self.register_buffer("cap", torch.tensor(fe.caps))
        self.register_buffer("bmask", torch.tensor([1.0 if (not TAG or n in BOUNDARY) else 0.0 for n in fe.nodes]))
        rev = {(a, b): e for e, (a, b) in enumerate(zip(fe.src, fe.dst))}
        self.register_buffer("rev", torch.tensor([rev[(b, a)] for a, b in zip(fe.src, fe.dst)]))

    def params(self, x, mo, ta, cc, src, dst, avail=None):
        z = self.enc.encode(x, mo, ta, cc, src, dst, self.enc.default_drop)      # (B, N, d)
        kx = F.softplus(self.k_head(z)) * torch.exp(self.k_scale)[None, :, None]
        g = F.softplus(self.g_head(torch.cat([z[:, src], z[:, dst], ta[:, -1]], -1)).squeeze(-1)) * self.cap / 10.0
        g = 0.5 * (g + g[:, self.rev])                                           # one value per undirected link
        if avail is not None:                                                    # A56: links not yet in service
            g = g * avail
        return kx[..., 0], kx[..., 1] * self.bmask, g                            # (B,N), (B,N), (B,E)


def solve(k, xb, g, src, dst, dN, N):
    """(diag(k + x) + L_g) dp = dN; g on directed edges (both directions present) -> use the mean of the pair."""
    B, E = g.shape
    L = torch.zeros(B, N, N, device=g.device)
    w = 0.5 * g                                                                  # each undirected link appears twice
    L.index_put_((torch.arange(B)[:, None].expand(B, E), src[None].expand(B, E), dst[None].expand(B, E)), -w,
                 accumulate=True)
    L.index_put_((torch.arange(B)[:, None].expand(B, E), dst[None].expand(B, E), src[None].expand(B, E)), -w,
                 accumulate=True)
    diag = -L.sum(-1)
    A = L + torch.diag_embed(diag + k + xb + 1e-3)
    dp = torch.linalg.solve(A, dN.unsqueeze(-1)).squeeze(-1)
    dR = k * dp
    dF = g * (dp[:, dst] - dp[:, src])                                          # directed src -> dst
    return dp, dR, dF


class Data:
    def __init__(self, fe):
        self.fe = fe
        d = np.load(f"data/processed/gnn/dataset{DSFX}.npz")
        ci = {c: i for i, c in enumerate(fe.meta["classes"])}
        gen, load = d["gen"], d["load"]
        nd = [ci[c] for c in ["nuclear", "biomass", "waste", "hydro_ror", "wind", "solar", "other"]]
        self.N_ = np.nan_to_num(load) - np.nansum(np.nan_to_num(gen[:, nd]), 1)          # MW
        self.load_ok = np.isfinite(load) & (np.nan_to_num(load) > 0)
        self.nd_ok = np.isfinite(gen[:, nd]).any(1)                                        # A56 (B2)
        self.R = fe.resid * 1000.0
        Fsd = fe.F[fe.src, fe.dst]
        self.Fe = np.nan_to_num(Fsd)
        self.Fok = d["flow_ok"][fe.src, fe.dst] if "flow_ok" in d.files else np.isfinite(Fsd)   # A56 (B6)
        # availability: a link exists from the first hour with |flow| > 1 MW (commissioning); before that g = 0
        first = np.array([np.argmax(np.abs(self.Fe[e]) > 1.0) if (np.abs(self.Fe[e]) > 1.0).any() else fe.H
                          for e in range(len(fe.src))])
        self.avail = (np.arange(fe.H)[None, :] >= first[:, None])                           # (E, H)
        self.P, self.Pok = np.nan_to_num(fe.price), np.isfinite(fe.price)
        H = fe.H
        dR = np.full_like(self.R, np.nan)
        dR[:, LAG:] = self.R[:, LAG:] - self.R[:, :-LAG]
        self.sR = np.nan_to_num(np.nanstd(dR[:, :hour_of(SPLITS["train"][1])], 1), nan=0.0) + 1.0
        Pn = np.where(self.Pok, self.P, np.nan)
        dP = np.full_like(Pn, np.nan)
        dP[:, LAG:] = Pn[:, LAG:] - Pn[:, :-LAG]
        self.sP = np.nan_to_num(np.nanstd(dP[:, :hour_of(SPLITS["train"][1])], 1), nan=0.0) + 1.0    # A55: zones without prices
        idx = {n: i for i, n in enumerate(fe.nodes)}
        self.outage = np.zeros(H, bool)
        self.events = {}
        # A56 (B4): identical event definition to outage_v2 (open spells, commissioning, merge < 72 h), with the
        # same -48 h / +24 h envelope, so no evaluation hour can enter training
        from wedge.gnn.outage_study import COMMISSIONED
        from wedge.gnn.outage_v2 import merge as merge_v2, spells_open
        fok = d["flow_ok"] if "flow_ok" in d.files else np.isfinite(fe.F)
        for b, (x, m) in LINKS.items():
            Fl = fe.F[idx[x], idx[m]]
            z = fok[idx[x], idx[m]] & np.isfinite(Fl) & (np.abs(Fl) < 1.0)
            if b in COMMISSIONED:
                z[:hour_of(COMMISSIONED[b])] = False
            ev = merge_v2(spells_open(z))
            self.events[b] = ev
            for s, e in ev:
                self.outage[max(s - 48, 0):min(e + 24, H)] = True
        self.all_ok = np.zeros(H, bool)                                                    # A56 (B2, B3)
        self.all_ok[1:] = (self.ok_hour_vec(np.arange(1, H))).all(0)
        # A60: training and validation hours also need every zone observed at t - LAG
        self.lag_ok = np.zeros(H, bool)
        self.lag_ok[LAG:] = (self.fe.disp_obs_ok[:, :-LAG] & self.load_ok[:, :-LAG] & self.nd_ok[:, :-LAG]).all(0)

    def ok_hour(self, t):
        return (self.fe.disp_obs_ok[:, t] & self.fe.disp_obs_ok[:, t - 1] & self.load_ok[:, t]
                & self.load_ok[:, t - 1] & self.nd_ok[:, t] & self.nd_ok[:, t - 1])

    def ok_hour_vec(self, ts):
        ts = np.asarray(ts)
        return (self.fe.disp_obs_ok[:, ts] & self.fe.disp_obs_ok[:, ts - 1] & self.load_ok[:, ts]
                & self.load_ok[:, ts - 1] & self.nd_ok[:, ts] & self.nd_ok[:, ts - 1])

    def batch(self, ts, dev):
        fe = self.fe
        xs, mos, tas, ccs = zip(*[fe.sample(int(t) - 1, W) for t in ts])          # state up to t-1
        to = lambda a: torch.tensor(np.stack(a), device=dev)                    # noqa: E731
        ts = np.asarray(ts)
        okn = np.stack([self.ok_hour(t) for t in ts])                            # (B, N)
        if LAG != 1:
            okn = okn & (self.fe.disp_obs_ok[:, ts - LAG] & self.load_ok[:, ts - LAG] & self.nd_ok[:, ts - LAG]).T
        dN = (self.N_[:, ts] - self.N_[:, ts - LAG]).T * okn
        dR = (self.R[:, ts] - self.R[:, ts - LAG]).T
        dF = (self.Fe[:, ts] - self.Fe[:, ts - LAG]).T
        okF = (self.Fok[:, ts] & self.Fok[:, ts - LAG]).T
        dP = (self.P[:, ts] - self.P[:, ts - LAG]).T
        okP = (self.Pok[:, ts] & self.Pok[:, ts - LAG]).T
        t_ = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev)       # noqa: E731
        av = self.avail[:, ts].T                                                  # (B, E)
        okF = okF & av
        return (to(xs), to(mos), to(tas), to(ccs), t_(dN), t_(dR), t_(okn), t_(dF), t_(okF), t_(dP), t_(okP),
                t_(av))


def loss_fn(model, dat, b, src, dst, dev):
    x, mo, ta, cc, dN, dR, okn, dF, okF, dP, okP, av = b
    k, xb, g = model.params(x, mo, ta, cc, src, dst, av)
    dp, pR, pF = solve(k, xb, g, src, dst, dN, dat.fe.N)
    sR = torch.tensor(dat.sR, device=dev, dtype=torch.float32)
    sP = torch.tensor(dat.sP, device=dev, dtype=torch.float32)
    cap = model.cap
    lR = (F.smooth_l1_loss(pR / sR, dR / sR, reduction="none") * okn).sum() / okn.sum().clamp(min=1)
    lF = (F.smooth_l1_loss(pF / cap, dF / cap, reduction="none") * okF).sum() / okF.sum().clamp(min=1)
    lP = (F.smooth_l1_loss(dp / sP, dP / sP, reduction="none") * okP).sum() / okP.sum().clamp(min=1)
    return lR + FLOW_W * lF + 0.5 * lP, lR.item(), lF.item(), lP.item()


def train(a):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"
    fe = Features()
    dat = Data(fe)
    model = NetResp(fe, local=a.local).to(dev)
    src, dst = torch.tensor(fe.src, device=dev), torch.tensor(fe.dst, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    tr_h = np.arange(max(W + 2, LAG + 1), hour_of(SPLITS["train"][1]))
    tr_h = tr_h[~dat.outage[tr_h] & dat.all_ok[tr_h] & dat.lag_ok[tr_h]]
    va_h = np.arange(*(hour_of(v) for v in SPLITS["val"]))
    va_h = np.random.default_rng(0).choice(va_h[~dat.outage[va_h] & dat.all_ok[va_h] & dat.lag_ok[va_h]], 1024,
                                           replace=False)
    print(f"training hours {len(tr_h)} (all zones observed, outage envelopes excluded, lag {LAG})", flush=True)
    tag = f"seed{a.seed}" + ("_local" if a.local else "")
    RUNS.mkdir(parents=True, exist_ok=True)
    best, log = 1e9, []
    for ep in range(a.epochs):
        model.train()
        t0, tl = time.time(), []
        for _ in range(a.steps):
            l, *parts = loss_fn(model, dat, dat.batch(rng.choice(tr_h, a.bs), dev), src, dst, dev)
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(parts)
        model.eval()
        with torch.no_grad():
            vl = [loss_fn(model, dat, dat.batch(va_h[i:i + 128], dev), src, dst, dev)[1:]
                  for i in range(0, len(va_h), 128)]
        v = np.mean(vl, 0)
        score = float(v[0] + FLOW_W * v[1] + 0.5 * v[2])
        log.append(dict(ep=ep, train=np.mean(tl, 0).tolist(), val=v.tolist(), sec=time.time() - t0))
        print(f"{tag} ep {ep:2d} train {np.mean(tl, 0).round(4)} val R/F/P {v.round(4)} ({time.time() - t0:.0f}s)",
              flush=True)
        if score < best:
            best = score
            torch.save(model.state_dict(), RUNS / f"{tag}.pt")
    (RUNS / f"{tag}.json").write_text(json.dumps(dict(best=best, log=log), indent=1), encoding="utf-8")


def load_models(fe, dev, local):
    ms = []
    for f in sorted(glob.glob(str(RUNS / "seed*.pt"))):
        if ("_local" in f) != local:
            continue
        m = NetResp(fe, local=local).to(dev)
        m.load_state_dict(torch.load(f, map_location=dev, weights_only=True))
        m.eval()
        ms.append(m)
    if not ms:
        raise FileNotFoundError(f"no {'local' if local else 'graph'} network-response checkpoints in {RUNS}")
    return ms


def intervene(models, dat, hours, xi, mi, eta, L, dev, return_ni=False):
    """Response of dR (MW) in every zone to removing the flow L_t on the link x -> m (g_l = 0). L_t > 0: L MW
    delivered to m (x sends L/eta); L_t < 0: |L| MW delivered to x (m sends |L|/eta). Returns the seed-mean of
    dR / L, shape (T, N), and with return_ni also the internal-link net-import change dNI / L (boundary excluded)."""
    fe = dat.fe
    src, dst = torch.tensor(fe.src, device=dev), torch.tensor(fe.dst, device=dev)
    lmask = torch.ones(len(fe.src), device=dev)
    for e, (s_, d_) in enumerate(zip(fe.src, fe.dst)):
        if {s_, d_} == {xi, mi}:
            lmask[e] = 0.0
    into = torch.zeros(len(fe.src), fe.N, device=dev)
    into[torch.arange(len(fe.src)), dst] = 1.0                                  # link e delivers into dst[e]
    out, outni = [], []
    for m in models:
        rs, ns = [], []
        with torch.no_grad():
            for i in range(0, len(hours), 128):
                hs = hours[i:i + 128]
                b = dat.batch(hs, dev)
                k, xb, g = m.params(*b[:4], src, dst, b[11])
                g = g * lmask
                dN = torch.zeros(len(hs), fe.N, device=dev)
                Lt = torch.tensor(np.asarray(L[i:i + 128], np.float32), device=dev)
                pos = Lt > 0
                dN[:, mi] += torch.where(pos, Lt, Lt / eta)                      # A56 (B1): losses on the sender
                dN[:, xi] += torch.where(pos, -Lt / eta, -Lt)
                _, dR, dF = solve(k, xb, g, src, dst, dN, fe.N)
                rs.append((dR / Lt[:, None]).cpu().numpy())
                ns.append(((dF @ into) / Lt[:, None]).cpu().numpy())
        out.append(np.concatenate(rs))
        outni.append(np.concatenate(ns))
    if return_ni:
        return np.mean(out, 0), np.mean(outni, 0)
    return np.mean(out, 0)


def local_emission_slopes(fe, hours, zones, dev):
    """v2 validated local responses dE_i / dR_i (t per MWh) for the given zones and hours (cached on disk)."""
    import hashlib
    zones = list(zones)
    from wedge.gnn.features import RUNS as GNN_RUNS, SPEC
    fp = "".join(f"{f.name}:{f.stat().st_size}:{int(f.stat().st_mtime)}" for f in sorted(GNN_RUNS.glob("seed*.pt"))
                 if "_drop" not in f.name)
    key = hashlib.md5(np.asarray(hours, np.int64).tobytes() + bytes(zones) + (fp + SPEC + DSFX).encode()
                      ).hexdigest()[:16]
    cache = pathlib.Path("data/processed/gnn/netresp_cache" + DSFX) / f"mslope_{key}.npy"
    if cache.exists():
        return np.load(cache)
    out = _slopes(fe, hours, zones, dev)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def _slopes(fe, hours, zones, dev):
    from wedge.gnn.marginal import load_models as load_v2
    from wedge.gnn.response_check import graph_slopes
    v2 = load_v2(fe, dev, "")
    out = np.zeros((len(hours), fe.N), np.float32)
    for n in zones:
        out[:, n] = graph_slopes(v2, fe, n, hours, dev)
    return out


def validate(a):
    """Compare model-implied outage responses with the event-study estimates (outage_study.py) on held-out events."""
    from wedge.gnn.outage_study import oof_predict
    dev = "cuda"
    fe = Features()
    dat = Data(fe)
    idx = {n: i for i, n in enumerate(fe.nodes)}
    ev_res = json.loads(pathlib.Path(f"data/processed/gnn/outage_study{DSFX}.json").read_text(encoding="utf-8"))
    d = np.load(f"data/processed/gnn/dataset{DSFX}.npz")
    ci = {c: i for i, c in enumerate(fe.meta["classes"])}
    gen, load = np.nan_to_num(d["gen"]), np.nan_to_num(d["load"])
    t = np.arange(fe.H)
    X = np.column_stack([a_.T for a_ in [load, gen[:, ci["wind"]], gen[:, ci["solar"]], fe.z_nd]]
                        + [np.nan_to_num(d["glob"]), np.sin(2 * np.pi * (t % 24) / 24),
                           np.cos(2 * np.pi * (t % 24) / 24), (t // 24) % 7, np.sin(2 * np.pi * t / 8766),
                           np.cos(2 * np.pi * t / 8766), t / 8766.0])
    res = {}
    for b in ("GB->NL", "GB->BE"):
        x, m = LINKS[b]
        xi, mi = idx[x], idx[m]
        Fl = fe.F[xi, mi]
        fin = np.isfinite(Fl)
        zero = fin & (np.abs(Fl) < 1.0)
        base_ok = fin & ~zero & (load[xi] > 0) & (load[mi] > 0)
        Fhat = oof_predict(X, np.nan_to_num(Fl), base_ok, t // 168)
        evs = dat.events[b]
        if LATE:                                  # held-out events for the event-calibrated models
            evs = [(s, e) for s, e in evs if s >= hour_of("2023-01-01")]
        oh = np.concatenate([np.arange(s, e) for s, e in evs])
        oh = oh[(oh > W + 2) & np.array([dat.ok_hour(h)[[xi, mi]].all() for h in oh])]
        L = Fhat[oh]
        sel = np.abs(L) > 50
        oh, L = oh[sel], L[sel]
        mslope = local_emission_slopes(fe, oh, range(fe.N), dev)
        res[b] = {"hours": int(len(oh))}
        for name, local in (("graph", False), ("local", True)):
            ms = load_models(fe, dev, local)
            if not ms:
                continue
            rho = intervene(ms, dat, oh, xi, mi, ETA[b], L, dev)                 # dR / L, (T, N)
            w = np.abs(L) / np.abs(L).sum()
            beta_R = {fe.nodes[j]: float(w @ rho[:, j]) for j in range(fe.N)}
            beta_E = {fe.nodes[j]: float(w @ (rho[:, j] * mslope[:, j])) for j in range(fe.N)}
            res[b][name] = {"R": beta_R, "E": beta_E, "E_network": float(sum(beta_E.values()))}
        # one-for-one benchmark (v2 counterfactual)
        res[b]["one_for_one"] = {"R": {m: 1.0, x: -1.0 / ETA[b]},
                                 "E": {m: float(np.mean(mslope[:, mi])), x: -float(np.mean(mslope[:, xi])) / ETA[b]}}
        emp = ev_res[b]
        key, sek = ("event_fe", "se_event_fe") if "event_fe" in emp["R_m"] else ("outage", "se")
        res[b]["empirical"] = {"R": {n: emp[f"R_{n}"][key] for n in fe.nodes if f"R_{n}" in emp},
                               "R_se": {n: emp[f"R_{n}"][sek] for n in fe.nodes if f"R_{n}" in emp},
                               "E": {n: emp[f"E_{n}"][key] for n in fe.nodes if f"E_{n}" in emp},
                               "E_se": {n: emp[f"E_{n}"][sek] for n in fe.nodes if f"E_{n}" in emp},
                               "estimator": key}
        # z-scores of the model deviations and the share of zones inside the 95% interval
        for name in ("graph", "local"):
            if name in res[b]:
                for q in ("R", "E"):
                    em, se, pr = res[b]["empirical"][q], res[b]["empirical"][q + "_se"], res[b][name][q]
                    z = [(pr.get(n, 0.0) - em[n]) / max(se[n], 1e-3) for n in em if np.isfinite(em[n])]
                    res[b][name][f"within95_{q}"] = float(np.mean(np.abs(z) < 1.96))
        for name in ("graph", "local", "one_for_one"):
            if name not in res[b]:
                continue
            for q in ("R", "E"):
                em = res[b]["empirical"][q]
                pr = res[b][name][q]
                err = [pr.get(n, 0.0) - em[n] for n in em]
                res[b][name][f"rmse_{q}"] = float(np.sqrt(np.mean(np.square(err))))
            print(b, name, "rmse R %.3f E %.3f" % (res[b][name]["rmse_R"], res[b][name]["rmse_E"]),
                  {n: round(res[b][name]["R"].get(n, 0), 2) for n in ("GB", "NL", "BE", "DE", "FR", "NO", "DK")},
                  flush=True)
        print(b, "empirical", {n: round(res[b]["empirical"]["R"][n], 2) for n in ("GB", "NL", "BE", "DE", "FR", "NO", "DK")},
              flush=True)
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / ("validate_outages_late.json" if LATE else "validate_outages.json")).write_text(json.dumps(res, indent=1),
                                                                                             encoding="utf-8")


def charged(a):
    """Network marginal responses on the charged borders: r_i per delivered MWh removed, with draws over seeds."""
    dev = "cuda"
    fe = Features()
    dat = Data(fe)
    idx = {n: i for i, n in enumerate(fe.nodes)}
    OUTD.mkdir(parents=True, exist_ok=True)
    for b, (x, m) in LINKS.items():
        xi, mi = idx[x], idx[m]
        hours = []
        for p in ("y2025", "charged", "charged_q3"):
            s, e = (hour_of(v) for v in SPLITS[p])
            hours += [h for h in range(s, e) if fe.F[xi, mi, h] > 100 and dat.all_ok[h]]    # A56 (B3)
        hours = np.array(hours)
        mslope = local_emission_slopes(fe, hours, range(fe.N), dev)
        L = np.full(len(hours), a.delta)
        for name, local in (("graph", False), ("local", True)):
            try:
                ms = load_models(fe, dev, local)
            except FileNotFoundError:                                              # A60: r3 has no node-local models
                if name == "graph":
                    raise
                continue
            draws = np.stack([intervene([mm], dat, hours, xi, mi, ETA[b], L, dev) for mm in ms])     # (S, T, N)
            r = draws * mslope[None]
            np.savez_compressed(OUTD / f"charged_{b.replace('->', '_')}_{name}.npz", hours=hours, rho=draws, r=r,
                                mslope=mslope, flow=fe.F[xi, mi, hours])
            w = fe.F[xi, mi, hours]
            mr = np.median(r, 0)
            mrho = np.median(draws, 0)
            wm = lambda v: float(v @ w / w.sum())                                   # noqa: E731
            print(b, name, "absorb m %.2f x %.2f | E_x %.3f E_m %.3f third %.3f network %.3f" % (
                wm(mrho[:, mi]), wm(-mrho[:, xi]), wm(-mr[:, xi]), wm(mr[:, mi]),
                wm(mr.sum(1) - mr[:, xi] - mr[:, mi]), wm(mr.sum(1))),
                {fe.nodes[j]: round(wm(mr[:, j]), 3) for j in range(fe.N) if abs(wm(mr[:, j])) > 0.01}, flush=True)


def finetune(a):
    """Event-calibrated fine-tuning (A52): starting from a trained model, match the within-event responses of
    dispatchable output in every zone to the lost flow on EARLY outage events (start before 2023) of the GB links,
    keeping the weather-shock loss as a regulariser. Late events (2023 onwards) stay held out for validation.
    Loss on event hours: for zone i, residual u_it = rho_i(t) L_t - dY_i(t), demeaned within event (event fixed
    effects, as in the event study), scaled by the zone's hourly-change scale."""
    dev = "cuda"
    fe = Features()
    dat = Data(fe)
    idx = {n: i for i, n in enumerate(fe.nodes)}
    src, dst = torch.tensor(fe.src, device=dev), torch.tensor(fe.dst, device=dev)
    cut = hour_of("2023-01-01")
    ev_data = []
    for b in ("GB->NL", "GB->BE"):
        x, m = LINKS[b]
        z = np.load(f"data/processed/gnn/outage_{b.replace('->', '_')}{DSFX}.npz")
        hrs, L, ev = z["hours"], z["Fhat"], z["evid"]
        start = {e: hrs[ev == e].min() for e in np.unique(ev)}
        keep = np.array([start[e] < cut for e in ev]) & (np.abs(L) > 50) & (hrs > W + 2)
        keep &= np.array([dat.ok_hour(h)[[idx[x], idx[m]]].all() for h in hrs])
        Y = np.stack([np.nan_to_num(z[f"R_{n}"]) if f"R_{n}" in z.files else np.zeros(len(hrs)) for n in fe.nodes], 1)
        okY = np.stack([np.isfinite(z[f"R_{n}"]) if f"R_{n}" in z.files else np.zeros(len(hrs), bool) for n in fe.nodes], 1)
        ev_data.append((b, idx[x], idx[m], hrs[keep], L[keep], ev[keep], Y[keep], okY[keep]))
        print(b, "early event hours", int(keep.sum()), "events", len(np.unique(ev[keep])), flush=True)
    variant = "_local" if a.local else ""
    src_runs = pathlib.Path("data/processed/gnn/runs_netresp_r2")
    out_runs = pathlib.Path("data/processed/gnn/runs_netresp_r2ft")
    out_runs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    tr_h = np.arange(W + 2, hour_of(SPLITS["train"][1]))
    tr_h = tr_h[~dat.outage[tr_h]]
    sR = torch.tensor(dat.sR, device=dev, dtype=torch.float32)
    model = NetResp(fe, local=a.local).to(dev)
    model.load_state_dict(torch.load(src_runs / f"seed{a.seed}{variant}.pt", map_location=dev, weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for step in range(a.steps):
        model.train()
        loss_ev = 0.0
        for b, xi, mi, hrs, L, ev, Y, okY in ev_data:
            sel = rng.choice(len(hrs), min(96, len(hrs)), replace=False)
            hs, Ls, evs, Ys, oks = hrs[sel], L[sel], ev[sel], Y[sel], okY[sel]
            bt = dat.batch(hs, dev)
            k, xb, g = model.params(*bt[:4], src, dst)
            lm = torch.ones(len(fe.src), device=dev)
            for e, (s_, d_) in enumerate(zip(fe.src, fe.dst)):
                if {s_, d_} == {xi, mi}:
                    lm[e] = 0.0
            dN = torch.zeros(len(hs), fe.N, device=dev)
            Lt = torch.tensor(Ls.astype(np.float32), device=dev)
            dN[:, mi] += Lt
            dN[:, xi] -= Lt / ETA[b]
            _, dR, _ = solve(k, xb, g * lm, src, dst, dN, fe.N)
            u = dR - torch.tensor(Ys.astype(np.float32), device=dev)
            okt = torch.tensor(oks.astype(np.float32), device=dev)
            evt = torch.tensor(evs, device=dev)
            for e in torch.unique(evt):
                s_ = evt == e
                if s_.sum() > 1:
                    u[s_] = u[s_] - (u[s_] * okt[s_]).sum(0) / okt[s_].sum(0).clamp(min=1)
            loss_ev = loss_ev + ((u / sR) ** 2 * okt).sum() / okt.sum().clamp(min=1)
        lw, *_ = loss_fn(model, dat, dat.batch(rng.choice(tr_h, 64), dev), src, dst, dev)
        loss = loss_ev + a.reg * lw
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            print(f"seed{a.seed}{variant} step {step} event {float(loss_ev):.4f} weather {float(lw):.4f}", flush=True)
    torch.save(model.state_dict(), out_runs / f"seed{a.seed}{variant}.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "validate", "charged", "finetune"])
    ap.add_argument("--reg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--delta", type=float, default=100.0)
    a = ap.parse_args()
    {"train": train, "validate": validate, "charged": charged, "finetune": finetune}[a.cmd](a)
