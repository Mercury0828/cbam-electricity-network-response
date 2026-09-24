"""Network-redistribution extension of the graph model (A51, specification v3).

Why. The flow counterfactual of v2 assigns the whole change in delivered import to the importer's own residual
(R_m += delta) and holds every other interconnector at its observed flow. Instrumental-variable estimates
(redistribution_iv.py) show that importers absorb far less than one for one (GB->NL about 0.09, GB->BE 0.42,
RS->HU about 0), because a coupled market passes the change on through other interconnectors. v3 lets the graph learn
that redistribution.

Model. As v2, plus
  * the target hour's flows and residuals are not inputs: at the last window step the node residual and net import
    carry their previous-hour values, and every link carries its previous-hour flow unless it is revealed;
  * a reveal flag on every link: in training a random subset of links (share p ~ U(0, 0.3) per sample) shows its
    actual target-hour flow, and the model predicts the others;
  * an edge flow head predicts every link's target-hour flow from the two node embeddings and the link features.
Counterfactual for border x -> m at hour t: reveal the link with its observed flow (base) and with the flow reduced by
delta (perturbed); all other links are predicted in both passes. With g the antisymmetrised predicted link flow,
  dNI_i = sum over links into i of dg,   plus  -delta at m and +delta/eta at x on the clamped link,
  R_i'  = R_i - dNI_i                    (balance: the zone's dispatchable plants supply what imports no longer do)
and the dispatch head reallocates R_i' in every zone. Outputs per delivered MWh removed: E_x (exporter zone),
E_m (importer zone), h_j (each third zone), the absorption share dR_m/delta, and the network total.
The no-graph control (drop TA, CC) cannot pass the clamp to other links, so it reverts to one-for-one absorption.

Usage:
  WEDGE_GNN_SPEC=v3 python src/wedge/gnn/v3.py train --seed 42 [--drop TA,CC]
  WEDGE_GNN_SPEC=v3 python src/wedge/gnn/v3.py marginal [--variant dropTACC] [--delta 100]
  WEDGE_GNN_SPEC=v3 python src/wedge/gnn/v3.py evaluate
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

from wedge.gnn.features import RUNS, SPEC, SPLITS, Features, hour_of      # noqa: E402
from wedge.gnn.model import MLP, MGSTGNNX                                 # noqa: E402

assert SPEC == "v3", "run with WEDGE_GNN_SPEC=v3"
W = 24
OUTD = pathlib.Path("data/processed/gnn/v3")
BORDERS = {"GB->NL": ("GB", "NL", 0.9794), "GB->BE": ("GB", "BE", 0.9794), "RS->HU": ("RS", "HU", 0.98345)}


class Net(MGSTGNNX):
    def __init__(self, n_nodes, node_in, mo_in, ta_in, cc_in, n_disp, d=64, layers=2, p=0.15):
        super().__init__(n_nodes, node_in, mo_in, ta_in, cc_in, n_disp, d, layers, p)
        self.flow = MLP(2 * d + ta_in + cc_in, d, 1, p)

    def forward(self, x, mo_x, ta_e, cc_e, src, dst, resid, disp_mask, drop=()):
        z = self.encode(x, mo_x, ta_e, cc_e, src, dst, tuple(drop) + tuple(self.default_drop))
        r = resid.unsqueeze(-1)
        logits = self.share(torch.cat([z, r, r * r], -1))
        logits = logits.masked_fill(disp_mask.unsqueeze(0) == 0, -1e4)
        s = torch.softmax(logits, -1)
        gen = s * F.relu(r)
        fl = self.flow(torch.cat([z[:, src], z[:, dst], ta_e[:, -1], cc_e[:, -1]], -1)).squeeze(-1)   # (B, E)
        return gen, self.price(z).squeeze(-1), s, fl


class Data:
    """v3 inputs: previous-hour residual, net import and flows at the target step, plus a reveal flag."""

    def __init__(self, fe):
        self.fe = fe
        E = len(fe.src)
        self.E = E
        self.f_all = (fe.F[fe.src, fe.dst] / fe.caps[:, None]).astype(np.float32)          # (E, H)
        self.f_ok = np.isfinite(fe.F[fe.src, fe.dst])
        self.f_all = np.nan_to_num(self.f_all)
        rev = {(a, b): e for e, (a, b) in enumerate(zip(fe.src, fe.dst))}
        self.rev = np.array([rev[(b, a)] for a, b in zip(fe.src, fe.dst)])               # reverse edge index
        self.und = np.array([min(e, self.rev[e]) for e in range(E)])                      # undirected id

    def sample(self, t, reveal):
        """reveal: bool (E,) with both directions of a link set alike. Returns X, MO, TA, CC for window ending t."""
        fe = self.fe
        X, MO, TA, CC = (a.copy() for a in fe.sample(t, W))
        X[-1, :, 3] = fe.z_resid[:, t - 1]
        X[-1, :, 4] = fe.z_ni[:, t - 1]
        f = np.where(reveal, self.f_all[:, t], self.f_all[:, t - 1])
        TA[-1, :, 1], TA[-1, :, 2] = np.abs(f), f
        CC[-1, :, 2] = np.clip(f, 0, None)
        flag = np.ones((W, self.E, 1), np.float32)
        flag[-1, :, 0] = reveal.astype(np.float32)
        return X, MO, np.concatenate([TA, flag], -1), np.concatenate([CC, flag], -1)

    def random_reveal(self, rng):
        p = rng.uniform(0, 0.3)
        u = rng.random(self.E) < p
        return u[self.und]


def build(fe, dev, variant=""):
    dat = Data(fe)
    x0 = dat.sample(1000, np.zeros(dat.E, bool))
    m = Net(fe.N, x0[0].shape[-1], x0[1].shape[-1], x0[2].shape[-1], x0[3].shape[-1], fe.gen_disp.shape[1]).to(dev)
    if variant.startswith("drop"):
        v = variant[4:]
        m.default_drop = tuple(v[i:i + 2] for i in range(0, len(v), 2))
    return m, dat


def batch(dat, ts, reveals, dev):
    fe = dat.fe
    xs, mos, tas, ccs = zip(*[dat.sample(int(t), rv) for t, rv in zip(ts, reveals)])
    to = lambda a: torch.tensor(np.stack(a), device=dev)                          # noqa: E731
    ts = np.array(ts)
    return (to(xs), to(mos), to(tas), to(ccs),
            torch.tensor(fe.gen_disp[:, :, ts].transpose(2, 0, 1), device=dev),
            torch.tensor(fe.resid[:, ts].T, device=dev),
            torch.tensor(fe.z_price[:, ts].T, device=dev),
            torch.tensor(fe.disp_obs_ok[:, ts].T, device=dev).float(),
            torch.tensor(fe.price_ok[:, ts].T, device=dev).float(),
            torch.tensor(dat.f_all[:, ts].T, device=dev),
            torch.tensor((dat.f_ok[:, ts] & ~np.stack(reveals, 1)).T, device=dev).float())


def corr(a, b, ok):
    a, b = a[ok > 0], b[ok > 0]
    if a.numel() < 3:
        return a.sum() * 0
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-9)


def loss_fn(model, b, src, dst, mask, scale, ef):
    x, mo, ta, cc, yg, r, yp, okg, okp, yf, okf = b
    gen, pr, _, fl = model(x, mo, ta, cc, src, dst, r, mask)
    lg = (F.smooth_l1_loss(gen / scale, yg / scale, reduction="none").sum(-1) * okg).sum() / okg.sum().clamp(min=1)
    lp = (F.smooth_l1_loss(pr, yp, reduction="none") * okp).sum() / okp.sum().clamp(min=1)
    lf = (F.smooth_l1_loss(fl, yf, reduction="none") * okf).sum() / okf.sum().clamp(min=1)
    ci_p = (gen * ef).sum(-1) / r.clamp(min=1e-3)
    ci_o = (yg * ef).sum(-1) / r.clamp(min=1e-3)
    lc = (corr(yp, ci_o, okg * okp) - corr(pr, ci_p, okg * okp)).abs()
    return lg + 1.5 * lp + 1.0 * lf + 0.1 * lc + 1e-3 * model.adj_l1, lg.item(), lp.item(), lf.item()


def train(a):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"
    fe = Features()
    variant = f"drop{''.join(a.drop.split(','))}" if a.drop else ""
    model, dat = build(fe, dev, variant)
    src, dst = torch.tensor(fe.src, device=dev), torch.tensor(fe.dst, device=dev)
    mask = torch.tensor(fe.disp_mask, device=dev)
    scale = torch.tensor(np.maximum(fe.rs.squeeze(1), 0.05), device=dev).view(1, -1, 1)
    ef = torch.tensor(fe.ef_disp, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rg = {k: (max(hour_of(s), W + 1), hour_of(e)) for k, (s, e) in SPLITS.items()}
    vr = np.random.default_rng(0)
    vt = vr.choice(np.arange(*rg["val"]), 1024, replace=False)
    vrev = [dat.random_reveal(vr) for _ in vt]
    tag = f"seed{a.seed}" + (f"_{variant}" if variant else "")
    RUNS.mkdir(parents=True, exist_ok=True)
    best, log = 1e9, []
    for ep in range(a.epochs):
        model.train()
        t0, tl = time.time(), []
        for _ in range(a.steps):
            ts = rng.integers(*rg["train"], size=a.bs)
            l, lg, lp, lf = loss_fn(model, batch(dat, ts, [dat.random_reveal(rng) for _ in ts], dev), src, dst, mask,
                                    scale, ef)
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append((lg, lp, lf))
        model.eval()
        with torch.no_grad():
            vl = [loss_fn(model, batch(dat, vt[i:i + 128], vrev[i:i + 128], dev), src, dst, mask, scale, ef)[1:]
                  for i in range(0, len(vt), 128)]
        vg, vp, vf = (float(np.mean([v[k] for v in vl])) for k in range(3))
        score = vg + 1.5 * vp + vf
        log.append(dict(ep=ep, train=np.mean(tl, 0).tolist(), val_gen=vg, val_price=vp, val_flow=vf,
                        sec=time.time() - t0))
        print(f"{tag} ep {ep:2d} train {np.mean(tl, 0).round(4)} val gen {vg:.4f} price {vp:.4f} flow {vf:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if score < best:
            best = score
            torch.save(model.state_dict(), RUNS / f"{tag}.pt")
    (RUNS / f"{tag}.json").write_text(json.dumps(dict(best=best, log=log), indent=1), encoding="utf-8")


def load(fe, dev, variant=""):
    ms, dat = [], None
    for f in sorted(glob.glob(str(RUNS / "seed*.pt"))):
        tag = pathlib.Path(f).stem
        if (tag.split("_", 1)[1] if "_" in tag else "") != variant:
            continue
        m, dat = build(fe, dev, variant)
        m.load_state_dict(torch.load(f, map_location=dev, weights_only=True))
        ms.append(m)
    return ms, dat


def counterfactual(model, dat, hours, xi, mi, eta, delta, dev, seed=None):
    """Returns per hour: dE (T, N) t per MWh delivered removed, absorption dR_m/delta, dR (T, N) MW per MW."""
    fe = dat.fe
    src, dst = torch.tensor(fe.src, device=dev), torch.tensor(fe.dst, device=dev)
    mask = torch.tensor(fe.disp_mask, device=dev)
    ef = torch.tensor(fe.ef_disp, device=dev)
    e_xm = int(np.where((fe.src == xi) & (fe.dst == mi))[0][0])
    e_mx = int(dat.rev[e_xm])
    rv = np.zeros(dat.E, bool)
    rv[[e_xm, e_mx]] = True
    into = torch.zeros(dat.E, fe.N, device=dev)
    into[torch.arange(dat.E), dst] = 1.0                                     # link e flows into node dst[e]
    outs = []
    for i in range(0, len(hours), 128):
        hs = hours[i:i + 128]
        base, pert = [], []
        for t in hs:
            X, MO, TA, CC = dat.sample(int(t), rv)
            base.append((X, MO, TA, CC))
            f_new = TA[-1, e_xm, 2] - delta / fe.caps[e_xm]
            Xp, TAp, CCp = X.copy(), TA.copy(), CC.copy()
            for e, v in ((e_xm, f_new), (e_mx, -f_new)):
                TAp[-1, e, 1], TAp[-1, e, 2] = abs(v), v
                CCp[-1, e, 2] = max(v, 0.0)
            pert.append((Xp, MO, TAp, CCp))
        r0 = torch.tensor(fe.resid[:, hs].T, device=dev)
        tb = [torch.tensor(np.stack(a), device=dev) for a in zip(*base)]
        tp = [torch.tensor(np.stack(a), device=dev) for a in zip(*pert)]
        with torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed + i)
            g0, _, _, f0 = model(*tb, src, dst, r0, mask)
            if seed is not None:
                torch.manual_seed(seed + i)
            _, _, _, f1 = model(*tp, src, dst, r0, mask)
            cap = torch.tensor(fe.caps, device=dev)
            df = (f1 - f0) * cap                                                  # MW per directed link
            rev = torch.tensor(dat.rev, device=dev)
            g = 0.5 * (df - df[:, rev])                                           # antisymmetrised
            g[:, [e_xm, e_mx]] = 0.0
            dni = g @ into                                                        # (B, N) MW
            dni[:, mi] += -delta
            dni[:, xi] += delta / eta
            r1 = r0 - dni / 1000.0
            if seed is not None:
                torch.manual_seed(seed + i)
            g1, _, _, _ = model(*tp, src, dst, r1, mask)
            e0 = (g0 * ef).sum(-1) * 1000.0
            e1 = (g1 * ef).sum(-1) * 1000.0
        outs.append(((e1 - e0).cpu().numpy() / delta, (-dni).cpu().numpy() / delta))
    dE = np.concatenate([o[0] for o in outs])
    dR = np.concatenate([o[1] for o in outs])
    return dE, dR


def marginal(a):
    dev = "cuda"
    fe = Features()
    models, dat = load(fe, dev, a.variant)
    print(SPEC, a.variant or "graph", len(models), "models", flush=True)
    idx = {n: i for i, n in enumerate(fe.nodes)}
    OUTD.mkdir(parents=True, exist_ok=True)
    for b, (x, m, eta) in BORDERS.items():
        xi, mi = idx[x], idx[m]
        hours = []
        for p in a.periods.split(","):
            s, e = (hour_of(v) for v in SPLITS[p])
            hours += [t for t in range(s, e) if fe.F[xi, mi, t] > a.delta and fe.disp_obs_ok[xi, t]
                      and fe.disp_obs_ok[mi, t]]
        hours = np.array(hours)
        dEs, dRs = [], []
        for model in models:
            for k in range(a.mc):
                model.train() if k else model.eval()
                dE, dR = counterfactual(model, dat, hours, xi, mi, eta, a.delta, dev, seed=1000 * k if k else None)
                dEs.append(dE.astype(np.float16))
                dRs.append(dR.astype(np.float16))
        dEs, dRs = np.stack(dEs), np.stack(dRs)                                  # (D, T, N)
        suf = (f"_d{int(a.delta)}" if a.delta != 100 else "") + (f"_{a.variant}" if a.variant else "")
        np.savez_compressed(OUTD / f"marginal_{b.replace('->', '_')}{suf}.npz", hours=hours, dE=dEs, dR=dRs,
                            flow=fe.F[xi, mi, hours])
        w = fe.F[xi, mi, hours]
        med = np.median(dEs.astype(np.float32), 0)
        medR = np.median(dRs.astype(np.float32), 0)
        wm = lambda v: float(v @ w / w.sum())                                    # noqa: E731
        oth = [j for j in range(fe.N) if j not in (xi, mi)]
        print(b, len(hours), "hours: E_x %.3f E_m %.3f third %.3f total %.3f | absorption m %.3f x %.3f"
              % (wm(-med[:, xi]), wm(med[:, mi]), wm(med[:, oth].sum(1)), wm(med.sum(1)), wm(medR[:, mi]),
                 wm(-medR[:, xi])), flush=True)


def evaluate(a):
    """Held-out flow accuracy with one revealed link (as in the counterfactual) against the previous-hour flow."""
    dev = "cuda"
    fe = Features()
    res = {}
    for variant in ("", "dropTACC"):
        models, dat = load(fe, dev, variant)
        if not models:
            continue
        src, dst = torch.tensor(fe.src, device=dev), torch.tensor(fe.dst, device=dev)
        mask = torch.tensor(fe.disp_mask, device=dev)
        ef = torch.tensor(fe.ef_disp, device=dev)
        rng = np.random.default_rng(1)
        for p in ("test", "y2025", "charged"):
            s, e = (hour_of(v) for v in SPLITS[p])
            ts = rng.choice(np.arange(max(s, W + 1), e), 2048, replace=False)
            rv = [np.isin(dat.und, rng.choice(np.unique(dat.und), 1)) for _ in ts]
            P, G = [], []
            for m in models:
                m.eval()
                fp, ge = [], []
                with torch.no_grad():
                    for i in range(0, len(ts), 128):
                        b = batch(dat, ts[i:i + 128], rv[i:i + 128], dev)
                        gen, _, _, fl = m(*b[:4], src, dst, b[5], mask)
                        fp.append(fl.cpu().numpy())
                        ge.append(((gen * ef).sum(-1) * 1000).cpu().numpy())
                P.append(np.concatenate(fp))
                G.append(np.concatenate(ge))
            P, G = np.mean(P, 0), np.mean(G, 0)
            obs = dat.f_all[:, ts].T * fe.caps
            lag = dat.f_all[:, ts - 1].T * fe.caps
            ok = dat.f_ok[:, ts].T & ~np.stack(rv)
            eo = (fe.gen_disp[:, :, ts] * fe.ef_disp[None, :, None]).sum(1).T * 1000
            oke = fe.disp_obs_ok[:, ts].T
            rm = lambda u, v, k: float(np.sqrt(((u - v) ** 2)[k].mean()))            # noqa: E731
            res[f"{variant or 'graph'}_{p}"] = dict(flow_rmse_mw=rm(P * fe.caps, obs, ok), lag_rmse_mw=rm(lag, obs, ok),
                                                    emis_rmse=rm(G, eo, oke))
            print(variant or "graph", p, {k: round(v, 1) for k, v in res[f"{variant or 'graph'}_{p}"].items()},
                  flush=True)
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "evaluate_v3.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "marginal", "evaluate"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--drop", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--mc", type=int, default=10)
    ap.add_argument("--delta", type=float, default=100.0)
    ap.add_argument("--periods", default="y2025,charged,charged_q3")
    a = ap.parse_args()
    {"train": train, "marginal": marginal, "evaluate": evaluate}[a.cmd](a)
