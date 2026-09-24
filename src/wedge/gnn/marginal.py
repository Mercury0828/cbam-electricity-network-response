"""Graph-model marginal emission responses on the charged borders (A49).

For each export hour t of a charged border x -> m, the delivered import is reduced by DELTA MW at the target hour:
  flow x -> m          F - DELTA                        (edge features of the last window step)
  importer residual    R_m + DELTA/1000 GW              (the importer supplies the removed import itself)
  exporter residual    R_x - DELTA/(eta*1000) GW        (the exporter no longer generates the import and its losses)
  net imports          updated consistently
The trained model re-allocates both residuals across dispatchable classes, and messages carry the change to the
neighbours. Per delivered MWh removed:
  E_x = -(emissions_x' - emissions_x)/DELTA,  E_m = (emissions_m' - emissions_m)/DELTA,
  third = (sum over other nodes of the change)/DELTA,  kappa = E_x - E_m (third-zone change reported separately).
Uncertainty: every seed checkpoint x MC-dropout passes gives one draw; per hour we keep all draws.
Output: data/processed/gnn/marginal_<border>.npz
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import RUNS, SPEC, SPLITS, Features, hour_of          # noqa: E402
from wedge.gnn.model import MGSTGNNX                              # noqa: E402

BORDERS = {"GB->NL": ("GB", "NL", 0.9794), "GB->BE": ("GB", "BE", 0.9794), "RS->HU": ("RS", "HU", 0.98345)}
DELTA = 100.0
W = 24


def load_models(fe, dev, variant=""):
    """variant "" = main graph model; "dropTACC" = same architecture trained without cross-node messages."""
    x0 = fe.sample(1000, W)
    ms = []
    for f in sorted(glob.glob(str(RUNS / "seed*.pt"))):
        tag = pathlib.Path(f).stem
        if (tag.split("_", 1)[1] if "_" in tag else "") != variant:
            continue
        m = MGSTGNNX(fe.N, x0[0].shape[-1], x0[1].shape[-1], x0[2].shape[-1], x0[3].shape[-1],
                     fe.gen_disp.shape[1]).to(dev)
        m.load_state_dict(torch.load(f, map_location=dev))
        if variant.startswith("drop"):
            v = variant[4:]
            m.default_drop = tuple(v[i:i + 2] for i in range(0, len(v), 2))              # "TACC" -> ("TA", "CC")
        ms.append(m)
    return ms


def perturbed(fe, t, xi, mi, eta, delta):
    """Return (x, mo, ta, cc, resid) for the base and the perturbed state at target hour t."""
    X, MO, TA, CC = (a.copy() for a in fe.sample(t, W))
    r = fe.resid[:, t].copy()
    Xp, TAp, CCp, rp = X.copy(), TA.copy(), CC.copy(), r.copy()
    dm, dx = delta / 1000.0, delta / (eta * 1000.0)
    rp[mi] += dm
    rp[xi] -= dx
    # node features at the last step: residual (col 3) and net import (col 4), standardised per node
    Xp[-1, mi, 3] += dm / fe.rs[mi, 0]
    Xp[-1, xi, 3] -= dx / fe.rs[xi, 0]
    Xp[-1, mi, 4] -= dm / fe.ni_s[mi, 0]          # the importer imports DELTA less
    Xp[-1, xi, 4] += dm / fe.ni_s[xi, 0]          # the exporter exports DELTA (delivered) less
    for e, (a, b) in enumerate(zip(fe.src, fe.dst)):
        if (a, b) == (xi, mi):
            TAp[-1, e, 2] -= delta / fe.caps[e]
            TAp[-1, e, 1] = abs(TAp[-1, e, 2])
            CCp[-1, e, -1] = max(TAp[-1, e, 2], 0.0)
        elif (a, b) == (mi, xi):
            TAp[-1, e, 2] += delta / fe.caps[e]
            TAp[-1, e, 1] = abs(TAp[-1, e, 2])
            CCp[-1, e, -1] = max(TAp[-1, e, 2], 0.0)
    return (X, MO, TA, CC, r), (Xp, MO, TAp, CCp, rp)


def emissions(model, fe, states, dev, mask, src, dst, drop=()):
    xs, mos, tas, ccs, rs = (torch.tensor(np.stack(a), device=dev) for a in zip(*states))
    gen, _, _ = model(xs, mos, tas, ccs, src, dst, rs, mask, drop=drop)              # (B, N, D) GW
    ef = torch.tensor(fe.ef_disp, device=dev)
    return (gen * ef).sum(-1) * 1000.0, gen * 1000.0                                  # t/h per node, MW by class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=10)
    ap.add_argument("--periods", default="y2025,charged,charged_q3")
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--drop", default="", help="channels removed at inference (MO,TA,CC), S0-style ablation")
    ap.add_argument("--borders", default=",".join(BORDERS))
    ap.add_argument("--variant", default="", help="trained variant, e.g. dropTACC (no cross-node messages)")
    a = ap.parse_args()
    dev = "cuda"
    fe = Features()
    models = load_models(fe, dev, a.variant)
    print(SPEC, a.variant or "graph", len(models), "models", flush=True)
    src, dst = torch.tensor(fe.src, device=dev), torch.tensor(fe.dst, device=dev)
    mask = torch.tensor(fe.disp_mask, device=dev)
    idx = {n: i for i, n in enumerate(fe.nodes)}
    drop = tuple(z for z in a.drop.split(",") if z)
    for b in a.borders.split(","):
        x, m, eta = BORDERS[b]
        xi, mi = idx[x], idx[m]
        hours = []
        for p in a.periods.split(","):
            s, e = (hour_of(v) for v in SPLITS[p])
            hours += [t for t in range(s, e) if fe.F[xi, mi, t] > a.delta and fe.disp_obs_ok[xi, t] and fe.disp_obs_ok[mi, t]]
        hours = np.array(hours)
        draws = []                                            # (n_draws, T, 3): E_x, E_m, third
        dnode, dcls = [], []                                  # per-node response; class response at x and m
        for model in models:
            for k in range(a.mc):
                if k == 0:
                    model.eval()
                else:
                    model.train()                              # MC dropout
                out, on, oc = [], [], []
                with torch.no_grad():
                    for i in range(0, len(hours), 128):
                        hs = hours[i:i + 128]
                        base, pert = zip(*[perturbed(fe, int(t), xi, mi, eta, a.delta) for t in hs])
                        # identical dropout masks for the base and the perturbed pass of the same draw, so the
                        # finite difference measures the response and not mask noise
                        torch.manual_seed(1000 * k + i)
                        e0, g0 = emissions(model, fe, base, dev, mask, src, dst, drop)
                        torch.manual_seed(1000 * k + i)
                        e1, g1 = emissions(model, fe, pert, dev, mask, src, dst, drop)
                        d = (e1 - e0).cpu().numpy() / a.delta
                        dg = (g1 - g0).cpu().numpy() / a.delta                   # (B, N, D) MW per MW
                        on.append(d.astype(np.float16))
                        oc.append(dg[:, [xi, mi]].astype(np.float16))
                        oth = d.sum(1) - d[:, xi] - d[:, mi]
                        out.append(np.stack([-d[:, xi], d[:, mi], oth], 1))
                draws.append(np.concatenate(out, 0))
                dnode.append(np.concatenate(on, 0))
                dcls.append(np.concatenate(oc, 0))
        draws = np.stack(draws, 0)
        pathlib.Path("data/processed/gnn").mkdir(parents=True, exist_ok=True)
        suf = (f"_d{int(a.delta)}" if a.delta != DELTA else "") + (f"_drop{''.join(drop)}" if drop else "")             + (f"_{a.variant}" if a.variant else "")
        outdir = pathlib.Path("data/processed/gnn" if SPEC == "v1" else f"data/processed/gnn/{SPEC}")
        outdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(outdir / f"marginal_{b.replace('->', '_')}{suf}.npz", hours=hours, draws=draws,
                            flow=fe.F[xi, mi, hours], dnode=np.stack(dnode, 0), dcls=np.stack(dcls, 0))
        med = np.median(draws, 0)
        w = fe.F[xi, mi, hours]
        print(b, len(hours), "hours; volume-weighted mean E_x %.3f E_m %.3f third %.3f" %
              tuple((med * w[:, None]).sum(0) / w.sum()), flush=True)


if __name__ == "__main__":
    main()
