"""Train the extended MG-STGNN (A49). Split as in the original study: train 2019-2022, validation 2023 H1,
test 2023 H2 - 2024; 2025 and the charged period 2026 are held out entirely.

Loss (weights as in S0): SmoothL1 on dispatchable generation by class (GW, scaled per node) + 1.5 x SmoothL1 on the
standardised price + 0.1 x price/emission-intensity correlation matching + 1e-3 x dynamic-adjacency L1.
Checkpoint selection on the validation loss. Usage: python src/wedge/gnn/train.py --seed 42
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import RUNS, SPLITS, Features, hour_of          # noqa: E402
from wedge.gnn.model import MGSTGNNX                              # noqa: E402

OUT = RUNS
W = 24


def batch(fe, ts, dev):
    xs, mos, tas, ccs = zip(*[fe.sample(t, W) for t in ts])
    to = lambda a: torch.tensor(np.stack(a), device=dev)                           # noqa: E731
    ts = np.array(ts)
    y_gen = torch.tensor(fe.gen_disp[:, :, ts].transpose(2, 0, 1), device=dev)   # (B, N, D)
    resid = torch.tensor(fe.resid[:, ts].T, device=dev)
    y_p = torch.tensor(fe.z_price[:, ts].T, device=dev)
    ok_g = torch.tensor(fe.disp_obs_ok[:, ts].T, device=dev).float()
    ok_p = torch.tensor(fe.price_ok[:, ts].T, device=dev).float()
    return to(xs), to(mos), to(tas), to(ccs), y_gen, resid, y_p, ok_g, ok_p


EF_DISP = None


def corr(a, b, ok):
    a, b = a[ok > 0], b[ok > 0]
    if a.numel() < 3:
        return a.sum() * 0
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-9)


def loss_fn(model, b, src, dst, mask, scale, drop=()):
    """S0 loss: lambda_gen * L_gen + lambda_p * L_price + lambda_c * L_corr + lambda_s * L_sparse (1.0/1.5/0.1/1e-3);
    the correlation term matches the price-emission-intensity correlation of predictions to observations."""
    x, mo, ta, cc, yg, r, yp, okg, okp = b
    gen, pr, _ = model(x, mo, ta, cc, src, dst, r, mask, drop=drop)
    lg = (F.smooth_l1_loss(gen / scale, yg / scale, reduction="none").sum(-1) * okg).sum() / okg.sum().clamp(min=1)
    lp = (F.smooth_l1_loss(pr, yp, reduction="none") * okp).sum() / okp.sum().clamp(min=1)
    ef = EF_DISP.to(gen.device)
    ci_p = (gen * ef).sum(-1) / r.clamp(min=1e-3)
    ci_o = (yg * ef).sum(-1) / r.clamp(min=1e-3)
    ok = okg * okp
    lc = (corr(yp, ci_o, ok) - corr(pr, ci_p, ok)).abs()
    ls = model.adj_l1
    return lg + 1.5 * lp + 0.1 * lc + 1e-3 * ls, lg.item(), lp.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--drop", default="", help="comma list of channels to remove (MO,TA,CC) for ablation")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda"
    fe = Features()
    global EF_DISP
    EF_DISP = torch.tensor(fe.ef_disp)
    rng ={k: (max(hour_of(s), W), hour_of(e)) for k, (s, e) in SPLITS.items()}
    x0 = fe.sample(rng["train"][0] + 10, W)
    drop = tuple(x for x in a.drop.split(",") if x)
    model = MGSTGNNX(fe.N, x0[0].shape[-1], x0[1].shape[-1], x0[2].shape[-1], x0[3].shape[-1],
                     fe.gen_disp.shape[1]).to(dev)
    src, dst = torch.tensor(fe.src, device=dev), torch.tensor(fe.dst, device=dev)
    mask = torch.tensor(fe.disp_mask, device=dev)
    scale = torch.tensor(np.maximum(fe.rs.squeeze(1), 0.05), device=dev).view(1, -1, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    vt = np.random.RandomState(0).choice(np.arange(*rng["val"]), 1024, replace=False)
    best, tag = 1e9, f"seed{a.seed}" + (f"_drop{''.join(drop)}" if drop else "")
    OUT.mkdir(parents=True, exist_ok=True)
    log = []
    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        tl = []
        for _ in range(a.steps):
            ts = np.random.randint(*rng["train"], size=a.bs)
            l, lg, lp = loss_fn(model, batch(fe, ts, dev), src, dst, mask, scale, drop)
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append((lg, lp))
        model.eval()
        with torch.no_grad():
            vl = [loss_fn(model, batch(fe, vt[i:i + 128], dev), src, dst, mask, scale, drop)[1:]
                  for i in range(0, len(vt), 128)]
        vg, vp = np.mean([v[0] for v in vl]), np.mean([v[1] for v in vl])
        score = vg + 1.5 * vp
        log.append(dict(ep=ep, train=np.mean(tl, 0).tolist(), val_gen=vg, val_price=vp, sec=time.time() - t0))
        print(f"{tag} ep {ep:2d} train {np.mean(tl, 0).round(4)} val gen {vg:.4f} price {vp:.4f} ({time.time() - t0:.0f}s)",
              flush=True)
        if score < best:
            best = score
            torch.save(model.state_dict(), OUT / f"{tag}.pt")
    (OUT / f"{tag}.json").write_text(json.dumps(dict(best=best, log=log), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
