"""Audit of the first-draft specification (v1): the CBAM-cost input and the post-June-2023 regime indicator are
identically zero in the training years. On the charged period we hold every other input fixed and compare
  actual      inputs as used in the first draft
  no_charge   CBAM cost set to zero in the node, merit-order and carbon-cost inputs
  no_unseen   additionally the regime indicator set to the last regime seen in training
for the emission prediction error and the graph-model marginal responses (eval mode, five seeds).
Run with WEDGE_GNN_SPEC=v1. Output: data/processed/gnn/audit_v1.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import SPEC, SPLITS, Features, hour_of      # noqa: E402
from wedge.gnn.marginal import BORDERS, emissions, load_models, perturbed   # noqa: E402

assert SPEC == "v1"


def main():
    dev = "cuda"
    fe = Features()
    models = load_models(fe, dev)
    src, dst = torch.tensor(fe.src, device=dev), torch.tensor(fe.dst, device=dev)
    mask = torch.tensor(fe.disp_mask, device=dev)
    idx = {n: i for i, n in enumerate(fe.nodes)}
    X0, MO0, CC0 = fe.X.copy(), fe.MO.copy(), fe.CC.copy()
    s, e = (hour_of(v) for v in SPLITS["charged"])
    obs = (fe.gen_disp * fe.ef_disp[None, :, None]).sum(1) * 1000              # (N, H)
    res = {}
    for cfg in ("actual", "no_charge", "no_unseen"):
        fe.X, fe.MO, fe.CC = X0.copy(), MO0.copy(), CC0.copy()
        if cfg != "actual":
            fe.X[..., 7] = 0.0
            fe.MO[..., 1] = 0.0
            fe.CC[..., 0] = 0.0
        if cfg == "no_unseen":
            fe.X[..., 14:17] = np.array([0.0, 1.0, 0.0], np.float32)
        r = {}
        hours = np.arange(s, e)
        pe = []
        for m in models:
            m.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(hours), 256):
                    hs = hours[i:i + 256]
                    xs, mos, tas, ccs = (torch.tensor(np.stack(a), device=dev) for a in zip(*[fe.sample(int(t), 24) for t in hs]))
                    rr = torch.tensor(fe.resid[:, hs].T, device=dev)
                    gen, _, _ = m(xs, mos, tas, ccs, src, dst, rr, mask)
                    out.append(((gen * torch.tensor(fe.ef_disp, device=dev)).sum(-1) * 1000).cpu().numpy())
            pe.append(np.concatenate(out))
        pe = np.mean(pe, 0)
        ok = fe.disp_obs_ok[:, hours].T
        r["emis_rmse_2026H1"] = float(np.sqrt(((pe - obs[:, hours].T) ** 2)[ok].mean()))
        r["pred"] = pe
        for b, (x, mm, eta) in BORDERS.items():
            xi, mi = idx[x], idx[mm]
            hs_b = np.array([t for t in hours if fe.F[xi, mi, t] > 100 and fe.disp_obs_ok[xi, t] and fe.disp_obs_ok[mi, t]])
            w = fe.F[xi, mi, hs_b]
            dd = []
            for m in models:
                m.eval()
                o = []
                with torch.no_grad():
                    for i in range(0, len(hs_b), 128):
                        base, pert = zip(*[perturbed(fe, int(t), xi, mi, eta, 100.0) for t in hs_b[i:i + 128]])
                        e0, _ = emissions(m, fe, base, dev, mask, src, dst)
                        e1, _ = emissions(m, fe, pert, dev, mask, src, dst)
                        d = (e1 - e0).cpu().numpy() / 100.0
                        o.append(np.stack([-d[:, xi], d[:, mi]], 1))
                dd.append(np.concatenate(o))
            dd = np.mean(dd, 0)
            r[b] = {"E_x": float(dd[:, 0] @ w / w.sum()), "E_m": float(dd[:, 1] @ w / w.sum()), "hourly": dd}
        res[cfg] = r
        print(cfg, round(r["emis_rmse_2026H1"], 1), {b: (round(r[b]["E_x"], 3), round(r[b]["E_m"], 3)) for b in BORDERS},
              flush=True)
    out = {}
    for cfg in ("no_charge", "no_unseen"):
        out[cfg] = {"emis_rmse_2026H1": res[cfg]["emis_rmse_2026H1"],
                    "mean_abs_change_pred_t_per_h": float(np.abs(res[cfg]["pred"] - res["actual"]["pred"]).mean())}
        for b in BORDERS:
            h0, h1 = res["actual"][b]["hourly"], res[cfg][b]["hourly"]
            out[cfg][b] = {"E_x": res[cfg][b]["E_x"], "E_m": res[cfg][b]["E_m"],
                           "mean_abs_hourly_change_E_x": float(np.abs(h1[:, 0] - h0[:, 0]).mean()),
                           "mean_abs_hourly_change_E_m": float(np.abs(h1[:, 1] - h0[:, 1]).mean())}
    out["actual"] = {"emis_rmse_2026H1": res["actual"]["emis_rmse_2026H1"],
                     **{b: {"E_x": res["actual"][b]["E_x"], "E_m": res["actual"][b]["E_m"]} for b in BORDERS}}
    pathlib.Path("data/processed/gnn/audit_v1.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
