"""Learned supply and transfer flexibilities of the network-response model on the charged hours (A58).

For the covered export hours of each charged border in January-June 2026 (charged_<border>_graph.npz), the trained
graph models give the supply flexibility k_i of every zone and the transfer flexibility g_e of every link (eq.
clearing). Reported per border, as medians over hours of the mean over trained models:
  k of the exporter and of the importer; g of every other link of the exporter and of the importer;
  own_share_importer = k_m / (k_m + sum of g over the importer's other links), the first-order share of a shortfall in
  the importer that its own plants meet before propagation (the charged link itself is clamped, g = 0);
  own_share_exporter likewise for a surplus in the exporter.
The ratios are unit-free; k and g share the units of the model's standardised price.
Usage: WEDGE_GNN_SPEC=v4 NETRESP_TAG=r2 python src/wedge/gnn/net_params.py
Output: <netresp OUTD>/net_params.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import SPLITS, Features, hour_of       # noqa: E402
from wedge.gnn.netresp import OUTD, Data, load_models           # noqa: E402

DEV = "cpu"


def main():
    fe = Features()
    dat = Data(fe)
    idx = {n: i for i, n in enumerate(fe.nodes)}
    ms = load_models(fe, DEV, False)
    src, dst = torch.tensor(fe.src, device=DEV), torch.tensor(fe.dst, device=DEV)
    a_, b_ = SPLITS["charged"]
    res = {}
    for b in ("GB->NL", "GB->BE", "RS->HU"):
        x, m = b.split("->")
        xi, mi = idx[x], idx[m]
        z = np.load(OUTD / f"charged_{b.replace('->', '_')}_graph.npz")
        hours = z["hours"]
        hours = hours[(hours >= hour_of(a_)) & (hours < hour_of(b_))]
        K, G = [], []
        for mdl in ms:
            ks, gs = [], []
            with torch.no_grad():
                for i in range(0, len(hours), 128):
                    bt = dat.batch(hours[i:i + 128], DEV)
                    k, xb, g = mdl.params(*bt[:4], src, dst, bt[11])
                    ks.append(k.numpy())
                    gs.append(g.numpy())
            K.append(np.concatenate(ks))
            G.append(np.concatenate(gs))
        K, G = np.mean(K, 0), np.mean(G, 0)                                     # (T, N), (T, E)
        links = [(e, fe.nodes[s_], fe.nodes[d_]) for e, (s_, d_) in enumerate(zip(fe.src, fe.dst))]
        charged = [e for e, s_, d_ in links if {s_, d_} == {x, m}]

        def other_links(zone):
            # one entry per physical link: the graph stores both directions with the same symmetrised g, and the
            # Laplacian of eq. clearing gives each physical link the weight g once (solve(): 0.5 g per direction)
            out, seen = [], set()
            for e, s_, d_ in links:
                nb = d_ if s_ == zone else s_
                if zone in (s_, d_) and e not in charged and nb not in seen:
                    seen.add(nb)
                    out.append((e, nb))
            return out
        o = {"hours": int(len(hours)), "k": {x: float(np.median(K[:, xi])), m: float(np.median(K[:, mi]))}}
        for zone, zi, key in ((m, mi, "importer"), (x, xi, "exporter")):
            ol = other_links(zone)
            o[f"g_{key}_links"] = {nb: float(np.median(G[:, e])) for e, nb in ol}
            gsum = G[:, [e for e, _ in ol]].sum(1)
            o[f"own_share_{key}"] = float(np.median(K[:, zi] / (K[:, zi] + gsum)))
            o[f"k_over_g_{key}"] = {nb: float(np.median(K[:, zi] / np.maximum(G[:, e], 1e-9))) for e, nb in ol}
        res[b] = o
        print(b, json.dumps(o)[:900], flush=True)
    (OUTD / "net_params.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
