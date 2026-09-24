"""Descriptive summary of the held-out HVDC outage validation (A59/A60) for one network-response tag.

Reads data/processed/gnn/outage_multi{rtag}_v4.json (rtag "" for r2) and writes outage_multi_summary{rtag}_v4.json:
  events, effective events per link; data pattern (links where other links carry more than half of the lost flow at
  one or both ends; end zones whose own output changes by less than 0.5 MW per MW); for graph, node-local and two-zone
  models over the 36 output and other-link responses (spec A, pooled direction): RMSE, share inside the interval,
  Pearson correlation with the outage estimates, and misses per link.
Usage: NETRESP_TAG=r2 python src/wedge/gnn/outage_multi_summary.py
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np

G = pathlib.Path("data/processed/gnn")
KEYS = ("R_a", "R_b", "NIother_a", "NIother_b")


def main():
    tag = os.environ.get("NETRESP_TAG", "r2")
    rtag = "" if tag in ("", "r2") else f"_{tag}"
    d = json.loads((G / f"outage_multi{rtag}_v4.json").read_text(encoding="utf-8"))
    links = [k for k in d if not k.startswith("_")]
    out = {"links": links, "events": {k: d[k]["events"] for k in links},
           "effective_events": {k: d[k]["effective_events"] for k in links},
           "events_total": int(sum(d[k]["events"] for k in links))}
    both, one, small = 0, 0, 0
    for k in links:
        a = d[k]["outcomes"]["NIother_a"]["A_block"]["did"][2] <= -0.5
        b = d[k]["outcomes"]["NIother_b"]["A_block"]["did"][2] >= 0.5
        both += int(a and b)
        one += int(a or b)
        small += sum(abs(d[k]["outcomes"][r]["A_block"]["did"][2]) < 0.5 for r in ("R_a", "R_b"))
    out["pattern"] = {"links_other_links_over_half_one_end": one, "links_other_links_over_half_both_ends": both,
                      "end_zones_own_output_below_half": int(small), "end_zones": 2 * len(links)}
    out["support"] = {k: int(sum(d[l][k] for l in links)) for k in ("events_with_blocks", "blocks",
                                                                     "blocks_with_window", "pseudo_blocks")}
    # descriptive, not pre-specified: the same comparison without the German end zones
    nd = {}
    for name in [n for n in ("graph", "local", "one_for_one") if n in d[links[0]]["model"]]:
        err = []
        for k in links:
            a, b = k.split("-")
            for r in KEYS:
                if (a if r.endswith("_a") else b) == "DE":
                    continue
                err.append(d[k]["model"][name][r]["A_block"]["model"][2] - d[k]["outcomes"][r]["A_block"]["did"][2])
        nd[name] = {"rmse": float(np.sqrt(np.mean(np.square(err)))), "n": len(err)}
    out["descriptive_without_german_end_zones"] = nd
    # descriptive: largest absolute errors and the share of squared error at the German end zones
    top = {}
    for name in [n for n in ("graph", "local", "one_for_one") if n in d[links[0]]["model"]]:
        rows = []
        for k in links:
            a, b = k.split("-")
            for r in KEYS:
                v = d[k]["outcomes"][r]["A_block"]["did"][2]
                m = d[k]["model"][name][r]["A_block"]["model"][2]
                rows.append((abs(m - v), k, a if r.endswith("_a") else b, r))
        tot = sum(e ** 2 for e, *_ in rows)
        de = sum(e ** 2 for e, _, z, _ in rows if z == "DE")
        top[name] = {"largest": [[round(e, 3), k, z, r] for e, k, z, r in sorted(rows, reverse=True)[:5]],
                     "german_share_sq_error": de / tot}
    out["descriptive_largest_errors"] = top
    for fes in ("A_block", "B_block_hour"):
        for name in [n for n in ("graph", "local", "one_for_one") if n in d[links[0]]["model"]]:
            x, y, ins, miss = [], [], [], {}
            for k in links:
                for r in KEYS:
                    e = d[k]["outcomes"][r][fes]["did"][2]
                    m = d[k]["model"][name][r][fes]["model"][2]
                    i = d[k]["model"][name][r][fes]["inside"][2]
                    x.append(e)
                    y.append(m)
                    ins.append(bool(i))
                    if not i:
                        miss[k] = miss.get(k, 0) + 1
            x, y = np.array(x), np.array(y)
            out[f"{fes}|{name}"] = {"rmse": float(np.sqrt(np.mean((y - x) ** 2))), "inside_share": float(np.mean(ins)),
                                    "pearson": float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 0 else None,
                                    "misses": miss, "n": int(len(x))}
    (G / f"outage_multi_summary{rtag}_v4.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if "|" in k and k.startswith("A_block")}, indent=0))
    print(out["pattern"], out["events_total"])


if __name__ == "__main__":
    main()
