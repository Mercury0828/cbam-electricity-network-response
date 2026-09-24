"""Held-out HVDC outage test restricted to confirmed outages.

Runs the estimator and model comparison of outage_multi.py on the seven HVDC links whose outages appear in the Elexon
REMIT or Nord Pool UMM archives, keeping only the events that published outage messages confirm
(outage_events_v4.json), and compares the model errors with those of the full run on the same links and responses.
Output: data/processed/gnn/outage_multi_confirmed_v4.json (with block files) and
        data/processed/gnn/outage_multi_confirmed_summary_v4.json
Usage: WEDGE_GNN_SPEC=v4 NETRESP_TAG=r2 python src/wedge/gnn/outage_multi_confirmed.py [summary]
       ('summary' rebuilds the summary from the saved JSON files without re-running the estimator)
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn import outage_multi as om                               # noqa: E402
from wedge.gnn.outage_events import day                                # noqa: E402

G = pathlib.Path("data/processed/gnn")
KEYS = ("R_a", "R_b", "NIother_a", "NIother_b")


def errors(d, links, name, fes="A_block", drop_zone=None):
    """RMSE of model minus outage estimate (pooled dose) and intervals containing the model; drop_zone leaves out the
    responses at that zone's ends of the links."""
    errs, ins = [], []
    for lk in links:
        r = d[lk]
        for k in KEYS:
            if drop_zone is not None and lk.split("-")[0 if k.endswith("_a") else 1] == drop_zone:
                continue
            e = r["outcomes"][k][fes]["did"][2]
            m = r["model"][name][k][fes]["model"][2]
            if np.isfinite(e) and np.isfinite(m):
                errs.append(m - e)
                ins.append(bool(r["model"][name][k][fes]["inside"][2]))
    return {"rmse": float(np.sqrt(np.mean(np.square(errs)))), "inside": int(sum(ins)), "n": len(errs)}


def main():
    ev = json.loads((G / "outage_events_v4.json").read_text(encoding="utf-8"))
    conf = {}
    for e in ev["events"]:
        if e["confirmed"] and e["link"] not in ("GB-NL", "GB-BE", "RS-HU"):
            conf.setdefault(tuple(e["link"].split("-")), set()).add(e["start"])
    links = [lk for lk in om.HVDC if lk in conf]
    print("links", links, {f"{a}-{b}": len(v) for (a, b), v in conf.items()}, flush=True)
    om.main(links=links, out_name="outage_multi_confirmed", event_ok=lambda a, b, s, e: day(s) in conf[(a, b)],
            min_events=1)
    summary()


def summary():
    d_c = json.loads((G / "outage_multi_confirmed_v4.json").read_text(encoding="utf-8"))
    d_f = json.loads((G / "outage_multi_v4.json").read_text(encoding="utf-8"))
    ran = [lk for lk in d_c if not lk.startswith("_") and "model" in d_c[lk]]
    out = {"links": ran, "events": {lk: d_c[lk]["events"] for lk in ran},
           "treated_hours": {lk: d_c[lk]["treated_hours"] for lk in ran},
           "effective_events": {lk: d_c[lk]["effective_events"] for lk in ran}}
    for name in ("graph", "local", "one_for_one"):
        out[name] = {"confirmed": errors(d_c, ran, name), "all_events_same_links": errors(d_f, ran, name),
                     # the German ends of Baltic Cable and NordLink carry the largest errors of every model
                     "confirmed_without_german_ends": errors(d_c, ran, name, drop_zone="DE"),
                     "all_events_same_links_without_german_ends": errors(d_f, ran, name, drop_zone="DE")}
        print(name, out[name], flush=True)
    (G / "outage_multi_confirmed_summary_v4.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    summary() if sys.argv[1:] == ["summary"] else main()
