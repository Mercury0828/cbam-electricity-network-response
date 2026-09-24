"""Post-processing of outage_v2 (A54): direction-specific identifying weights, REMIT first-publication
lead times, leave-one-event-out ranges per specification, and the run configuration.

Additions without re-estimation:
  20  effective number of events for the L+ coefficient: weights proportional to the within-group sum of squares of
      L+ residualised on L- (specification A: block groups; B: block x hour groups), on the outcome's own mask
  21  leave-one-event-out ranges quoted per specification (read from outage_v2 output)
  5   REMIT: planned/unplanned counts over contributing events; lead time from the FIRST publication of each message
  18  configuration saved (spec, dataset suffix, estimator settings)
Output: data/processed/gnn/outage_v2_post{DSFX}.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import DSFX, SPEC, Features, hour_of                  # noqa: E402
from wedge.gnn.outage_v2 import BLOCK, FOCAL, NB, demean                      # noqa: E402


def outcome_mask(fe, idx, k, x, m, hours):
    xi, mi = idx[x], idx[m]
    ok = fe.disp_obs_ok[mi, hours] & fe.disp_obs_ok[xi, hours]
    if k[:2] in ("E_", "R_") and k not in ("R_m", "R_x"):
        ok &= fe.disp_obs_ok[idx[k.split("_", 1)[1]], hours]
    return ok


def remit_first_publication():
    f = pathlib.Path("data/raw/remit/remit_ic.jsonl")
    msgs = {}
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        if r.get("participantId") not in ("BRITNED", "NEMO1") or (r.get("unavailableCapacity") or 0) < 500:
            continue
        k = r["mrid"]
        rec = msgs.setdefault(k, {"first_pub": r["publishTime"], "last": r})
        rec["first_pub"] = min(rec["first_pub"], r["publishTime"])
        if r.get("revisionNumber", 0) >= rec["last"].get("revisionNumber", 0):
            rec["last"] = r
    out = {"GB->NL": [], "GB->BE": []}
    for rec in msgs.values():
        r = rec["last"]
        if r.get("eventStatus") == "Dismissed":
            continue
        try:
            s = hour_of(r["eventStartTime"][:19])
            e = hour_of(r["eventEndTime"][:19]) if r.get("eventEndTime") else s + 1
            p = hour_of(rec["first_pub"][:19])
        except Exception:                                                         # noqa: BLE001
            continue
        b = "GB->NL" if r["participantId"] == "BRITNED" else "GB->BE"
        out[b].append((s, e, r.get("unavailabilityType"), p))
    return out


def main():
    fe = Features()
    idx = {n: i for i, n in enumerate(fe.nodes)}
    res_all = json.loads(pathlib.Path(f"data/processed/gnn/outage_v2{DSFX}.json").read_text(encoding="utf-8"))
    remit = remit_first_publication()
    post = {"config": {"spec": SPEC, "dataset_suffix": DSFX, "block_hours": BLOCK, "bootstrap_draws": NB,
                       "pseudo_per_block": 5, "merge_gap_h": 72, "min_spell_h": 24,
                       "source_json": f"outage_v2{DSFX}.json"}}
    for b, (x, m) in FOCAL.items():
        z = np.load(f"data/processed/gnn/outage_v2_blocks_{b.replace('->', '_')}{DSFX}.npz")
        hours, blk, ev, L = z["hours"], z["block"], z["event"], z["L"]
        r = res_all[b]
        out = {"effective_events_pooled_dose": r["effective_events"]}
        # direction-specific identifying weights for the L+ coefficient
        for k in ("NIother_m", "R_m", f"E_{m}", "R_x", "NIother_x"):
            ok = outcome_mask(fe, idx, k, x, m, hours)
            h, bl, e_, l_ = hours[ok], blk[ok], ev[ok], L[ok]
            for fes in ("A_block", "B_block_hour"):
                g = bl if fes == "A_block" else bl * 24 + (h % 24)
                _, gi = np.unique(g, return_inverse=True)
                lp, lm = demean(np.maximum(l_, 0), gi), demean(np.minimum(l_, 0), gi)
                coef = (lp @ lm) / (lm @ lm) if (lm @ lm) > 0 else 0.0
                res_ = lp - coef * lm                                          # L+ residualised on L-
                w = np.bincount(np.unique(e_, return_inverse=True)[1], weights=res_ ** 2)
                w = w / w.sum()
                out[f"effective_events_Lplus|{k}|{fes}"] = float(1.0 / (w ** 2).sum())
                out[f"top_event_weight_Lplus|{k}|{fes}"] = float(w.max())
                out[f"n_hours|{k}"] = int(ok.sum())
        # leave-one-event-out ranges per specification for the headline outcomes
        for k in ("NIother_m", "R_m", f"E_{m}"):
            for spec in ("full", "reduced"):
                key = f"{k}|{spec}"
                if key not in r["outcomes"]:
                    continue
                for fes in ("A_block", "B_block_hour"):
                    e = r["outcomes"][key][fes]
                    out[f"loo|{key}|{fes}"] = {"plus": [e["loo_min"][0], e["loo_max"][0]],
                                               "pooled": [e["loo_min"][2], e["loo_max"][2]]}
        # REMIT classification with first-publication lead time, contributing events only
        cls = []
        for row in r["events_table"]:
            s = hour_of(row["start"])
            e = s + row["hours"]
            ov = [(ms, me, ty, pu) for (ms, me, ty, pu) in remit[b] if ms < e and me > s]
            typ, lead = "unclassified", None
            if ov:
                hrs = {}
                for ms, me, ty, pu in ov:
                    hrs[ty] = hrs.get(ty, 0) + min(me, e) - max(ms, s)
                typ = max(hrs, key=hrs.get)
                lead = float((s - min(pu for (_, _, _, pu) in ov)) / 24.0)
            cls.append({"event": row["event"], "start": row["start"], "remit": typ, "first_pub_lead_days": lead,
                        "contributing": row["treated_hours"] > 0})
        out["remit_events"] = cls
        for scope, rows in (("envelope", cls), ("contributing", [c for c in cls if c["contributing"]])):
            cnt = {}
            for c in rows:
                cnt[c["remit"]] = cnt.get(c["remit"], 0) + 1
            out[f"remit_counts_{scope}"] = cnt
        post[b] = out
        print(b, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()
                  if k.startswith(("effective", "remit_counts"))}, flush=True)
    pathlib.Path(f"data/processed/gnn/outage_v2_post{DSFX}.json").write_text(json.dumps(post, indent=1),
                                                                             encoding="utf-8")


if __name__ == "__main__":
    main()
