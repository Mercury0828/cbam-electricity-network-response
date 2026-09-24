"""Event list of all outage tests and their confirmation in published outage messages.

For every link and event of the outage tests (outage_v2: BritNed GB-NL and Nemo Link GB-BE, 2019-2026; outage_multi:
ten HVDC links, events from July 2023; outage_rs: RS-HU), lists the zero-flow spell (start, end, length), the treated
hours and blocks that enter the estimates, the normal-operation windows of its blocks, its share of the identifying
weight, and whether a published outage message confirms it. Message sources:
  Elexon REMIT         GB interconnector operators (NEMO1, BRITNED from November 2025, NGNSL, NGVLL)
  Nord Pool UMM        transmission messages (BritNed Development from 2022, Statnett, Energinet, Svenska kraftnat,
                       TenneT, Baltic Cable AB, and messages forwarded by ENTSO-E); src/wedge/fetch/umm_nordpool.py
  JAO                  hourly offered capacity of the daily auctions RS-HU and HU-RS (src/wedge/fetch/
                       jao_outage_check.py); an hour counts when both directions offered zero capacity
A message hour is a full outage when a message reports available capacity of at most 5% of the normal (installed)
capacity in at least one direction. An event is confirmed when such hours cover at least half of its zero-flow hours;
its type (planned or unplanned) is the one covering most of them. The rows also give the directions whose messages
cover at least half of the zero-flow hours, the share covered in both directions, up to five message ids, and the
identifying weight of the event for the pooled dose and for the direction-specific slopes (L+ and L-, each
residualised on the other within blocks, specification A). Treated hours, blocks, windows and weights are read from the block
files of the frozen runs, and the reconstructed events are checked against them.
Outputs: data/processed/gnn/outage_events{DSFX}.json and outage_events{DSFX}.csv (one row per event).
Usage: WEDGE_GNN_SPEC=v4 python src/wedge/gnn/outage_events.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import DSFX, Features, hour_of                                 # noqa: E402
from wedge.gnn.outage_multi import COMMISSIONED as COMM_MULTI, HELD, HVDC               # noqa: E402
from wedge.gnn.outage_v2 import T0, design, merge, spells_open                          # noqa: E402

ELEXON = {"NEMO1": ("GB", "BE"), "BRITNED": ("GB", "NL"), "NGNSL": ("GB", "NO"), "NGVLL": ("DK", "GB")}
NAMES = {("GB", "NL"): "BritNed", ("GB", "BE"): "Nemo Link", ("DE", "SE"): "Baltic Cable", ("GR", "IT"): "GRITA",
         ("NL", "NO"): "NorNed", ("PL", "SE"): "SwePol", ("DK", "NL"): "COBRAcable", ("ME", "IT"): "MONITA",
         ("BE", "DE"): "ALEGrO", ("DE", "NO"): "NordLink", ("GB", "NO"): "North Sea Link", ("DK", "GB"): "Viking Link",
         ("RS", "HU"): "RS-HU tie lines"}
FULL = 0.05


def day(h):
    return datetime.fromtimestamp(T0 + 3600 * int(h), tz=timezone.utc).strftime("%Y-%m-%d")


def hour_iso(s):
    """Fractional hour index of an ISO time stamp (UTC)."""
    return (datetime.fromisoformat(s[:19]).replace(tzinfo=timezone.utc).timestamp() - T0) / 3600.0


def message_hours(H):
    """Full-outage hours per unordered link from Elexon REMIT and Nord Pool UMM:
    {frozenset(pair): {(source, type, direction): bool array}} and {same key: [(message id, start, end)]}.
    Direction: Nord Pool 'out>in' bidding-zone countries; Elexon 'G' (into GB) or 'D' (out of GB) from the asset id."""
    out, ids = {}, {}

    def add(pair, s, e, typ, src, direction, mid):
        k = frozenset(pair)
        d = out.setdefault(k, {})
        s, e = max(int(s), 0), min(int(e), H)
        if e <= s:
            return
        key = (src, "planned" if str(typ).lower().startswith("plan") or typ == 2 else "unplanned", direction)
        d.setdefault(key, np.zeros(H, bool))[s:e] = True
        ids.setdefault(k, {}).setdefault(key, []).append((mid, s, e))

    f = pathlib.Path("data/raw/remit/remit_ic.jsonl")
    last = {}
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        if r.get("participantId") not in ELEXON or r.get("messageType") != "UnavailabilitiesOfElectricityFacilities":
            continue
        k = r.get("mrid")
        if k not in last or r.get("revisionNumber", 0) >= last[k].get("revisionNumber", 0):
            last[k] = r
    for r in last.values():
        if r.get("eventStatus") == "Dismissed":
            continue
        normal, avail = abs(r.get("normalCapacity") or 0.0), r.get("availableCapacity")
        if not normal or avail is None:
            continue
        asset = r.get("assetId") or ""
        direction = "G" if ("G-" in asset or asset.endswith("_G")) else ("D" if ("D-" in asset or asset.endswith("_D")) else "?")
        prof = r.get("outageProfile") or [{"startTime": r.get("eventStartTime"), "endTime": r.get("eventEndTime"),
                                          "capacity": avail}]
        for q in prof:
            cap = q.get("capacity", avail)
            if q.get("startTime") and q.get("endTime") and cap is not None and abs(cap) <= FULL * normal:
                add(ELEXON[r["participantId"]], np.floor(hour_iso(q["startTime"])), np.ceil(hour_iso(q["endTime"])),
                    r.get("unavailabilityType", "unplanned"), "Elexon REMIT", direction, r.get("mrid"))
    f = pathlib.Path("data/raw/umm/nordpool_umm_transmission.jsonl")
    best = {}
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        k = r.get("messageId")
        if k not in best or (r.get("version") or 0) >= (best[k].get("version") or 0):
            best[k] = r
    for r in best.values():
        if r.get("eventStatus") == 3:                                   # dismissed
            continue
        for u in r.get("transmissionUnits") or []:
            a, b = (u.get("outAreaName") or "")[:2], (u.get("inAreaName") or "")[:2]
            inst = u.get("installedCapacity") or 0
            for q in u.get("timePeriods") or []:
                av = q.get("availableCapacity")
                if av is None or not inst or av > FULL * inst or not q.get("eventStart") or not q.get("eventStop"):
                    continue
                add((a, b), np.floor(hour_iso(q["eventStart"])), np.ceil(hour_iso(q["eventStop"])),
                    r.get("unavailabilityType"), "Nord Pool UMM", f"{a}>{b}", r.get("messageId"))
    return out, ids


def jao_offered(H):
    """Hourly offered capacity (MW) of the JAO daily auctions RS-HU and HU-RS (NaN where no auction):
    {corridor: array}. Product hour h of the delivery day starts h hours after marketPeriodStart (UTC)."""
    import glob
    res = {}
    for c in ("RS-HU", "HU-RS"):
        arr = np.full(H, np.nan)
        for f in sorted(glob.glob(f"data/raw/jao/jao_Daily_{c}_*.json")):
            for auc in json.load(open(f, encoding="utf-8")) or []:
                if auc.get("cancelled"):
                    continue
                t0 = hour_iso(auc["marketPeriodStart"][:19])
                for rr in auc.get("results") or []:
                    hh = int(str(rr.get("productHour", "00"))[:2])
                    h = int(round(t0)) + hh
                    if 0 <= h < H and rr.get("offeredCapacity") is not None:
                        arr[h] = float(rr["offeredCapacity"])
        res[c] = arr
    return res


def confirm(msgs, pair, zero_hours, ids=None):
    """Confirmation of one event: messages reporting the link fully unavailable (in at least one direction) cover at
    least half of the zero-flow hours. Returns source, type, coverage, directions covering >= half, both-direction
    coverage, and up to five message ids."""
    d = msgs.get(frozenset(pair), {})
    if not len(zero_hours):
        return None, None, 0.0, [], 0.0, []
    cov = {k: v[zero_hours].sum() for k, v in d.items()}
    covered = np.zeros(len(zero_hours), bool)
    a, b = pair
    other = b if a == "GB" else a

    def physical(dr):
        """Elexon G (into GB) / D (out of GB) and Nord Pool 'x>y' labels as one physical direction of the link."""
        if dr == "G":
            return f"{other}>GB"
        if dr == "D":
            return f"GB>{other}"
        return dr

    by_dir = {}
    for (src, typ, dr), v in d.items():
        covered |= v[zero_hours]
        pd_ = physical(dr)
        by_dir[pd_] = by_dir.get(pd_, np.zeros(len(zero_hours), bool)) | v[zero_hours]
    share = float(covered.mean())
    dirs = sorted(dr for dr, v in by_dir.items() if v.mean() >= 0.5)
    both = 0.0
    d1, d2 = f"{a}>{b}", f"{b}>{a}"
    if d1 in by_dir and d2 in by_dir:
        both = float(np.mean(by_dir[d1] & by_dir[d2]))
    mids = []
    if ids is not None:
        for key, lst in ids.get(frozenset(pair), {}).items():
            for mid, s, e in lst:
                if s < zero_hours.max() + 1 and e > zero_hours.min() and mid not in mids:
                    mids.append(mid)
    if share < 0.5:
        return None, None, share, dirs, both, mids[:5]
    (src, typ, _), _ = max(cov.items(), key=lambda kv: kv[1])
    return src, typ, share, dirs, both, mids[:5]


def direction_weights(z, n_ev):
    """Identifying weight of each event for the direction-specific slopes (specification A): within-block demeaned
    L+ residualised on demeaned L- (and vice versa), squared and summed by event."""
    evid, blk, L = z["event"], z["block"], z["L"]
    lp, lm = np.maximum(L, 0), np.minimum(L, 0)
    dp, dm = lp.copy(), lm.copy()
    for bb in np.unique(blk):
        sel = blk == bb
        dp[sel] -= lp[sel].mean()
        dm[sel] -= lm[sel].mean()
    out = []
    for x, y in ((dp, dm), (dm, dp)):
        b = (x @ y) / (y @ y) if (y @ y) > 0 else 0.0
        r = x - b * y
        w = np.bincount(evid, weights=r * r, minlength=n_ev)[:n_ev]
        out.append(w / w.sum() if w.sum() > 0 else w)
    return out


def event_rows(link, ev, zero, z, weight_from_npz, msgs, ids=None, jao=None, remit_frozen=None):
    """Rows for one link: ev list of (s, e); zero: zero-flow mask; z: block file of the frozen run; jao: hourly
    offered capacity of both directions (RS-HU), replacing the message check."""
    hrs, evid, blk, L = z["hours"], z["event"], z["block"], z["L"]
    pparent = z["p_parent"] if "p_parent" in z.files else np.array([], int)
    pblock = z["p_block"] if "p_block" in z.files else np.array([], int)
    w = np.zeros(len(ev))
    for bb in np.unique(blk):
        sel = blk == bb
        w[evid[sel][0]] += ((L[sel] - L[sel].mean()) ** 2).sum()
    w = w / w.sum() if w.sum() > 0 else w
    wp, wm = direction_weights(z, len(ev))
    rows = []
    for n, (s, e) in enumerate(ev):
        zh = np.where(zero[s:e])[0] + s
        if jao is not None:
            a1, a2 = jao["RS-HU"][zh], jao["HU-RS"][zh]
            have = np.isfinite(a1) & np.isfinite(a2)
            if have.any():
                share = float(((a1 == 0) & (a2 == 0))[have].mean())
                src, typ = ("JAO", "zero offered capacity") if share >= 0.5 else (None, None)
                dirs, both, mids = (["RS>HU", "HU>RS"] if share >= 0.5 else []), share, []
                note = f"auction data for {have.mean():.0%} of zero-flow hours"
            else:
                src, typ, share, dirs, both, mids = None, None, None, [], 0.0, []
                note = "no JAO auctions in this period"
        else:
            src, typ, share, dirs, both, mids = confirm(msgs, link, zh, ids)
            note = ""
        blocks_n = np.unique(blk[evid == n])
        wins = len(np.unique(pblock[np.isin(pparent, blocks_n)])) if len(pblock) else 0
        rows.append({"link": f"{link[0]}-{link[1]}", "name": NAMES.get(link, ""), "event": n, "start": day(s),
                     "end": day(e - 1), "hours": int(e - s), "zero_hours": int(len(zh)),
                     "treated_hours": int((evid == n).sum()), "blocks": int(len(blocks_n)), "windows": int(wins),
                     "blocks_with_window": int(len(np.intersect1d(blocks_n, np.unique(pparent)))) if len(pparent) else 0,
                     "weight": float(w[n]), "weight_plus": float(wp[n]), "weight_minus": float(wm[n]),
                     "confirmed": src is not None, "source": src or "",
                     "type": typ or "", "message_coverage": (round(share, 3) if share is not None else None),
                     "directions": dirs, "both_directions_coverage": round(both, 3), "message_ids": mids,
                     "note": note, "remit_frozen": (remit_frozen[n] if remit_frozen is not None else "")})
    return rows


def main():
    fe = Features()
    d = np.load(f"data/processed/gnn/dataset{DSFX}.npz")
    idx = {n: i for i, n in enumerate(fe.nodes)}
    H = fe.H
    fok = d["flow_ok"] if "flow_ok" in d.files else np.isfinite(fe.F)
    msgs, ids = message_hours(H)
    jao_cap = jao_offered(H)
    rows, checks = [], {}
    # BritNed and Nemo Link (outage_v2 design; identical events and blocks)
    frozen = json.loads(pathlib.Path(f"data/processed/gnn/outage_v2{DSFX}.json").read_text(encoding="utf-8"))
    DS = design(fe, d, np.random.default_rng(11))
    for b, D in DS.items():
        x, m = D["x"], D["m"]
        z = np.load(f"data/processed/gnn/outage_v2_blocks_{b.replace('->', '_')}{DSFX}.npz")
        assert np.array_equal(np.where(D["so_all"])[0], z["hours"])
        F = fe.F[idx[x], idx[m]]
        zero = fok[idx[x], idx[m]] & np.isfinite(F) & (np.abs(F) < 1.0)
        rem = [r_["remit"] for r_ in frozen[b]["events_table"]]
        rows += event_rows((x, m), D["ev"], zero, z, True, msgs, ids=ids, remit_frozen=rem)
        checks[b] = "blocks identical to outage_v2"
    # ten HVDC links (outage_multi event definition, events from July 2023)
    zeros = {}
    for a, b in HVDC + [("RS", "HU")]:
        ai, bi = idx[a], idx[b]
        F1 = fe.F[ai, bi]
        ok = fok[ai, bi] & np.isfinite(F1)
        tot = np.abs(np.nan_to_num(F1))
        first = np.where(ok & (tot > 1.0))[0]
        zz = ok & (tot < 1.0)
        if len(first):
            zz[:first[0]] = False
        if (a, b) in COMM_MULTI:
            zz[:hour_of(COMM_MULTI[(a, b)])] = False
        zeros[(a, b)] = zz
    for a, b in HVDC:
        ev = [(s, e) for s, e in merge(spells_open(zeros[(a, b)])) if s >= hour_of(HELD)]
        z = np.load(f"data/processed/gnn/outage_multi_blocks_{a}_{b}{DSFX}.npz")
        for n in np.unique(z["event"]):                               # reconstruction check
            s, e = ev[n]
            hh = z["hours"][z["event"] == n]
            assert hh.min() >= s and hh.max() < e, (a, b, n)
        rows += event_rows((a, b), ev, zeros[(a, b)], z, True, msgs, ids=ids)
        checks[f"{a}-{b}"] = "treated hours inside reconstructed events"
    # RS-HU: all three outages of the sensitivity run, JAO confirmation for the primary run
    ev = [(s, e) for s, e in merge(spells_open(zeros[("RS", "HU")])) if s >= hour_of("2023-07-01")]
    z = np.load(f"data/processed/gnn/outage_rs_all_blocks_RS_HU{DSFX}.npz")
    rows += event_rows(("RS", "HU"), ev, zeros[("RS", "HU")], z, True, msgs, jao=jao_cap)
    # per-link summary
    summ = {}
    for lk in dict.fromkeys(r_["link"] for r_ in rows):
        rr = [r_ for r_ in rows if r_["link"] == lk]
        used = [r_ for r_ in rr if r_["treated_hours"] > 0]
        summ[lk] = {"name": rr[0]["name"], "events": len(rr), "events_with_blocks": len(used),
                    "treated_hours": sum(r_["treated_hours"] for r_ in rr),
                    "confirmed_events": sum(r_["confirmed"] for r_ in used),
                    "confirmed_weight": float(sum(r_["weight"] for r_ in used if r_["confirmed"])),
                    "planned": sum(r_["type"] == "planned" for r_ in used),
                    "unplanned": sum(r_["type"] == "unplanned" for r_ in used),
                    "sources": sorted({r_["source"] for r_ in used if r_["source"]})}
        print(lk, summ[lk], flush=True)
    ten = [f"{a}-{b}" for a, b in HVDC]
    summ["_ten_hvdc_links"] = {k: sum(summ[lk][k] for lk in ten)
                               for k in ("events", "events_with_blocks", "treated_hours", "confirmed_events")}
    summ["_ten_hvdc_links"]["blocks"] = sum(r_["blocks"] for r_ in rows if r_["link"] in ten)
    summ["_ten_hvdc_links"]["blocks_with_window"] = sum(r_["blocks_with_window"] for r_ in rows if r_["link"] in ten)
    out = {"_meta": {"full_outage_threshold": FULL, "confirmation": "message hours cover >= 50% of zero-flow hours",
                     "checks": checks}, "summary": summ, "events": rows}
    pathlib.Path(f"data/processed/gnn/outage_events{DSFX}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    with open(f"data/processed/gnn/outage_events{DSFX}.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows([{k: (";".join(map(str, v)) if isinstance(v, list) else v) for k, v in r_.items()} for r_ in rows])


if __name__ == "__main__":
    main()
