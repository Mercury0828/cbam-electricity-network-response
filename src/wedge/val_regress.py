"""Validation (c): an independently specified empirical marginal-emission response check (A41).

Triangulation, NOT ground truth. Hawkes-style first-difference
regressions on hourly data, estimated separately for 2025 and Jan-Jun 2026:

  GB:  d CO2_GB(h) = b_x * d NetExport_GB(h) + g * d VRE_GB(h) + hour-of-day FE + month FE + e
  NL:  d CO2_NL(h) = b_m * d NetImport_NL(h) + g * d VRE_NL(h) + hour-of-day FE + month FE + e

CO2 = sum_k generation_k x midpoint EF_k (the model's own EF midpoints; the check tests the RESPONSE,
not the EFs). b_x estimates tCO2 per MWh of additional GB net export (all links), comparable to the
model's exporter-side e_out = E_x * eta; -b_m estimates tCO2 avoided per MWh of NL net import,
comparable to the model's e_in = E_m. Confounding caveat: net trade responds to the same shocks as
generation; VRE and FE absorb part of it; the coefficients are conditional associations.
Model counterpart: volume-weighted means of the scenario-midpoint E_x*eta and E_m over resolved
GB->NL export hours (benchmark.collect), same period.
"""
from __future__ import annotations

import glob
import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_gb_fr_2025_12 as G                            # noqa: E402
from wedge import run_year_gbfr as RG                                  # noqa: E402
from wedge.run_year import _merge                                      # noqa: E402

GB_EF = {k: sum(t.ef_el_range) / 2 for k, t in G.GB_TECHS.items()}
NL_EF = {k: sum(t.ef_el_range) / 2 for k, t in G.NL_TECHS.items()}
GB_VRE = ("Wind Onshore", "Wind Offshore", "Solar")
NL_VRE = ("Wind offshore", "Wind onshore", "Solar")


def gb_series(year):
    gen, *_ = RG.load_gb_fr(pathlib.Path(f"data/raw/gb_fr_{year}"), "nl")
    ic = {}
    for f in sorted(glob.glob(f"data/raw/gb_fr_{year}/gb_interconnector_*.json")):
        for rec in json.loads(pathlib.Path(f).read_text(encoding="utf-8")).get("data", []):
            ic[(rec["startTime"], rec["interconnectorName"])] = float(rec["generation"])
    net_imp = {}
    for (t, _), v in ic.items():
        h = RG._hour(t)
        net_imp[h] = net_imp.get(h, 0.0) + v / 2.0
    hours = set(net_imp)
    for k in ("Fossil Gas", "Nuclear"):
        hours &= set(gen.get(k, {}))
    out = {}
    for h in hours:
        co2 = sum(gen[k].get(h, 0.0) * GB_EF.get(k, 0.0) for k in gen)
        vre = sum(gen.get(k, {}).get(h, 0.0) for k in GB_VRE)
        out[h] = (co2, -net_imp[h], vre)                   # (CO2 t/h, net EXPORT MW, VRE MW)
    return out


def nl_series(year):
    g = _merge(f"data/raw/gb_fr_{year}/nl_generation_*.json")
    trade = g.get("Cross border electricity trading", {})   # EC: + = net import (verified sign, A31)
    out = {}
    for h, x in trade.items():
        if h not in g.get("Fossil gas", {}):
            continue
        co2 = sum(g[k].get(h, 0.0) * NL_EF.get(k, 0.0) for k in NL_EF if k in g)
        vre = sum(g.get(k, {}).get(h, 0.0) for k in NL_VRE)
        out[h] = (co2, x, vre)
    return out


def regress(series, year, last_month):
    hs = sorted(h for h in series if datetime.fromtimestamp(h, tz=timezone.utc).year == year
                and datetime.fromtimestamp(h, tz=timezone.utc).month <= last_month)
    rows = []
    for a, b in zip(hs, hs[1:]):
        if b - a != 3600:
            continue
        d = [series[b][i] - series[a][i] for i in range(3)]
        t = datetime.fromtimestamp(b, tz=timezone.utc)
        rows.append((d[0], d[1], d[2], t.hour, t.month))
    y = np.array([r[0] for r in rows])
    X = [np.array([r[1] for r in rows]), np.array([r[2] for r in rows])]
    for hh in range(1, 24):
        X.append(np.array([1.0 if r[3] == hh else 0.0 for r in rows]))
    for mm in sorted({r[4] for r in rows})[1:]:
        X.append(np.array([1.0 if r[4] == mm else 0.0 for r in rows]))
    X.append(np.ones(len(rows)))
    X = np.column_stack(X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    # HAC-lite: week-block bootstrap of the coefficient
    rng = np.random.default_rng(3)
    wk = np.array([(r[4], i // 168) for i, r in enumerate(rows)])
    blocks = {}
    for i, key in enumerate(map(tuple, wk)):
        blocks.setdefault(key, []).append(i)
    keys = list(blocks)
    bs = []
    for _ in range(200):
        idx = np.concatenate([blocks[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        b, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        bs.append(b[0])
    bs.sort()
    return dict(n=len(rows), coef=float(beta[0]), ci95=[float(bs[5]), float(bs[194])],
                r2=float(1 - res.var() / y.var()))


def model_counterpart():
    from wedge.benchmark import C26, collect
    from wedge.run_tiers import month
    links, ta, pa, ea = RG.IMPORTERS["nl"]

    def prices(h):
        m = C26[str(month(h).month)]
        return dict(p_x_dispatch=m["gb_dispatch_eur"], p_x_uka=m["uka_eur"],
                    p_x_credit_full=m["gb_dispatch_eur"])
    rows = collect(lambda: RG.load_gb_fr(pathlib.Path("data/raw/gb_fr_2026"), "nl"), G.GB_TECHS,
                   getattr(G, ta), prices, getattr(G, ea), -1)
    lab = [r for r in rows if r["sc"] is not None]
    w = sum(r["mwh"] for r in lab)
    ex = sum(r["mwh"] * np.mean([s[3] * s[2] for s in r["sc"]]) for r in lab) / w
    em = sum(r["mwh"] * np.mean([s[4] for s in r["sc"]]) for r in lab) / w
    return dict(e_out_mean=float(ex), e_in_mean=float(em), hours=len(lab))


def main():
    res = {}
    gb25, gb26 = gb_series(2025), gb_series(2026)
    nl25, nl26 = nl_series(2025), nl_series(2026)
    res["GB_bx_2025"] = regress(gb25, 2025, 12)
    res["GB_bx_2026H1"] = regress(gb26, 2026, 6)
    res["NL_bm_2025"] = regress(nl25, 2025, 12)
    res["NL_bm_2026H1"] = regress(nl26, 2026, 6)
    res["model_GB_NL_2026H1"] = model_counterpart()
    for k, v in res.items():
        print(k, v)
    pathlib.Path("data/processed/val_regress.json").write_text(json.dumps(res, indent=1),
                                                               encoding="utf-8")


if __name__ == "__main__":
    main()


# --- RS/HU extension (A41): the GB/NL ratio must NOT be transferred to a lignite exporter ---------
def ec_series(path_glob, techs, vre_names):
    ef = {k: sum(t.ef_el_range) / 2 for k, t in techs.items()}
    g = _merge(path_glob)
    trade = g.get("Cross border electricity trading", {})        # + = net import (A31)
    out = {}
    for h, x in trade.items():
        co2 = sum(g[k].get(h, 0.0) * ef.get(k, 0.0) for k in ef if k in g)
        vre = sum(g.get(k, {}).get(h, 0.0) for k in vre_names)
        out[h] = (co2, x, vre)
    return out


def rs_hu_main():
    from wedge import inputs_rs_hu_2025_12 as R
    from wedge.run_year import daily_ttf_fuel, techs_for          # noqa: F401
    res = {}
    for y, last in ((2025, 12), (2026, 6)):
        rs = ec_series(f"data/raw/rs_hu_{y}/rs_generation_*.json", R.RS_TECHS, ("Wind onshore",))
        hu = ec_series(f"data/raw/rs_hu_{y}/hu_generation_*.json", R.HU_TECHS, ("Wind onshore", "Solar"))
        # RS: net EXPORT = -trade
        rs_x = {h: (v[0], -v[1], v[2]) for h, v in rs.items()}
        res[f"RS_bx_{y}"] = regress(rs_x, y, last)
        res[f"HU_bm_{y}"] = regress(hu, y, last)
    for k, v in res.items():
        print(k, v)
    old = json.loads(pathlib.Path("data/processed/val_regress.json").read_text(encoding="utf-8"))
    old.update(res)
    pathlib.Path("data/processed/val_regress.json").write_text(json.dumps(old, indent=1),
                                                               encoding="utf-8")


def daily(series):
    """Aggregate an hourly series {h: (co2, trade, vre)} to daily sums keyed by the day's first hour."""
    out = {}
    for h, v in series.items():
        d = h - h % 86400
        a = out.setdefault(d, [0.0, 0.0, 0.0, 0])
        for i in range(3):
            a[i] += v[i]
        a[3] += 1
    return {d: tuple(a[:3]) for d, a in out.items() if a[3] == 24}


def regress_daily(series, year, last_month):
    """Day-to-day first differences, month FE (no hour FE), week-block bootstrap."""
    ds = sorted(d for d in series if datetime.fromtimestamp(d, tz=timezone.utc).year == year
                and datetime.fromtimestamp(d, tz=timezone.utc).month <= last_month)
    rows = []
    for a, b in zip(ds, ds[1:]):
        if b - a != 86400:
            continue
        t = datetime.fromtimestamp(b, tz=timezone.utc)
        rows.append(tuple(series[b][i] - series[a][i] for i in range(3)) + (t.weekday(), t.month))
    y = np.array([r[0] for r in rows])
    X = [np.array([r[1] for r in rows]), np.array([r[2] for r in rows])]
    for wd in range(1, 7):
        X.append(np.array([1.0 if r[3] == wd else 0.0 for r in rows]))
    for mm in sorted({r[4] for r in rows})[1:]:
        X.append(np.array([1.0 if r[4] == mm else 0.0 for r in rows]))
    X.append(np.ones(len(rows)))
    X = np.column_stack(X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rng = np.random.default_rng(4)
    n = len(rows)
    bs = []
    for _ in range(300):
        starts = rng.integers(0, max(n - 7, 1), n // 7 + 1)
        idx = np.concatenate([np.arange(s, min(s + 7, n)) for s in starts])
        b, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        bs.append(b[0])
    bs.sort()
    return dict(n=n, coef=float(beta[0]), ci95=[float(bs[7]), float(bs[292])])


def daily_main():
    from wedge import inputs_rs_hu_2025_12 as R
    res = {}
    for y, last in ((2025, 12), (2026, 6)):
        gb, nl = gb_series(y), nl_series(y)
        rs = ec_series(f"data/raw/rs_hu_{y}/rs_generation_*.json", R.RS_TECHS, ("Wind onshore",))
        hu = ec_series(f"data/raw/rs_hu_{y}/hu_generation_*.json", R.HU_TECHS, ("Wind onshore", "Solar"))
        rs_x = {h: (v[0], -v[1], v[2]) for h, v in rs.items()}
        for name, ser in (("GB_bx", gb), ("NL_bm", nl), ("RS_bx", rs_x), ("HU_bm", hu)):
            res[f"{name}_daily_{y}"] = regress_daily(daily(ser), y, last)
    for k, v in res.items():
        print(k, v)
    old = json.loads(pathlib.Path("data/processed/val_regress.json").read_text(encoding="utf-8"))
    old.update(res)
    pathlib.Path("data/processed/val_regress.json").write_text(json.dumps(old, indent=1),
                                                               encoding="utf-8")


def model_rs_counterpart():
    """Model counterpart for RS->HU (resolved export hours Jan-Jun 2026), stored for the figure."""
    from wedge.benchmark import collect
    from wedge import inputs_rs_hu_2025_12 as R, run_year as RY

    def rl():
        rs, hu, rp, hp, fl, rc, hc = RY.load_rs_hu(pathlib.Path("data/raw/rs_hu_2026"))
        return rs, rp, hu, hp, fl, rc, hc
    rows = collect(rl, R.RS_TECHS, R.HU_TECHS,
                   lambda h: dict(p_x_dispatch=4.0, p_x_uka=4.0, p_x_credit_full=4.0),
                   R.ETA_LINK_RANGE, +1)
    lab = [r for r in rows if r["sc"] is not None]
    w = sum(r["mwh"] for r in lab)
    res = dict(e_out_mean=float(sum(r["mwh"] * np.mean([s[3] * s[2] for s in r["sc"]]) for r in lab) / w),
               e_in_mean=float(sum(r["mwh"] * np.mean([s[4] for s in r["sc"]]) for r in lab) / w),
               hours=len(lab))
    old = json.loads(pathlib.Path("data/processed/val_regress.json").read_text(encoding="utf-8"))
    old["model_RS_HU_2026H1"] = res
    pathlib.Path("data/processed/val_regress.json").write_text(json.dumps(old, indent=1), encoding="utf-8")
    print(res)


def be_main():
    """R1-05: a Belgian importer regression (the T4 baseline previously transferred the NL coefficient)."""
    BE_EF = {k: sum(t.ef_el_range) / 2 for k, t in G.BE_TECHS.items()}

    def be_series(year):
        g = _merge(f"data/raw/gb_fr_{year}/be_generation_*.json")
        trade = g.get("Cross border electricity trading", {})
        out = {}
        for h, x in trade.items():
            if h not in g.get("Fossil gas", {}):
                continue
            co2 = sum(g[k].get(h, 0.0) * BE_EF.get(k, 0.0) for k in BE_EF if k in g)
            vre = sum(g.get(k, {}).get(h, 0.0) for k in ("Wind offshore", "Wind onshore", "Solar"))
            out[h] = (co2, x, vre)
        return out
    res = {}
    for y, last in ((2025, 12), (2026, 6)):
        ser = be_series(y)
        res[f"BE_bm_{y}"] = regress(ser, y, last)
        res[f"BE_bm_daily_{y}"] = regress_daily(daily(ser), y, last)
    for k, v in res.items():
        print(k, v)
    old = json.loads(pathlib.Path("data/processed/val_regress.json").read_text(encoding="utf-8"))
    old.update(res)
    pathlib.Path("data/processed/val_regress.json").write_text(json.dumps(old, indent=1), encoding="utf-8")
