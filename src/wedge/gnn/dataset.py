"""Hourly graph dataset for the extended MG-STGNN (A49), January 2019 - August 2026, UTC.

Nodes: the 15 zones of the earlier MG-STGNN study plus Great Britain and Serbia.
Per node and hour:
  gen[k]   generation by class (MW), classes CLASSES below
  load     MW (Energy-Charts "Load"; for GB, total generation plus net imports, so the balance holds by construction)
  price    EUR/MWh (day-ahead of the named bidding zone; GB: Elexon Market Index Price, converted at monthly ECB FX)
  netimp   net import (MW) summed over all borders, from the balance: load - generation + pumping consumption
  emis     tCO2/h = sum_k gen[k] * EF[k]
Per directed edge (i -> j) among nodes and hour: flow i -> j (MW, >= 0), from Energy-Charts cross-border physical
flows (GW, positive = import into the reporting country) and, for GB, Elexon interconnector outturn.
Global monthly inputs: EUA, GB carbon cost (UKA + CPS), gas and coal prices (gnn_prices_monthly.json).
Output: data/processed/gnn/dataset.npz and meta.json
"""
from __future__ import annotations

import glob
import json
import pathlib
from collections import defaultdict
from datetime import datetime, timezone

import os

import numpy as np

# A53: dataset v4 closes the boundary around the charged borders (neighbours of RS and HU, Ireland for GB, Portugal
# for ES) and fixes the GB interconnector mapping (Elexon names with spaces were not matched, and several links to
# one neighbour were averaged instead of summed). v2 (17 nodes) stays reproducible with WEDGE_DATASET=v2.
DATASET = os.environ.get("WEDGE_DATASET", "v4")
NODES = ["AT", "BE", "CH", "CZ", "DE", "DK", "ES", "FR", "GB", "HU", "IT", "NL", "NO", "PL", "RS", "SE", "SK"]
if DATASET == "v4":
    NODES = NODES + ["RO", "BG", "HR", "SI", "GR", "BA", "ME", "MK", "IE", "PT"]
CLASSES = ["nuclear", "lignite", "coal", "gas", "oil", "biomass", "waste", "hydro_ror", "hydro_res", "pumped",
           "wind", "solar", "other"]
# tCO2 per MWh electrical, class midpoints (IPCC 2006 fuel factors / typical efficiencies; biomass zero as in the EU ETS)
EF = {"nuclear": 0.0, "lignite": 1.10, "coal": 0.90, "gas": 0.40, "oil": 0.75, "biomass": 0.0, "waste": 0.30,
      "hydro_ror": 0.0, "hydro_res": 0.0, "pumped": 0.0, "wind": 0.0, "solar": 0.0, "other": 0.30}
EC_MAP = {"Nuclear": "nuclear", "Fossil brown coal / lignite": "lignite", "Fossil brown coal": "lignite",
          "Fossil hard coal": "coal", "Fossil coal-derived gas": "coal", "Fossil gas": "gas", "Fossil oil": "oil",
          "Fossil oil shale": "oil", "Fossil peat": "lignite", "Biomass": "biomass", "Waste": "waste",
          "Hydro Run-of-River": "hydro_ror", "Hydro water reservoir": "hydro_res", "Hydro pumped storage": "pumped",
          "Wind onshore": "wind", "Wind offshore": "wind", "Solar": "solar", "Geothermal": "other", "Others": "other",
          "Other renewables": "other", "Marine": "other"}
EC_NAME = {"Austria": "AT", "Belgium": "BE", "Switzerland": "CH", "Czech Republic": "CZ", "Germany": "DE",
           "Denmark": "DK", "Spain": "ES", "France": "FR", "Hungary": "HU", "Italy": "IT", "Netherlands": "NL",
           "Norway": "NO", "Poland": "PL", "Serbia": "RS", "Sweden": "SE", "Slovakia": "SK",
           "Great Britain": "GB", "United Kingdom": "GB", "Romania": "RO", "Bulgaria": "BG", "Croatia": "HR",
           "Slovenia": "SI", "Greece": "GR", "Bosnia-Herzegovina": "BA", "Bosnia and Herzegovina": "BA",
           "Montenegro": "ME", "North Macedonia": "MK", "Ireland": "IE", "Portugal": "PT"}
GB_FUEL = {"CCGT": "gas", "OCGT": "gas", "COAL": "coal", "NUCLEAR": "nuclear", "BIOMASS": "biomass", "OIL": "oil",
           "NPSHYD": "hydro_ror", "PS": "pumped", "OTHER": "other"}
GB_IC = {"France(IFA)": "FR", "France(IFA2)": "FR", "France(ElecLink)": "FR", "Netherlands(BritNed)": "NL",
         "Belgium(Nemolink)": "BE", "Denmark(Viking)": "DK", "Norway(NSL)": "NO"}
if DATASET == "v4":                   # Elexon names as they appear in the raw files 2019-2026
    GB_IC = {"France(IFA)": "FR", "IFA2 (INTIFA2)": "FR", "Eleclink (INTELEC)": "FR", "Netherlands(BritNed)": "NL",
             "Belgium (Nemolink)": "BE", "Denmark (Viking link)": "DK", "North Sea Link (INTNSL)": "NO",
             "Ireland(East-West)": "IE", "Northern Ireland(Moyle)": "IE", "Ireland (Greenlink)": "IE"}
# Physical interconnections among the 17 nodes (links not yet in service carry zero flow before commissioning).
TOPOLOGY = [("AT", "CH"), ("AT", "CZ"), ("AT", "DE"), ("AT", "HU"), ("AT", "IT"), ("BE", "DE"), ("BE", "FR"),
            ("BE", "NL"), ("BE", "GB"), ("CH", "DE"), ("CH", "FR"), ("CH", "IT"), ("CZ", "DE"), ("CZ", "PL"),
            ("CZ", "SK"), ("DE", "DK"), ("DE", "FR"), ("DE", "NL"), ("DE", "NO"), ("DE", "PL"), ("DE", "SE"),
            ("DK", "NL"), ("DK", "NO"), ("DK", "SE"), ("DK", "GB"), ("ES", "FR"), ("FR", "IT"), ("FR", "GB"),
            ("GB", "NL"), ("GB", "NO"), ("HU", "SK"), ("HU", "RS"), ("NL", "NO"), ("NO", "SE"), ("PL", "SK"),
            ("PL", "SE")]
if DATASET == "v4":
    TOPOLOGY = TOPOLOGY + [("RS", "RO"), ("RS", "BG"), ("RS", "HR"), ("RS", "BA"), ("RS", "ME"), ("RS", "MK"),
                           ("HU", "RO"), ("HU", "HR"), ("HU", "SI"), ("RO", "BG"), ("BG", "MK"), ("BG", "GR"),
                           ("HR", "SI"), ("HR", "BA"), ("SI", "AT"), ("SI", "IT"), ("GR", "MK"), ("GR", "IT"),
                           ("BA", "ME"), ("ME", "IT"), ("IE", "GB"), ("PT", "ES")]
RAW = pathlib.Path("data/raw/gnn")
T0 = int(datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp())
T1 = int(datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp())
H = (T1 - T0) // 3600


def hidx(ts):
    return (np.asarray(ts, dtype=np.int64) - T0) // 3600


def hourly_add(dst, cnt, ts, vals):
    """Average sub-hourly values of ONE series into hourly bins (dst, cnt accumulate)."""
    h = hidx(ts)
    v = np.asarray([np.nan if x is None else x for x in vals], float)
    ok = (h >= 0) & (h < H) & np.isfinite(v)
    np.add.at(dst, h[ok], v[ok])
    np.add.at(cnt, h[ok], 1)


class ClassSum:
    """Several source series map to one class: average each series within the hour, then SUM across series."""

    def __init__(self, n, k):
        self.acc = {}
        self.n, self.k = n, k

    def add(self, node, cls, series, ts, vals):
        key = (node, cls, series)
        if key not in self.acc:
            self.acc[key] = (np.zeros(H), np.zeros(H))
        hourly_add(self.acc[key][0], self.acc[key][1], ts, vals)

    def result(self):
        out = np.full((self.n, self.k, H), np.nan)
        for (node, cls, _), (s, c) in self.acc.items():
            m = np.where(c > 0, s / np.maximum(c, 1), np.nan)
            cur = out[node, cls]
            out[node, cls] = np.where(np.isnan(cur), m, np.where(np.isnan(m), cur, cur + m))
        return out


def load_ec(cc, cs, load, lcnt, price, pcnt, flow, fcnt, ni):
    for f in sorted(glob.glob(str(RAW / "ec_y" / cc.lower() / f"{cc.lower()}_generation_*.json"))):
        d = json.load(open(f, encoding="utf8"))
        ts = d["unix_seconds"]
        for x in d["production_types"]:
            n = x["name"]
            if n == "Load":
                hourly_add(load[ni], lcnt[ni], ts, x["data"])
            elif n in EC_MAP:
                cs.add(ni, CLASSES.index(EC_MAP[n]), n, ts, x["data"])
    for f in sorted(glob.glob(str(RAW / "ec_y" / cc.lower() / f"{cc.lower()}_price_*.json"))):
        d = json.load(open(f, encoding="utf8"))
        hourly_add(price[ni], pcnt[ni], d["unix_seconds"], d["price"])
    for f in sorted(glob.glob(str(RAW / "ec_y" / cc.lower() / f"{cc.lower()}_cbpf_*.json"))):
        d = json.load(open(f, encoding="utf8"))
        ts = d["unix_seconds"]
        for x in d["countries"]:
            nb = EC_NAME.get(x["name"])
            if nb is None or nb not in NODES or nb == cc:
                continue
            j = NODES.index(nb)
            v = [None if y is None else 1000.0 * y for y in x["data"]]      # GW -> MW, + = import into cc
            hourly_add(flow[j, ni], fcnt[j, ni], ts, v)                       # stored as signed j -> cc


def load_gb(cs, price, pcnt, flow, fcnt, gbp):
    gi = NODES.index("GB")
    for f in sorted(glob.glob(str(RAW / "gb" / "*" / "gb_fuelhh_*.json"))):
        for r in json.load(open(f, encoding="utf8")).get("data", []):
            k = GB_FUEL.get(r.get("fuelType"))
            if k is None:
                continue
            ts = int(datetime.fromisoformat(r["startTime"].replace("Z", "+00:00")).timestamp())
            cs.add(gi, CLASSES.index(k), r.get("fuelType"), [ts], [r["generation"]])
    for f in sorted(glob.glob(str(RAW / "gb" / "*" / "gb_agpt_*.json"))):
        for blk in json.load(open(f, encoding="utf8")).get("data", []):
            ts = int(datetime.fromisoformat(blk["startTime"].replace("Z", "+00:00")).timestamp())
            for x in blk.get("data", []):
                t = x.get("psrType", "")
                k = "wind" if t.startswith("Wind") else ("solar" if t == "Solar" else None)
                if k:
                    cs.add(gi, CLASSES.index(k), t, [ts], [x.get("quantity")])
    for f in sorted(glob.glob(str(RAW / "gb" / "*" / "gb_mid_*.json"))):
        for r in json.load(open(f, encoding="utf8")).get("data", []):
            if r.get("dataProvider") != "APXMIDP" or not r.get("volume"):
                continue
            ts = int(datetime.fromisoformat(r["startTime"].replace("Z", "+00:00")).timestamp())
            m = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")
            if m in gbp:                                   # chunks can reach past the study window
                hourly_add(price[gi], pcnt[gi], [ts], [float(r["price"]) / gbp[m]])
    seen = set()
    per_link = {}                         # link -> (sum, count): average each link within the hour, then sum links
    for f in sorted(glob.glob(str(RAW / "gb" / "*" / "gb_ic_*.json"))):
        for r in json.load(open(f, encoding="utf8")).get("data", []):
            name = r.get("interconnectorName")
            nb = GB_IC.get(name)
            key = (r.get("startTime"), name)
            if nb is None or nb not in NODES or key in seen:
                continue
            seen.add(key)
            ts = int(datetime.fromisoformat(r["startTime"].replace("Z", "+00:00")).timestamp())
            if DATASET != "v4":
                j = NODES.index(nb)
                hourly_add(flow[j, gi], fcnt[j, gi], [ts], [r["generation"]])    # + = import into GB (j -> GB)
                continue
            if name not in per_link:
                per_link[name] = (nb, np.zeros(H), np.zeros(H))
            hourly_add(per_link[name][1], per_link[name][2], [ts], [r["generation"]])
    if DATASET == "v4":
        tot = {}
        for name, (nb, s, c) in per_link.items():
            m = np.where(c > 0, s / np.maximum(c, 1), np.nan)
            cur = tot.get(nb)
            tot[nb] = m if cur is None else np.where(np.isnan(cur), m, np.where(np.isnan(m), cur, cur + m))
        for nb, series in tot.items():
            j = NODES.index(nb)
            ok = np.isfinite(series)
            flow[j, gi, ok] += series[ok]                                       # one hourly value per neighbour
            fcnt[j, gi, ok] += 1


def main():
    N, K = len(NODES), len(CLASSES)
    cs = ClassSum(N, K)
    load, lcnt = np.zeros((N, H)), np.zeros((N, H))
    price, pcnt = np.zeros((N, H)), np.zeros((N, H))
    flow, fcnt = np.zeros((N, N, H)), np.zeros((N, N, H))
    prices = json.load(open("data/processed/gnn_prices_monthly.json", encoding="utf8"))
    gbp = {m: v["gbp_per_eur"] for m, v in prices.items()}
    for cc in NODES:
        if cc == "GB":
            continue
        load_ec(cc, cs, load, lcnt, price, pcnt, flow, fcnt, NODES.index(cc))
        print("loaded", cc, flush=True)
    load_gb(cs, price, pcnt, flow, fcnt, gbp)
    gen = cs.result()
    print("loaded GB", flush=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        load = np.where(lcnt > 0, load / lcnt, np.nan)
        price = np.where(pcnt > 0, price / pcnt, np.nan)
        sflow = np.where(fcnt > 0, flow / fcnt, np.nan)      # signed j -> i as reported by importer i
    # Combine the two reports of each border into one signed series F[i, j] = flow i -> j (MW)
    F = np.full((N, N, H), np.nan)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            a = sflow[i, j]           # reported by j: + = import into j from i  => i -> j
            b = -sflow[j, i]          # reported by i: + = import into i from j  => -(i -> j)
            F[i, j] = np.where(np.isfinite(a) & np.isfinite(b), 0.5 * (a + b), np.where(np.isfinite(a), a, b))
    edges = [(NODES.index(a), NODES.index(b)) for a, b in TOPOLOGY]
    # GB load from balance: generation + net import
    gi = NODES.index("GB")
    gb_net_imp = np.nansum(np.stack([F[j, gi] for j in range(N) if j != gi]), axis=0)
    load[gi] = np.nansum(gen[gi], axis=0) + gb_net_imp
    ef = np.array([EF[c] for c in CLASSES])
    emis = np.nansum(gen * ef[None, :, None], axis=1)
    months = [datetime.fromtimestamp(T0 + 3600 * h, tz=timezone.utc).strftime("%Y-%m") for h in range(H)]
    um = sorted(set(months))
    glob_feats = np.array([[prices[m]["eua_eur"], prices[m]["gb_carbon_eur"], prices[m]["gas_eur_mwh_th"],
                            prices[m]["coal_eur_mwh_th"]] for m in um])
    gidx = np.array([um.index(m) for m in months])
    out = pathlib.Path("data/processed/gnn")
    out.mkdir(parents=True, exist_ok=True)
    sfx = "" if DATASET == "v2" else f"_{DATASET}"
    np.savez_compressed(out / f"dataset{sfx}.npz", gen=gen.astype(np.float32), load=load.astype(np.float32),
                        price=price.astype(np.float32), flow=np.nan_to_num(F).astype(np.float32),
                        flow_ok=np.isfinite(F), emis=emis.astype(np.float32),
                        glob=glob_feats[gidx].astype(np.float32), t0=T0)
    meta = dict(nodes=NODES, classes=CLASSES, ef=EF, edges=[(NODES[i], NODES[j]) for i, j in edges], hours=H,
                t0="2019-01-01T00:00Z", glob=["eua_eur", "gb_carbon_eur", "gas_eur_mwh_th", "coal_eur_mwh_th"],
                coverage={n: dict(load=float(np.isfinite(load[i]).mean()), price=float(np.isfinite(price[i]).mean()),
                                  gen=float(np.isfinite(gen[i]).any(axis=0).mean())) for i, n in enumerate(NODES)})
    (out / f"meta{sfx}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(len(edges), "edges"); print(json.dumps(meta["coverage"], indent=0)[:1500])


if __name__ == "__main__":
    main()
