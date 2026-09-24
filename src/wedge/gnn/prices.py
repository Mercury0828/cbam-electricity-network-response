"""Monthly fuel and carbon price panel, January 2019 - August 2026, for the graph model (A49).

Sources (all archived under data/inputs/source/):
  EUA   EEX primary-market auction reports, mean auction clearing price of the month (EUR/t).
  UKA   DESNZ Cost Containment Mechanism full table, monthly average prices (GBP/t), May 2021 onwards.
        GB was in the EU ETS until December 2020, so 2019-2020 GB allowance cost = EUA. The UK ETS had no
        secondary price before its first auction (19 May 2021); January-April 2021 use the May 2021 average
        (declared assumption, 4 months).
  CPS   18 GBP/t throughout (unchanged since 2016).
  FX    ECB reference rates, monthly averages (GBP per EUR, USD per EUR).
  Fuel  World Bank Pink Sheet (September 2026 vintage): "Natural gas, Europe" ($/mmbtu -> EUR/MWh_th,
        1 mmbtu = 0.29307 MWh) and "Coal, South African" ($/t -> EUR/MWh_th, 6,000 kcal/kg = 6.978 MWh/t),
        the same series as build_inputs.py.
Output: data/processed/gnn_prices_monthly.json  {"YYYY-MM": {...}}
"""
from __future__ import annotations

import csv
import glob
import html
import io
import json
import pathlib
import re
import urllib.request
from collections import defaultdict

import pandas as pd

SRC = pathlib.Path("data/inputs/source")
CPS_GBP = 18.0
MONTHS = [f"{y}-{m:02d}" for y in range(2019, 2027) for m in range(1, 13) if (y, m) <= (2026, 8)]
CCM_URL = ("https://www.gov.uk/government/publications/taking-part-in-the-uk-emissions-trading-scheme-markets/"
           "cost-containment-mechanism-ccm-trigger-prices-and-average-monthly-prices-full-table")
MON = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
MON.update({"June": 6, "July": 7, "Sept": 9})


def eua():
    acc = defaultdict(list)
    for f in sorted(glob.glob(str(SRC / "eex" / "*.xls*"))):
        df = pd.read_excel(f, header=None)
        hdr = next(i for i in range(15) if "Date" in [str(x) for x in df.iloc[i].tolist()])
        cols = [str(x) for x in df.iloc[hdr].tolist()]
        di = cols.index("Date")
        pi = next(i for i, c in enumerate(cols) if c.startswith("Auction Price"))
        for _, r in df.iloc[hdr + 1:].iterrows():
            d, p = r.iloc[di], r.iloc[pi]
            try:
                v = float(p)
                if v == v:                               # cancelled auctions carry no price
                    acc[pd.Timestamp(d).strftime("%Y-%m")].append(v)
            except (ValueError, TypeError):
                continue
    return {m: sum(v) / len(v) for m, v in acc.items()}


def uka(cache=SRC / "desnz_ccm_full_table.html"):
    if not cache.exists():
        cache.write_bytes(urllib.request.urlopen(urllib.request.Request(CCM_URL, headers={"User-Agent": "Mozilla/5.0"}),
                                                 timeout=60).read())
    t = cache.read_text(encoding="utf8", errors="ignore")
    out = {}
    for r in re.findall(r"<tr>(.*?)</tr>", t, re.S):
        c = [html.unescape(re.sub("<[^>]+>", "", x)).strip() for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
        if len(c) < 6 or not re.match(r"[A-Z][a-z]{2}-\d\d$", c[0]):
            continue
        yy = 2000 + int(c[0][-2:])
        mm = MON[c[0][:3]]
        names = [x.strip() for x in re.split(r",|\band\b", c[3]) if x.strip()]
        vals = c[4:4 + len(names)]
        # monitoring months precede the row month; walk back from the row month
        k = len(names)
        for j, (nm, v) in enumerate(zip(names, vals)):
            back = k - j
            y2, m2 = yy, mm - back
            while m2 <= 0:
                m2 += 12
                y2 -= 1
            num = re.sub(r"[^\d.]", "", v)
            if num:
                out[f"{y2}-{m2:02d}"] = float(num)
    return out


def ecb(cur):
    u = (f"https://data-api.ecb.europa.eu/service/data/EXR/M.{cur}.EUR.SP00.A?format=csvdata"
         f"&startPeriod=2019-01&endPeriod=2026-08")
    cache = SRC / f"ecb_{cur}_EUR_monthly.csv"
    if not cache.exists():
        cache.write_bytes(urllib.request.urlopen(u, timeout=60).read())
    rows = csv.DictReader(io.StringIO(cache.read_text(encoding="utf8")))
    return {r["TIME_PERIOD"]: float(r["OBS_VALUE"]) for r in rows}


def worldbank():
    df = pd.read_excel(SRC / "wb_cmo_monthly_2026-09.xlsx", sheet_name="Monthly Prices", header=None)
    hdr = df.iloc[4].tolist()
    gi = next(i for i, c in enumerate(hdr) if isinstance(c, str) and c.startswith("Natural gas, Europe"))
    ci = next(i for i, c in enumerate(hdr) if isinstance(c, str) and c.startswith("Coal, South African"))
    out = {}
    for _, r in df.iloc[6:].iterrows():
        k = str(r.iloc[0])
        if re.match(r"\d{4}M\d\d$", k) and int(k[:4]) >= 2019:
            out[f"{k[:4]}-{k[5:]}"] = (float(r.iloc[gi]), float(r.iloc[ci]))
    return out


def main():
    e, u, gbp, usd, wb = eua(), uka(), ecb("GBP"), ecb("USD"), worldbank()
    panel = {}
    for m in MONTHS:
        y = int(m[:4])
        uk_gbp = None
        if y >= 2021:
            uk_gbp = u.get(m) if m >= "2021-05" else u.get("2021-05")
        uka_eur = e[m] if y <= 2020 else uk_gbp / gbp[m]
        gas_usd, coal_usd = wb[m]
        panel[m] = dict(
            eua_eur=round(e[m], 4), uka_eur=round(uka_eur, 4), cps_eur=round(CPS_GBP / gbp[m], 4),
            gb_carbon_eur=round(uka_eur + CPS_GBP / gbp[m], 4), gbp_per_eur=gbp[m], usd_per_eur=usd[m],
            gas_eur_mwh_th=round(gas_usd / usd[m] / 0.29307, 4),
            coal_eur_mwh_th=round(coal_usd / usd[m] / 6.978, 4),
            uka_source="EUA (GB in EU ETS)" if y <= 2020 else ("UKA May 2021 (pre-auction months)" if m < "2021-05" else "DESNZ CCM"))
    p = pathlib.Path("data/processed/gnn_prices_monthly.json")
    p.write_text(json.dumps(panel, indent=1), encoding="utf-8")
    for m in ("2019-01", "2021-03", "2022-08", "2025-06", "2026-03", "2026-08"):
        print(m, panel[m])


if __name__ == "__main__":
    main()
