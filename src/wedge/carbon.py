"""Carbon prices in dispatch, one place (A37).

🔴 A37 (input correction): GB generators pay the UK ETS
allowance price AND the Carbon Price Support rate of Climate Change Levy (CPS, GBP 18/tCO2 policy
rate; per-fuel rates set by HMRC - round gb-carbon-price-support). The pre-A37 code used UKA only,
understating GB marginal carbon cost. Dispatch now uses UKA + CPS. The UKA-only value is kept
separately because a carbon cost in dispatch is NOT automatically the legally creditable Art. 9
amount; credit scenarios are modelled separately (tau.py).

The source table data/processed/carbon_monthly_2026.json is not modified; the GB dispatch price is
derived here.
"""
from __future__ import annotations

import json
import pathlib

CPS_GBP_PER_T = 18.0          # GBP/tCO2, frozen policy rate (verify: round gb-carbon-price-support)
CPS_ON = True                 # False reproduces pre-A37 numbers exactly


def load_c26() -> dict:
    c = json.loads(pathlib.Path("data/processed/carbon_monthly_2026.json").read_text(encoding="utf-8"))
    for m in c.values():
        m["cps_eur"] = CPS_GBP_PER_T * m["gbp_eur"] if CPS_ON else 0.0
        m["gb_dispatch_eur"] = m["uka_eur"] + m["cps_eur"]
    return c
