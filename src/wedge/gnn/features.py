"""Feature tensors for the extended MG-STGNN (A49), built from data/processed/gnn/dataset.npz.

All inputs at window step tau use information available by hour tau; the price enters lagged by one hour, so the
target-hour price is never an input. The dispatch head receives the observed residual that dispatchable classes
supply at the target hour (the sum of their generation), and learns how it is split across classes.
"""
from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone

import os

import numpy as np

P = pathlib.Path("data/processed/gnn")
# Specification. v1 (first draft) fed the CBAM cost and a post-June-2023 regime indicator, both identically zero in the
# training years, so their effect was never identified by data. v2 (main) drops both: the charge enters only the
# accounting, and the carbon-cost channel carries the exporter's domestic carbon cost and the carbon-cost differential,
# which vary throughout 2019-2022.
SPEC = os.environ.get("WEDGE_GNN_SPEC", "v2")
# v4 = v2 model specification on dataset v4 (boundary closed, GB interconnector mapping fixed; A53)
DSFX = "_v4" if SPEC == "v4" else ""
RUNS = pathlib.Path("data/processed/gnn/runs" if SPEC == "v1" else f"data/processed/gnn/runs_{SPEC}")
DISP = ["lignite", "coal", "gas", "oil", "hydro_res", "pumped"]
NONDISP = ["nuclear", "biomass", "waste", "hydro_ror", "wind", "solar", "other"]
E_DEC = {"GB": 0.430, "RS": 1.041}
CERT_Q = {(2026, 1): 75.36, (2026, 2): 75.28, (2026, 3): 75.28}       # Q3 2026 not yet published: Q2 carried
SPLITS = {"train": ("2019-01-01", "2023-01-01"), "val": ("2023-01-01", "2023-07-01"),
          "test": ("2023-07-01", "2025-01-01"), "y2025": ("2025-01-01", "2026-01-01"),
          "charged": ("2026-01-01", "2026-07-01"), "charged_q3": ("2026-07-01", "2026-09-01")}


def hour_of(s):
    return int((datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()
                - datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp()) // 3600)


class Features:
    def __init__(self):
        d = np.load(P / f"dataset{DSFX}.npz")
        self.meta = json.load(open(P / f"meta{DSFX}.json", encoding="utf8"))
        nodes, classes = self.meta["nodes"], self.meta["classes"]
        self.nodes, self.N = nodes, len(nodes)
        gen, load, price, flow = d["gen"], d["load"], d["price"], d["flow"]
        H = gen.shape[-1]
        self.H = H
        ci = {c: i for i, c in enumerate(classes)}
        self.ef_disp = np.array([self.meta["ef"][c] for c in DISP], np.float32)
        self.ef_all = np.array([self.meta["ef"][c] for c in classes], np.float32)
        disp = np.nan_to_num(gen[:, [ci[c] for c in DISP]])                       # (N, D, H) MW
        self.disp_obs_ok = np.isfinite(gen[:, [ci[c] for c in DISP]]).any(1)        # (N, H)
        self.disp_mask = (np.nanmax(np.nan_to_num(gen[:, [ci[c] for c in DISP]]), axis=2) > 50).astype(np.float32)
        nondisp = np.nansum(gen[:, [ci[c] for c in NONDISP]], 1)
        vre = np.nansum(gen[:, [ci["wind"], ci["solar"]]], 1)
        self.gen_disp = disp / 1000.0                                               # GW, targets
        self.resid = disp.sum(1) / 1000.0                                           # GW
        self.emis_nondisp = np.nansum(np.nan_to_num(gen) * self.ef_all[None, :, None], 1) \
            - (disp * self.ef_disp[None, :, None]).sum(1)                           # t/h from non-disp classes
        # edges (directed both ways)
        idx = {n: i for i, n in enumerate(nodes)}
        und = [(idx[a], idx[b]) for a, b in self.meta["edges"]]
        self.src = np.array([a for a, b in und] + [b for a, b in und], dtype=np.int64)
        self.dst = np.array([b for a, b in und] + [a for a, b in und], dtype=np.int64)
        self.F = flow                                                               # F[i, j] = i -> j (MW)
        self.cap = {}
        for a, b in und:
            self.cap[(a, b)] = self.cap[(b, a)] = max(float(np.percentile(np.abs(flow[a, b]), 99)), 50.0)
        netimp = np.zeros((self.N, H), np.float32)
        for a, b in und:
            netimp[b] += flow[a, b]
            netimp[a] -= flow[a, b]
        self.netimp = netimp / 1000.0
        # prices and costs
        g = np.nan_to_num(d["glob"])                                                               # (H, 4) eua, gb carbon, gas, coal
        if SPEC == "v4":
            # A55: a zone whose day-ahead price is missing in most training hours (ME, BA, MK, IE) gets no price at
            # all, so the price input and target are identically absent in training and inference; otherwise its
            # standardisation would come from a handful of hours and later prices would be unseen inputs.
            tr_end = hour_of(SPLITS["train"][1])
            cov = np.isfinite(price[:, :tr_end]).mean(1)
            price = np.where((cov < 0.5)[:, None], np.nan, price)
            self.price_dropped = [nodes[i] for i in np.where(cov < 0.5)[0]]
        self.price = price                                                          # EUR/MWh (NaN where missing)
        carbon = np.tile(g[:, 0], (self.N, 1))
        carbon[idx["GB"]] = g[:, 1]
        carbon[idx["RS"]] = 4.0
        for n in ("BA", "ME", "MK"):                  # no carbon price in dispatch (non-EU Western Balkans)
            if n in idx:
                carbon[idx[n]] = 0.0
        self.carbon = carbon
        t = datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp() + 3600 * np.arange(H)
        dts = [datetime.fromtimestamp(x, tz=timezone.utc) for x in t]
        cbam = np.zeros((self.N, H), np.float32)
        for h, dt in enumerate(dts):
            q = CERT_Q.get((dt.year, (dt.month - 1) // 3 + 1))
            if q:
                for n, e in E_DEC.items():
                    cbam[idx[n], h] = q * e
        self.cbam = cbam
        hod = np.array([dt.hour for dt in dts])
        dow = np.array([dt.weekday() for dt in dts])
        ym = np.array([dt.year * 100 + dt.month for dt in dts])
        regime = np.stack([ym < 202110, (ym >= 202110) & (ym <= 202306), ym > 202306], 0).astype(np.float32)
        cal = np.stack([np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24),
                        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)], 0).astype(np.float32)
        plag = np.concatenate([np.full((self.N, 1), np.nan), price[:, :-1]], 1)
        # standardisation (train statistics, per node)
        tr = slice(0, hour_of(SPLITS["train"][1]))

        def z(a):
            with np.errstate(all="ignore"):
                m = np.nan_to_num(np.nanmean(a[:, tr], 1, keepdims=True))
                s = np.nan_to_num(np.nanstd(a[:, tr], 1, keepdims=True), nan=1.0) + 1e-6
            s = np.where(s < 1e-3, 1.0, s)
            return np.nan_to_num((a - m) / s).astype(np.float32), m, s
        self.z_load, *_ = z(load / 1000.0)
        self.z_vre, *_ = z(vre / 1000.0)
        self.z_nd, *_ = z(nondisp / 1000.0)
        self.z_resid, self.rm, self.rs = z(self.resid)
        self.z_ni, _, self.ni_s = z(self.netimp)
        self.z_plag, *_ = z(plag)
        self.z_price, self.pm, self.ps = z(price)
        self.price_ok = np.isfinite(price)
        glob_z = (g - np.nanmean(g[tr], 0)) / (np.nanstd(g[tr], 0) + 1e-6)
        self.z_carbon = ((carbon - 60.0) / 30.0).astype(np.float32)
        self.z_cbam = (cbam / 50.0).astype(np.float32)
        self.gasz = np.tile(glob_z[:, 2], (self.N, 1)).astype(np.float32)
        self.coalz = np.tile(glob_z[:, 3], (self.N, 1)).astype(np.float32)
        self.cal, self.regime = cal, regime
        self.ptilde = np.nan_to_num(plag / 100.0).astype(np.float32)
        self.caps = np.array([self.cap[(a, b)] for a, b in zip(self.src, self.dst)], np.float32)
        allh = np.arange(H)
        self.X = self.node_x(allh)                     # (H, N, node_in)
        self.MO = self.mo_x(allh)                      # (H, N, mo_in)
        self.TA, self.CC = self.edge_x(allh)           # (H, E, *)

    def node_x(self, hs, resid_override=None, ni_override=None):
        """(len(hs), N, node_in) features at hours hs."""
        r = self.z_resid[:, hs] if resid_override is None else resid_override
        ni = self.z_ni[:, hs] if ni_override is None else ni_override
        cols = [self.z_load[:, hs], self.z_vre[:, hs], self.z_nd[:, hs], r, ni, self.z_plag[:, hs],
                self.z_carbon[:, hs], self.z_cbam[:, hs], self.gasz[:, hs], self.coalz[:, hs]]
        if SPEC != "v1":
            cols = [self.z_load[:, hs], self.z_vre[:, hs], self.z_nd[:, hs], r, ni, self.z_plag[:, hs],
                    self.z_carbon[:, hs], self.gasz[:, hs], self.coalz[:, hs]]
            cols += [np.tile(c[hs], (self.N, 1)) for c in self.cal]
            return np.stack(cols, -1).transpose(1, 0, 2)
        cols += [np.tile(c[hs], (self.N, 1)) for c in self.cal] + [np.tile(c[hs], (self.N, 1)) for c in self.regime]
        return np.stack(cols, -1).transpose(1, 0, 2)

    def mo_x(self, hs):
        """Merit-order channel inputs: own carbon cost, CBAM cost, fuel prices and the previous hour's shares of the
        dispatchable classes (S0 feeds the generation mix; lagged here so the target hour's mix is never an input)."""
        with np.errstate(all="ignore"):
            sh = np.nan_to_num(self.gen_disp / np.maximum(self.resid[:, None, :], 1e-3))       # (N, D, H)
        hl = np.maximum(np.asarray(hs) - 1, 0)
        cols = [self.z_carbon[:, hs], self.z_cbam[:, hs], self.gasz[:, hs], self.coalz[:, hs]]
        if SPEC != "v1":
            cols = [self.z_carbon[:, hs], self.gasz[:, hs], self.coalz[:, hs]]
        cols += [sh[:, k, hl] for k in range(sh.shape[1])]
        return np.stack(cols, -1).transpose(1, 0, 2).astype(np.float32)

    def edge_x(self, hs, flow_override=None):
        Fh = self.F[:, :, hs] if flow_override is None else flow_override
        f = Fh[self.src, self.dst] / self.caps[:, None]                         # (E, T) signed src -> dst
        dp = self.ptilde[self.dst][:, hs] - self.ptilde[self.src][:, hs]
        ta = np.stack([dp, np.abs(f), f], -1).transpose(1, 0, 2)
        if SPEC == "v1":
            cc = np.stack([self.z_cbam[self.src][:, hs], np.clip(f, 0, None)], -1).transpose(1, 0, 2)
        else:
            dc = self.z_carbon[self.dst][:, hs] - self.z_carbon[self.src][:, hs]
            cc = np.stack([self.z_carbon[self.src][:, hs], dc, np.clip(f, 0, None)], -1).transpose(1, 0, 2)
        return ta.astype(np.float32), cc.astype(np.float32)

    def sample(self, t, W=24):
        s = slice(t - W + 1, t + 1)
        return self.X[s], self.MO[s], self.TA[s], self.CC[s]
