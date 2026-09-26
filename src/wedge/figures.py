"""Paper figures from frozen artefacts (self-check T6). Vector PDF + PNG at final Elsevier sizes.

Colour with B/W redundancy: Okabe-Ito colour-blind-safe palette, and series also differ by marker, fill and line
style so that a greyscale print stays readable. Final widths:
single column 90 mm, double column 190 mm; fonts >= 7 pt at final size.
All numbers are read from data/processed/*.json; nothing is typed in.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import matplotlib.patheffects as pe                                      # noqa: E402

P = pathlib.Path("data/processed")
OUT = pathlib.Path("paper/figs")
MM = 1 / 25.4
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.prop_cycle": matplotlib.cycler(color=["k"]),
                     "font.family": "DejaVu Sans", "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
GREYS = ["#1a1a1a", "#595959", "#999999", "#d0d0d0"]
# Okabe-Ito palette; one colour per border and per estimator throughout the paper
OI = {"blue": "#0072B2", "orange": "#E69F00", "verm": "#D55E00", "green": "#009E73", "sky": "#56B4E9",
      "purple": "#CC79A7", "yellow": "#F0E442", "black": "#000000"}
BCOL = {"GB->NL": OI["blue"], "GB->BE": OI["orange"], "RS->HU": OI["verm"], "RS->HU (model levels)": OI["verm"]}
ECOL = {"graph": OI["blue"], "redispatch": OI["orange"], "regression": OI["black"]}
HATCH = ["", "///", "...", "xx"]
NAMES = {"DE": "Germany", "PL": "Poland", "BE": "Belgium", "NL": "Netherlands", "CZ": "Czechia", "BG": "Bulgaria",
         "AT": "Austria", "FR": "France", "IT": "Italy", "BA": "Bosnia and Herz.", "RO": "Romania", "SK": "Slovakia"}


def load(name):
    return json.loads((P / name).read_text(encoding="utf-8"))


def save(fig, name, tight=True):
    OUT.mkdir(parents=True, exist_ok=True)
    kw = {} if tight else {"bbox_inches": fig.bbox_inches}
    fig.savefig(OUT / f"{name}.pdf", **kw)
    fig.savefig(OUT / f"{name}.png", dpi=300, **kw)
    plt.close(fig)


def fig_regime_map():
    d = load("regime_map.json")
    s, px = np.array(d["grid"]["s"]), np.array(d["grid"]["p_x"])
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 70 * MM), sharey=True)
    for ax, (b, lab) in zip(axes, (("GB->NL", "GB→NL"), ("RS->HU (model levels)", "RS→HU"))):
        z = np.array([[c["material_share"] for c in row] for row in d["map"][b]])
        zero = np.array([[c["zero_action_share"] for c in row] for row in d["map"][b]])
        im = ax.pcolormesh(px, s, z, cmap="viridis", vmin=0, vmax=1, shading="nearest")
        cs = ax.contour(px, s, zero, levels=[0.5], colors="k", linestyles="--", linewidths=1.0)
        for c in getattr(cs, "collections", []) or []:
            c.set_path_effects([pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()])
        if not getattr(cs, "collections", None):
            cs.set_path_effects([pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()])
        ax.set_title(lab)
        ax.set_xlabel("Exporter dispatch carbon cost (EUR/t)")
        if b.startswith("GB"):
            for (x, y, m, t) in ((87, 75, "o", "2026"), (66, 75, "s", "CPS removed"),
                                 (75, 75, "^", "Convergence")):
                ax.plot(x, y, m, ms=5, mfc="white", mec="k", mew=0.9, label=t)
            ax.plot([], [], "k--", lw=0.9, label="Zero charge on half of volume")
            fig.legend(loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.45, -0.1))
    axes[0].set_ylabel("Carbon valuation (EUR/t)")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("Share of volume with material\nimporter information")
    save(fig, "fig_regime_map")


def fig_rule_regret():
    d = load("regime_map.json")["scenarios"]["GB->NL"]
    states = [("S26", "2026"), ("W27", "Matched\nprices"), ("NOCPS", "CPS removed"),
              ("CONV", "Price\nconvergence")]
    rules = ["zero charge", "gross default", "default + credit", "all-sources proxy", "all-sources proxy + credit"]
    labels = ["Zero charge", "Gross default", "Default + credit", "All-sources proxy", "All-sources proxy + credit"]
    marks = ["v", "o", "s", "^", "D"]
    cols = [OI["black"], OI["verm"], OI["blue"], OI["orange"], OI["green"]]
    fig, ax = plt.subplots(figsize=(190 * MM, 66 * MM))
    x = np.arange(len(states))
    wdt = 0.15
    for i, (r, lab) in enumerate(zip(rules, labels)):
        if r == "zero charge":                     # regret of a = 0 is the non-negative benchmark itself
            lo = [d[s]["feasible_oracle_u"][0] for s, _ in states]
            hi = [d[s]["feasible_oracle_u"][1] for s, _ in states]
        else:
            lo = [d[s][r]["regret"][0] for s, _ in states]
            hi = [d[s][r]["regret"][1] for s, _ in states]
        xs = x + (i - 2) * wdt
        ax.vlines(xs, lo, hi, color=cols[i], lw=1.8)
        ax.plot(xs, (np.array(lo) + np.array(hi)) / 2, marks[i], color=cols[i], mfc="white", mew=1.1, ms=4.5,
                label=lab)
    ax.set_xticks(x, [t for _, t in states])
    ax.set_ylabel("Regret range across\nscenarios (EUR/MWh)")
    ax.set_ylim(-1, 34)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, handletextpad=0.6,
              columnspacing=1.6)
    save(fig, "fig_rule_regret")


def fig_decomposition():
    d = load("benchmark_2026h1.json")["s=cert"]
    fig, ax = plt.subplots(figsize=(84 * MM, 60 * MM))
    terms = [("factor", "Declaration-factor\nmismatch"), ("credit", "Credit\nmismatch"),
             ("importer", "Importer valuation\n/ timing")]
    for j, (b, mk) in enumerate((("GB->NL", "o"), ("GB->BE", "s"), ("RS->HU", "^"))):
        dc = d[b]["rules"]["G"]["decomposition"]
        for i, (k, _) in enumerate(terms):
            lo, hi = dc[k]
            y = i + (j - 1) * 0.22
            ax.plot([lo, hi], [y, y], "-", color=BCOL[b], lw=1.4)
            ax.plot([(lo + hi) / 2], [y], mk, mfc="white", mec=BCOL[b], mew=1.1, ms=4,
                    label=b.replace("->", "→") if i == 0 else None)
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_yticks(range(3), [t for _, t in terms])
    ax.invert_yaxis()
    ax.set_xlabel("Gross-default gap contribution (EUR/MWh)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig_decomposition")


def fig_trading():
    dist = load("cap_distribution_rev.json")         # A48: revised specification (lag D-2, symmetric months)
    ev = load("cap_event_study_rev.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(186 * MM, 60 * MM), gridspec_kw={"wspace": 0.3})
    th = sorted(dist, key=float)
    b = [dist[t]["beta"] * 100 for t in th]
    lo = [dist[t]["ci95"][0] * 100 for t in th]
    hi = [dist[t]["ci95"][1] * 100 for t in th]
    x = np.arange(len(th))
    a1.errorbar(x, b, yerr=[np.array(b) - lo, np.array(hi) - b], fmt="o", mfc="white", mec=OI["blue"],
                ecolor=OI["blue"], capsize=2, ms=4, mew=1.1)
    a1.axhline(0, color="k", lw=0.5, ls=":")
    a1.set_xticks(x, [f">{float(t):g}" for t in th])
    a1.set_xlabel("Capacity price threshold (EUR/MWh)")
    a1.set_ylabel("Change in P(price > c), GB→EU\nrelative to EU→GB (pp)")
    a1.set_title("(a)")
    ms = sorted(ev, key=int)
    m = np.array([int(k) for k in ms])
    bb = np.array([ev[k]["beta"] for k in ms])
    l2 = np.array([ev[k]["ci95"][0] for k in ms])
    h2 = np.array([ev[k]["ci95"][1] for k in ms])
    a2.fill_between(m, l2, h2, color=OI["sky"], alpha=0.35, lw=0)
    a2.plot(m, bb, "o-", color=OI["blue"], mfc="white", ms=3.5, lw=0.9)
    a2.axvline(12.5, color="k", lw=0.8, ls="--")
    a2.axhline(0, color="k", lw=0.5, ls=":")
    a2.set_xticks([1, 7, 13, 19], ["Jan 2025", "Jul 2025", "Jan 2026", "Jul 2026"])
    a2.set_ylabel("Outward − inward valuation,\nvs Dec 2025 (EUR/MWh)")
    a2.set_title("(b)")
    save(fig, "fig_trading")


def fig_validation():
    v = load("val_regress.json")
    mg, mr = v["model_GB_NL_2026H1"], v["model_RS_HU_2026H1"]
    rows = [("GB exporter", "GB_bx", mg["e_out_mean"], 1, mg["e_out_range"]),
            ("NL importer", "NL_bm", mg["e_in_mean"], -1, mg["e_in_range"]),
            ("RS exporter", "RS_bx", mr["e_out_mean"], 1, mr["e_out_range"]),
            ("HU importer", "HU_bm", mr["e_in_mean"], -1, mr["e_in_range"])]
    fig, ax = plt.subplots(figsize=(84 * MM, 66 * MM))
    for i, (lab, key, model, sgn, rng) in enumerate(rows):
        hk = f"{key}_2026H1" if key in ("GB_bx", "NL_bm") else f"{key}_2026"
        for k, (res_key, mk, off) in enumerate(((hk, "o", -0.15), (f"{key}_daily_2026", "s", 0.15))):
            c = v[res_key]
            val, lo, hi = sgn * c["coef"], sorted([sgn * c["ci95"][0], sgn * c["ci95"][1]])[0], \
                sorted([sgn * c["ci95"][0], sgn * c["ci95"][1]])[1]
            ax.errorbar(val, i + off, xerr=[[val - lo], [hi - val]], fmt=mk, mfc="white", mec=ECOL["regression"],
                        color=ECOL["regression"], ecolor=ECOL["regression"], capsize=2, ms=4,
                        label=("Regression, hourly" if k == 0 else "Regression, daily") if i == 0 else None)
        ax.plot(rng, [i, i], color=ECOL["redispatch"], lw=2.2, alpha=0.35, solid_capstyle="butt",
                label="Model scenario range" if i == 0 else None)
        ax.plot(model, i, "D", color=ECOL["redispatch"], ms=4, label="Redispatch model, scenario mean" if i == 0 else None)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Marginal emissions (t CO₂ per MWh traded)")
    ax.set_xlim(-0.05, 1.25)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=2, handletextpad=0.6,
              columnspacing=1.0, fontsize=7)
    save(fig, "fig_validation")


def fig_frontier():
    d = load("regime_map.json")
    s = np.array(d["grid"]["s"])
    px_idx = {"GB->NL": list(d["grid"]["p_x"]).index(90.0), "GB->BE": list(d["grid"]["p_x"]).index(90.0),
              "RS->HU (model levels)": list(d["grid"]["p_x"]).index(0.0)}
    fig, ax = plt.subplots(figsize=(90 * MM, 60 * MM))
    for (b, lab, ls, mk) in (("GB->NL", "GB→NL", "-", "o"),
                             ("GB->BE", "GB→BE", "--", "s"),
                             ("RS->HU (model levels)", "RS→HU", ":", "^")):
        r = [d["map"][b][i][px_idx[b]]["mean_R"] for i in range(len(s))]
        ax.plot(s, r, ls, color=BCOL[b], marker=mk, mfc="white", ms=3, lw=1.1, markevery=2, label=lab)
    ax.set_xlabel("Carbon valuation (EUR/t)")
    ax.set_ylabel("Unavoidable regret (EUR/MWh)")
    ax.legend(frameon=False)
    save(fig, "fig_frontier")


def fig_onset():
    d = load("onset_curves.json")
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 62 * MM), sharey=True)
    for ax, side, title in ((axes[0], "exp", "(a) Exports above 5% of capacity"),
                            (axes[1], "imp", "(b) Imports above 5% of capacity")):
        for y, ls, mk, col in ((2025, "--", "o", OI["orange"]), (2026, "-", "s", OI["blue"])):
            c = d[f"{side}_{y}"]
            xs = np.array([float(k) for k in c])
            ys = np.array([v[0] for v in c.values()])
            keep = (xs >= -60) & (xs <= 80)
            ax.plot(xs[keep], ys[keep], ls, color=col, marker=mk, mfc="white", ms=3, lw=0.9, markevery=2,
                    label=f"Jan-Aug {y}")
        ax.set_title(title)
        ax.set_xlabel("Realised spread, EU minus GB (EUR/MWh)")
    axes[0].set_ylabel("Share of hours")
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, "fig_onset")


def fig_three_estimators():
    """E_x and E_m on the charged borders, January-June 2026: graph model (median and 5-95% range of seed x
    MC-dropout draws, volume-weighted), redispatch model (scenario mean and range) and daily regressions."""
    v = load("val_regress.json")
    gnn = load("gnn/v2/gnn_marginal_summary.json")      # A50: main specification v2
    rows = [("GB→NL exporter", "GB->NL", 0, v["model_GB_NL_2026H1"]["e_out_mean"], v["model_GB_NL_2026H1"]["e_out_range"],
             v["GB_bx_daily_2026"]),
            ("GB→NL importer", "GB->NL", 1, v["model_GB_NL_2026H1"]["e_in_mean"], v["model_GB_NL_2026H1"]["e_in_range"],
             v["NL_bm_daily_2026"]),
            ("RS→HU exporter", "RS->HU", 0, v["model_RS_HU_2026H1"]["e_out_mean"], v["model_RS_HU_2026H1"]["e_out_range"],
             v["RS_bx_daily_2026"]),
            ("RS→HU importer", "RS->HU", 1, v["model_RS_HU_2026H1"]["e_in_mean"], v["model_RS_HU_2026H1"]["e_in_range"],
             v["HU_bm_daily_2026"])]
    fig, ax = plt.subplots(figsize=(84 * MM, 70 * MM))
    for i, (lab, b, k, rm, rr, reg) in enumerate(rows):
        g = gnn[b]["charged"]
        key = "e_out" if k == 0 else "E_m"
        gm, (glo, ghi) = g[key]["median"], g[key]["p05_p95"]
        ax.plot([glo, ghi], [i - 0.2, i - 0.2], color=ECOL["graph"], lw=1.4)
        ax.plot(gm, i - 0.2, "o", color=ECOL["graph"], mfc=ECOL["graph"], ms=4, label="Graph model" if i == 0 else None)
        ax.plot(rr, [i, i], color=ECOL["redispatch"], lw=2.4, alpha=0.4, solid_capstyle="butt")
        ax.plot(rm, i, "D", color=ECOL["redispatch"], mfc="white", mew=1.1, ms=4, label="Redispatch model" if i == 0 else None)
        sg = 1 if k == 0 else -1
        c, lo, hi = sg * reg["coef"], *sorted([sg * reg["ci95"][0], sg * reg["ci95"][1]])
        ax.errorbar(c, i + 0.2, xerr=[[c - lo], [hi - c]], fmt="s", mfc="white", mec=ECOL["regression"],
                    ecolor=ECOL["regression"], capsize=2, ms=4, mew=1.1, label="Daily regression" if i == 0 else None)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Marginal emissions (t CO₂ per MWh)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=2, fontsize=7.5)
    save(fig, "fig_three_estimators")


def fig_learned_rule():
    """(a) Worst-case exporter-only bound rho* and (b) realised value of importer information (regret of the learned
    exporter-only rule minus regret of the learned full-graph rule), January-June 2026, by carbon valuation."""
    d = load("gnn/v2/rule_learning.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(180 * MM, 62 * MM), gridspec_kw={"wspace": 0.28})
    svals = [("p_m", "Importer\nprice"), ("cert", "Certificate\nprice"), ("150", "150"), ("200", "200")]
    x = np.arange(len(svals))
    for b, ls, mk in (("GB->NL", "-", "o"), ("GB->BE", "--", "s"), ("RS->HU", ":", "^")):
        lab = b.replace("->", "→")
        a1.plot(x, [d[b][s]["mean_rho_star"] for s, _ in svals], ls, color=BCOL[b], marker=mk, mfc="white", ms=4.5,
                lw=0.9, label=lab)
        a2.plot(x, [d[b][s]["information_value"] for s, _ in svals], ls, color=BCOL[b], marker=mk, mfc="white", ms=4.5,
                lw=0.9, label=lab)
        if b == "RS->HU":
            a2.plot(x, [d[b][s]["information_value_both"] for s, _ in svals], "-.", color=BCOL[b], marker=mk,
                    mfc=BCOL[b], ms=4.5, lw=0.9, label="RS→HU, both responses")
            a1.plot([], [], "-.", color=BCOL[b], marker=mk, mfc=BCOL[b], ms=4.5, lw=0.9,
                    label="RS→HU, both (b)")
    for ax, ttl, yl in ((a1, "(a)", "Bound ρ* (EUR/MWh)"), (a2, "(b)", "Value of importer information (EUR/MWh)")):
        ax.set_xticks(x, [t for _, t in svals])
        ax.set_xlabel("Carbon valuation (EUR/t)")
        ax.set_ylabel(yl)
        ax.set_title(ttl)
        ax.axhline(0, color="k", lw=0.5, ls=":")
    a1.set_ylim(top=a1.get_ylim()[1] * 1.3)
    a1.legend(frameon=False, fontsize=7.5, loc="upper left")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_learned_rule.{ext}", dpi=300, pad_inches=0.06)
    plt.close(fig)


def fig_outage():
    """Interconnector outages (A52/A53): response per MWh of lost GB import, difference-in-slopes against
    pseudo-events with event-bootstrap 95% intervals, dataset v4. (a) where the lost energy is replaced,
    (b) where emissions change."""
    d = load("gnn/outage_did_v4.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(186 * MM, 66 * MM), gridspec_kw={"wspace": 0.55})
    rows1 = [("R_m", "Importer's own plants"), ("NIother_m", "Importer's other links"),
             ("R_x", "GB plants"), ("NIother_x", "GB other links (net import)")]
    rows2 = [("E_GB", "GB"), ("E_m", "Importing zone"), ("E_DE", "Germany"), ("E_PL", "Poland"), ("E_FR", "France")]
    for j, (b, mk) in enumerate((("GB->NL", "o"), ("GB->BE", "s"))):
        r = d[b]
        col = BCOL[b]
        off = -0.13 if j == 0 else 0.13
        lab = b.replace("->", "→")
        for i, (k, _) in enumerate(rows1):
            v = r[k]
            a1.errorbar(v["did"], i + off, xerr=[[v["did"] - v["did_ci95"][0]], [v["did_ci95"][1] - v["did"]]], fmt=mk,
                        color=col, mfc="white", mew=1.1, ms=4, capsize=2, label=lab if i == 0 else None)
        m = b.split("->")[1]
        for i, (k, _) in enumerate(rows2):
            v = r[f"E_{m}"] if k == "E_m" else r[k]
            a2.errorbar(v["did"], i + off, xerr=[[v["did"] - v["did_ci95"][0]], [v["did_ci95"][1] - v["did"]]], fmt=mk,
                        color=col, mfc="white", mew=1.1, ms=4, capsize=2)
    for ax, rows, xl, ttl in ((a1, rows1, "MW per MW of lost import", "(a)"),
                              (a2, rows2, "t CO₂ per MWh of lost import", "(b)")):
        ax.set_yticks(range(len(rows)), [r_[1] for r_ in rows])
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_xlabel(xl)
        ax.set_title(ttl)
    a1.legend(frameon=False, loc="upper left", fontsize=7.5)
    save(fig, "fig_outage")


def fig_outage_v2(fes="A_block", direction=None):
    """Outage responses (outage_v2, dataset v4) with the network-response model evaluated by the identical estimator
    (outage_model_compare). direction: 0 = would-be GB export (L+), 1 = would-be GB import (L-), 2 = pooled.
    Empirical: difference-in-slopes with joint event-bootstrap 95% interval; markers: model slopes."""
    emp = load("gnn/outage_v2_v4.json")
    try:
        mod = load("gnn/outage_model_compare_v4.json")
    except FileNotFoundError:
        mod = None
    rows = [("R_m", "Importer's own plants"), ("NIother_m", "Importer's other links"), ("R_x", "GB plants"),
            ("NIother_x", "GB other links")]
    fig, axes = plt.subplots(1, 2, figsize=(186 * MM, 64 * MM), sharex=True, gridspec_kw={"wspace": 0.08})
    mk = {"graph": ("o", OI["blue"], "Graph network model"), "local": ("^", OI["green"], "Node-local parameters"),
          "one_for_one": ("x", OI["verm"], "Two-zone baseline")}
    # GB->NL: hours in which the link would have carried GB exports (L+); GB->BE: pooled (L+ not identified there)
    dirs = {"GB->NL": 0, "GB->BE": 2} if direction is None else {"GB->NL": direction, "GB->BE": direction}
    for ax, b in zip(axes, ("GB->NL", "GB->BE")):
        direction_b = dirs[b]
        for i, (k, lab) in enumerate(rows):
            e = emp[b]["outcomes"][f"{k}|full"][fes]
            v, (lo, hi) = e["did"][direction_b], e["ci95"][direction_b]
            ax.plot([lo, hi], [i, i], color="k", lw=1.6, solid_capstyle="butt")
            ax.plot(v, i, "s", color="k", mfc="white", mew=1.1, ms=4.5, label="Outage estimate, 95% CI" if i == 0 else None)
            if mod:
                for j, (name, (m_, c_, lab_)) in enumerate(mk.items()):
                    if name in mod[b] and k in mod[b][name]["rows"]:
                        mv = mod[b][name]["rows"][k][fes]["model"][direction_b]
                        ax.plot(mv, i + 0.22 + 0.1 * (j - 1), m_, color=c_, ms=4.5, mew=1.2,
                                label=lab_ if i == 0 else None)
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_yticks(range(len(rows)), [r[1] for r in rows] if b == "GB->NL" else [""] * len(rows))
        ax.invert_yaxis()
        ax.set_title(b.replace("->", "→") + (" (would-be GB exports)" if direction_b == 0 else " (all hours)"))
        ax.set_xlabel("MW per MW of lost flow")
    axes[0].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(1.0, -0.22), ncol=4)
    save(fig, "fig_outage_v2")


def fig_netresp():
    """Network emission responses to the removal of one delivered MWh on each charged border, January-June 2026
    (charged_summary.json): exporter, importer, the three third zones with the largest responses, the other zones and
    the net change, with the range over trained models; two-zone accounting of the same local responses as markers."""
    d = load("gnn/netresp_r2_v4/charged_summary.json")
    fig, axes = plt.subplots(1, 3, figsize=(190 * MM, 62 * MM), sharex=True, gridspec_kw={"wspace": 0.62})
    for ax, (b, lab) in zip(axes, (("GB_NL", "GB→NL"), ("GB_BE", "GB→BE"), ("RS_HU", "RS→HU"))):
        s = d[f"{b}|graph|charged"]
        x, m = b.split("_")
        top = sorted(s["third_by_zone"].items(), key=lambda kv: -abs(kv[1]))[:3]
        rest = s["third"][0] - sum(v for _, v in top)
        rows = [(f"{x} (exporter)", -s["E_x"][0], (-s["E_x"][2], -s["E_x"][1]), -s["two_zone"]["E_x"]),
                (f"{m} (importer)", s["E_m"][0], (s["E_m"][1], s["E_m"][2]), s["two_zone"]["E_m"])]
        rows += [(NAMES.get(z, z), v, None, 0.0) for z, v in top]
        rows += [("Other zones", rest, None, 0.0),
                 ("Net change", s["network"][0], (s["network"][1], s["network"][2]), s["two_zone"]["network"])]
        col = BCOL[b.replace("_", "->")]
        for i, (name, v, rng, tz) in enumerate(rows):
            net = name == "Net change"
            ax.barh(i, v, height=0.62, color="white" if net else col, edgecolor=col if net else "k",
                    lw=1.4 if net else 0.6, alpha=1.0 if net else 0.85)
            if rng:
                ax.plot(rng, [i, i], color="k", lw=1.0)
            ax.plot(tz, i, "D", ms=4.2, mfc="white", mec="k", mew=1.0,
                    label="Two-zone baseline" if i == 0 else None)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_yticks(range(len(rows)), [r_[0] for r_ in rows])
        ax.invert_yaxis()
        ax.set_title(lab)
    axes[1].set_xlabel("t CO₂ per MWh removed")
    from matplotlib.patches import Patch
    h, l_ = axes[0].get_legend_handles_labels()
    h = [Patch(facecolor="white", edgecolor="k", lw=0.8)] + h
    l_ = ["Network model, range over trained models"] + l_
    fig.legend(h, l_, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.14))
    save(fig, "fig_netresp")


def fig_rule_regret_net():
    """Regret of credit rules and of a zero charge in four institutional states, GB->NL, January-June 2026, against
    (a) the network benchmark and (b) the two-zone accounting of the same local responses (net_states.json). Bars:
    range over trained network models; the gross default and the uncredited all-sources proxy lie off scale."""
    d = load("gnn/netresp_r2_v4/net_states.json")["GB->NL|graph"]["states"]
    states = [("S26", "2026"), ("W27", "Matched\nprices"), ("NOCPS", "CPS\nremoved"), ("CONV", "Price\nconvergence")]
    rules = [("zero", "Zero charge", "v", OI["black"]),
             ("default_credit", "Default + credit", "s", OI["blue"]),
             ("default_annual_credit", "Default + annual credit", "o", OI["sky"]),
             ("all_sources_credit", "All-sources proxy + credit", "D", OI["green"])]

    def annual(st):
        return "default_annual_credit_81" if st in ("S26", "W27") else ("default_annual_credit_uka" if st == "NOCPS"
                                                                        else None)
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 64 * MM), sharey=True, gridspec_kw={"wspace": 0.06})
    x = np.arange(len(states))
    for ax, key, ttl in ((axes[0], "regret", "(a) Network benchmark"),
                         (axes[1], "regret_two_zone", "(b) Two-zone baseline")):
        for i, (r, lab, mk, col) in enumerate(rules):
            ys, lo, hi, xs = [], [], [], []
            for j, (st, _) in enumerate(states):
                rr = annual(st) if r == "default_annual_credit" else r
                if rr is None or rr not in d[st][key]:
                    continue
                xs.append(x[j] + (i - 1.5) * 0.17)
                ys.append(d[st][key][rr])
                if key == "regret":
                    lo.append(d[st]["regret_range"][rr][0])
                    hi.append(d[st]["regret_range"][rr][1])
            if key == "regret":
                ax.vlines(xs, lo, hi, color=col, lw=1.6)
            ax.plot(xs, ys, mk, color=col, mfc="white", mew=1.2, ms=4.8, label=lab)
        ax.set_xticks(x, [t for _, t in states])
        ax.set_title(ttl)
        ax.set_ylim(-0.3, 7.2)
    axes[0].set_ylabel("Link-only regret $|a-u|$\n(EUR/MWh)")
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(1.03, -0.2), frameon=False, handletextpad=0.5,
                   columnspacing=1.4)
    save(fig, "fig_rule_regret_net")


def fig_regime_net():
    """Share of January-June 2026 export volume on which network information lowers the regret of a learned
    exporter-only rule by more than 1 EUR/MWh, by carbon valuation and exporter carbon cost (net_frontier.json)."""
    d = load("gnn/netresp_r2_v4/net_frontier.json")
    pr = load("gnn_prices_monthly.json")
    h1 = [f"2026-0{k}" for k in range(1, 7)]
    gb = float(np.mean([pr[k]["gb_carbon_eur"] for k in h1]))
    uka = float(np.mean([pr[k]["uka_eur"] for k in h1]))
    eua = float(np.mean([pr[k]["eua_eur"] for k in h1]))
    cert = 75.32                                      # mean of the Q1 and Q2 2026 certificate prices (75.36, 75.28)
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 70 * MM), sharey=True)
    for ax, (b, lab) in zip(axes, (("GB->NL", "GB→NL"), ("GB->BE", "GB→BE"))):
        g = d[b]["grid"]
        s, px = np.array(g["s"]), np.array(g["p_x"])
        z = np.array(g["share_value_above_1"]).T                  # (s, p_x)
        im = ax.pcolormesh(px, s, z, cmap="viridis", vmin=0, vmax=0.5, shading="nearest")
        ax.axhline(eua, color="white", lw=0.9, ls="--")
        for (xx, yy, m_, t) in ((gb, cert, "o", "2026"), (uka, cert, "s", "CPS removed"), (eua, eua, "^", "Convergence")):
            ax.plot(xx, yy, m_, ms=5, mfc="white", mec="k", mew=0.9, label=t)
        ax.set_title(lab)
        ax.set_xlabel("Exporter dispatch carbon cost (EUR/t)")
    axes[0].plot([], [], "--", color="0.4", lw=0.9, label="EU allowance price")
    axes[0].set_ylabel("Carbon valuation (EUR/t)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.45, -0.1))
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("Share of volume with network\ninformation worth > 1 EUR/MWh")
    save(fig, "fig_regime_net")


def fig_outage_multi(fes="A_block"):
    """Held-out HVDC outages (outage_multi_v4.json): outage estimate (x, with 95% interval) against the response of
    the graph network model and of the two-zone accounting (y), all hours, for the output of the two end zones and
    their net imports on other links."""
    d = load("gnn/outage_multi_v4.json")
    keys = [k for k in d if not k.startswith("_") and "model" in d[k]]
    rows = [("R_a", "o"), ("R_b", "o"), ("NIother_a", "s"), ("NIother_b", "s")]
    fig, axes = plt.subplots(1, 2, figsize=(160 * MM, 74 * MM), sharex=True, sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (name, ttl, col) in zip(axes, (("graph", "(a) Graph network model", OI["blue"]),
                                           ("one_for_one", "(b) Two-zone baseline", OI["verm"]))):
        for key in keys:
            for k, mk in rows:
                e = d[key]["outcomes"][k][fes]
                v, (lo, hi) = e["did"][2], e["ci95"][2]
                m = d[key]["model"][name][k][fes]["model"][2]
                inside = d[key]["model"][name][k][fes]["inside"][2]
                ax.plot(np.clip([lo, hi], -2.28, 2.28), [m, m], color="0.6", lw=0.8, zorder=1)
                ax.plot(v, m, mk, ms=4.2, mec=col, mfc=col if inside else "white", mew=1.0, zorder=2)
        lim = (-2.3, 2.3)
        ax.plot(lim, lim, "k--", lw=0.7)
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.axvline(0, color="k", lw=0.4, ls=":")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_title(ttl)
        ax.set_xlabel("Outage estimate (MW per MW)")
    axes[0].set_ylabel("Model response (MW per MW)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", mec="k", mfc="k", ms=4.2), Line2D([], [], marker="s", ls="", mec="k",
                                                                            mfc="k", ms=4.2),
         Line2D([], [], marker="o", ls="", mec="k", mfc="white", ms=4.2), Line2D([], [], color="0.6", lw=0.8)]
    fig.legend(h, ["Own plants", "Other links", "Open: outside the 95% interval", "95% interval of the estimate"],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    save(fig, "fig_outage_multi")



# ---------------------------------------------------------------------------------------------------------------
# Maps of the study area and of the network responses (Natural Earth 1:50m boundaries, public domain,
# data/inputs/source/naturalearth/; projected to ETRS89-LAEA Europe, EPSG:3035)

NE = pathlib.Path("data/inputs/source/naturalearth")
ZONE_ORDER = ["AT", "BE", "CH", "CZ", "DE", "DK", "ES", "FR", "GB", "HU", "IT", "NL", "NO", "PL", "RS", "SE", "SK",
              "RO", "BG", "HR", "SI", "GR", "BA", "ME", "MK", "IE", "PT"]
# anchor point of each zone (lon, lat): link end points and labels
ANCHOR = {"AT": (14.6, 47.6), "BE": (4.6, 50.6), "CH": (8.1, 46.8), "CZ": (15.4, 49.8), "DE": (10.2, 51.2),
          "DK": (9.6, 56.1), "ES": (-3.6, 40.2), "FR": (2.4, 46.7), "GB": (-1.6, 52.8), "HU": (19.4, 47.2),
          "IT": (12.2, 43.3), "NL": (5.7, 52.3), "NO": (8.8, 60.9), "PL": (19.2, 52.1), "RS": (20.8, 44.0),
          "SE": (15.2, 60.2), "SK": (19.6, 48.7), "RO": (24.9, 45.9), "BG": (25.3, 42.7), "HR": (15.9, 45.3),
          "SI": (14.8, 46.1), "GR": (22.0, 39.4), "BA": (17.8, 44.2), "ME": (19.3, 42.8), "MK": (21.7, 41.6),
          "IE": (-7.9, 53.4), "PT": (-8.0, 39.7)}
CHARGED_LINKS = [("GB", "NL"), ("GB", "BE"), ("GB", "FR"), ("GB", "DK"), ("GB", "IE"),
                 ("RS", "HU"), ("RS", "RO"), ("RS", "BG"), ("RS", "HR")]
HELD_OUT = {("DE", "SE"): "Baltic Cable", ("GR", "IT"): "GRITA", ("NL", "NO"): "NorNed", ("PL", "SE"): "SwePol",
            ("DK", "NL"): "COBRAcable", ("ME", "IT"): "MONITA", ("BE", "DE"): "ALEGrO", ("DE", "NO"): "NordLink",
            ("GB", "NO"): "North Sea Link", ("DK", "GB"): "Viking Link"}
_ZONES = {}


def _zones():
    """Zone polygons in EPSG:3035 (GB = Great Britain; IE = island of Ireland; DE = Germany and Luxembourg) and the
    other countries as background, clipped to Europe."""
    if _ZONES:
        return _ZONES["z"], _ZONES["bg"], _ZONES["to3035"]
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import unary_union
    clip = box(-12.5, 33.5, 36.0, 72.0)
    cty = gpd.read_file(NE / "ne_50m_admin_0_countries.zip")
    sub = gpd.read_file(NE / "ne_50m_admin_0_map_subunits.zip")
    iso = cty.set_index("ISO_A2_EH")["geometry"]
    su = sub.set_index("SU_A3")["geometry"]
    geom = {}
    for z in ZONE_ORDER:
        if z == "GB":
            g = unary_union([su["ENG"], su["SCT"], su["WLS"]])
        elif z == "IE":
            g = unary_union([iso["IE"], su["NIR"]])
        elif z == "DE":
            g = unary_union([iso["DE"], iso["LU"]])
        else:
            g = iso[z]
        geom[z] = g.intersection(clip)
    z = gpd.GeoDataFrame({"zone": list(geom)}, geometry=list(geom.values()), crs="EPSG:4326").to_crs(3035)
    covered = unary_union(list(geom.values()))
    bg = cty[~cty["ISO_A2_EH"].isin(ZONE_ORDER + ["LU"])].copy()
    bg["geometry"] = bg.geometry.intersection(clip).difference(covered)
    bg = bg[~bg.geometry.is_empty].to_crs(3035)
    to3035 = Transformer.from_crs(4326, 3035, always_xy=True)
    _ZONES.update(z=z, bg=bg, to3035=to3035)
    return z, bg, to3035


def _pt(code, to3035):
    return to3035.transform(*ANCHOR[code])


def _clip(gdf, xlim, ylim):
    from shapely.geometry import box
    out = gdf.copy()
    out["geometry"] = out.geometry.intersection(box(xlim[0], ylim[0], xlim[1], ylim[1]))
    return out[~out.geometry.is_empty]


def _hatch(ax, geom, color, spacing=65e3, lw=0.5):
    """Diagonal hatch drawn as line segments clipped to the polygon (no PDF pattern tiles)."""
    from shapely.geometry import LineString
    x0, y0, x1, y1 = geom.bounds
    c = x0 - (y1 - y0)
    while c < x1:
        seg = LineString([(c, y0), (c + (y1 - y0), y1)]).intersection(geom)
        for part in getattr(seg, "geoms", [seg]):
            if part.geom_type == "LineString" and not part.is_empty:
                xs, ys = part.xy
                ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt", zorder=4)
        c += spacing


def _base(ax, bg, xlim, ylim):
    _clip(bg, xlim, ylim).plot(ax=ax, color="#f3f3f3", edgecolor="#d9d9d9", lw=0.4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _link(ax, a, b, to3035, rad=0.0, shrink=7, **kw):
    from matplotlib.patches import FancyArrowPatch
    p = FancyArrowPatch(_pt(a, to3035), _pt(b, to3035), connectionstyle=f"arc3,rad={rad}", shrinkA=shrink,
                        shrinkB=shrink, **kw)
    ax.add_patch(p)
    return p


def _halo(txt):
    txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
    return txt


def _arc_mid(a, b, to3035, rad, t=0.5):
    """Point at parameter t of the arc3 connection between two zone anchors (quadratic Bezier, arc3 control point)."""
    (x1, y1), (x2, y2) = _pt(a, to3035), _pt(b, to3035)
    cx, cy = (x1 + x2) / 2 + rad * (y2 - y1), (y1 + y2) / 2 - rad * (x2 - x1)
    return ((1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2,
            (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2)


HELD_OUT_ORDER = [("GB", "NO"), ("DK", "GB"), ("NL", "NO"), ("DK", "NL"), ("BE", "DE"), ("DE", "NO"), ("DE", "SE"),
                  ("PL", "SE"), ("ME", "IT"), ("GR", "IT")]
HELD_OUT_T = {("NL", "NO"): 0.74, ("DK", "NL"): 0.30, ("DE", "NO"): 0.72}
# side of the arc on which each number sits (+1: towards the bulge of the arc, -1: the other side)
HELD_OUT_SIDE = {("GB", "NO"): 1, ("DK", "GB"): -1, ("NL", "NO"): -1, ("DK", "NL"): -1, ("BE", "DE"): -1,
                 ("DE", "NO"): 1, ("DE", "SE"): -1, ("PL", "SE"): 1, ("ME", "IT"): 1, ("GR", "IT"): -1}


def _bezier(a, b, to3035, rad, n=240):
    """Points of the arc3 connection between two zone anchors, as a quadratic Bezier (x, y arrays)."""
    (x1, y1), (x2, y2) = _pt(a, to3035), _pt(b, to3035)
    cx, cy = (x1 + x2) / 2 + rad * (y2 - y1), (y1 + y2) / 2 - rad * (x2 - x1)
    t = np.linspace(0, 1, n)
    return ((1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2,
            (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2)


def _numbered_arcs(ax, links, to3035, obstacles, rad=0.18, end_gap=120e3, badge_gap=100e3, **kw):
    """Draw each link as a dashed arc with a numbered badge sitting in a gap of the arc. The badge goes to the arc
    point (or a point up to 30 km off the arc) farthest from the obstacle lines (borders, other arcs, badges already
    placed), so no line passes under it."""
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union
    arcs = {k: _bezier(a, b, to3035, rad) for k, (a, b) in enumerate(links)}
    placed = []
    for k, (a, b) in enumerate(links):
        xs, ys = arcs[k]
        seg = np.hypot(np.diff(xs), np.diff(ys))
        s = np.concatenate([[0], np.cumsum(seg)])
        others = unary_union([obstacles] + [LineString(np.column_stack(arcs[j])) for j in arcs if j != k]
                             + [Point(q).buffer(140e3) for q in placed])
        eg = min(end_gap, 0.15 * s[-1])
        best, best_d, best_q = None, -1.0, None
        for i in range(1, len(s) - 1):
            if not (eg + badge_gap < s[i] < s[-1] - eg - badge_gap):
                continue
            tx, ty = xs[i + 1] - xs[i - 1], ys[i + 1] - ys[i - 1]
            tn = np.hypot(tx, ty)
            for off in (0.0, 30e3, -30e3):
                q = (xs[i] + off * ty / tn, ys[i] - off * tx / tn)
                d_ = others.distance(Point(q)) - 0.3 * abs(off)
                if d_ > best_d:
                    best, best_d, best_q = i, d_, q
        sb = s[best]
        for lo, hi in ((eg, sb - badge_gap), (sb + badge_gap, s[-1] - eg)):
            m = (s >= lo) & (s <= hi)
            ax.plot(xs[m], ys[m], **kw)
        placed.append(best_q)
        ax.text(best_q[0], best_q[1], str(k + 1), ha="center", va="center", fontsize=7, zorder=8,
                bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec=kw.get("color", "k"), lw=0.9))


def fig_map_network():
    """Study area: the 27 zones of the graph by the carbon cost in dispatch, the 58 interconnections and the charged
    links of Great Britain and Serbia (a), and the links whose outages test the model (b). Lines join zone points and
    do not trace the cables."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    z, bg, to3035 = _zones()
    edges = load("gnn/meta_v4.json")["edges"]
    group = {n: "eu" for n in ZONE_ORDER}
    group.update(GB="gb", RS="rs", BA="none", ME="none", MK="none")
    fill = {"eu": "#c6d9ea", "gb": "#E69F00", "rs": "#8a3200", "none": "#9a9a9a"}
    xlim, ylim = (2.6e6, 6.3e6), (1.5e6, 4.95e6)
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 118 * MM), gridspec_kw={"wspace": 0.03})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.25)
    for k, ax in enumerate(axes):
        _base(ax, bg, xlim, ylim)
        zc = _clip(z, xlim, ylim)
        for g_, col in fill.items():
            sel = zc[zc["zone"].map(group) == g_]
            sel.plot(ax=ax, color=col if k == 0 else "#c6d9ea", edgecolor="white", lw=0.5)
        if k == 0:                                     # the full network in (a); (b) shows only the tested links
            for a, b in edges:
                _link(ax, a, b, to3035, arrowstyle="-", color="#8c8c8c", lw=0.45, zorder=3)
        for code in ZONE_ORDER:
            x, y = _pt(code, to3035)
            _halo(ax.text(x, y, code, ha="center", va="center", fontsize=7, zorder=6))
    ax = axes[0]
    for a, b in CHARGED_LINKS:
        col = OI["orange"] if a == "GB" else OI["verm"]
        _link(ax, a, b, to3035, rad=0.12, arrowstyle="-|>,head_length=3.2,head_width=1.8", color=col, lw=1.6,
              zorder=5, path_effects=[pe.withStroke(linewidth=3.2, foreground="white")])
    ax.set_title("(a) Zones by carbon cost in dispatch, charged links", fontsize=8)
    ax.legend(handles=[Patch(facecolor=fill["eu"], label="EU allowance price"),
                       Patch(facecolor=fill["gb"], label="UK allowance + Carbon Price Support (GB)"),
                       Patch(facecolor=fill["rs"], label="Nominal carbon cost (RS)"),
                       Patch(facecolor=fill["none"], label="No carbon price in dispatch (BA, ME, MK)"),
                       Line2D([], [], color="#8c8c8c", lw=0.6, label="Zone-pair connection (58)"),
                       Line2D([], [], color=OI["orange"], lw=1.6, marker=">", ms=4, label="Charged export link, GB"),
                       Line2D([], [], color=OI["verm"], lw=1.6, marker=">", ms=4, label="Charged export link, RS")],
              loc="upper left", bbox_to_anchor=(0.02, -0.01), ncol=1, fontsize=7, frameon=False,
              handlelength=2.2, columnspacing=1.2)
    ax = axes[1]
    for a, b in (("GB", "NL"), ("GB", "BE")):
        _link(ax, a, b, to3035, rad=0.12, arrowstyle="-", color=OI["blue"], lw=2.4, zorder=5)
    _link(ax, "RS", "HU", to3035, rad=0.0, arrowstyle="-", color=OI["purple"], lw=2.2, ls=(0, (4, 1.2, 1, 1.2)),
          zorder=5)
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union
    zc, bc = _clip(z, xlim, ylim), _clip(bg, xlim, ylim)
    obstacles = unary_union(list(zc.geometry.boundary) + list(bc.geometry.boundary)
                            + [Point(_pt(c, to3035)).buffer(70e3) for c in ZONE_ORDER]
                            + [LineString(np.column_stack(_bezier(a, b, to3035, r_))) for a, b, r_ in
                               (("GB", "NL", 0.12), ("GB", "BE", 0.12), ("RS", "HU", 0.0))])
    _numbered_arcs(ax, HELD_OUT_ORDER, to3035, obstacles, color=OI["green"], lw=1.4, ls=(0, (3, 1.6)), zorder=5)
    ax.set_title("(b) Links whose outages test the model", fontsize=8)
    key = [Line2D([], [], color=OI["blue"], lw=2.4, label="Focal links, BritNed (GB–NL) and Nemo Link (GB–BE)"),
           Line2D([], [], color=OI["purple"], lw=2.2, ls=(0, (4, 1.2, 1, 1.2)), label="Serbia–Hungary tie lines"),
           Line2D([], [], color=OI["green"], lw=1.4, ls=(0, (3, 1.6)), label="Held-out HVDC links, outages from July 2023:")]
    leg = ax.legend(handles=key, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=1, fontsize=7, frameon=False,
                    handlelength=2.2)
    names = [f"{n} {HELD_OUT[(a, b)]} ({a}–{b})" for n, (a, b) in enumerate(HELD_OUT_ORDER, start=1)]
    for c, col in enumerate((names[:5], names[5:])):
        ax.text(0.08 + 0.48 * c, -0.165, "\n".join(col), transform=ax.transAxes, ha="left", va="top", fontsize=7,
                linespacing=1.25)
    save(fig, "fig_map_network", tight=False)


# label position of each zone on the response maps (lon, lat); NL, BE and BG are moved into the sea with a leader
# line, and the GB label sits over Scotland, clear of the removed link
RESP_LABEL = {"NL": (4.6, 55.4), "BE": (0.6, 49.1), "BG": (29.6, 43.6), "GB": (-3.6, 56.9), "RS": (16.2, 41.6)}
NO_LEADER = {"GB"}


def fig_map_response():
    """Emission change of every zone per delivered MWh removed from each charged border, January-June 2026, network
    model (mean over trained models; charged_summary.json r_by_zone). Zones with a decrease are hatched; the exporter,
    the importer and the three third zones with the largest responses are labelled, as in fig_netresp."""
    from matplotlib import colors
    from matplotlib.cm import ScalarMappable
    z, bg, to3035 = _zones()
    d = load("gnn/netresp_r2_v4/charged_summary.json")
    norm = colors.AsinhNorm(linear_width=0.02, vmin=-0.25, vmax=0.25)
    cmap = plt.get_cmap("RdBu_r")
    xlim, ylim = (2.65e6, 6.2e6), (1.5e6, 4.35e6)
    fig, axes = plt.subplots(1, 3, figsize=(190 * MM, 64 * MM), gridspec_kw={"wspace": 0.03})
    fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.2)
    for ax, (b, lab) in zip(axes, (("GB_NL", "(a) GB→NL"), ("GB_BE", "(b) GB→BE"), ("RS_HU", "(c) RS→HU"))):
        s = d[f"{b}|graph|charged"]
        r = s["r_by_zone"]
        x, m = b.split("_")
        _base(ax, bg, xlim, ylim)
        zz = _clip(z, xlim, ylim)
        zz["r"] = zz["zone"].map(r)
        zz["rgba"] = [cmap(norm(v)) for v in zz["r"]]
        zz.plot(ax=ax, color=list(zz["rgba"]), edgecolor="#a0a0a0", lw=0.35)
        for _, row in zz[zz["r"] <= -0.01].iterrows():      # decrease: hatched, light or dark hatch by fill lightness
            rr, gg, bb, _a = row["rgba"]
            light = 0.299 * rr + 0.587 * gg + 0.114 * bb > 0.55
            _hatch(ax, row.geometry, "#303030" if light else "white")
        _link(ax, x, m, to3035, arrowstyle="-", color="k", lw=1.1, ls=(0, (2.5, 1.5)), zorder=5)
        (xa, ya), (xb, yb) = _pt(x, to3035), _pt(m, to3035)
        ax.plot((xa + xb) / 2, (ya + yb) / 2, marker="x", ms=5, mew=1.3, color="k", zorder=6)
        top = [k for k, _ in sorted(s["third_by_zone"].items(), key=lambda kv: -abs(kv[1]))[:3]]
        for code in [x, m] + top:
            v = r[code]
            px, py = _pt(code, to3035)
            if code in RESP_LABEL:
                lx, ly = to3035.transform(*RESP_LABEL[code])
                if code not in NO_LEADER:
                    ax.plot([px, lx], [py, ly], color="k", lw=0.5, zorder=6)
            else:
                lx, ly = px, py
            _halo(ax.text(lx, ly, f"{code} {v:+.2f}".replace("-", "−"), ha="center", va="center", fontsize=7,
                          zorder=7))
        ax.set_title(lab, fontsize=8)
    cax = fig.add_axes([0.28, 0.1, 0.44, 0.035])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    ticks = [-0.2, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.2]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:g}".replace("-", "−") for t in ticks])
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Emission change, t CO₂ per MWh removed (asinh scale; hatched: decrease of at least 0.01)",
                 fontsize=7.5)
    save(fig, "fig_map_response")


def main():
    for f in (fig_trading, fig_onset, fig_outage_v2, fig_outage_multi, fig_netresp, fig_rule_regret_net, fig_regime_net,
              fig_map_network, fig_map_response):
        f()
        print("wrote", f.__name__)


if __name__ == "__main__":
    main()
