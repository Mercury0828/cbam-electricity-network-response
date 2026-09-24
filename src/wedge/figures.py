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


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
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


def main():
    for f in (fig_trading, fig_onset, fig_outage_v2, fig_outage_multi, fig_netresp, fig_rule_regret_net, fig_regime_net):
        f()
        print("wrote", f.__name__)


if __name__ == "__main__":
    main()
