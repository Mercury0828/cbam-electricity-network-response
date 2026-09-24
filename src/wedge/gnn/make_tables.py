"""LaTeX tables for the network version (A57), read from frozen artefacts only.

tab_outage_validation: outage responses (difference in slopes with event-bootstrap 95% intervals) against the graph
network model, the same model with node-local parameters, and the two-zone accounting, all evaluated with the identical
estimator on the identical hours (outage_model_compare_v4.json). GB->NL uses the hours in which the link would have
carried GB exports (L+); GB->BE uses all hours (L+ is not identified there).
Output: paper/tables/tab_outage_validation.tex
"""
from __future__ import annotations

import json
import pathlib

G = pathlib.Path("data/processed/gnn")
OUT = pathlib.Path("paper/tables")
ROWS = [("R_m", "Importer's own plants"), ("NIother_m", "Importer's other links"), ("R_x", "GB plants"),
        ("NIother_x", "GB other links")]
DIR = {"GB->NL": (0, "GB$\\to$NL, would-be GB exports"), "GB->BE": (2, "GB$\\to$BE, all hours")}
POOLED_ROWS = [("R_x", "GB plants"), ("NIother_x", "GB other links"), ("E_GB", "GB emissions (t per MWh)")]


def fmt(v):
    s = f"{v:.2f}"
    return ("0.00" if s == "-0.00" else s).replace("-", "$-$")


def cell(v, ci):
    mark = "" if ci[0] <= v <= ci[1] else "$^{\\dagger}$"
    return fmt(v) + mark


def main(fes="A_block"):
    cmp_ = json.loads((G / "outage_model_compare_v4.json").read_text(encoding="utf-8"))
    lines = [r"\begin{tabular}{L{0.3\textwidth}C{0.17\textwidth}C{0.11\textwidth}C{0.11\textwidth}C{0.11\textwidth}}",
             r"\toprule",
             r"Response per MW of lost flow & Outage estimate [95\% interval] & Graph network & Node-local & Two-zone \\",
             r"\midrule"]
    for b, (j, head) in DIR.items():
        lines.append(r"\multicolumn{5}{l}{\textit{" + head + r"}} \\")
        for k, lab in ROWS:
            e = cmp_[b]["graph"]["rows"][k][fes]
            v, ci = e["empirical_did"][j], e["ci95"][j]
            cells = [cell(cmp_[b][name]["rows"][k][fes]["model"][j], ci) for name in ("graph", "local", "one_for_one")]
            lines.append(f"{lab} & {fmt(v)} [{fmt(ci[0])}, {fmt(ci[1])}] & " + " & ".join(cells) + r" \\")
    # GB side, both links pooled (outage_pooled_gb_v4.json), all hours
    pool = json.loads((G / "outage_pooled_gb_v4.json").read_text(encoding="utf-8"))
    lines.append(r"\multicolumn{5}{l}{\textit{GB side, both links pooled, all hours}} \\")
    for k, lab in POOLED_ROWS:
        e = pool["rows"][k][fes]
        v, ci = e["did"][2], e["ci95"][2]
        cells = [cell(e["model"][name][2], ci) for name in ("graph", "local", "one_for_one")]
        lines.append(f"{lab} & {fmt(v)} [{fmt(ci[0])}, {fmt(ci[1])}] & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tab_outage_validation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def netresp_table():
    """tab_netresp: network responses on the charged borders, January-June 2026 (charged_summary.json), with the
    two-zone accounting of the same local emission responses."""
    S = json.loads((G / "netresp_r2_v4" / "charged_summary.json").read_text(encoding="utf-8"))
    rows = [("Importer's own plants (MW per MW)", "absorb_importer", None),
            ("Exporter's own plants (MW per MW)", "absorb_exporter", None),
            ("Exporter emissions avoided, $E_x$", "E_x", "E_x"),
            ("Importer emissions added, $E_m$", "E_m", "E_m"),
            ("Third-zone emissions added", "third", None),
            ("Net network change", "network", "network")]
    heads = [("GB_NL", r"GB$\to$NL"), ("GB_BE", r"GB$\to$BE"), ("RS_HU", r"RS$\to$HU")]
    lines = [r"\begin{tabular}{L{0.34\textwidth}C{0.18\textwidth}C{0.18\textwidth}C{0.18\textwidth}}", r"\toprule",
             " & " + " & ".join(h for _, h in heads) + r" \\", r"\midrule"]
    for lab, k, k2 in rows:
        cells = []
        for b, _ in heads:
            s = S[f"{b}|graph|charged"]
            v, lo, hi = s[k]
            c = f"{fmt(v)} [{fmt(lo)}, {fmt(hi)}]"
            if k2:
                c += f" ({fmt(s['two_zone'][k2])})"
            cells.append(c)
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_netresp.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def regret_table():
    """tab_regret_net: regret |a - u| of each rule against the network benchmark, EUR/MWh, January-June 2026, in the
    2026 state and after removal of the Carbon Price Support (net_states.json, graph variant)."""
    S = json.loads((G / "netresp_r2_v4" / "net_states.json").read_text(encoding="utf-8"))
    rows = [("Gross default", "gross_default", "gross_default"),
            ("All-sources proxy", "all_sources", "all_sources"),
            ("Default, contemporaneous credit$^{a}$", "default_credit", "default_credit"),
            ("All-sources proxy, contemporaneous credit$^{a}$", "all_sources_credit", "all_sources_credit"),
            ("Default, annual credit$^{b}$", "default_annual_credit_81", "default_annual_credit_uka"),
            ("Default, contemporaneous credit, allowance only", "default_credit_uka", None),
            ("Default, annual credit 60.50~EUR/t, allowance only", "default_annual_credit_uka", None),
            ("Zero charge (diagnostic)", "zero", "zero")]
    cols = [("GB->NL|graph", "S26"), ("GB->NL|graph", "NOCPS"), ("GB->BE|graph", "S26"), ("GB->BE|graph", "NOCPS"),
            ("RS->HU|graph", "S26")]
    lines = [r"\begin{tabular}{L{0.37\textwidth}C{0.095\textwidth}C{0.095\textwidth}C{0.095\textwidth}"
             r"C{0.095\textwidth}C{0.095\textwidth}}", r"\toprule",
             r" & \multicolumn{2}{c}{GB$\to$NL} & \multicolumn{2}{c}{GB$\to$BE} & RS$\to$HU \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-6}",
             r"Rule & 2026 & CPS removed & 2026 & CPS removed & 2026 \\", r"\midrule"]
    for lab, k26, kno in rows:
        cells = []
        for b, st in cols:
            k = k26 if st == "S26" else kno
            reg = S[b]["states"][st]["regret"]
            cells.append(f"{reg[k]:.2f}" if k and k in reg else "--")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    # A66: along the response path, |a chi - u| (net_taxbase.json, graph variant, median over trained models)
    T = json.loads((G / "netresp_r2_v4" / "net_taxbase.json").read_text(encoding="utf-8"))
    lines += [r"\midrule", r"\multicolumn{6}{l}{\textit{Along the response path, $|a\chi_\ell-u|$}} \\"]
    chi = [f"{T['_summary'][b.split('|')[0]]['chi']:.2f}" for b, st in cols]           # hourly median over models
    lines.append(r"Charged-import retention $\chi_\ell$ & " + " & ".join(chi) + r" \\")
    for lab, k26, kno in rows:
        cells = []
        for b, st in cols:
            k = k26 if st == "S26" else kno
            reg = T[b]["states"][st]["path"]
            cells.append(f"{reg[k]:.2f}" if k and k in reg else "--")
        if any(c != "--" for c in cells):
            lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_regret_net.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def robust_table():
    """tab_robust: largest volume-weighted mean regret on GB->NL, 2026 state / post-CPS state, EUR/MWh, under the
    network model (max over trained models), its node-local variant, the two-zone accounting of the graph model's
    local responses, and the redispatch-model variants of the earlier analysis (t2_ablation.json, t4_baselines.json,
    max over scenarios)."""
    S = json.loads((G / "netresp_r2_v4" / "net_states.json").read_text(encoding="utf-8"))
    t2 = json.loads(pathlib.Path("data/processed/t2_ablation.json").read_text(encoding="utf-8"))
    t4 = json.loads(pathlib.Path("data/processed/t4_baselines.json").read_text(encoding="utf-8"))
    keys_net = ["gross_default", "default_credit", "all_sources", "all_sources_credit"]
    keys_t2 = ["gross default", "default + credit", "all-sources proxy", "all-sources proxy + credit"]
    keys_t4 = ["gross default", "default + credit (exporter carbon cost)", "all-sources proxy",
               "all-sources proxy + credit"]

    def net(variant, two_zone=False):
        st = S[f"GB->NL|{variant}"]["states"]
        if two_zone:
            return [(st["S26"]["regret_two_zone"][k], st["NOCPS"]["regret_two_zone"][k]) for k in keys_net]
        return [(st["S26"]["regret_range"][k][1], st["NOCPS"]["regret_range"][k][1]) for k in keys_net]

    def old2(v):
        return [(t2[v]["GB->NL"]["S26"]["regret"][k], t2[v]["GB->NL"]["NOCPS"]["regret"][k]) for k in keys_t2]

    def old4(v):
        return [(t4["GB->NL"][f"S26/{v}"]["worst_case_regret"][k], t4["GB->NL"][f"NOCPS/{v}"]["worst_case_regret"][k])
                for k in keys_t4]
    rows = [("Network model", net("graph")), ("Network model, node-local encoder", net("local")),
            ("Two-zone baseline, graph model", net("graph", True)), ("Redispatch comparator", old2("base")),
            ("Hourly average factors", old4("AEF")), ("Regression marginal factors", old4("REG")),
            ("Redispatch, joint efficiencies", old2("joint")), ("Redispatch, repaired hours dropped", old2("exclude")),
            ("Redispatch, no zero floor", old2("no_floor"))]
    lines = [r"\begin{tabular}{L{0.3\textwidth}C{0.14\textwidth}C{0.14\textwidth}C{0.14\textwidth}C{0.14\textwidth}}",
             r"\toprule",
             r"Variant & Gross default & Default with credit & All-sources proxy & All-sources with credit \\",
             r"\midrule"]
    for lab, vals in rows:
        lines.append(f"{lab} & " + " & ".join(f"{a:.2f} / {b:.2f}" for a, b in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_robust.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def accuracy_table():
    """tab_accuracy: RMSE of dispatchable-plant emissions (t/h) and day-ahead price (EUR/MWh) by period
    (v4/evaluate.json). The average-share baseline has no price model and carries the previous hour's price, so its
    price column is the lagged-price baseline. The no-graph control appears once its models exist."""
    d = json.loads((G / "v4" / "evaluate.json").read_text(encoding="utf-8"))
    per = [("val", "Validation"), ("test", "Test"), ("y2025", "2025"), ("charged", "2026")]

    def g(p, k, m):
        return d[p][k][m] if k in d[p] else None
    lines = [r"\begin{tabular}{L{0.13\textwidth}C{0.09\textwidth}C{0.09\textwidth}C{0.09\textwidth}C{0.09\textwidth}"
             r"C{0.09\textwidth}C{0.09\textwidth}C{0.09\textwidth}C{0.09\textwidth}}", r"\toprule",
             r" & \multicolumn{4}{c}{Emissions} & \multicolumn{4}{c}{Price} \\",
             r"Period & Graph & Control & Shares & Trees & Graph & Control & Lagged & Trees \\", r"\midrule"]
    for p, lab in per:
        e = [g(p, k, "emis_rmse") for k in ("MG-STGNN-X", "nograph", "avg-share", "xgb")]
        q = [g(p, k, "price_rmse") for k in ("MG-STGNN-X", "nograph", "avg-share", "xgb")]
        cells = [f"{v:.0f}" if v is not None else "--" for v in e] + [f"{v:.1f}" if v is not None else "--" for v in q]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_accuracy.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def response_table():
    """tab_response: calibration slope of observed hour-to-hour emission changes on the local response times the
    change in dispatchable output, 2025 and January-June 2026 (v4/response_check.json)."""
    d = json.loads((G / "v4" / "response_check.json").read_text(encoding="utf-8"))
    zones = [("GB", "Great Britain"), ("NL", "Netherlands"), ("BE", "Belgium"), ("RS", "Serbia"), ("HU", "Hungary")]
    lines = [r"\begin{tabular}{L{0.18\textwidth}C{0.12\textwidth}C{0.12\textwidth}C{0.12\textwidth}C{0.12\textwidth}"
             r"C{0.12\textwidth}}", r"\toprule", r"Zone & Graph & Control & Empirical & Trees & Average \\", r"\midrule"]
    for z, lab in zones:
        cells = [f"{d[z][k]['calibration_slope']:.2f}" if k in d[z] else "--"
                 for k in ("graph", "nograph", "hourly_mef", "trees", "average")]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_response.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def fmt_ci(v, ci):
    return f"{fmt(v)} [{fmt(ci[0])}, {fmt(ci[1])}]"


def outage_components_table():
    """tab_outage_components: outage slope (Eq. outage-reg), normal-window slope (Eq. normal-reg) and their difference
    for the rows of tab_outage_validation, specifications A and B (outage_v2_v4.json; GB side pooled from
    outage_subsets_v4.json, subset 'all', which reproduces outage_pooled_gb_v4.json)."""
    v2 = json.loads((G / "outage_v2_v4.json").read_text(encoding="utf-8"))
    sub = json.loads((G / "outage_subsets_v4.json").read_text(encoding="utf-8"))
    lines = [r"\begin{tabular}{L{0.3\textwidth}C{0.09\textwidth}C{0.09\textwidth}C{0.09\textwidth}C{0.09\textwidth}"
             r"C{0.09\textwidth}C{0.09\textwidth}}", r"\toprule",
             r" & \multicolumn{3}{c}{Specification A} & \multicolumn{3}{c}{Specification B} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Response per MW of lost flow & Outage $\beta$ & Normal $\gamma$ & $\beta-\gamma$ & Outage $\beta$ & "
             r"Normal $\gamma$ & $\beta-\gamma$ \\", r"\midrule"]
    for b, (j, head) in DIR.items():
        lines.append(r"\multicolumn{7}{l}{\textit{" + head + r"}} \\")
        for k, lab in ROWS:
            o = v2[b]["outcomes"][f"{k}|full"]
            cells = [fmt(o[fes][q][j]) for fes in ("A_block", "B_block_hour") for q in ("outage", "pseudo", "did")]
            lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines.append(r"\multicolumn{7}{l}{\textit{GB side, both links pooled, all hours}} \\")
    for k, lab in (("R_x", "GB plants"), ("NIother_x", "GB other links"), ("E_x", "GB emissions (t per MWh)")):
        o = sub["GB_pooled"]["all"][k]
        cells = [fmt(o[fes][q][2]) for fes in ("A_block", "B_block_hour") for q in ("outage", "pseudo", "did")]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_outage_components.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def events_table():
    """tab_events: every BritNed and Nemo Link event with its treated hours, blocks, windows, identifying weight and
    confirmation in published outage messages (outage_events_v4.json)."""
    ev = json.loads((G / "outage_events_v4.json").read_text(encoding="utf-8"))
    src = {"Nord Pool UMM": "Nord Pool", "Elexon REMIT": "Elexon", "JAO": "JAO", "": "--"}
    lines = [r"\begin{tabular}{L{0.26\textwidth}R{0.08\textwidth}R{0.08\textwidth}C{0.1\textwidth}C{0.09\textwidth}"
             r"R{0.08\textwidth}L{0.17\textwidth}}", r"\toprule",
             r"Event (first and last day) & Zero-flow hours & Treated hours & Blocks (with window) & Windows & "
             r"Weight (\%) & Outage message \\", r"\midrule"]
    for lk, head in (("GB-NL", r"BritNed, GB$\to$NL"), ("GB-BE", r"Nemo Link, GB$\to$BE")):
        lines.append(r"\multicolumn{7}{l}{\textit{" + head + r"}} \\")
        for e in [e for e in ev["events"] if e["link"] == lk]:
            msg = f"{src[e['source']]} ({e['type'][0].upper()})" if e["confirmed"] else "none"
            lines.append(f"{e['start']} to {e['end'][5:]} & {e['zero_hours']:,} & {e['treated_hours']:,} & "
                         f"{e['blocks']} ({e['blocks_with_window']}) & {e['windows']} & {100 * e['weight']:.1f} & {msg}"
                         + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_events.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def events_multi_table():
    """tab_events_multi: outages of the held-out HVDC links and of the Serbia-Hungary tie lines by link, with
    their confirmation in published outage messages (outage_events_v4.json; event-level list in the supplementary
    file outage_events_v4.csv)."""
    ev = json.loads((G / "outage_events_v4.json").read_text(encoding="utf-8"))
    order = ["DE-SE", "GR-IT", "NL-NO", "PL-SE", "DK-NL", "ME-IT", "BE-DE", "DE-NO", "GB-NO", "DK-GB", "RS-HU"]
    lines = [r"\begin{tabular}{L{0.3\textwidth}C{0.1\textwidth}R{0.08\textwidth}C{0.1\textwidth}C{0.12\textwidth}"
             r"C{0.09\textwidth}L{0.12\textwidth}}", r"\toprule",
             r"Link & Outages (with a block) & Treated hours & Blocks (with window) & Confirmed outages (weight, \%) & "
             r"Planned / unplanned & Messages \\", r"\midrule"]
    tot = [0, 0, 0, 0, 0, 0]
    for lk in order:
        s = ev["summary"][lk]
        rows = [e for e in ev["events"] if e["link"] == lk]
        nb = sum(e["blocks"] for e in rows)
        nbw = sum(e["blocks_with_window"] for e in rows)
        name = s["name"].replace("RS-HU tie lines", "Serbia--Hungary tie lines")
        src = ", ".join(x.replace("Nord Pool UMM", "Nord Pool").replace("Elexon REMIT", "Elexon") for x in s["sources"]) or "none"
        pu = f"{s['planned']} / {s['unplanned']}" if (s["planned"] + s["unplanned"]) else "--"
        conf = f"{s['confirmed_events']} ({100 * s['confirmed_weight']:.0f})" if s["confirmed_events"] else "0"
        lines.append(f"{name} ({lk.replace('-', '--')}) & {s['events']} ({s['events_with_blocks']}) & "
                     f"{s['treated_hours']:,} & {nb} ({nbw}) & {conf} & {pu} & {src}" + r" \\")
        if lk != "RS-HU":
            for i, v in enumerate((s["events"], s["events_with_blocks"], s["treated_hours"], nb, nbw,
                                   s["confirmed_events"])):
                tot[i] += v
    lines.append(r"\midrule")
    lines.append(f"Ten HVDC links & {tot[0]} ({tot[1]}) & {tot[2]:,} & {tot[3]} ({tot[4]}) & {tot[5]} & & " + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_events_multi.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def synthetic_table():
    """tab_synthetic: recovery check of the outage estimation pipeline (outage_synthetic10_v4.json, ten draws), specification A. For each
    known response: the target, the mean estimate over the noise draws, and the mean of the estimate minus the
    estimate under no response on the same draw ('paired'); BritNed would-be-export and pooled slopes, Nemo pooled."""
    s = json.loads((G / "outage_synthetic10_v4.json").read_text(encoding="utf-8"))           # ten noise draws
    fes = "A_block"
    groups = [("GB->NL", 0), ("GB->NL", 2), ("GB->BE", 2)]
    truths = [("local", "Importer's plants only"), ("rerouting", "Split 0.2 / 0.8"), ("state", "State-dependent"),
              ("placebo", "No response")]
    lines = [r"\begin{tabular}{L{0.23\textwidth}" + r"C{0.068\textwidth}C{0.07\textwidth}C{0.07\textwidth}" * 3 + "}",
             r"\toprule",
             r" & \multicolumn{3}{c}{BritNed, $L^{+}$} & \multicolumn{3}{c}{BritNed, pooled} & "
             r"\multicolumn{3}{c}{Nemo Link, pooled} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"Outcome & Target & Mean & Paired & Target & Mean & Paired & Target & Mean & Paired \\", r"\midrule"]
    for t, head in truths:
        lines.append(r"\multicolumn{10}{l}{\textit{" + head + r"}} \\")
        for k, lab in ROWS:
            cells = []
            for b, j in groups:
                runs, tg = s[b]["runs"], s[b]["targets"][t][k][fes][j]
                d = [r[k][fes]["did"][j] for r in runs[t]]
                p = [r[k][fes]["did"][j] - q[k][fes]["did"][j] for r, q in zip(runs[t], runs["placebo"])]
                same = t == "placebo" or (t == "local" and k in ("NIother_m", "NIother_x"))   # identical data
                cells += [fmt(tg), fmt(sum(d) / len(d)), "--" if same else fmt(sum(p) / len(p))]
            lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines.append(r"\multicolumn{10}{l}{\textit{Importer's plants only, normal windows regressed on "
                 r"$\hat F_\ell-F_\ell$}} \\")
    alt = [fmt(s[b]["alternative_normal_window_dose"]["R_m"][fes]["did_mean"][j]) for b, j in groups]
    lines.append(r"Importer's own plants & " + f"1.00 & {alt[0]} & -- & 1.00 & {alt[1]} & -- & 1.00 & {alt[2]} & --"
                 + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_synthetic.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def subsets_table():
    """tab_outage_subsets: outage responses on all events, on the events that published outage messages confirm, and
    on the blocks that received a normal-operation window, with the graph model (outage state and state of the day
    before the event) and the two-zone baseline on all events (outage_subsets_v4.json), specification A."""
    s = json.loads((G / "outage_subsets_v4.json").read_text(encoding="utf-8"))
    v2 = json.loads((G / "outage_v2_v4.json").read_text(encoding="utf-8"))
    pool = json.loads((G / "outage_pooled_gb_v4.json").read_text(encoding="utf-8"))
    fes = "A_block"
    blocks = [(r"BritNed, would-be GB exports", "GB->NL", 0, ROWS[:2]),
              (r"BritNed, all hours", "GB->NL", 2, ROWS[:2]),
              (r"Nemo Link, all hours", "GB->BE", 2, ROWS[:2]),
              (r"GB side, both links pooled, all hours", "GB_pooled", 2,
               [("R_x", "GB plants"), ("NIother_x", "GB other links"), ("E_x", "GB emissions (t per MWh)")])]
    lines = [r"\begin{tabular}{L{0.19\textwidth}C{0.19\textwidth}C{0.19\textwidth}C{0.19\textwidth}C{0.07\textwidth}"
             r"C{0.07\textwidth}}", r"\toprule",
             r"Response per MW of lost flow & All events & Confirmed events & Blocks with a window & Graph network & "
             r"Two-zone \\", r"\midrule"]
    for head, b, j, rows in blocks:
        lines.append(r"\multicolumn{6}{l}{\textit{" + head + r"}} \\")
        for k, lab in rows:
            # all events: the frozen estimates and intervals of tab_outage_validation (identical point estimates)
            if b == "GB_pooled":
                o = pool["rows"][{"E_x": "E_GB"}.get(k, k)][fes]
            else:
                o = v2[b]["outcomes"][f"{k}|full"][fes]
            cells = [fmt_ci(o["did"][j], o["ci95"][j])]
            for sname in ("confirmed", "matched"):
                if b == "GB_pooled":
                    o = s[b][sname][k][fes]
                    cells.append(fmt_ci(o["did"][j], o["ci95"][j]))
                else:
                    o = s[b][sname]["outcomes"][k]["empirical"][fes]
                    cells.append(fmt_ci(o["did"][j], o["ci95"][j]))
            if b == "GB_pooled":
                m = s[b]["all"][k][fes]["model"]
            else:
                m = {nm: v[fes] for nm, v in s[b]["all"]["outcomes"][k]["model"].items()}
            cells += [fmt(m["graph"][j]), fmt(m["two_zone"][j])]
            lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_outage_subsets.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def horizon_table():
    """tab_horizon: network decomposition on the charged borders, January-June 2026, under the network-response model
    trained on hour-to-hour changes (main, r2) and on changes against the same hour of the previous day (r3), mean over
    trained models with the range across them (charged_summary.json of both tags)."""
    S2 = json.loads((G / "netresp_r2_v4" / "charged_summary.json").read_text(encoding="utf-8"))
    S3 = json.loads((G / "netresp_r3_v4" / "charged_summary.json").read_text(encoding="utf-8"))
    rows = [("Importer's own plants (MW per MW)", "absorb_importer"), ("Exporter's own plants (MW per MW)", "absorb_exporter"),
            ("Exporter emissions avoided, $E_x$", "E_x"), ("Importer emissions added, $E_m$", "E_m"),
            ("Third-zone emissions added", "third"), ("of which Germany", "DE"), ("Net network change", "network")]
    heads = [("GB_NL", r"GB$\to$NL"), ("GB_BE", r"GB$\to$BE"), ("RS_HU", r"RS$\to$HU")]
    lines = [r"\begin{tabular}{L{0.25\textwidth}" + r"C{0.105\textwidth}" * 6 + "}", r"\toprule",
             " & " + " & ".join(r"\multicolumn{2}{c}{" + h + "}" for _, h in heads) + r" \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r" & Hourly & 24-h diff. & Hourly & 24-h diff. & Hourly & 24-h diff. \\", r"\midrule"]
    for lab, k in rows:
        cells = []
        for b, _ in heads:
            for S in (S2, S3):
                s = S[f"{b}|graph|charged"]
                if k == "DE":
                    v = s["third_by_zone"]["DE"]
                    v = v[0] if isinstance(v, list) else v
                    cells.append(fmt(v))
                else:
                    v, lo, hi = s[k]
                    cells.append(f"{fmt(v)} [{fmt(lo)}, {fmt(hi)}]")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_horizon.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def rs_events_table():
    """tab_rs_events: Serbia-Hungary outages one by one (outage_rs_events_v4.json): difference in slopes (outage minus
    normal-window slope, specification A, pooled dose), graph network and two-zone baseline. The two confirmed outages
    come from the primary run, the unconfirmed outage of August 2023 from the run with all three outages."""
    ev = json.loads((G / "outage_rs_events_v4.json").read_text(encoding="utf-8"))
    cols = []
    for run, day in (("outage_rs", "2024-09"), ("outage_rs", "2025-08"), ("outage_rs_all", "2023-08")):
        for e in ev[run]["events"].values():
            if e["first_treated_day"].startswith(day):
                cols.append((day, e))
    names = {"2024-09": "Sep. 2024", "2025-08": "Aug. 2025", "2023-08": "Aug. 2023$^{a}$"}
    rows = [("R_a", "Serbian plants"), ("R_b", "Hungarian plants"), ("NIother_a", "Serbian other links"),
            ("NIother_b", "Hungarian other links"), ("E_a", "Serbian emissions (t per MWh)"),
            ("E_b", "Hungarian emissions (t per MWh)")]
    lines = [r"\begin{tabular}{L{0.22\textwidth}" + r"C{0.075\textwidth}" * 9 + "}", r"\toprule",
             " & " + " & ".join(r"\multicolumn{3}{c}{" + f"{names[d]}, {e['treated_hours']} h" + "}" for d, e in cols)
             + r" \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"Response per MW of lost flow" + r" & Outage & Graph & Two-zone" * 3 + r" \\", r"\midrule"]
    for k, lab in rows:
        cells = []
        for _, e in cols:
            o = e["outcomes"][k]
            cells += [fmt(o["difference"]), fmt(o["model"]["graph"]), fmt(o["model"]["one_for_one"])]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_rs_events.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
    netresp_table()
    regret_table()
    robust_table()
    accuracy_table()
    response_table()
    outage_components_table()
    events_table()
    events_multi_table()
    synthetic_table()
    subsets_table()
    horizon_table()
    rs_events_table()
