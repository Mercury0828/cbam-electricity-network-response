"""Run the graph-model pipeline of the paper stage by stage.

Every stage is a list of commands (script + arguments + environment) in the order used for the paper. Each script
documents its inputs and outputs in its module docstring; outputs go to data/processed/gnn/.

Usage (from the repository root):
    python tools/run_gnn_pipeline.py --list
    python tools/run_gnn_pipeline.py --stage outages
    python tools/run_gnn_pipeline.py --all
Training stages need a CUDA GPU; run them one at a time.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

BASE = {"WEDGE_GNN_SPEC": "v4"}
R2 = {"NETRESP_TAG": "r2"}
R3 = {"NETRESP_TAG": "r3", "NETRESP_LAG": "24"}
GNN = "src/wedge/gnn/"

STAGES = {
    # hourly graph dataset and monthly fuel and carbon prices (needs the raw downloads, see README)
    "build": [({}, ["src/wedge/gnn/prices.py"]), ({}, [GNN + "dataset.py"])],
    # graph model: five seeds, and the control without cross-border messages (three seeds)
    "train_graph": [({}, [GNN + "train.py", "--seed", str(s)]) for s in (42, 123, 7, 2024, 999)]
    + [({}, [GNN + "train.py", "--seed", str(s), "--drop", "TA,CC"]) for s in (42, 123, 7)],
    # network-response layer: main (hour-to-hour shocks) with node-local variant, and the 24-hour-difference variant
    "train_netresp": [(R2, [GNN + "netresp.py", "train", "--seed", str(s), "--epochs", "40"] + loc)
                      for loc in ([], ["--local"]) for s in (42, 123, 7)]
    + [(R3, [GNN + "netresp.py", "train", "--seed", str(s), "--epochs", "40"]) for s in (42, 123, 7)],
    # responses on the charged borders, local emission responses and prediction accuracy
    "responses": [(R2, [GNN + "netresp.py", "charged"]), (R3, [GNN + "netresp.py", "charged"]),
                  ({}, [GNN + "marginal.py"]), ({}, [GNN + "summarize.py"]), ({}, [GNN + "evaluate.py"]),
                  ({}, [GNN + "response_check.py"])],
    # interconnector outages: events, estimator, model comparison, held-out links, Serbia-Hungary, recovery check
    "outages": [({}, [GNN + "outage_study.py"]), ({}, [GNN + "outage_v2.py"]), ({}, [GNN + "outage_v2_post.py"]),
                ({}, [GNN + "outage_v2_matching.py"]), ({}, [GNN + "outage_did.py"]),
                (R2, [GNN + "outage_model_compare.py"]), (R2, [GNN + "outage_model_summary.py"]),
                (R2, [GNN + "outage_pooled_gb.py"]), (R2, [GNN + "outage_multi.py"]),
                (R2, [GNN + "outage_multi_summary.py"]), ({}, [GNN + "outage_events.py"]),
                (R2, [GNN + "outage_subsets.py"]), (R2, [GNN + "outage_multi_confirmed.py"]),
                (R2, [GNN + "outage_rs.py"]), (R2, [GNN + "outage_rs_events.py"]),
                ({"SYNTH_REPS": "10", "SYNTH_TAG": "10"}, [GNN + "outage_synthetic.py"]),
                (R3, [GNN + "outage_model_compare.py"]), (R3, [GNN + "outage_model_summary.py"]),
                (R3, [GNN + "outage_pooled_gb.py"]), (R3, [GNN + "outage_multi.py"]),
                (R3, [GNN + "outage_multi_summary.py"])],
    # network responses, reference charge, rule comparison, charged imports along the path, information frontier
    "charges": [(R2, [GNN + "net_summary.py"]), (R2, [GNN + "net_params.py"]), (R2, [GNN + "net_rules.py"]),
                (R2, [GNN + "net_states.py"]), (R2, [GNN + "net_feasibility.py"]), (R2, [GNN + "net_taxbase.py"]),
                (R2, [GNN + "net_taxbase.py", "summary"]), (R2, [GNN + "net_frontier.py"]),
                (R2, [GNN + "rule_learning.py"]), (R3, [GNN + "net_summary.py"]), (R3, [GNN + "net_states.py"]),
                (R3, [GNN + "net_taxbase.py"]), (R3, [GNN + "net_taxbase.py", "summary"]),
                ({}, [GNN + "collect_numbers.py", "_v4"])],
    # LaTeX tables and figures of the paper (written to paper/tables and paper/figs)
    "outputs": [({}, [GNN + "make_tables.py"]), ({}, ["src/wedge/figures.py"])],
}


def run(stage):
    for env, args in STAGES[stage]:
        e = dict(os.environ, **BASE, **env)
        label = " ".join(f"{k}={v}" for k, v in env.items())
        print(f"[{stage}] {label} python {' '.join(args)}", flush=True)
        rc = subprocess.call([sys.executable, *args], env=e)
        if rc != 0:
            sys.exit(f"[{stage}] failed with exit code {rc}: {' '.join(args)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not (a.stage or a.all):
        for k, v in STAGES.items():
            print(f"{k}: {len(v)} commands")
            for env, args in v:
                print("   ", " ".join(f"{kk}={vv}" for kk, vv in env.items()), " ".join(args))
        return
    for stage in (STAGES if a.all else [a.stage]):
        run(stage)


if __name__ == "__main__":
    main()
