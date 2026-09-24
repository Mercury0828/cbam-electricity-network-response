# Where charged electricity imports are replaced

Code for the paper **"Where charged electricity imports are replaced: interconnector outages and a graph
network-response model for the EU CBAM"** by Zirui Tong, Jiachen Shen and Jian Shi (University of Houston).

The EU Carbon Border Adjustment Mechanism (CBAM) charges electricity imported into the EU with a default emission
factor of the exporting country. The code in this repository

- estimates where a lost import is replaced, using interconnector outages (BritNed, Nemo Link, ten further HVDC links
  and the Serbia-Hungary tie lines) as link-specific natural experiments;
- trains a mechanism-guided spatiotemporal graph neural network with an energy-balance dispatch head and a
  state-dependent linear network-response layer on 27 European bidding zones, and tests it against the outages;
- computes a reference charge from the network responses and compares CBAM declaration rules against it, including
  the charges on imports that a reduction reroutes over the exporter's other charged links.

The processed data, the trained model weights and every artefact behind the tables and figures of the paper are in
the companion data repository:
**https://github.com/Mercury0828/cbam-electricity-network-response-data**

## Repository layout

```
src/wedge/            Python package (the internal project name is "wedge")
  fetch/              downloaders for the raw data (Elexon, Energy-Charts, JAO, Nord Pool UMM, ENTSO-E probe)
  gnn/                graph dataset, graph model, network-response layer, outage analyses, charges, tables
  *.py                redispatch comparator, trading comparisons, early analyses (run by tools/make_all.py)
tests/                unit tests
tools/make_all.py     regenerates the redispatch and trading artefacts in data/processed/*.json
tools/run_gnn_pipeline.py   runs the graph-model pipeline stage by stage
paper/tables, paper/figs    output folders for the LaTeX tables and the figures
```

## Environment

Two Python environments reproduce the paper exactly. XGBoost results depend on the version, so use the pinned
versions.

| Environment | Python | Packages | Used for |
|---|---|---|---|
| `.venv-gnn` | 3.12 | `requirements-gnn.txt` (PyTorch 2.5.1 with CUDA 12.1, XGBoost 3.4.1) | graph model, network-response layer, outage analyses, charges, the LaTeX tables |
| `.venv-core` | 3.11 | `requirements-core.txt` (matplotlib 3.10.8) | redispatch comparator, trading comparisons, the figures of the paper |

```
# Linux and macOS; on Windows replace bin/ by Scripts\ in every command of this README
python3.12 -m venv .venv-gnn
.venv-gnn/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv-gnn/bin/pip install -r requirements-gnn.txt
python3.11 -m venv .venv-core
.venv-core/bin/pip install -r requirements-core.txt
```

Every command below names its interpreter explicitly. Figure 1 is a TikZ drawing and needs a LaTeX installation with
`pdflatex`.

## Data

Clone the data repository into `data/` so that the scripts find `data/processed/...`:

```
git clone https://github.com/Mercury0828/cbam-electricity-network-response-data data
```

The raw downloads are not redistributed. `src/wedge/fetch/` downloads them from the original sources into
`data/raw/`; `data/manifest/download_manifest.jsonl` in the data repository lists every request with its time and
SHA-256 hash. Elexon, Energy-Charts and the Nord Pool UMM platform need no key. The JAO auction API needs a personal
token in the environment variable `JAO_API_TOKEN` (the ENTSO-E probe uses `ENTSOE_API_TOKEN`); tokens are read from
the environment only and never written to disk.

| Script | Data |
|---|---|
| `fetch/gnn_bulk.py` | hourly generation by type, prices and flows of the graph zones (Energy-Charts, Elexon) |
| `fetch/gb_fuelhh.py` | Elexon FUELHH, used to repair gaps in the GB per-type feed |
| `fetch/remit_ic.py`, `fetch/umm_nordpool.py` | interconnector unavailability messages (Elexon REMIT, Nord Pool UMM) |
| `fetch/jao.py`, `fetch/jao_outage_check.py` | JAO capacity auctions (GB links; Serbia-Hungary around the outages) |
| `fetch/gb_fr.py`, `fetch/rs_hu.py`, `fetch/ec_zone.py` | border-month and zone data of the redispatch comparator |

### Price inputs

`src/wedge/gnn/prices.py` and `src/wedge/build_inputs.py` read public price files from `data/inputs/source/`. The
data repository contains the World Bank monthly commodity prices, the ACER LNG price assessments, the ECB exchange
rates and the UK allowance price table. The EEX EU ETS primary auction reports are not redistributed; download them
into `data/inputs/source/eex/`:

```
https://public.eex-group.com/eex/eua-auction-report/emission-spot-primary-market-auction-report-2019-data.xls
https://public.eex-group.com/eex/eua-auction-report/emission-spot-primary-market-auction-report-<YEAR>-data.xlsx   (YEAR = 2020 ... 2026)
```

## Reproducing the paper

1. **Tables and figures from the processed artefacts** (minutes, no GPU, no raw downloads):
   ```
   .venv-gnn/bin/python src/wedge/gnn/make_tables.py     # 13 LaTeX tables  -> paper/tables/
   .venv-core/bin/python src/wedge/figures.py           # 7 data figures   -> paper/figs/
   cd paper/figs && pdflatex fig_schematic.tex && cd ../..   # Figure 1 (TikZ schematic)
   ```
   A fresh clone of both repositories regenerated all 13 tables byte-identical and all 7 data figures pixel-identical
   to those of the paper (checked on 2026-09-24). `.venv-gnn/bin/python tools/run_gnn_pipeline.py --stage outputs`
   runs the first two commands with one interpreter; with the matplotlib 3.11.2 of `requirements-gnn.txt` the figures
   differ from the paper's only in rendering details.
2. **Re-running the analyses on the trained models** (outage estimates, network responses, charges):
   ```
   .venv-gnn/bin/python tools/run_gnn_pipeline.py --stage responses
   .venv-gnn/bin/python tools/run_gnn_pipeline.py --stage outages
   .venv-gnn/bin/python tools/run_gnn_pipeline.py --stage charges
   ```
   The runner sets `WEDGE_GNN_SPEC=v4` and, for the network-response scripts, `NETRESP_TAG` (`r2` for the main model,
   `r3` with `NETRESP_LAG=24` for the 24-hour-difference variant); unset any other `NETRESP_*` or `WEDGE_*` variables
   in the shell first. Some steps use the GPU (network re-solves, the recovery check); run GPU jobs one at a time.
   Four inputs of these stages are raw downloads that the data repository does not redistribute. Download them first:
   - outage-message archives read by `outage_events.py`, `outage_v2.py` and `outage_v2_post.py`:
     `fetch/remit_ic.py` and `fetch/umm_nordpool.py` (into `data/raw/remit/` and `data/raw/umm/`);
   - JAO daily auctions on the Serbia-Hungary border read by `outage_events.py`: `fetch/jao_outage_check.py` (needs
     `JAO_API_TOKEN`; into `data/raw/jao/`);
   - Elexon interconnector flows by cable read by `net_taxbase.py`: `fetch/gnn_bulk.py` (into `data/raw/gnn/gb/`).
   Every other step of these stages reads only the data repository.
3. **From the raw data**: download the raw data (see Data), then
   ```
   .venv-gnn/bin/python tools/run_gnn_pipeline.py --stage build
   .venv-gnn/bin/python tools/run_gnn_pipeline.py --stage train_graph
   .venv-gnn/bin/python tools/run_gnn_pipeline.py --stage train_netresp
   ```
   and continue with step 2. `.venv-gnn/bin/python tools/run_gnn_pipeline.py --list` prints every command.
4. **Redispatch comparator, liability scenarios and trading comparisons** (read the raw downloads of
   `fetch/gb_fr.py`, `fetch/rs_hu.py`, `fetch/ec_zone.py` and `fetch/jao.py`):
   ```
   .venv-core/bin/python tools/make_all.py
   .venv-core/bin/python src/wedge/aggregate.py
   .venv-core/bin/python src/wedge/did_d1.py
   .venv-core/bin/python src/wedge/cap_event_rev2.py
   .venv-core/bin/python src/wedge/cap_event_rev3.py
   .venv-core/bin/python src/wedge/boot_stability.py
   .venv-core/bin/python src/wedge/onset_curves.py
   ```

### Paper-to-artefact index

Every number in the paper is stored in the data repository. Generated means that the script above writes the table
or figure file; typed means that the table is written in the manuscript from the listed artefact.

| Paper item | Produced by | Artefact in the data repository | Output |
|---|---|---|---|
| Figure 1 | `paper/figs/fig_schematic.tex` (TikZ) | — | generated (pdflatex) |
| Tables 1-4 (related studies, symbols, sources, rules) | manuscript | — | typed, no results |
| Figure 2, Tables 7, A.1, A.5 | `gnn/outage_v2.py`, `gnn/outage_model_compare.py`, `gnn/outage_pooled_gb.py`, `gnn/outage_subsets.py` | `processed/gnn/outage_v2_v4.json`, `outage_model_compare_v4.json`, `outage_pooled_gb_v4.json`, `outage_subsets_v4.json` | generated |
| Figure 3 | `gnn/outage_multi.py`, `gnn/outage_multi_summary.py` | `processed/gnn/outage_multi_v4.json`, `outage_multi_summary_v4.json` | generated |
| Tables 5, 6 | `gnn/evaluate.py`, `gnn/response_check.py` | `processed/gnn/v4/evaluate.json`, `response_check.json` | generated |
| Figure 4, Tables 8, B.1 | `gnn/netresp.py charged`, `gnn/net_summary.py` | `processed/gnn/netresp_r2_v4/`, `netresp_r3_v4/` (`charged_summary.json`) | generated |
| Figure 5, Table 9 | `gnn/net_rules.py`, `gnn/net_states.py`, `gnn/net_taxbase.py` | `processed/gnn/netresp_r2_v4/net_states.json`, `net_taxbase.json` | generated |
| Figure 6 and the information frontier | `gnn/net_frontier.py`, `gnn/rule_learning.py` | `processed/gnn/netresp_r2_v4/net_frontier.json` | generated |
| Table 10 | `aggregate.py` | `processed/aggregate_2026h1.json` | typed |
| Table 11 | `gnn/net_states.py`, `t2_ablation.py`, `t4_baselines.py` | `processed/gnn/netresp_r2_v4/net_states.json`, `processed/t2_ablation.json`, `t4_baselines.json` | generated |
| Tables A.2-A.4, A.6 | `gnn/outage_synthetic.py`, `gnn/outage_events.py`, `gnn/outage_rs_events.py` | `processed/gnn/outage_synthetic10_v4.json`, `outage_events_v4.json`, `outage_rs_events_v4.json` | generated |
| Figures D.1, D.2 | `onset_curves.py`, `cap_event_rev2.py` | `processed/onset_curves.json`, `cap_distribution_rev.json`, `cap_event_study_rev.json` | generated |
| Table D.1 | `did_d1.py`, `cap_event_rev3.py`, `boot_stability.py` | `processed/did_d1_directional.json`, `cap_event_rev_links.json`, `boot_stability.json` | typed |
| Table E.1 | `frontier.py`, `d2_money.py`, `d2_frontier.py`, `cap_price.py`, `r2_cert.py` | `processed/frontier_gb.json`, `d2_money.json`, `d2_frontier.json`, `cap_price.json`, `r2_cert.json` | typed |
| Supplementary data | `gnn/outage_events.py` | `processed/gnn/outage_events_v4.csv` | generated |

Scripts are in `src/wedge/` (the `gnn/` prefix is `src/wedge/gnn/`).

## License

The code is released under the MIT License (`LICENSE`). The data repository is released under CC BY 4.0; the
third-party data it derives from remain subject to their providers' terms (see `DATA_SOURCES.md` there).

## Citation

See `CITATION.cff`. The citation will be updated with the journal reference and the archive DOI.
