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

The graph-model results were produced with Python 3.12 and the packages in `requirements-gnn.txt` (PyTorch 2.5.1 with
CUDA 12.1, XGBoost 3.4.1). The redispatch comparator and the trading comparisons (`tools/make_all.py`) were produced
with Python 3.11 and `requirements-core.txt`. XGBoost results depend on the version, so use the pinned versions to
reproduce the numbers exactly.

```
python -m venv .venv-gnn
.venv-gnn/bin/pip install -r requirements-gnn.txt      # Windows: .venv-gnn\Scripts\pip
```

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

All graph-model scripts read `WEDGE_GNN_SPEC=v4`; the network-response scripts also read `NETRESP_TAG` (`r2` for the
main model, `r3` with `NETRESP_LAG=24` for the 24-hour-difference variant). `tools/run_gnn_pipeline.py` sets them.

1. **Tables and figures from the processed artefacts** (minutes, no GPU):
   ```
   python tools/run_gnn_pipeline.py --stage outputs
   ```
   `src/wedge/gnn/make_tables.py` writes the LaTeX tables to `paper/tables/` and `src/wedge/figures.py` the figures to
   `paper/figs/`. The tables are identical to those of the paper in either environment. The figures of the paper were
   drawn with matplotlib 3.10.8 (`requirements-core.txt`); with matplotlib 3.11.2 (`requirements-gnn.txt`) they differ
   only in rendering details.
2. **Re-running the analyses on the trained models** (outage estimates, network responses, charges):
   ```
   python tools/run_gnn_pipeline.py --stage responses
   python tools/run_gnn_pipeline.py --stage outages
   python tools/run_gnn_pipeline.py --stage charges
   ```
   Some steps use the GPU (network re-solves, the recovery check); run GPU jobs one at a time. Three inputs of these
   stages are raw downloads that the data repository does not redistribute: the outage-message archives read by
   `outage_events.py`, `outage_v2.py` and `outage_v2_post.py` (download with `fetch/remit_ic.py` and
   `fetch/umm_nordpool.py` into `data/raw/remit/` and `data/raw/umm/`), and the Elexon interconnector flows by cable
   read by `net_taxbase.py` (download with `fetch/gnn_bulk.py` into `data/raw/gnn/gb/`). Every other step of stage 2
   reads only the data repository.
3. **From the raw data**: download the raw data (above), then
   ```
   python tools/run_gnn_pipeline.py --stage build
   python tools/run_gnn_pipeline.py --stage train_graph
   python tools/run_gnn_pipeline.py --stage train_netresp
   ```
   and continue with step 2. `python tools/run_gnn_pipeline.py --list` prints every command.
4. **Redispatch comparator and trading comparisons**: `python tools/make_all.py` (Python 3.11 environment).

Main scripts behind the results:

| Result | Scripts (`src/wedge/gnn/` unless stated) |
|---|---|
| Outage responses, estimator slopes, events and subsets | `outage_v2.py`, `outage_v2_post.py`, `outage_events.py`, `outage_subsets.py` |
| Model against the outages (BritNed, Nemo Link, GB side) | `outage_model_compare.py`, `outage_pooled_gb.py` |
| Held-out outages of ten further HVDC links | `outage_multi.py`, `outage_multi_summary.py`, `outage_multi_confirmed.py` |
| Serbia-Hungary outages | `outage_rs.py`, `outage_rs_events.py` |
| Recovery check of the outage estimator | `outage_synthetic.py` |
| Prediction accuracy and local emission responses | `evaluate.py`, `response_check.py`, `marginal.py` |
| Network responses on the charged borders, feasibility | `netresp.py charged`, `net_summary.py`, `net_feasibility.py` |
| Reference charge and rule comparison | `net_rules.py`, `net_states.py` |
| Charged imports along the response path | `net_taxbase.py` |
| Information frontier and learned rules | `net_frontier.py`, `rule_learning.py` |
| Redispatch comparator, trading comparisons | `src/wedge/*.py` via `tools/make_all.py` |

## License

The code is released under the MIT License (`LICENSE`). The data repository is released under CC BY 4.0; the
third-party data it derives from remain subject to their providers' terms (see `DATA_SOURCES.md` there).

## Citation

See `CITATION.cff`. The citation will be updated with the journal reference and the archive DOI.
