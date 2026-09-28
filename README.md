# GreedySmartCharger

Reproducible companion code for **When Is Greedy Enough? Reserve-Based Energy Management for Grid-Connected Residential Nanogrids** (submission draft, Stephen Makonin).

This repository provides the fixed household experiment, compact reference results, independent checks and commands to regenerate the analysis or rerun the controllers. It contains **no manuscript, cover letter, highlights, LaTeX submission files or paper PDFs**. The exploratory development repository is preserved separately in a private archive.

The main evaluation covers **138 Ausgrid households, 269 continuous 30-day windows, three known synthetic tariffs and five controller families**. All policies permit grid charging. The full matrix contains 68,595 logical rows, 54,069 successful unique physical traces and 807 retained, expected OMG parameter-domain failures. These failures are part of the design, not missing favorable results.

## Start here

Use Python **3.14** (the recorded run used 3.14.4), a virtual environment, and a C compiler on macOS or Linux. Windows users can use WSL. The optional pure-Python backend is slower. Only standard Python tooling and the three pinned scientific packages are needed.

```sh
git clone https://github.com/smakonin/GreedySmartCharger.git
cd GreedySmartCharger
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py build
python reproduce.py test
python reproduce.py verify
```

`build` compiles the unchanged C recurrence locally with floating-point contraction and fast-math disabled. No platform-specific binary is distributed. `test` includes native/reference parity over 400 seeded cases and physical, meter and certificate tests. `verify` checks release hashes, equal-household cost aggregation and the six primary bootstrap screens.

### Recreate the reported analysis, without running controllers

```sh
python reproduce.py review --output runs/review
```

This decompresses the saved case summaries into a new directory, recomputes **all full-matrix analysis fields**, compares them with the reference summary at absolute tolerance `1e-10`, and regenerates six result figures in PNG, SVG and PDF. Reference timing and hindsight diagnostics are reused and identified as saved measurements. This command does not pretend to rerun controllers or hardware timings. The chart contents match the study; submission-specific typography and the conceptual system illustration are outside this repository.

### Download the source data and rerun controllers

```sh
python reproduce.py fetch-data
python reproduce.py run --mode smoke --workers 1 --output runs/smoke
```

The download is a 57 MB qualified preservation copy. Both the ZIP and used annual CSV members are checked against the exact recorded SHA-256 hashes. Source licensing, attribution, clock handling and provenance limits are in [data/README.md](data/README.md).

The smoke test runs **one complete 30-day household window across all 85 configurations and all three tariffs** (255 logical rows), including the expected OMG failures. It compares every recorded numerical metric other than elapsed time with the saved results at absolute tolerance `1e-8`. It is a portability test, not the whole study. Select a registered window with `--case A003_20120701`, for example; available identifiers are in `experiments/household-v3/cases.json`.

For the five primary policies across all 269 windows:

```sh
python reproduce.py run --mode primary --workers 6 --output runs/primary
```

For the **complete matrix and diagnostics used by the article**:

```sh
python reproduce.py run --mode full --workers 6 --output runs/full
python reproduce.py postprocess --output runs/full
```

The original complete controller run took about **1.84 hours with six workers** on its recorded macOS environment, before postprocessing. Allow **at least 20 GiB free disk** and more time on slower machines. The full run writes about 9 GB of traces. This is not a runtime promise. Diagnostic MILP solves have their original 10-second limit, so bounds, statuses and timings can vary with hardware and solver versions. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

Every run uses a new output directory and retains failure records. Existing runs are never overwritten. Outputs and downloaded data are ignored by Git. Numeric agreement is checked rather than assumed; a difference beyond tolerance produces a failing command and a diagnostic file. Neither state variables nor controller parameters are adjusted to force a match.

## Repository map

| Path | Purpose |
| --- | --- |
| `reproduce.py` | Reviewer-facing commands |
| `src/` | Simulation, fixed controllers, reserves, analysis and native C source |
| `tests/` | Physical accounting, causal meter, certificate and native-equivalence tests |
| `experiments/household-v3/` | Exact frozen cases, configurations, matrix and diagnostic selection |
| `data/metadata/` | Original source adjudication, cohort and quality screen |
| `reference/` | Compact saved summaries, diagnostic records and six illustrative traces |
| `provenance/` | Original hashes/environment, portability record and release integrity manifest |
| `runs/` | Generated outputs; absent until commands run |

The simulation permits grid charging, disallows export and preserves SOC and cost accounting across midnight. Forecasts use completed meter intervals and 28 days of history. Immediate local feedback on constant half-hour reconstructed power is an explicit model assumption. Tariffs are synthetic currency-unit scenarios, not historical retail bills. The reference includes adverse and null findings: sharper reserves do not improve average costs, and energy savings can increase cycling and peaks.

HUE was used in earlier development experiments with synthetic PV and different information assumptions. Those outcomes are **not pooled** with this main Ausgrid comparison. The dataset is described by Makonin, *Data in Brief* 23 (2019), 103744, [doi:10.1016/j.dib.2019.103744](https://doi.org/10.1016/j.dib.2019.103744).

## License and citation

The source code is [MIT licensed](LICENSE). Ausgrid source data and derived household traces retain Ausgrid's **CC BY 3.0 Australia** attribution terms; the code license does not relicense the data. `CITATION.cff` identifies this software release. The article is a submission draft, not a published or accepted article. No article DOI is asserted.
