# Reproduction scope

The public companion extracts the dependency closure needed by the completed v3 household study. Original numerical-source hashes are in `provenance/original_source_hashes.json`. The native C recurrence is byte-identical to the recorded experiment. Packaging changes replace development-directory paths, compile the shared library for the current platform, and add a command-line entry point. They do not retune controllers, change forecast information, relax numerical tolerances, or alter the registered matrix.

The fixed families are PV-First Greedy (PFG), threshold-based adaptation (THB), Online Modified Greedy (OMG), nominal model predictive control (MPC) and Shadow-Certified Reserve (SCR). Published-family adaptations and the execution-layer allocation rule are preserved in code and registered metadata. Grid charging is allowed for all policies. No export, demand charge, network dynamics or degradation model is added.

## What each command establishes

- `verify`: release integrity plus an independent recomputation of household cost means and six bootstrap screens from saved case points. It does not rerun the simulator.
- `review`: complete regeneration of saved-summary statistical analysis with field-by-field numeric comparison. Hindsight outcomes and runtime samples remain the recorded diagnostics. Six quantitative figures are regenerated from inspectable values and the six included illustrative traces.
- `run --mode smoke`: one window, all 255 logical rows, three tariffs, both resolutions, all sensitivity groups and expected failure behavior. Compares every available numeric result metric except timing against that window's archived records.
- `run --mode primary`: 4,035 logical primary rows, with the same per-metric reference comparison. This is not the sensitivity matrix, and is not accepted by `postprocess` as the complete study.
- `run --mode full`: all 68,595 logical rows and full trace files, preserving aliases and expected failures.
- `postprocess`: full-trace timestamp-rounding audit; complete matrix/hash audit and fixed 10-home/20-window flow checks; 60 perfect-forecast MPC and 60 hindsight MILP references; 6,000 serial decision timings; 4,800 same-state certificate probes; flat-model check; complete analysis and figures.

Only the full matrix is accepted by full postprocessing. Earlier development trials and interrupted attempts are neither pooled nor presented as new household evidence. Their original amendment and environment records remain referenced by the provenance metadata.

## Portability and numerical differences

The original environment was Python 3.14.4, NumPy 2.5.3, SciPy 1.18.1 and Matplotlib 3.11.2 on arm64 macOS 27.0. The C continuation was built by Apple clang 21 with `-O3 -fno-fast-math -ffp-contract=off`. The new build records its compiler, exact command, source hash and library hash in `.build/native-build.json`.

Use the pinned requirements for the closest reproduction. A pure-Python reference backend is available using `GSC_BACKEND=python`; it runs the same continuation recurrence much more slowly. Native/reference parity tests must pass on a newly built environment. Different compiler/platform floating-point behavior may affect a tie or a guarded action. A failed comparison is evidence to investigate, not permission to widen a tolerance.

New NPZ files contain packaging/timing metadata, so byte-for-byte trace ZIP equality is not required for a numerical reproduction. The per-run comparator checks numerical outcomes and the executor checks physical accounting and certificate limits. For the same platform, tighter/equal outcomes are reported by the local validation record. These checks are distinct from a formal floating-point or hardware guarantee.

The hindsight MILP restricts mode within each half hour and uses a 10-second time limit. Hardware and solver progress can change the feasible incumbent, dual bound, gap and status. Retain all of them. Runtime samples are hardware dependent and cannot be expected to match the original milliseconds. The paper's perfect-information references are not implementable policies or universal continuous-switching lower bounds.

## Figure and table mapping

- Household paired costs: `analysis/primary_household_points.csv`; `household_cost_differences` figure.
- Cost, cycles and peaks: `analysis/summary.json` primary entries; `cost_cycling_peak_tradeoffs`.
- Throughput accounting: `summary.json` wear entries; `throughput_penalty_sensitivity`.
- Terminal sensitivity: `configuration_household_means.csv`; `terminal_sensitivity`.
- Household regimes: `summary.json` regimes; `history_regimes`.
- Selected adverse trajectories: selection rule and six reference NPZ traces; `selected_protection_trajectories`.
- Main cost, sufficiency, certificate and timing tables: `summary.json` primary, screens, certificate and runtime entries. Design values are in `experiments/household-v3`.

The system block diagram is a conceptual illustration, not an experimental result. Article text, submission files and that diagram's publication assets are not stored in this repository.
