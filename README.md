# JustInTimeDecoding
Data-generation used for simulating the JIT decoder in a cubic lattice, including twisted errors induced by the nonabelian stabilizers of the twisted quantum double model and the heralding heuristic proposed in arxiv/2604.02033.  

## Module layout

Model-independent machinery first, then one module per model.

- geometry.py: lattice indexing, time-depth rule, last-step measured-edge mask.
- lattice.py: cubic (2+1)d lattice construction — incidence matrix, edge
  endpoints, neighbor-edge lookup. Shared by every quantum double model.
- decoder.py: JIT decoding protocol and the Z2 logical-error check
  (`is_logical_error_z2`, plus `z2_logical_error(error_type)`, the per-channel
  check the general runner asks for).
- runner.py: the general layered data generator. A run is a stack of
  `LayerSpec`s; each layer declares its `ChannelSpec`s (channel key, the group
  representing the channel, and the logical-error function of that group) and
  the model-specific callables `generate_delegated_errors` / `herald_links`.
  Per repetition it samples the physical errors of every layer and channel,
  decodes each layer channel by channel (just in time, or globally for a layer
  with `decoding="global"`), stops at the first layer with a logical error, and
  otherwise delegates that layer's flux into the next one and heralds its
  links. Entry points: `run_layered_simulation`, `run_physical_error_sweep`.
- twisted.py: everything specific to the twisted quantum double — twisted-Z
  error generation, the Completing-the-Loop heralding (both the legacy matching
  builder and the `herald_links` adapter), the twist masks, and
  `make_twisted_layer_specs(p_phys, heralded)`, which is the two-layer /
  three-Z2-channels-per-layer stack the general runner needs.
- TQD_runner.py: the original two-layer twisted quantum double driver
  (`run_full_simulation`, `run_x_only_simulation`), kept for reproducing the
  published datasets. `make_twisted_layer_specs` reproduces the same protocol
  through the general runner.
- multilayer.py: N-layer generalization where every layer is corrected just in
  time, interleaved per time step; flux loops of layer i delegate errors into
  layer i+1. Per-layer delegated-error generation and decoder-side accounting
  (heralded 0-weight links) are passed in as functions (LayerSpec); defaults
  reproduce the twisted-error behavior of the two-layer code.
  - run_multilayer_simulation (JIT cascade + global-cascade baseline)
- cluster/: Slurm array worker and collector for multilayer sweeps
  (multilayer_worker.py, multilayer_collect.py), driven by multilayer.slurm.sh.
- cli.py: command-line wrapper preserving legacy argument order.

## Quick usage

```bash
python -m shared_simulation.cli results/xz_errs 9 0.02 0.02 10000 0
```

Or from Python:

```python
from shared_simulation.TQD_runner import run_full_simulation

counters = run_full_simulation(
    linear_size=9,
    px=0.02,
    pz=0.02,
    repetitions=10000,
    output_dir="results/xz_errs",
    boundary="OBC",
    run_id=0,
    use_jit=True,
)
print(counters)
```

## General layered runner

Any quantum double model on the cubic lattice is described to `runner.py` as a
stack of layers; the twisted quantum double is the two-layer instance of it:

```python
from shared_simulation.runner import run_layered_simulation
from shared_simulation.twisted import make_twisted_layer_specs

result = run_layered_simulation(
    linear_size=9,
    layer_specs=make_twisted_layer_specs(0.02, heralded=True),
    repetitions=1000,
    model="tqd",
    output_dir="results/tqd",
    run_id=0,
    tag="herald",
)
print(result["logical_error_rate"])
```

A different model only replaces the three callables and the channel groups:

```python
from shared_simulation.decoder import z2_logical_error
from shared_simulation.runner import ChannelSpec, LayerSpec

layer = LayerSpec(
    channels=(ChannelSpec(key="e", group="Z2", is_logical_error=z2_logical_error("z")),),
    noise_probability=0.02,
    decoding="jit",                       # or "global" for an offline layer
    generate_delegated_errors=my_generator,  # (context, parent) -> {channel: mask}
    herald_links=my_herald,                  # (context, parent) -> {channel: mask}
)
```

`parent` (a `LayerView`) exposes the layer below: its correction, the offline
MWPM correction of the same syndrome (both decoder-visible, and what
Completing-the-Loop compares), and — for generation only — the ground-truth
residual. A channel with a group other than Z2 supplies its own
`is_logical_error`; that is the only place the group enters.

Note: the package-level `LayerSpec` is the general runner's; the interleaved
multi-layer pipeline has its own, `multilayer.LayerSpec`.

## Twisted quantum double: p_log vs p_phys study

Grid: `L in {9, 11}`, heralding in {plain, Completing-the-Loop},
`p_phys = 0 + (3e-2 - 0)/40 * i` for `i = 0..39` (0 to 2.925e-2 in steps of
7.5e-4), 10^6 reps/point, two layers with three Z2 channels each. The plain and
heralded curves are computed on the *same* noise realizations (the per-rep seed
excludes the heralding flag), so their difference is a paired comparison — and
the plot colors those two accounting options, distinguishing L by marker and
line style.

- cluster/tqd_worker.py: resumable array worker; one task = one rep-chunk of one
  (L, heralding) group across all 40 p values (`--print-plan` for the task list).
- cluster/tqd_collect.py: sums the chunks, prints the table, writes a CSV and a
  summary pickle, and plots p_log vs p_phys on linear axes with Wilson 95%
  intervals (`--yscale log` for the small-p tail).
- cluster/tqd_study.slurm.sh: 200-task array driver (12 h per submission).

Scale: 10^6 reps/point is ~28k core-hours (~139 h per task on a 200-job
array), so no single job finishes a chunk. Each submission advances every
chunk by its wall budget and checkpoints; the study takes roughly a dozen
submissions. Because the p values are sampled round-robin, the partial data
is a complete curve at lower statistics rather than a half-empty one, so it
is worth collecting and plotting between submissions.

```bash
sbatch cluster/tqd_study.slurm.sh
```

Check what is finished — the checkpoints, not the queue, are the authority:

```bash
python cluster/tqd_worker.py --print-status --output-dir results/tqd
```

It prints the completed fraction per (L, heralding) group, the core-hours
left with a submission-count estimate, and either `STUDY COMPLETE` or the
`sbatch --array=...` line that resumes the unfinished tasks. Aggregate and
plot at any point:

```bash
python cluster/tqd_collect.py --results-dir results/tqd
```

### Same study, constant-speed commit

The parallel run of the study above with the *only* difference being the JIT
commit rule: `decoder.constant_speed_commit` instead of
`decoder.classic_commit`. Same grid, same 200-task plan, same per-rep seeds —
it is the same `cluster/tqd_worker.py`, invoked with `--commit constant-speed`
and its own output directory — so the two curves are a paired comparison of the
commit rule alone.

```bash
bash cluster/tqd_cs_submit.sh          # array + collect/plot in one go
```

That submits `cluster/tqd_cs_study.slurm.sh` (200-task array, results into
`results/tqd_cs`) and then `cluster/tqd_cs_collect.slurm.sh` with
`--dependency=afterok` on it, so the table, CSV and both figures
(`tqd_cs_plog_vs_pphys_linear.pdf`, `..._log.pdf`) are regenerated as soon as
the submission ends. Progress and resume, as for the classic study:

```bash
python cluster/tqd_worker.py --print-status \
    --output-dir results/tqd_cs --commit constant-speed
bash cluster/tqd_cs_submit.sh 7,19-42   # resume the ids it prints
```

Two things to keep in mind when reading the result:

- `constant_speed_commit` raises `decoder.CommitRejected` on a proposal it is
  not defined on — a time-like cluster that does not end on a defect in its
  future-most slice. The worker tallies those repetitions per p, skips them (a
  deterministic reseed would only refuse them again) and the collector drops
  them from the `p_log` denominator, printing the rate; `--rejected-as-errors`
  gives the conservative bound instead. Measured over 4,800 repetitions across
  both L, both heralding options and p up to 2.925e-2 the rate is **zero**, so a
  nonzero `rejected` column is worth investigating rather than expected
  attrition.
- `results/tqd/tqd_plog_vs_pphys_linear.pdf` was collected at 783,042 of the
  10^6 reps/point, so stop re-submitting at a comparable fraction to match its
  statistics (or run to 10^6 for a sharper curve).

### Same study, constant-speed commit with a classic last step

The constant-speed data have no threshold because of the *last* JIT step: it
sees the full lattice and must close every open syndrome pair, but the walk
closes only clusters of at most two edges per step, so any pair still three or
more sites apart is left as an open residual string, which the X logical-error
check flags. A 3-chain in either of the last two slices is enough, giving a
floor of order `L^2 p^3` at every L. `--commit constant-speed-flush` keeps the
walk on every step but commits classically on the last one
(`decoder.jit_decode_full(..., final_commit=classic_commit)`,
`LayerSpec.final_commit`), which removes that floor and leaves only the genuine
cost of the walk (larger residual loops, more delegated twisted errors).

```bash
bash cluster/tqd_csf_submit.sh          # array + collect/plot in one go
python cluster/tqd_worker.py --print-status \
    --output-dir results/tqd_csf --commit constant-speed-flush
bash cluster/tqd_csf_submit.sh 7,19-42  # resume the ids it prints
```

Results go to `results/tqd_csf` (chunk tag `csf_`). The deterministic check
behind the diagnosis is `diagnostics/cs_endgame_check.py`: it enumerates the
weight-3 configurations in the last slices and shows which ones the plain
constant-speed rule leaves open and that the flush closes them.

### Large-L extension: L in {13, 15, 17}

Same study, three more lattice sizes, so the family becomes
`L in {9, 11, 13, 15, 17}`. It is the same worker again — only `--L-list` and
the output directory change — and it runs either commit rule:

```bash
bash cluster/tqd_largeL_submit.sh                 # constant-speed (default)
COMMIT=classic bash cluster/tqd_largeL_submit.sh  # the classic rule instead
```

That submits `cluster/tqd_largeL.slurm.sh` (200-task array over the six
`(L, heralding)` groups, into `results/tqd_cs_largeL` or `results/tqd_largeL`)
and then `cluster/tqd_largeL_collect.slurm.sh` with `--dependency=afterok` on
it. The collect job reads the base directory *and* the extension in one pass,
so the figures show all five sizes; it skips the base directory if it holds no
chunks yet.

**The plots are split by accounting option** (`--split-heralding`): one figure
for plain and one for heralded, each with a curve per L and color encoding L —
the combined two-size figure does not stay legible at five. Both scales are
produced:

```
results/tqd_cs_largeL/tqd_cs_plog_vs_pphys_linear_plain.pdf
results/tqd_cs_largeL/tqd_cs_plog_vs_pphys_linear_herald.pdf
results/tqd_cs_largeL/tqd_cs_plog_vs_pphys_log_plain.pdf
results/tqd_cs_largeL/tqd_cs_plog_vs_pphys_log_herald.pdf
```

Any collect run can be split the same way, and `--results-dir` now takes
several directories:

```bash
python cluster/tqd_collect.py     --results-dir results/tqd_cs results/tqd_cs_largeL     --plot results/tqd_cs_largeL/tqd_cs_plog_vs_pphys_linear.pdf     --split-heralding
```

Progress and resume for the extension:

```bash
python cluster/tqd_worker.py --print-status --L-list 13,15,17     --output-dir results/tqd_cs_largeL --commit constant-speed
bash cluster/tqd_largeL_submit.sh 7,19-42
```

Scale: measured 0.054 / 0.086 / 0.074 / 0.106 / 0.124 / 0.202 s per repetition
for (13, 15, 17) x (plain, heralded) with the constant-speed rule — about
**7,200 core-hours** for the full 10^6 reps/point, ~36 h per task on the
200-job array, so roughly three submissions. `COST_PER_REP` carries entries for
these sizes so the chunk allocation (17/26/23/33/38/63 tasks) is balanced.

## Multi-layer JIT

```python
from shared_simulation.multilayer import (
    make_default_layer_specs,
    run_multilayer_simulation,
)

result = run_multilayer_simulation(
    linear_size=9,
    layer_specs=make_default_layer_specs([0.02, 0.02, 0.02]),
    repetitions=10000,
    output_dir="results/multilayer",
    boundary="OBC",
    run_id=0,
)
print(result)
```

Custom physics per layer: pass your own `generate_delegated_errors` /
`account_delegated_errors` callables in each `LayerSpec` (see the contracts in
the multilayer.py module docstring). Error configurations are plain dicts
whose key set is declared per run via `LayerSpec.channels` — the pipeline is
not tied to the color code; the b/g/r keys (`TWISTED_CHANNELS`) are used only
by the default twisted-error functions.

Cluster sweep: edit the configuration block in `multilayer.slurm.sh`
(200-task array), then `sbatch multilayer.slurm.sh` from the repository root
and aggregate with `python cluster/multilayer_collect.py` once the array
finishes.

## Complexity entropy (depth-restricted)

Numerically accessible proxy for the complexity entropy of arXiv:2403.04828,
specialized to stabilizer states: all Renyi orders of the measured
distribution coincide, so the depth-d Clifford-restricted complexity entropy
reduces to minimizing the GF(2) rank of the tableau's X-block over depth-d
brickwork circuits (see the module docstring of complexity_entropy.py for
the exact statement and its relation to the paper's hypothesis-testing
measure).

- complexity_entropy.py: pure-numpy Sp(2n,2) tableau machinery, toric-code
  builder, brickwork circuit layers, simulated-annealing circuit search
  (certified upper bounds on H^(d), monotone in d by construction), Z-basis
  sampler and an unbiased collision-entropy (Renyi-2) estimator with
  jackknife error bars for non-stabilizer sample data (e.g. D4 quantum
  double).
- cluster/centropy_worker.py: resumable Slurm array worker sweeping
  L in {4,6,8,10,12}, depth in {1..20}, 16 annealing restarts per point
  (`--print-plan` for the task list, `--self-test` for consistency checks).
- cluster/centropy_collect.py: aggregates CENT_*.pkl chunks, independently
  re-verifies every stored best circuit, builds monotone envelopes
  H_env(d) anchored at the exact H(0) = L^2 - 1, tests the coarse-graining
  scaling H ~ (L/d)^2, and plots curves / scaling / collapse.
- cluster/centropy_study.slurm.sh: 200-task padded array driver (extra ids
  are no-ops; tighten with `--print-plan`).

Cluster sweep: `sbatch cluster/centropy_study.slurm.sh` from the repository
root, re-submit as needed until all chunks report complete, then aggregate
with `python cluster/centropy_collect.py --results-dir results/centropy`.
