# JustInTimeDecoding
Data-generation used for simulating the JIT decoder in a cubic lattice, including twisted errors induced by the nonabelian stabilizers of the quantum twusted double model and the heralding heuristic proposed in *insert arXiv identifier here*.  

## Module layout

- geometry.py: lattice indexing, time-depth rule, last-step measured-edge mask,
  canonical output filename builder.
- lattice.py: incidence matrix construction, neighbor-edge lookup,
  vectorization masks, local edge-mask helper.
- decoder.py: logical-error checks and JIT decoding protocol.
- twisted.py: twisted-Z error generation and loop-closing matching builder.
- runner.py: high-level simulation entry points:
  - run_full_simulation
  - run_x_only_simulation
  - gather_effective_length_data
  - run_res_30d_grid
- cli.py: command-line wrapper preserving legacy argument order.

## Quick usage

```bash
python -m shared_simulation.cli results/xz_errs 9 0.02 0.02 10000 0
```

Or from Python:

```python
from shared_simulation.runner import run_full_simulation

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
