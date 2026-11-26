# ESMFold2

This family builds [`biohub/ESMFold2`](https://huggingface.co/biohub/ESMFold2) and
[`biohub/ESMFold2-Fast`](https://huggingface.co/biohub/ESMFold2-Fast) (`EsmFold2Model`) for the `structure_prediction`
Task (`IStructurePrediction`, CLI `trtmc predict-structure`): one protein chain in, an all-atom structure (mmCIF, or
PDB through the C++ Task API) with per-residue pLDDT and pTM out. A bundle is general: the native runtime featurizes
any sequence up to the bundle's maximum length, and every TensorRT plan has dynamic token, atom and sample dimensions.

| Checkpoint | Revision | Folding trunk | MSA encoder | Parameters (F32) |
|---|---|---|---|---|
| `biohub/ESMFold2` | `69869f737beffec5294845ede23db5fc0b4f509e` | 48 layers | yes (query only) | 6.58 B (24.5 GiB) |
| `biohub/ESMFold2-Fast` | `45fe8656f5b3ef493c17fcf9abe9a2968902e712` | 24 layers | no | 6.53 B (24.3 GiB) |

The reference is the transformers port of the model (`transformers==5.24.1`, `EsmFold2Model.fold`); the family
reimplements its single-chain protein path. Both checkpoints carry the same ESMC-6B language model (80 layers, width
2560), which this family builds itself.

## Build and run

```sh
trtmc build biohub/ESMFold2 -o esmfold2.bundle --precision bf16 --max-sequence-length 512
trtmc predict-structure esmfold2.bundle --input protein.fasta --output prediction.cif --seed 0
```

- `--precision bf16` (default) is the qualified build. `fp32` (TF32 off) is an experimental strict-parity build for
  debugging against the fp32 reference.
- `--max-sequence-length` (residues, default 512) sets the plans' largest profile: 2 to 1024. TensorRT reserves each
  plan's activation memory for that profile when the bundle loads, so the bundle's maximum, not the request, sets the
  device memory (see Performance).
- Input: FASTA with one record, or a bare sequence; one chain of 2 to the bundle's maximum residues, letters
  `ACDEFGHIKLMNPQRSTVWY` and `X` (an unknown residue with backbone atoms only, `UNK`, as in the reference). Anything
  else (lowercase, `B`, `Z`, `U`, `O`, `*`, gaps, several records) is rejected with an error.
- Output: the best sample (highest pTM) at `--output`, the other samples as `<stem>_sample_<i>.cif`, and a metadata
  JSON per sample: `ptm`, `complex_plddt`, per-residue `plddt` (0-1, also the B-factor column), `rank`,
  `sample_index`, `seed` and the controls used. The PDB writer (Task `output_format = kPdb`) reproduces the reference's
  `output_to_pdb` record layout.

### Controls

| Task field | ESMFold2 meaning | Default |
|---|---|---|
| `recycling_steps` | trunk loops (`num_loops`): `recycling_steps + 1` trunk iterations, 0 to 64 | 3 (the model cards' examples use `num_loops=3`; the checkpoint config's fallback is 20) |
| `sampling_steps` | configured diffusion steps of the Karras schedule, 1 to 1000 (levels above 256 are skipped, as in the reference) | 0 = the checkpoint's 14 (10 run) |
| `diffusion_samples` | samples per request (the denoiser batch), 1 to the checkpoint's 32 | 0 = 32 |
| `seed` | seed of the family's random streams | -1 = 0 |

The CLI sets `--num-steps` and `--seed`; the other fields are set through the C++ Task API (`StructurePredictionConfig`)
or `trtmc-bench`.

### Randomness

The reference draws its noise from torch (initial pair state, per-loop dropout of the language-model pair — applied at
inference by design — initial coordinates, per-step rotations, translations and churn noise). The family draws the same
quantities from its own counter-based generator (SplitMix64 of seed, stream and index), so a seed gives the same noise
on every GPU and host, and the E2E test runs the reference with only these draws replaced to compare sample by sample.
Native and reference folds of one seed therefore differ only by arithmetic; folds of different seeds differ like any
two diffusion samples.

## What runs where

- TensorRT plans (strongly typed, bf16 graphs keep LayerNorm, softmax, the Fourier noise embedding, coordinates and the
  pair output in fp32, as the reference's bf16 policy does): `esmc` (language model and the layer mix), `embed`
  (atom encoder, single and pair inputs, LM pair), `msa` (ESMFold2 only), `loop` (LM pair update, parcae step and
  folding trunk; once per trunk iteration), `decode`, `conditioning` (diffusion conditioning and per-layer pair bias),
  `denoise` (one diffusion step for all samples) and `confidence` (one sample).
- Host (`runtime/`): FASTA parsing and featurization (residue tables of the reference), the random streams, the EDM
  sampler (noise schedule, random augmentation, Kabsch alignment in double), pTM aggregation, ranking, mmCIF and PDB.

## Validation

1x NVIDIA B300 (143 GiB), CUDA 13.x, TensorRT 11.4; the reference on PyTorch 2.15 with transformers 5.24.1, TF32 off.

- Per plan, fp32, reference intermediates as inputs (ubiquitin, 76 residues): relative L2 at most 1.5e-6 (ESMC),
  5.6e-6 (denoiser) and 1e-6 or less for every other plan, both checkpoints.
- Whole CLI fold, fp32 bundle against the fp32 reference with the family's draws (ubiquitin, 32 samples): all-atom
  RMSD in the shared frame 0.0005 A, lDDT 1.0, pLDDT and pTM within 1e-5.
- BF16 qualification: the bf16 bundle against the fp32 reference, 4 proteins (Trp-cage, villin HP35, GB1, ubiquitin;
  20 to 76 residues) x 2 seeds x 32 samples, next to the reference's own bf16 run with the same draws (worst sample;
  frame RMSD as the median over samples):

| Metric | ESMFold2 native bf16 | ESMFold2 reference bf16 | Fast native bf16 | Fast reference bf16 |
|---|---|---|---|---|
| C-alpha RMSD after superposition (A) | 0.28 | 0.25 | 0.65 | 0.67 |
| C-alpha lDDT | 0.991 | 0.994 | 0.980 | 0.976 |
| pLDDT mean absolute difference | 0.014 | 0.009 | 0.023 | 0.016 |
| pTM absolute difference | 0.020 | 0.017 | 0.026 | 0.028 |
| Median all-atom RMSD in the shared frame (A) | 0.38 | 0.33 | 0.46 | 0.44 |
| All-atom RMSD after superposition (A) | 0.61 | 0.53 | 0.96 | 1.04 |
| All-atom lDDT | 0.950 | 0.963 | 0.928 | 0.907 |

  The native bf16 build deviates from the fp32 reference as much as the reference's own bf16 run (the all-atom rows come
  from a second build of the same cases; their worst samples sit in the flexible ends of Trp-cage and villin). A few
  samples per fold turn their global pose under bf16 rounding in both (frame RMSD of single samples up to 10 A), so the
  shared-frame check takes the median (a reduction for the global pose only: the evidence keeps every sample's
  coordinates in the shared frame), and every sample is gated on its own atoms after superposition and by lDDT. The
  premerge thresholds (`tests/thresholds/`) keep about 2x; a fold from another seed (median frame RMSD 15-18 A), a trunk
  without the language-model pair (lDDT 0.40-0.46) and one sample with its side-chain atoms moved by 2 A or its atoms
  out of order inside each residue (all-atom lDDT 0.60-0.76) fail them.

## Performance

1x NVIDIA B300 (143 GiB), CUDA 13.x, TensorRT 11.4. Controls of the model cards' example: 3 trunk loops, 50 configured
sampling steps (34 run), 32 diffusion samples. Inputs: random single-chain sequences, uniform over the 20 standard
residues (`tests/benchmark_reference.py` defines them and writes them as FASTA). Every timed call is one complete
request, from the sequence to the ranked structure text of all 32 samples: featurization, trunk, diffusion sampler,
confidence head, ranking and structure text. Native: `trtmc-bench` (`public_task_call_wall` of `predict_structure`:
mmCIF text, plus a metadata JSON per sample with the scores and per-residue pLDDT, 3 KB at 128 and 21 KB at 1024
residues), one bf16 bundle per row with `--max-sequence-length` equal to the length; bundle load (9 to 11 s) and engine
build (about 7 min) excluded. Eager and `torch.compile`: `tests/benchmark_reference.py`, the reference `EsmFold2Model`
in bf16 (PyTorch 2.15, transformers 5.24.1) through `prepare_protein_features`, `fold` and `output_to_pdb` (PDB text;
the reference has no mmCIF writer), between CUDA synchronizations; `torch.compile` compiles the trunk, the diffusion
module and the confidence head (default mode) in its first, excluded calls. Both sides make 2 warmup calls, then report
the median of 5 timed calls (`--warmup 2 --iterations 5` on both); peak memory: `nvidia-smi` device memory, sampled
every 100 ms.

| Residues | Native | Eager PyTorch | `torch.compile` | Native peak | Eager peak | `torch.compile` peak |
|---|---|---|---|---|---|---|
| 128 | 0.67 s | 1.42 s | 0.76 s | 14.3 GiB | 19.9 GiB | 19.2 GiB |
| 256 | 1.66 s | 3.51 s | 1.89 s | 17.8 GiB | 31.7 GiB | 32.3 GiB |
| 512 | 4.94 s | 13.1 s | 7.36 s | 30.5 GiB | 87.2 GiB | 88.0 GiB |
| 1024 | 18.0 s | out of memory | not run | 79.5 GiB | > 143 GiB | - |

The native runtime is 2.1x to 2.7x faster than eager PyTorch and 1.1x to 1.5x faster than `torch.compile` (which first
compiles for 326 to 652 s per length), needs 1.3x to 2.9x less memory than either, and folds 1024 residues where eager
PyTorch runs out of memory. One request folds one sequence; the 32 samples run as one batch, so throughput is one
structure (32 samples) per latency (0.67 s at 128 residues: 48 samples/s). Every fold of the validation and the
benchmarks finished with finite coordinates and confidences (46 folds, 1,472 samples; the runtime rejects a non-finite
output).

### Limits

- Sequence length: 2 to 1024 residues per bundle. Device memory grows with the square of the bundle's maximum (79.5 GiB
  at 1024), and TensorRT cannot build the plans at 2046 residues; longer chains need chunked pair updates.
- Samples per request: at most the checkpoint's 32 (the denoiser's profile).

## Not supported

- Complexes, nucleic acids, ligands, modified residues, covalent bonds and user MSAs (the publisher's `esm` package
  builds those inputs with RDKit and a chemical-component pickle); the MSA encoder of `ESMFold2` runs on the query
  sequence only, as the reference's protein featurizer does.
- The distogram head, PAE / PDE outputs and ipTM (single chain: reported as 0).
- Batching several sequences in one request.

## Tests

```sh
cmake --build build --target trtmc trtmc_backend_trt trtmc_model_esmfold2 test_esmfold2_host
ctest --test-dir build --output-on-failure -R '^esmfold2_host$'
TRTMC_NATIVE_BUILD_DIR=build python -m pytest families/esmfold2/tests -m "not gpu and not trt" \
  --ignore=families/esmfold2/tests/test_e2e.py -q
python -m pytest families/esmfold2/tests -m "gpu or trt" --ignore=families/esmfold2/tests/test_e2e.py -q
TRTMC_BINARY=build/trtmc TRTMC_RUNTIME_ROOT=build \
  python -m pytest families/esmfold2/tests/test_e2e.py --e2e-testcase esmfold2-ubiquitin-bf16 -q
```

`test_host.py` compares the runtime's featurization, random streams, noise schedule, augmentation, alignment, pTM
weights and PDB text with the reference (CPU). `test_atom_window.py` (GPU) runs random atom layers as a TensorRT plan
and as the reference, row by row, and requires plans with the window radius off by one to fail. `test_e2e.py` builds
the bundle from the cached checkpoint, folds the case with the CLI and compares every sample with the reference fold of
the same seed.

The family's adapted third-party code is attributed in `NOTICE`; the checkpoints' license is recorded in
`ASSET_LICENSES.md`.
