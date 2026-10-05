# Formal run, October 2026: ledger and assignment

The frozen inputs of the formal run on two GB300 hosts (DESIGN.md Section 9 and Section 12.1, gates 2 and 6).

| File | What it holds |
|---|---|
| `calibration.json` | The ledger's inputs: each profile's smoke-based prediction (`smoke_predictions`), the pilot's measured wall seconds (`pilot_seconds`), and the measured ratios of pilot time to prediction per Task (`task_ratios`, `small_task_ratio`, `still_image_ratio`; video and edits 1.0), the allowance for the three blocked profiles, the failure allowance (10%), the per-host gate (22 hours), and the settle (`settle`). |
| `ledger.py` | Derives the ledger from `calibration.json`: `python ledger.py calibration.json > ledger.json`. |
| `ledger.json` | Predicted seconds per profile (174 profiles, 38.0 hours of GPU time, the settle of DESIGN.md 4.6 included: 10 s per side and timed request of the default class). |
| `assignment.json` | `trtmc-aiperf-qual assign` over `ledger.json` for hosts `h1` and `h2`: checkpoint groups longest first, each to the host with less predicted time, ties by name, then host order; each host's list is its planned order. 87 profiles and 19.0 hours a host (20.9 with the failure allowance). |

The ratios come from the pilot at the formal sample sizes with the final configuration (eight Acc copies a side under
one CUDA MPS daemon, both sides' Acc at once, shared Acc selections computed during the build, copies started
together): lfm2-350m 544 s, falcon-rw-1b 414 s, qwen35-4b 975 s, resnet50 190 s, qwen3-vl-2b 692 s; MiniMax-H3
10,901 s, SANA-WM 6,421 s, wan22 5,036 s, qwen-image-edit-2511 4,470 s, qwen36-27b 5,264 s (earlier configuration, kept
as measured).

## Commands

Each host runs its share in the assigned order, then the merged result is checked before the matrix is published:

```bash
# on h1 (and likewise on h2 with --host h2)
trtmc-aiperf-qual assign --environment gb300-perf-serving.yaml --ledger ledger.json --host h1 --host h2 \
    --output assignment.json                      # reproduces assignment.json
trtmc-aiperf-qual run-all --environment gb300-perf-serving.yaml --assignment assignment.json --host h1 \
    --no-prefetch --out-root /runs/results/formal
# after both hosts finished
trtmc-aiperf-qual merge-check --assignment assignment.json h1-root h2-root
trtmc-aiperf-qual summary --assignment assignment.json h1-root h2-root --output summary.md --html summary.html
```

Each host runs a copy of `gb300-perf-serving.yaml` whose `repo` is the checked-out commit under test and whose
`retention.bundle` is `delete_built_unless_error` (a bundle the run built is deleted after its profile; each host
starts with about 300 GB free and its own share's existing bundles).

`--no-prefetch`: a checkpoint downloaded while another profile is timed skews that timing (the gate-1 pilot's
patchtst-granite-official went white while personaplex-7b's 15 GB downloaded), so each profile resolves its own
checkpoints before its build; every checkpoint of either share is cached on its host beforehand (one 0.2 GB one
was not). Nothing else runs on a host during the run.

Every host's `plan.json` records the assignment's digest, its host name, and the campaign inputs; `campaign.jsonl`
records each profile's host, position, and start time; every report records the host, GPU, TensorRT libraries, and
the qualified bundle's sha256.
