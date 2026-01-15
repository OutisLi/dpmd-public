# Force-spike experiments

The research protocol, acceptance criteria, experiment ledger, and conclusions are maintained in [force_spike.md](../../doc/outisli/force_spike.md). This directory contains the training adapters, evaluation tools, and diagnostics used by those experiments.

## Training and evaluation

| Purpose                                               | Entry points                                                  |
| ----------------------------------------------------- | ------------------------------------------------------------- |
| Training with recorded experimental options           | `train_refresh.py`                                            |
| Training and checkpoint observation in one allocation | `run_recipe.py`, `slurm_recipe.sh`, `watch_run.py`            |
| Single-GPU remote or Slurm launch                     | `remote_one.sh`, `slurm_one.sh`                               |
| Hydrogen launch and evaluation                        | `local_h.sh`, `remote_h.sh`, `slurm_h.sh`, `h_eval_chain.sh`  |
| OMat24 checkpoint evaluation                          | `twin_eval_chain.sh`, `twin_eval.sh`, `copy_refresh_ckpts.sh` |
| Complete isolated-pair curves                         | `dimer_survey.py`, `survey_series.py`, `survey_batch.sh`      |
| Fixed-set validation                                  | `matrix_accuracy.sh`, `pro_scan_fast.py`, `h_accuracy.py`     |
| Training events and matched-checkpoint summaries      | `needle_window.py`, `read_matrix.py`                          |
| Release checkpoint evaluation and comparison          | `release_eval_chain.sh`, `release_compare.py`                 |
| Experimental export and force-loss checks             | `verify_experiment_export.py`                                 |

Launch inputs and checkpoint destinations belong under `runs/<run_name>/`. Each experiment retains its input, structural options, checkpoints, pair curves, and validation arrays. Evaluation chains must finish before their allocation is released. Read the entry point's arguments and the run's configuration before launching or resuming an experiment.

## Model adapters and diagnostics

`radial_experiment.py`, `attention_normalization.py`, `message_envelope.py`, `edge_type_neighbors.py`, `seed_radial_gate.py`, `fitting_normalization.py`, and `plain_fitting.py` implement the structural experiments. Reference and geometry adapters are in `original_reference.py`, `reference_loss.py`, `cluster_anchors.py`, and `replay_anchors.py`.

`diagnose/repro_spike.py` restores the experimental options recorded beside a checkpoint. Its imports include historical adapters in `train_curv.py` and the `sezm_attnres*` modules; these remain necessary to reconstruct earlier checkpoints. Load an experimental model through the recorded options before comparing its predictions.

`diagnose/` contains the curvature decompositions, radial/type/seed interventions, captured-frame replays, and per-atom contact classification. `heal_spike.py`, `traj_curvature.py`, and `eval_model.py` also provide helpers used by the hydrogen and pair-evaluation entry points. The OMat24 frame probes in `contact_anchor_probe.py`, `hard_contact_series.py`, and `gain_probe.py` are called by the checkpoint evaluation chain.

## Local artifacts

Training runs and their generated data are ignored by Git. The ignore rules also cover numerical arrays, checkpoints, CSV exports, figures, logs, caches, and the external `fairchem_src/` reference checkout. These local artifacts are separate from the versioned experiment source; cleaning source files does not remove the recorded experimental data.
