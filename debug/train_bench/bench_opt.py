#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201, ANN001, ANN201, TID253, B905
"""Phase timing of the full optimization step.

The step benchmark (``bench.py``) times ``zero_grad + forward + backward``;
the real training loop additionally clips the gradient norm and runs the
optimizer. This harness times all four phases with CUDA events on the real
trainer, isolating the update cost the step benchmark never sees.
Usage: bench_opt.py work/pro.json [--steps N]
"""

import argparse
import json
import logging
from pathlib import (
    Path,
)

import torch

from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt_expt.train.gradient import (
    clip_grad_norm_,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)

logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input")
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--warmup", type=int, default=5)
args = parser.parse_args()

config = json.loads(Path(args.input).read_text())
config = normalize(update_deepmd_input(config, warning=False, dump=None))
trainer = get_trainer(config)
lr = float(trainer.lr_schedule.value(0))
max_norm = float(trainer.gradient_max_norm)
input_dict, label_dict, _ = trainer.get_data(is_train=True)
print(f"optimizer: {trainer.opt_type}  grad_max_norm: {max_norm}")

PHASES = ("forward", "backward", "clip", "optimizer")


def run_step(events):
    trainer.optimizer.zero_grad(set_to_none=True)
    events["t0"].record()
    _, loss, _ = trainer.wrapper(
        **input_dict, cur_lr=lr, label=label_dict, task_key="Default"
    )
    events["t1"].record()
    loss.backward()
    events["t2"].record()
    if max_norm > 0.0:
        clip_grad_norm_(trainer.wrapper.parameters(), max_norm, stable=True)
    events["t3"].record()
    trainer.optimizer.step()
    events["t4"].record()


for _ in range(args.warmup):
    run_step(
        {
            k: torch.cuda.Event(enable_timing=True)
            for k in ("t0", "t1", "t2", "t3", "t4")
        }
    )
torch.cuda.synchronize()

records = []
for _ in range(args.steps):
    events = {
        k: torch.cuda.Event(enable_timing=True) for k in ("t0", "t1", "t2", "t3", "t4")
    }
    run_step(events)
    torch.cuda.synchronize()
    records.append(
        [
            events["t0"].elapsed_time(events["t1"]),
            events["t1"].elapsed_time(events["t2"]),
            events["t2"].elapsed_time(events["t3"]),
            events["t3"].elapsed_time(events["t4"]),
        ]
    )

cols = list(zip(*records))
total = 0.0
for name, col in zip(PHASES, cols):
    ordered = sorted(col)
    med = ordered[len(ordered) // 2]
    total += med
    print(
        f"{name:>9}: median {med:8.3f} ms  min {ordered[0]:8.3f}  max {ordered[-1]:8.3f}"
    )
print(f"{'total':>9}: median {total:8.3f} ms")
