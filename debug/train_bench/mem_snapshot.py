#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201, ANN201, TID253
"""Peak-memory decomposition of one training step.

Runs a few steps from the bench harness with allocation history enabled and
buckets the live allocations at the peak by size, printing the largest
buckets. Usage: mem_snapshot.py work/neo.json
"""

import json
import logging
import sys
from collections import (
    defaultdict,
)
from pathlib import (
    Path,
)

import torch

from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)

logging.basicConfig(level=logging.ERROR)

config = json.loads(Path(sys.argv[1]).read_text())
config = normalize(update_deepmd_input(config, warning=False, dump=None))
trainer = get_trainer(config)
lr = float(trainer.lr_schedule.value(0))
input_dict, label_dict, _ = trainer.get_data(is_train=True)


def step():
    trainer.wrapper.zero_grad(set_to_none=True)
    _, loss, _ = trainer.wrapper(
        **input_dict, cur_lr=lr, label=label_dict, task_key="Default"
    )
    loss.backward()


for _ in range(3):
    step()
torch.cuda.synchronize()

torch.cuda.memory._record_memory_history(max_entries=200000, stacks="python")
torch.cuda.reset_peak_memory_stats()
step()
torch.cuda.synchronize()
snap = torch.cuda.memory._snapshot()
torch.cuda.memory._record_memory_history(enabled=None)

peak = torch.cuda.max_memory_allocated() / 2**30
print(f"peak allocated during one step: {peak:.2f} GiB")

# Replay the allocation trace to the in-step peak and bucket the live
# allocations at that instant by block size.
events = []
for trace in snap["device_traces"]:
    for ev in trace:
        if ev["action"] in ("alloc", "free_completed"):
            events.append(ev)

live = {}
cur = 0
best = 0
best_live = {}
for ev in events:
    addr, sz = ev["addr"], ev["size"]
    if ev["action"] == "alloc":
        live[addr] = ev
        cur += sz
        if cur > best:
            best = cur
            best_live = dict(live)
    else:
        if addr in live:
            cur -= live.pop(addr)["size"]

print(f"traced peak: {best / 2**30:.2f} GiB across {len(best_live)} blocks")
buckets = defaultdict(lambda: [0, 0, None])
for ev in best_live.values():
    b = buckets[ev["size"]]
    b[0] += 1
    b[1] += ev["size"]
    if b[2] is None:
        b[2] = ev.get("frames", [])
rows = sorted(buckets.items(), key=lambda kv: -kv[1][1])[:10]
print(f"{'size':>14} {'count':>6} {'total MiB':>10}")
for sz, (cnt, tot, frames) in rows:
    print(f"{sz:14,d} {cnt:6d} {tot / 2**20:10.1f}")
    shown = 0
    for fr in frames or []:
        name, fn = fr["name"], fr["filename"]
        if "torch/" in fn or "mem_snapshot" in fn:
            continue
        print(f"        {name}  ({fn.rsplit('/', 1)[-1]}:{fr['line']})")
        shown += 1
        if shown >= 3:
            break
