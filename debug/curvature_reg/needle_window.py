# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""Count batch errors using the optimizer's zero-based step index.

Legacy logs count calls within each process. Their session headers are matched to
``launch.log``; a restart from ``model.ckpt-N.pt`` resumes at the stored step
``N - 1``. A rollback can revisit an optimizer step, and every observed event is
retained. Adding session lengths instead would count discarded work as progress.
Explicit ``session`` headers carry the origin for logs written by the diagnostic
hook. Unknown restart origins are errors rather than estimated step numbers.

Usage: needle_window.py <lo> <hi> <run> [<run> ...]
"""

import re
import sys
from pathlib import (
    Path,
)

ROOT = Path("/nas/outisli/Software/deepmd-kit/debug/curvature_reg/runs")


def launch_steps(run: Path) -> list[int]:
    """Read the optimizer step at each recorded launch or restart."""
    path = run / "launch.log"
    if not path.exists():
        return []
    starts = []
    for line in path.read_text().splitlines():
        if "train_refresh.py" not in line or not re.match(
            r"(?:launched|restarted) at ", line
        ):
            continue
        checkpoint = re.search(r"--restart\s+\S*model(?:_ema)?\.ckpt-(\d+)\.pt", line)
        checkpoint = checkpoint or re.search(r"from step (\d+)", line)
        if "--restart" in line and checkpoint is None:
            raise ValueError(f"Unknown restart checkpoint in {path}: {line}")
        starts.append(max(int(checkpoint.group(1)) - 1, 0) if checkpoint else 0)
    return starts


def entries(run: str) -> list[tuple[int, float, str]]:
    """Return (optimizer step, largest error, nearest-pair distance) for every attempt."""
    out: list[tuple[int, float, str]] = []
    directory = ROOT / run
    starts = launch_steps(directory)
    for log in sorted(directory.glob("batch_max_error_rank*.log")):
        sessions: list[tuple[int | None, list[tuple[int, float, str]]]] = []
        rows: list[tuple[int, float, str]] = []
        origin = pending_origin = None
        for line in log.read_text().splitlines():
            marker = re.fullmatch(
                r"# session start_step=(\d+) step_kind=global_zero_based", line
            )
            if marker:
                pending_origin = int(marker.group(1))
                continue
            if line.startswith("# step "):
                if rows:
                    sessions.append((origin, rows))
                rows = []
                origin, pending_origin = pending_origin, None
                continue
            parts = line.split()
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            rows.append(
                (int(parts[0]), float(parts[1]), parts[4] if len(parts) > 4 else "-")
            )
        if rows:
            sessions.append((origin, rows))
        if any(origin is None for origin, _ in sessions) and len(starts) != len(
            sessions
        ):
            if len(sessions) != 1 or starts:
                raise ValueError(
                    f"Cannot locate restart origins for {log}: "
                    f"{len(sessions)} log sessions and {len(starts)} recorded launches"
                )
        for index, (origin, rows) in enumerate(sessions):
            offset = (
                0
                if origin is not None
                else (starts[index] if starts else 0) - rows[0][0]
            )
            out.extend((step + offset, error, pair) for step, error, pair in rows)
    return sorted(out)


def main() -> None:
    """Print the needle counts of each run over the window."""
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    print(
        f"{'run':34s} {'reached':>8s} {'small':>6s} {'tears':>6s} {'largest':>10s}  tears in window"
    )
    for run in sys.argv[3:]:
        rows = entries(run)
        if not rows:
            print(f"{run:34s} {'no log':>8s}")
            continue
        window = [r for r in rows if lo <= r[0] < hi]
        small = sum(1 for _, e, _ in window if 100.0 < e <= 1000.0)
        tears = [(s, e, p) for s, e, p in window if e > 1000.0]
        largest = max((e for _, e, _ in window), default=0.0)
        detail = "; ".join(f"{s} {e:.0f} eV/A at {p} A" for s, e, p in tears[:3])
        print(
            f"{run:34s} {rows[-1][0]:8d} {small:6d} {len(tears):6d} {largest:10.0f}  {detail}"
        )


if __name__ == "__main__":
    main()
