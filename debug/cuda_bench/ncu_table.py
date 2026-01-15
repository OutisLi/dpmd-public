# SPDX-License-Identifier: LGPL-3.0-or-later
"""Print selected Nsight Compute metrics of a report as a compact table."""

from __future__ import (
    annotations,
)

import argparse
import csv
import io
import re
import subprocess

METRICS = {
    "us": "gpu__time_duration.sum",
    "regs": "launch__registers_per_thread",
    "warps%": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "issue%": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "L1%": "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "L1hit": "l1tex__t_sector_hit_rate.pct",
    "L2%": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "L2hit": "lts__t_sector_hit_rate.pct",
    "DRAM%": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "L2>L1MB": "l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum",
    "insts(M)": "smsp__inst_executed.sum",
    "spill": "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--filter", default=".")
    args = parser.parse_args()
    out = subprocess.run(
        [
            "ncu",
            "--import",
            args.report,
            "--page",
            "raw",
            "--csv",
            "--metrics",
            ",".join(METRICS.values()),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = list(csv.reader(io.StringIO(out)))
    header = rows[0]
    print(f"{'kernel':58s} " + " ".join(f"{k:>9s}" for k in METRICS))  # noqa: T201
    for row in rows[2:]:
        record = dict(zip(header, row, strict=False))
        name = record["Kernel Name"]
        if not re.search(args.filter, name):
            continue
        name = re.sub(r"\(.*", "", name.replace("(anonymous namespace)", "anon"))
        name = name.replace("void ", "").replace("deepmd_dpa4c::", "")[:58]
        values = []
        for key, metric in METRICS.items():
            raw = record.get(metric, "").replace(",", "")
            try:
                value = float(raw)
            except ValueError:
                values.append(f"{'-':>9s}")
                continue
            if key == "L2>L1MB":
                value = value * 32 / 1e6
            elif key == "insts(M)":
                value = value / 1e6
            elif key == "spill":
                value = value / 1e6
            values.append(f"{value:9.1f}")
        print(f"{name:58s} " + " ".join(values))  # noqa: T201


if __name__ == "__main__":
    main()
