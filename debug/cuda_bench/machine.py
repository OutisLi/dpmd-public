# SPDX-License-Identifier: LGPL-3.0-or-later
"""Load the machine profile of ``machine.sh`` into the environment.

The shell scripts of the harness source ``machine.sh``; the Python scripts
call :func:`load` before they read a ``BENCH_*`` variable, so that both take
the paths of the same profile. A variable that is already set keeps its value;
``LD_LIBRARY_PATH`` takes the value that ``machine.sh`` derives from it.
"""

import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
#: Variables of the profile outside the ``BENCH_*`` family.
_EXTRA = (
    "DPA4C_CHECKPOINTS",
    "OMPI_ALLOW_RUN_AS_ROOT",
    "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM",
)


def load() -> None:
    """Export the variables of the machine profile that are not set yet.

    Raises
    ------
    RuntimeError
        If no profile matches the machine.
    """
    if os.environ.get("BENCH_MACHINE"):
        return
    script = os.path.join(HERE, "machine.sh")
    result = subprocess.run(
        ["bash", "-c", '. "$0" >&2 && env -0', script],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"machine.sh failed: {result.stderr.decode().strip()}")
    for entry in result.stdout.split(b"\0"):
        key, _, value = entry.decode().partition("=")
        if key.startswith("BENCH_") or key in _EXTRA:
            os.environ.setdefault(key, value)
        elif key == "LD_LIBRARY_PATH":
            os.environ[key] = value
