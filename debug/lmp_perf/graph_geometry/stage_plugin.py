# SPDX-License-Identifier: LGPL-3.0-or-later
"""Stage a CMake plugin installation without touching its formal prefix."""

from __future__ import (
    annotations,
)

import argparse
import json
import re
import subprocess
from pathlib import (
    Path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend-libraries", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source = args.build.resolve() / "lmp/plugin/cmake_install.cmake"
    script = source.read_text()
    prefix = re.search(r'set\(CMAKE_INSTALL_PREFIX "([^"]+)"\)', script)[1]
    link_directory = prefix + "/lib/deepmd_lmp"
    assert script.count(link_directory) == 2, source
    # The custom symlink commands embed the configured prefix and ignore
    # CMake's install-prefix override. Both target the isolated stage here.
    script = script.replace(link_directory, str(output / "lib/deepmd_lmp"))
    if args.backend_libraries:
        installed_rpath = re.search(r'NEW_RPATH ("[^"]*"|\[\[.*?\]\])', script)[1]
        assert script.count(installed_rpath) == 2, source
        final_rpath = (
            "[[$ORIGIN:$ORIGIN/..:" + str(args.backend_libraries.resolve()) + "]]"
        )
        script = script.replace(installed_rpath, final_rpath)
    temporary = output.with_suffix(".cmake")
    temporary.write_text(script)
    command = ["cmake", f"-DCMAKE_INSTALL_PREFIX={output}", "-P", str(temporary)]
    subprocess.run(command, check=True)
    plugin = output / "lib/libdeepmd_lmp.so"
    link = output / "lib/deepmd_lmp/dpplugin.so"
    assert plugin.is_file() and link.resolve() == plugin, output
    dynamic = subprocess.check_output(["readelf", "-d", str(plugin)], text=True)
    record = {"source": str(source), "command": command, "dynamic_section": dynamic}
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
