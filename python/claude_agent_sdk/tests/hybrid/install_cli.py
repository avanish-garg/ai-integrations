"""Install the published baseline CLI beside the fork SDK's virtualenv Python."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    """Extract the pinned release binary without replacing the installed SDK."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-version", default="0.2.153")
    args = parser.parse_args()
    plugin = Path(__file__).resolve().parents[2]
    python = (
        plugin / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    if Path(sys.prefix).resolve() != (plugin / ".venv").resolve():
        parser.error(f"run this helper with {python}")
    filename = "claude.exe" if os.name == "nt" else "claude"
    destination = python.parent / filename
    with tempfile.TemporaryDirectory(prefix="claude-published-cli-") as temp:
        stage = Path(temp)
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--target",
                str(stage),
                "--no-deps",
                "--only-binary",
                ":all:",
                f"claude-agent-sdk=={args.sdk_version}",
            ],
            check=True,
        )
        binary = stage / "claude_agent_sdk" / "_bundled" / filename
        if not binary.is_file():
            parser.error("the selected published SDK wheel has no bundled CLI")
        subprocess.run([str(binary), "--version"], check=True)
        shutil.copy2(binary, destination)
        if os.name != "nt":
            destination.chmod(destination.stat().st_mode | 0o111)
    print(f"CLI installed at {destination}; make test adds it to PATH.")


if __name__ == "__main__":
    main()
