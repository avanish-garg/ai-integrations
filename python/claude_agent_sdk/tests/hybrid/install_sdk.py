"""Build and install a sibling SDK wheel with a selected CLI for local probes."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sysconfig
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-dir", type=Path, required=True)
    parser.add_argument("--cli", type=Path)
    args = parser.parse_args()
    plugin = Path(__file__).resolve().parents[2]
    python = (
        plugin / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    sdk = args.sdk_dir.resolve()
    if not (sdk / "src/claude_agent_sdk/_internal/main_agent_recovery.py").exists():
        parser.error("the sibling SDK checkout does not contain main-agent recovery")
    cli = args.cli
    if cli is None:
        location = subprocess.check_output(
            [
                str(python),
                "-c",
                "import claude_agent_sdk; print(claude_agent_sdk.__file__)",
            ],
            text=True,
        ).strip()
        cli = (
            Path(location).parent
            / "_bundled"
            / ("claude.exe" if os.name == "nt" else "claude")
        )
    cli = cli.resolve()
    if not cli.is_file():
        parser.error("no bundled CLI found; pass --cli with the binary to test")
    env = {
        **os.environ,
        "UV_NO_EDITABLE": "1",
        "UV_CACHE_DIR": os.environ.get(
            "UV_CACHE_DIR", str(Path(tempfile.gettempdir()) / "claude-hybrid-uv-cache")
        ),
    }
    with tempfile.TemporaryDirectory(prefix="claude-main-recovery-") as temp:
        stage = Path(temp)
        for filename in ("pyproject.toml", "README.md", "LICENSE"):
            shutil.copy2(sdk / filename, stage / filename)
        shutil.copytree(
            sdk / "src",
            stage / "src",
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "claude", "claude.exe"
            ),
        )
        bundled = stage / "src/claude_agent_sdk/_bundled"
        bundled.mkdir(exist_ok=True)
        shutil.copy2(cli, bundled / ("claude.exe" if os.name == "nt" else "claude"))
        subprocess.run(["uv", "build", "--wheel"], cwd=stage, env=env, check=True)
        wheel = next((stage / "dist").glob("*.whl"))
        tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        subprocess.run(
            [
                "uv",
                "tool",
                "run",
                "--from",
                "wheel",
                "wheel",
                "tags",
                "--platform-tag",
                tag,
                "--remove",
                str(wheel),
            ],
            cwd=stage,
            env=env,
            check=True,
        )
        wheel = next((stage / "dist").glob("*.whl"))
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--no-deps",
                "--reinstall",
                str(wheel),
            ],
            cwd=plugin,
            env=env,
            check=True,
        )
    subprocess.run([str(cli), "--version"], check=True)
    print("Local SDK installed. Use UV_NO_SYNC=1 with the existing make targets.")


if __name__ == "__main__":
    main()
