"""nightly_report.py maps workflow job names to plugins; ci.yml owns those names."""

from __future__ import annotations

from pathlib import Path

import nightly_report
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _job(name: str, conclusion: str | None) -> dict:
    return {"name": name, "conclusion": conclusion}


def test_classify_aggregates_results_by_plugin() -> None:
    failing, passing = nightly_report.classify(
        [
            _job("Python (openai_agents) / openai_agents (ubuntu-latest, py3.14)", "success"),
            _job("Python (openai_agents) / openai_agents (ubuntu-latest, py3.10)", "failure"),
            _job("Python (mcp) / mcp (macos-latest, py3.14)", "success"),
            _job("Python (mcp) / matrix", "success"),
            _job("Python (mcp) / mcp (windows-latest, py3.14)", None),
            _job("Conventions and tooling tests", "failure"),
            _job("ci-status", "failure"),
        ]
    )
    assert failing == {"openai_agents"}
    assert passing == {"mcp"}


def test_report_matches_ci_workflow_job_names() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())
    python_jobs = {
        job["name"]
        for job in workflow["jobs"].values()
        if isinstance(job, dict) and str(job.get("uses", "")).endswith("_python-plugin.yml")
    }
    assert python_jobs == {"Python"}, (
        "ci.yml renamed a Python job; update JOB_RE in scripts/ci/nightly_report.py"
    )
    for job in python_jobs:
        match = nightly_report.JOB_RE.match(f"{job} (fakeplug) / fakeplug (ubuntu-latest, py3.14)")
        assert match is not None and match.group("plugin") == "fakeplug"
