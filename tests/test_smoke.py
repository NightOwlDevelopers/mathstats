"""Public smoke tests for mathstats (not Harbor scoring tests)."""

import subprocess
import sys


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "mathstats.cli", *args],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": "src"},
    )


def test_help_lists_options():
    r = _run("--help")
    assert r.returncode == 0
    assert "--stat" in r.stdout
    assert "--axis" in r.stdout
