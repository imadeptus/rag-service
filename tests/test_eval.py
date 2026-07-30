import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_eval(*threshold_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "eval/run_eval.py",
            "--k",
            "3",
            *threshold_args,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_eval_cli_passes_when_quality_meets_thresholds():
    result = run_eval("--min-hit", "0.85", "--min-mrr", "0.7")

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["documents"] == 5
    assert report["questions"] >= 15
    assert report["hit@3"] >= 0.85
    assert report["mrr"] >= 0.7
    assert report["status"] == "PASS"


def test_eval_cli_fails_when_quality_misses_thresholds():
    result = run_eval("--min-hit", "1.01")

    assert result.returncode == 1, result.stdout + result.stderr
    assert json.loads(result.stdout)["status"] == "FAIL"


def test_eval_threshold_uses_unrounded_metric(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "known.md").write_text("alpha beta", encoding="utf-8")
    golden_path = tmp_path / "golden.jsonl"
    rows = [
        {"question": "alpha", "relevant_doc_ids": ["known"]},
        {"question": "beta", "relevant_doc_ids": ["known"]},
        {"question": "offcorpus", "relevant_doc_ids": ["missing"]},
    ]
    golden_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "eval/run_eval.py",
            "--docs",
            str(docs_dir),
            "--golden",
            str(golden_path),
            "--k",
            "1",
            "--min-hit",
            "0.667",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
