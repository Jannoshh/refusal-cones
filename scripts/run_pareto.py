#!/usr/bin/env python3
"""
Run Pareto discovery on Modal and download results locally.

Usage:
    uv run python scripts/run_pareto.py
    uv run python scripts/run_pareto.py --high-budget
    uv run python scripts/run_pareto.py --n-init 50 --n-iter 100
"""

import argparse
import subprocess
import re
import sys
from pathlib import Path
from datetime import datetime


def run_modal_and_download(args):
    """Run Modal command and download results."""

    # Build modal command
    if args.high_budget:
        cmd = ["uv", "run", "modal", "run", "modal_app.py::run_high_budget_discovery"]
    else:
        cmd = [
            "uv", "run", "modal", "run", "modal_app.py::run_pareto_discovery",
            "--n-init-samples", str(args.n_init),
            "--n-pareto-iterations", str(args.n_iter),
            "--max-measurements", str(args.max_measurements),
        ]

    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    # Run and capture output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    output_lines = []
    timestamp = None

    # Stream output and capture timestamp
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)

        # Look for timestamp in saved paths
        # Pattern: pareto_discovery_20260119_212053.json or high_budget_20260119_204649.json
        match = re.search(r'(\d{8}_\d{6})\.json', line)
        if match:
            timestamp = match.group(1)

    process.wait()

    if process.returncode != 0:
        print(f"\nModal command failed with code {process.returncode}")
        return False

    if not timestamp:
        print("\nCould not find timestamp in output. Check Modal dashboard for results.")
        return False

    print("\n" + "=" * 60)
    print(f"Timestamp: {timestamp}")

    # Create local results directory
    if args.high_budget:
        prefix = "high_budget"
        subdir = f"high_budget_{timestamp}"
    else:
        prefix = "pareto"
        subdir = f"pareto_ace_{timestamp}"

    results_dir = Path("results") / subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading results to: {results_dir}")

    # Files to download
    if args.high_budget:
        files = [
            f"high_budget_{timestamp}.json",
            f"high_budget_vectors_{timestamp}.pt",
            f"high_budget_plot_{timestamp}.png",
        ]
    else:
        files = [
            f"pareto_discovery_{timestamp}.json",
            f"pareto_vectors_{timestamp}.pt",
            f"pareto_plot_{timestamp}.png",
        ]

    # Download each file
    for filename in files:
        local_path = results_dir / filename
        download_cmd = [
            "uv", "run", "modal", "volume", "get",
            "refusal-cones-results", filename,
            str(local_path), "--force"
        ]

        result = subprocess.run(download_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Downloaded: {filename}")
        else:
            print(f"  Failed to download: {filename}")
            if result.stderr:
                print(f"    Error: {result.stderr.strip()}")

    print("\n" + "=" * 60)
    print(f"Results saved to: {results_dir}")
    print("\nFiles:")
    for f in results_dir.iterdir():
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Run Pareto discovery on Modal")
    parser.add_argument("--high-budget", action="store_true",
                        help="Run high-budget discovery (more iterations)")
    parser.add_argument("--n-init", type=int, default=30,
                        help="Number of initial samples (default: 30)")
    parser.add_argument("--n-iter", type=int, default=70,
                        help="Number of Pareto iterations (default: 70)")
    parser.add_argument("--max-measurements", type=int, default=150,
                        help="Maximum measurements (default: 150)")

    args = parser.parse_args()

    success = run_modal_and_download(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
