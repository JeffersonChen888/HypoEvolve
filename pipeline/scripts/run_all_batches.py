#!/usr/bin/env python3
"""
Master launcher for all 4 lethal_genes_2 batches.

Launches all 4 batch scripts with a shared run_id for unified output organization.

Behavior:
    - If --run-id is provided: All 4 batches launch in parallel with that run_id
    - If --run-id is NOT provided: Batch 1 launches first and creates run_id,
      then batches 2-4 launch with the same run_id

Usage:
    python pipeline3/scripts/run_all_batches.py [--population-size N] [--generations N] [--model MODEL]

Examples:
    # Let Batch 1 create run_id automatically (recommended)
    python pipeline3/scripts/run_all_batches.py --population-size 6 --generations 3 --model gpt-4o

    # Provide specific run_id (all batches launch in parallel)
    python pipeline3/scripts/run_all_batches.py --run-id run_20251112_235900 --population-size 6 --generations 3

    # Quick test (smaller config)
    python pipeline3/scripts/run_all_batches.py --population-size 3 --generations 2 --model gpt-4o

Output:
    All results saved to: pipeline3/output/lethal_genes_2/{run_id}/
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch all 4 lethal_genes_2 batches in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=6,
        help="Population size for genetic algorithm (default: 6)"
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=3,
        help="Number of generations (default: 3)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model to use (default: gpt-4o). Options: gpt-5, gpt-4o, o3-mini, gemini-2.5-pro, qwen2.5:32b"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Shared run ID for all batches (default: auto-generated timestamp)"
    )

    return parser.parse_args()


def launch_batch(batch_number: int, run_id: str, args):
    """
    Launch a single batch script in the background.

    Args:
        batch_number: Batch number (1-4)
        run_id: Shared run ID for all batches (None to let batch create its own)
        args: Command line arguments

    Returns:
        Popen process object
    """
    script_name = f"run_lethal_genes_2_batch{batch_number}.py"
    script_path = Path(__file__).parent / script_name

    cmd = [
        sys.executable,
        str(script_path),
        "--population-size", str(args.population_size),
        "--generations", str(args.generations),
        "--model", args.model
    ]

    # Only add --run-id if provided
    if run_id:
        cmd.extend(["--run-id", run_id])

    if args.verbose:
        cmd.append("--verbose")

    print(f"Launching Batch {batch_number}: {' '.join(cmd)}")

    # Launch in background
    process = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    return process


def detect_run_id_from_batch1():
    """
    Detect the run_id created by batch 1 by checking the output directory.

    Returns:
        str: The detected run_id, or None if not found
    """
    output_base = project_root / "pipeline3" / "output" / "lethal_genes_2"

    if not output_base.exists():
        return None

    # Find most recent run_* directory
    run_dirs = sorted([d for d in output_base.iterdir() if d.is_dir() and d.name.startswith("run_")],
                     key=lambda x: x.stat().st_mtime, reverse=True)

    if run_dirs:
        return run_dirs[0].name

    return None


def main():
    """Main launcher function."""
    args = parse_arguments()

    print("=" * 80)
    print("LETHAL GENES 2 - PARALLEL BATCH LAUNCHER")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Population size: {args.population_size}")
    print(f"  Generations: {args.generations}")
    print(f"  Model: {args.model}")
    print(f"  Verbose: {args.verbose}")
    print("=" * 80)
    print()
    print("Batch Distribution:")
    print("  Batch 1: Prompts 1-5   (5 prompts)")
    print("  Batch 2: Prompts 6-10  (5 prompts)")
    print("  Batch 3: Prompts 11-15 (5 prompts)")
    print("  Batch 4: Prompts 16-19 (4 prompts)")
    print("  Total: 19 prompts")
    print("=" * 80)
    print()

    processes = {}
    start_time = time.time()

    # Determine run_id strategy
    if args.run_id:
        # User provided run_id - use it for all batches
        run_id = args.run_id
        print(f"Using provided run_id: {run_id}")
        print()

        # Launch all 4 batches in parallel with shared run_id
        for batch_num in range(1, 5):
            proc = launch_batch(batch_num, run_id, args)
            processes[batch_num] = proc
            print(f"✓ Batch {batch_num} launched (PID: {proc.pid})")
            time.sleep(1)  # Small delay between launches
    else:
        # No run_id provided - let batch 1 create it, then detect and share
        print("No run_id provided - Batch 1 will create one automatically")
        print()
        print("=" * 80)
        print("STEP 1: Launching Batch 1 first...")
        print("=" * 80)

        # Launch batch 1 without run_id (it will create one)
        proc1 = launch_batch(1, None, args)
        processes[1] = proc1
        print(f"✓ Batch 1 launched (PID: {proc1.pid})")
        print()

        # Wait for batch 1 to create the output directory
        print("Waiting for Batch 1 to create run_id folder...")
        run_id = None
        for attempt in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            run_id = detect_run_id_from_batch1()
            if run_id:
                print(f"✓ Detected run_id from Batch 1: {run_id}")
                break

        if not run_id:
            print("✗ ERROR: Could not detect run_id from Batch 1 after 30 seconds")
            print("Batch 1 may have failed to start. Check batch1_orchestrator.log")
            return 1

        print()
        print("=" * 80)
        print(f"STEP 2: Launching Batches 2-4 with shared run_id: {run_id}")
        print("=" * 80)

        # Launch remaining batches with the detected run_id
        for batch_num in range(2, 5):
            proc = launch_batch(batch_num, run_id, args)
            processes[batch_num] = proc
            print(f"✓ Batch {batch_num} launched (PID: {proc.pid})")
            time.sleep(1)

    output_dir = f"pipeline3/output/lethal_genes_2/{run_id}/"

    print()
    print("=" * 80)
    print("All 4 batches launched successfully!")
    print("=" * 80)
    print()
    print("Monitoring progress...")
    print(f"All results will be saved to: {output_dir}")
    print(f"  - batch1_results.json, batch1_orchestrator.log")
    print(f"  - batch2_results.json, batch2_orchestrator.log")
    print(f"  - batch3_results.json, batch3_orchestrator.log")
    print(f"  - batch4_results.json, batch4_orchestrator.log")
    print(f"  - Individual gene pair folders (19 total)")
    print()
    print("You can monitor individual batch logs:")
    print(f"  tail -f {output_dir}batch1_orchestrator.log")
    print(f"  tail -f {output_dir}batch2_orchestrator.log")
    print(f"  tail -f {output_dir}batch3_orchestrator.log")
    print(f"  tail -f {output_dir}batch4_orchestrator.log")
    print()
    print("=" * 80)
    print()

    # Wait for all processes to complete
    print("Waiting for all batches to complete...")
    print()

    completed = {}
    while len(completed) < 4:
        for batch_num, proc in processes.items():
            if batch_num not in completed:
                retcode = proc.poll()
                if retcode is not None:
                    completed[batch_num] = retcode
                    elapsed = time.time() - start_time
                    status = "SUCCESS" if retcode == 0 else f"FAILED (code {retcode})"
                    print(f"[{elapsed/60:.1f} min] Batch {batch_num}: {status}")

        if len(completed) < 4:
            time.sleep(10)  # Check every 10 seconds

    end_time = time.time()
    total_time = end_time - start_time

    # Final summary
    print()
    print("=" * 80)
    print("ALL BATCHES COMPLETED")
    print("=" * 80)
    print(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print()

    success_count = sum(1 for code in completed.values() if code == 0)
    failed_count = sum(1 for code in completed.values() if code != 0)

    print(f"Successful batches: {success_count}/4")
    print(f"Failed batches: {failed_count}/4")
    print()

    if failed_count > 0:
        print("Failed batch details:")
        for batch_num, code in completed.items():
            if code != 0:
                print(f"  Batch {batch_num}: return code {code}")
    print()

    print(f"All results saved to: {output_dir}")
    print()
    print("Results structure:")
    print(f"  {output_dir}batch1_results.json  (5 prompts)")
    print(f"  {output_dir}batch2_results.json  (5 prompts)")
    print(f"  {output_dir}batch3_results.json  (5 prompts)")
    print(f"  {output_dir}batch4_results.json  (4 prompts)")
    print(f"  {output_dir}<gene_pair_1>/")
    print(f"  {output_dir}<gene_pair_2>/")
    print("  ... (19 gene pair folders total)")
    print()

    print("=" * 80)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
