#!/usr/bin/env python3
"""
Orchestrator script for lethal_genes_2 mode batch processing.

This script processes all prompt files from data/lethal_genes/individual_prompts/
directory, running the genetic algorithm for each gene pair separately.

Usage:
    python pipeline3/scripts/run_lethal_genes_2.py [--population-size N] [--generations N] [--model MODEL]

Example:
    python pipeline3/scripts/run_lethal_genes_2.py --population-size 5 --generations 5 --model gpt-5
    python pipeline3/scripts/run_lethal_genes_2.py --test-run  # Process only first 2 prompts
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pipeline3.utils.prompt_loader import list_prompt_files, extract_gene_pair_name


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run lethal_genes_2 mode on all individual prompt files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=5,
        help="Population size for genetic algorithm (default: 5)"
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="Number of generations (default: 5)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model to use (default: gpt-4o). Options: gpt-5, gpt-4o, o3-mini, gemini-2.5-pro, qwen2.5:32b"
    )

    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Test mode: process only first 2 prompt files"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--prompts-dir",
        type=str,
        default="data/lethal_genes/individual_prompts",
        help="Directory containing prompt files (default: data/lethal_genes/individual_prompts)"
    )

    parser.add_argument(
        "--continue-from",
        type=str,
        default=None,
        help="Continue from specific gene pair (e.g., 'KLF5_ARID1A'). Skip all prompts before this one."
    )

    return parser.parse_args()


def run_single_prompt(prompt_file: str, args, results_tracker: dict) -> dict:
    """
    Run pipeline for a single prompt file.

    Args:
        prompt_file: Path to prompt file
        args: Command line arguments
        results_tracker: Dictionary to track results across all runs

    Returns:
        Dictionary with run results
    """
    gene_pair_name = extract_gene_pair_name(prompt_file)

    logging.info("=" * 80)
    logging.info(f"Processing: {gene_pair_name}")
    logging.info(f"Prompt file: {prompt_file}")
    logging.info("=" * 80)

    # Build command
    cmd = [
        sys.executable,
        "pipeline3/main.py",
        "dummy_research_goal",  # Not used in lethal_genes_2 mode
        "--mode", "lethal_genes_2",
        "--prompt-file", prompt_file,
        "--population-size", str(args.population_size),
        "--generations", str(args.generations),
        "--model", args.model,
        "--save-json"
    ]

    if args.verbose:
        cmd.append("--verbose")

    start_time = time.time()

    try:
        # Run the pipeline
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per prompt
        )

        end_time = time.time()
        execution_time = end_time - start_time

        run_result = {
            "gene_pair": gene_pair_name,
            "prompt_file": prompt_file,
            "status": "success" if result.returncode == 0 else "failed",
            "return_code": result.returncode,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }

        if result.returncode == 0:
            logging.info(f"✓ Success: {gene_pair_name} completed in {execution_time:.1f}s")
            results_tracker["successful"] += 1
        else:
            logging.error(f"✗ Failed: {gene_pair_name} (return code: {result.returncode})")
            logging.error(f"Error output: {result.stderr[-500:]}")  # Last 500 chars
            run_result["error"] = result.stderr[-500:]
            results_tracker["failed"] += 1

        # Store stdout/stderr for debugging
        run_result["stdout_tail"] = result.stdout[-1000:] if result.stdout else ""
        run_result["stderr_tail"] = result.stderr[-1000:] if result.stderr else ""

        return run_result

    except subprocess.TimeoutExpired:
        end_time = time.time()
        execution_time = end_time - start_time
        logging.error(f"✗ Timeout: {gene_pair_name} exceeded 2 hour limit")
        results_tracker["failed"] += 1

        return {
            "gene_pair": gene_pair_name,
            "prompt_file": prompt_file,
            "status": "timeout",
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "error": "Execution exceeded 2 hour timeout"
        }

    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        logging.error(f"✗ Exception: {gene_pair_name} - {str(e)}")
        results_tracker["failed"] += 1

        return {
            "gene_pair": gene_pair_name,
            "prompt_file": prompt_file,
            "status": "exception",
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


def save_progress(results: list, output_file: str):
    """Save progress to JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "results": results,
                "total_runs": len(results),
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        logging.info(f"Progress saved to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save progress: {e}")


def print_summary(results: list, total_time: float, results_tracker: dict):
    """Print execution summary."""
    print("\n" + "=" * 80)
    print("LETHAL_GENES_2 BATCH EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total prompts processed: {len(results)}")
    print(f"Successful: {results_tracker['successful']}")
    print(f"Failed: {results_tracker['failed']}")
    print(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    if results:
        avg_time = sum(r['execution_time'] for r in results) / len(results)
        print(f"Average time per prompt: {avg_time:.1f}s")

    # Show failed runs
    failed_runs = [r for r in results if r['status'] != 'success']
    if failed_runs:
        print("\nFailed runs:")
        for run in failed_runs:
            print(f"  - {run['gene_pair']}: {run['status']}")
            if 'error' in run:
                print(f"    Error: {run['error'][:100]}")

    print("=" * 80)


def main():
    """Main entry point."""
    args = parse_arguments()

    # Setup logging
    log_file = f"lethal_genes_2_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info("Starting lethal_genes_2 batch processing")
    logging.info(f"Population size: {args.population_size}")
    logging.info(f"Generations: {args.generations}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Test mode: {args.test_run}")

    # Get prompt files
    prompts_dir = os.path.join(project_root, args.prompts_dir)
    try:
        prompt_files = list_prompt_files(prompts_dir)
    except Exception as e:
        logging.error(f"Failed to list prompt files: {e}")
        return 1

    logging.info(f"Found {len(prompt_files)} prompt files in {prompts_dir}")

    # Filter based on --continue-from if specified
    if args.continue_from:
        # Find index of prompt to continue from
        continue_idx = None
        for idx, pf in enumerate(prompt_files):
            if args.continue_from in pf:
                continue_idx = idx
                break

        if continue_idx is not None:
            prompt_files = prompt_files[continue_idx:]
            logging.info(f"Continuing from {args.continue_from}, processing {len(prompt_files)} remaining prompts")
        else:
            logging.warning(f"Could not find prompt matching '{args.continue_from}', processing all prompts")

    # Test mode: only process first 2 prompts
    if args.test_run:
        prompt_files = prompt_files[:2]
        logging.info(f"Test mode: processing only first {len(prompt_files)} prompts")

    # Results tracking
    results = []
    results_tracker = {"successful": 0, "failed": 0}
    output_file = f"lethal_genes_2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    start_time = time.time()

    # Process each prompt file
    for idx, prompt_file in enumerate(prompt_files, 1):
        logging.info(f"\nProcessing prompt {idx}/{len(prompt_files)}")

        run_result = run_single_prompt(prompt_file, args, results_tracker)
        results.append(run_result)

        # Save progress after each run
        save_progress(results, output_file)

    end_time = time.time()
    total_time = end_time - start_time

    # Final summary
    print_summary(results, total_time, results_tracker)

    # Save final results
    save_progress(results, output_file)
    logging.info(f"Final results saved to {output_file}")
    logging.info(f"Log file: {log_file}")

    # Return exit code based on success
    return 0 if results_tracker["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
