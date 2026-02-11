#!/usr/bin/env python3
"""
Batch 3: Process prompts 11-15

Usage:
    python pipeline3/scripts/run_lethal_genes_2_batch3.py [--population-size N] [--generations N] [--model MODEL]

Example:
    python pipeline3/scripts/run_lethal_genes_2_batch3.py --population-size 6 --generations 3 --model gpt-4o
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

# Batch configuration
BATCH_NUMBER = 3
START_INDEX = 10  # 0-indexed
END_INDEX = 15    # Exclusive (processes prompts 10-14, which are 11-15 in 1-indexed)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=f"Run lethal_genes_2 mode - Batch {BATCH_NUMBER} (prompts {START_INDEX+1}-{END_INDEX})",
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
        "--prompts-dir",
        type=str,
        default="data/lethal_genes/individual_prompts",
        help="Directory containing prompt files (default: data/lethal_genes/individual_prompts)"
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Shared run ID for grouping all batches together (default: auto-generated timestamp)"
    )

    return parser.parse_args()


def run_single_prompt(prompt_file: str, args, run_id: str, results_tracker: dict, results_file: Path) -> dict:
    """
    Run pipeline3 for a single prompt file.

    Args:
        prompt_file: Path to the prompt file
        args: Command line arguments
        run_id: Shared run ID for this batch
        results_tracker: Dictionary tracking all results
        results_file: Path to results JSON file

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
        "--run-id", run_id,  # Pass shared run_id
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
            logging.info(f"✓ Successfully completed {gene_pair_name} in {execution_time:.1f}s")
        else:
            logging.error(f"✗ Failed {gene_pair_name} with return code {result.returncode}")
            logging.error(f"STDOUT: {result.stdout[-500:]}")  # Last 500 chars
            logging.error(f"STDERR: {result.stderr[-500:]}")
            run_result["error"] = result.stderr[-500:]

        # Save progress after each run
        results_tracker["results"].append(run_result)
        save_results(results_tracker, results_file)

        return run_result

    except subprocess.TimeoutExpired:
        end_time = time.time()
        execution_time = end_time - start_time
        logging.error(f"✗ Timeout after {execution_time:.1f}s for {gene_pair_name}")

        run_result = {
            "gene_pair": gene_pair_name,
            "prompt_file": prompt_file,
            "status": "timeout",
            "return_code": -1,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "error": "Execution timeout after 2 hours"
        }

        results_tracker["results"].append(run_result)
        save_results(results_tracker, results_file)

        return run_result

    except Exception as e:
        logging.error(f"✗ Exception running {gene_pair_name}: {str(e)}")

        run_result = {
            "gene_pair": gene_pair_name,
            "prompt_file": prompt_file,
            "status": "error",
            "return_code": -1,
            "execution_time": 0,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

        results_tracker["results"].append(run_result)
        save_results(results_tracker, results_file)

        return run_result


def save_results(results_tracker: dict, results_file: Path):
    """Save results to JSON file."""
    try:
        with open(results_file, 'w') as f:
            json.dump(results_tracker, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save results: {e}")


def main():
    """Main orchestrator function."""
    args = parse_arguments()

    # Use provided run_id, or read from batch 1's saved run_id, or create new one
    if args.run_id:
        run_id = args.run_id
    else:
        # Try to read run_id from batch 1's saved file
        run_id_file = project_root / "pipeline3" / "output" / "lethal_genes_2" / ".last_run_id"
        if run_id_file.exists():
            try:
                with open(run_id_file, 'r') as f:
                    run_id = f.read().strip()
            except Exception as e:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_id = f"run_{timestamp}"
        else:
            # No saved run_id, create new one
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{timestamp}"

    # Create output directory structure
    output_base = project_root / "pipeline3" / "output" / "lethal_genes_2" / run_id
    output_base.mkdir(parents=True, exist_ok=True)

    # Setup logging FIRST (before any logging calls)
    log_file = output_base / f"batch{BATCH_NUMBER}_orchestrator.log"
    results_file = output_base / f"batch{BATCH_NUMBER}_results.json"

    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True  # Force reconfiguration even if logging was already initialized
    )

    # Now log the run_id source
    if args.run_id:
        logging.info(f"Using provided run_id: {run_id}")
    else:
        run_id_file = project_root / "pipeline3" / "output" / "lethal_genes_2" / ".last_run_id"
        if run_id_file.exists():
            logging.info(f"Read run_id from batch 1: {run_id}")
        else:
            logging.info(f"No saved run_id found, created new: {run_id}")

    logging.info("=" * 80)
    logging.info(f"LETHAL GENES 2 - BATCH {BATCH_NUMBER} ORCHESTRATOR")
    logging.info(f"Processing prompts {START_INDEX+1}-{END_INDEX}")
    logging.info("=" * 80)
    logging.info(f"Configuration:")
    logging.info(f"  Run ID: {run_id}")
    logging.info(f"  Population size: {args.population_size}")
    logging.info(f"  Generations: {args.generations}")
    logging.info(f"  Model: {args.model}")
    logging.info(f"  Prompts directory: {args.prompts_dir}")
    logging.info(f"  Output directory: {output_base}")
    logging.info(f"  Log file: {log_file}")
    logging.info(f"  Results file: {results_file}")
    logging.info("=" * 80)

    # Find all prompt files
    prompts_dir = os.path.join(project_root, args.prompts_dir)
    prompt_files = list_prompt_files(prompts_dir)

    if not prompt_files:
        logging.error(f"No prompt files found in {prompts_dir}")
        return 1

    # Select prompts for this batch
    batch_prompts = prompt_files[START_INDEX:END_INDEX]

    logging.info(f"Found {len(prompt_files)} total prompt files")
    logging.info(f"Batch {BATCH_NUMBER} will process {len(batch_prompts)} prompts:")
    for i, pf in enumerate(batch_prompts, start=START_INDEX+1):
        gene_pair = extract_gene_pair_name(pf)
        logging.info(f"  {i}. {gene_pair}")
    logging.info("=" * 80)

    # Results tracker
    results_tracker = {
        "batch_number": BATCH_NUMBER,
        "run_id": run_id,
        "start_index": START_INDEX + 1,  # 1-indexed for display
        "end_index": END_INDEX,
        "total_prompts": len(batch_prompts),
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "population_size": args.population_size,
            "generations": args.generations,
            "model": args.model
        },
        "results": []
    }

    # Process each prompt
    start_time = time.time()

    for i, prompt_file in enumerate(batch_prompts, start=1):
        gene_pair = extract_gene_pair_name(prompt_file)
        logging.info(f"\n{'='*80}")
        logging.info(f"PROMPT {i}/{len(batch_prompts)}: {gene_pair}")
        logging.info(f"{'='*80}")

        run_result = run_single_prompt(prompt_file, args, run_id, results_tracker, results_file)

    end_time = time.time()
    total_time = end_time - start_time

    # Final summary
    logging.info("\n" + "=" * 80)
    logging.info(f"BATCH {BATCH_NUMBER} COMPLETED")
    logging.info("=" * 80)
    logging.info(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logging.info(f"Total prompts processed: {len(batch_prompts)}")

    success_count = sum(1 for r in results_tracker["results"] if r["status"] == "success")
    failed_count = sum(1 for r in results_tracker["results"] if r["status"] in ["failed", "timeout", "error"])

    logging.info(f"Successful: {success_count}")
    logging.info(f"Failed: {failed_count}")
    logging.info(f"Results saved to: {results_file}")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Output directory: {output_base}")
    logging.info("=" * 80)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
