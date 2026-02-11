#!/usr/bin/env python3
"""
Resume Batch 1: Process remaining prompts (ARID1A_CHK1 and WRN_MSI)

Usage:
    python pipeline3/scripts/resume_batch1.py --run-id run_20251113_152031

This script resumes batch 1 from where it left off, processing:
- prompt_04_ARID1A_CHK1.txt
- prompt_05_WRN_microsatellite_instability_MSI.txt
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

# Resume configuration - start from ARID1A_CHK1 (index 3)
BATCH_NUMBER = 1
START_INDEX = 3  # Start from prompt_04_ARID1A_CHK1.txt
END_INDEX = 5    # End at prompt_05 (exclusive, so processes 3 and 4)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=f"Resume lethal_genes_2 mode - Batch {BATCH_NUMBER} (prompts {START_INDEX+1}-{END_INDEX})",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID to continue (e.g., run_20251113_152031)"
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

    return parser.parse_args()


def run_single_prompt(prompt_file: str, args, run_id: str):
    """
    Run pipeline3 for a single prompt file.

    Args:
        prompt_file: Path to the prompt file
        args: Command line arguments
        run_id: Shared run ID for this batch
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

        if result.returncode == 0:
            logging.info(f"✓ Successfully completed {gene_pair_name} in {execution_time:.1f}s")
        else:
            logging.error(f"✗ Failed {gene_pair_name} with return code {result.returncode}")
            logging.error(f"STDOUT: {result.stdout[-500:]}")  # Last 500 chars
            logging.error(f"STDERR: {result.stderr[-500:]}")

    except subprocess.TimeoutExpired:
        end_time = time.time()
        execution_time = end_time - start_time
        logging.error(f"✗ Timeout after {execution_time:.1f}s for {gene_pair_name}")

    except Exception as e:
        logging.error(f"✗ Exception running {gene_pair_name}: {str(e)}")


def main():
    """Main orchestrator function."""
    args = parse_arguments()

    run_id = args.run_id

    # Create output directory structure
    output_base = project_root / "pipeline3" / "output" / "lethal_genes_2" / run_id
    output_base.mkdir(parents=True, exist_ok=True)

    # Setup logging to console only
    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ],
        force=True  # Force reconfiguration even if logging was already initialized
    )

    logging.info("=" * 80)
    logging.info(f"LETHAL GENES 2 - BATCH {BATCH_NUMBER} RESUME")
    logging.info(f"Processing remaining prompts {START_INDEX+1}-{END_INDEX}")
    logging.info("=" * 80)
    logging.info(f"Configuration:")
    logging.info(f"  Run ID: {run_id}")
    logging.info(f"  Population size: {args.population_size}")
    logging.info(f"  Generations: {args.generations}")
    logging.info(f"  Model: {args.model}")
    logging.info(f"  Prompts directory: {args.prompts_dir}")
    logging.info(f"  Output directory: {output_base}")
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
    logging.info(f"Resuming batch {BATCH_NUMBER} - processing {len(batch_prompts)} remaining prompts:")
    for i, pf in enumerate(batch_prompts, start=START_INDEX+1):
        gene_pair = extract_gene_pair_name(pf)
        logging.info(f"  {i}. {gene_pair}")
    logging.info("=" * 80)

    # Process each prompt
    start_time = time.time()

    for i, prompt_file in enumerate(batch_prompts, start=1):
        gene_pair = extract_gene_pair_name(prompt_file)
        logging.info(f"\n{'='*80}")
        logging.info(f"PROMPT {i}/{len(batch_prompts)}: {gene_pair}")
        logging.info(f"{'='*80}")

        run_single_prompt(prompt_file, args, run_id)

    end_time = time.time()
    total_time = end_time - start_time

    # Final summary
    logging.info("\n" + "=" * 80)
    logging.info(f"BATCH {BATCH_NUMBER} RESUME COMPLETED")
    logging.info("=" * 80)
    logging.info(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logging.info(f"Total prompts processed: {len(batch_prompts)}")
    logging.info(f"Output directory: {output_base}")
    logging.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
