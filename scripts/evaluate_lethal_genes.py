#!/usr/bin/env python3
"""
Lethal Genes Evaluation Script for Pipeline3

This script runs Pipeline3 on all gene pairs from the input TSV file
and generates comprehensive JSON outputs with evolutionary lineage tracking.

Usage:
    python scripts/evaluate_lethal_genes.py --gene-pairs-file data/lethal_genes/pairs.tsv
    python scripts/evaluate_lethal_genes.py --gene-pairs-file data/lethal_genes/pairs.tsv --output-dir results/lethal_genes
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate all gene pairs using Pipeline3 lethal_genes mode",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--gene-pairs-file",
        type=str,
        required=True,
        help="Path to TSV file containing gene pairs (GeneA<tab>GeneB<tab>type)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/lethal_genes",
        help="Output directory for results (default: output/lethal_genes)"
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
        default=3,
        help="Number of generations (default: 3)"
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip gene pairs that already have results"
    )

    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Test run with only first 3 gene pairs"
    )

    return parser.parse_args()


def load_gene_pairs(filepath: str) -> List[Tuple[str, str, str]]:
    """
    Load gene pairs from TSV file.

    Args:
        filepath: Path to TSV file

    Returns:
        List of (gene_a, gene_b, type) tuples
    """
    gene_pairs = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)  # Skip header

            for row in reader:
                if len(row) >= 3:
                    gene_a, gene_b, pair_type = row[0], row[1], row[2]
                    gene_pairs.append((gene_a, gene_b, pair_type))

        logging.info(f"Loaded {len(gene_pairs)} gene pairs from {filepath}")
        return gene_pairs

    except Exception as e:
        logging.error(f"Failed to load gene pairs from {filepath}: {e}")
        sys.exit(1)


def run_pipeline_for_gene_pair(gene_a: str, gene_b: str,
                               args: argparse.Namespace,
                               pair_output_dir: str) -> Dict[str, Any]:
    """
    Run Pipeline3 for a single gene pair.

    Args:
        gene_a: First gene name
        gene_b: Second gene name
        args: Command line arguments
        pair_output_dir: Output directory for this pair

    Returns:
        Dictionary with execution results and metadata
    """
    start_time = time.time()

    # Create research goal
    research_goal = f"Evaluate synthetic lethality for gene pair: {gene_a} and {gene_b}"

    # Build command
    cmd = [
        sys.executable,
        "pipeline3/main.py",
        research_goal,
        "--mode", "lethal_genes",
        "--population-size", str(args.population_size),
        "--generations", str(args.generations),
        "--output-dir", pair_output_dir
    ]

    logging.info(f"Running Pipeline3 for {gene_a} x {gene_b}")
    logging.info(f"Command: {' '.join(cmd)}")

    try:
        # Run pipeline
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        end_time = time.time()
        execution_time = end_time - start_time

        if result.returncode == 0:
            logging.info(f"✓ Successfully completed {gene_a} x {gene_b} in {execution_time:.1f}s")
            status = "success"
        else:
            logging.error(f"✗ Pipeline failed for {gene_a} x {gene_b}")
            logging.error(f"Error output: {result.stderr}")
            status = "failed"

        return {
            "gene_a": gene_a,
            "gene_b": gene_b,
            "status": status,
            "execution_time": execution_time,
            "return_code": result.returncode,
            "stdout": result.stdout[-1000:] if result.stdout else "",  # Last 1000 chars
            "stderr": result.stderr[-1000:] if result.stderr else ""
        }

    except subprocess.TimeoutExpired:
        end_time = time.time()
        execution_time = end_time - start_time
        logging.error(f"✗ Pipeline timed out for {gene_a} x {gene_b} after {execution_time:.1f}s")

        return {
            "gene_a": gene_a,
            "gene_b": gene_b,
            "status": "timeout",
            "execution_time": execution_time,
            "return_code": -1,
            "stdout": "",
            "stderr": "Execution timed out after 1 hour"
        }

    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        logging.error(f"✗ Exception while running {gene_a} x {gene_b}: {e}")

        return {
            "gene_a": gene_a,
            "gene_b": gene_b,
            "status": "error",
            "execution_time": execution_time,
            "return_code": -1,
            "stdout": "",
            "stderr": str(e)
        }


def consolidate_results(output_dir: str, gene_pairs: List[Tuple[str, str, str]]) -> Dict[str, Any]:
    """
    Consolidate all individual results into a single summary.

    Args:
        output_dir: Base output directory
        gene_pairs: List of evaluated gene pairs

    Returns:
        Consolidated results dictionary
    """
    consolidated = {
        "timestamp": datetime.now().isoformat(),
        "total_gene_pairs": len(gene_pairs),
        "gene_pair_results": []
    }

    for gene_a, gene_b, pair_type in gene_pairs:
        pair_dir = os.path.join(output_dir, f"{gene_a}_{gene_b}")

        # Try to load the detailed JSON for this pair
        detailed_json_pattern = f"lethal_genes_{gene_a}_{gene_b}_detailed_*.json"

        try:
            # Find the most recent detailed JSON
            json_files = list(Path(pair_dir).glob(detailed_json_pattern))

            if json_files:
                latest_json = max(json_files, key=os.path.getmtime)

                with open(latest_json, 'r', encoding='utf-8') as f:
                    pair_data = json.load(f)

                # Extract key information for consolidated summary
                best_hyp = pair_data.get("best_hypothesis", {})
                algo_results = pair_data.get("algorithm_results", {})

                pair_summary = {
                    "gene_a": gene_a,
                    "gene_b": gene_b,
                    "pair_type": pair_type,
                    "best_fitness_score": best_hyp.get("fitness_score", 0),
                    "biological_plausibility": best_hyp.get("biological_plausibility", "")[:200],
                    "clinical_relevance": best_hyp.get("clinical_relevance", "")[:200],
                    "total_hypotheses": algo_results.get("total_hypotheses_generated", 0),
                    "generations_completed": algo_results.get("generations_completed", 0),
                    "fitness_improvement_percent": algo_results.get("fitness_improvement_percent", 0),
                    "detailed_json_file": str(latest_json)
                }

                consolidated["gene_pair_results"].append(pair_summary)

            else:
                logging.warning(f"No detailed JSON found for {gene_a} x {gene_b}")
                consolidated["gene_pair_results"].append({
                    "gene_a": gene_a,
                    "gene_b": gene_b,
                    "pair_type": pair_type,
                    "status": "missing_results"
                })

        except Exception as e:
            logging.error(f"Failed to process results for {gene_a} x {gene_b}: {e}")
            consolidated["gene_pair_results"].append({
                "gene_a": gene_a,
                "gene_b": gene_b,
                "pair_type": pair_type,
                "status": "error",
                "error": str(e)
            })

    return consolidated


def main():
    """Main execution function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Parse arguments
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load gene pairs
    gene_pairs = load_gene_pairs(args.gene_pairs_file)

    # Test run mode
    if args.test_run:
        gene_pairs = gene_pairs[:3]
        logging.info(f"TEST RUN: Evaluating only first {len(gene_pairs)} gene pairs")

    # Run pipeline for each gene pair
    execution_results = []

    for i, (gene_a, gene_b, pair_type) in enumerate(gene_pairs, 1):
        logging.info("=" * 80)
        logging.info(f"Processing pair {i}/{len(gene_pairs)}: {gene_a} x {gene_b} (type: {pair_type})")
        logging.info("=" * 80)

        # Create output directory for this pair
        pair_output_dir = os.path.join(args.output_dir, f"{gene_a}_{gene_b}")
        os.makedirs(pair_output_dir, exist_ok=True)

        # Check if results already exist
        if args.skip_existing:
            existing_results = list(Path(pair_output_dir).glob(f"lethal_genes_{gene_a}_{gene_b}_detailed_*.json"))
            if existing_results:
                logging.info(f"⊘ Skipping {gene_a} x {gene_b} (results already exist)")
                continue

        # Run pipeline
        result = run_pipeline_for_gene_pair(gene_a, gene_b, args, pair_output_dir)
        execution_results.append(result)

        # Small delay between runs
        time.sleep(2)

    # Consolidate all results
    logging.info("=" * 80)
    logging.info("Consolidating results...")
    logging.info("=" * 80)

    consolidated = consolidate_results(args.output_dir, gene_pairs)

    # Save consolidated summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    consolidated_file = os.path.join(args.output_dir, f"consolidated_summary_{timestamp}.json")

    try:
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated, f, indent=2, ensure_ascii=False)
        logging.info(f"Consolidated summary saved to: {consolidated_file}")
    except Exception as e:
        logging.error(f"Failed to save consolidated summary: {e}")

    # Save execution log
    execution_log_file = os.path.join(args.output_dir, f"execution_log_{timestamp}.json")
    try:
        with open(execution_log_file, 'w', encoding='utf-8') as f:
            json.dump(execution_results, f, indent=2)
        logging.info(f"Execution log saved to: {execution_log_file}")
    except Exception as e:
        logging.error(f"Failed to save execution log: {e}")

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Total gene pairs: {len(gene_pairs)}")
    print(f"Successful: {sum(1 for r in execution_results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in execution_results if r['status'] == 'failed')}")
    print(f"Errors: {sum(1 for r in execution_results if r['status'] == 'error')}")
    print(f"Timeouts: {sum(1 for r in execution_results if r['status'] == 'timeout')}")
    print(f"\nConsolidated summary: {consolidated_file}")
    print(f"Execution log: {execution_log_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
