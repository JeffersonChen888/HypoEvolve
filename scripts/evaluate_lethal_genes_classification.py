#!/usr/bin/env python3
"""
Evaluate lethal genes classification against ground truth.

This script runs the lethal_genes pipeline on gene pairs from the ground truth
file and compares predictions against the actual labels.

Usage:
    # Run all 51 pairs
    python scripts/evaluate_lethal_genes_classification.py --output-dir output/lethal_genes/run_20260114

    # Run batch (pairs 0-9)
    python scripts/evaluate_lethal_genes_classification.py --output-dir output/lethal_genes/run_20260114 --start-index 0 --count 10

    # Skip existing pairs (resume)
    python scripts/evaluate_lethal_genes_classification.py --output-dir output/lethal_genes/run_20260114 --skip-existing

    # Generate report only (after all batches complete)
    python scripts/evaluate_lethal_genes_classification.py --output-dir output/lethal_genes/run_20260114 --report-only
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GROUND_TRUTH_FILE = PROJECT_ROOT / "data" / "lethal_genes" / "Positive and Negative pair curated filtered known.tsv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate lethal genes classification against ground truth"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results (all batches accumulate here)"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index in ground truth file (0-indexed)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of pairs to process (default: all remaining)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip gene pairs that already have results"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate report from existing results (don't run pipeline)"
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=6,
        help="GA population size (default: 6)"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=3,
        help="Number of GA generations (default: 3)"
    )
    return parser.parse_args()


def load_ground_truth():
    """
    Load ground truth TSV file.

    Returns:
        List of (gene_a, gene_b, label) tuples where label is 'well-based' or 'random'
    """
    pairs = []
    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            # Map: paper -> well-based, random -> random
            label = "well-based" if row["type"] == "paper" else "random"
            pairs.append((row["GeneA"], row["GeneB"], label))
    return pairs


def run_pipeline(gene_a: str, gene_b: str, output_dir: str,
                 pop_size: int, generations: int) -> bool:
    """
    Run the lethal_genes pipeline for a single gene pair.

    Args:
        gene_a: First gene name
        gene_b: Second gene name
        output_dir: Directory for this gene pair's output
        pop_size: Population size for GA
        generations: Number of generations

    Returns:
        True if successful, False otherwise
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build research goal in expected format
    research_goal = f"{gene_a}:{gene_b}"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "pipeline" / "main.py"),
        research_goal,
        "--mode", "lethal_genes",
        "--population-size", str(pop_size),
        "--generations", str(generations),
        "--output-dir", output_dir
    ]

    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per pair
            cwd=str(PROJECT_ROOT)
        )

        if result.returncode != 0:
            print(f"  Error: Pipeline failed with return code {result.returncode}")
            print(f"  Stderr: {result.stderr[:500] if result.stderr else 'None'}")
            return False

        return True

    except subprocess.TimeoutExpired:
        print(f"  Error: Pipeline timed out after 2 hours")
        return False
    except Exception as e:
        print(f"  Error: {str(e)}")
        return False


def extract_prediction(pair_dir: str) -> tuple:
    """
    Extract FINAL_PREDICTION and fitness score from results JSON.

    Args:
        pair_dir: Directory containing pipeline results (may have nested run_* subdirs)

    Returns:
        Tuple of (prediction, fitness_score) or (None, None) if not found
    """
    # Look for detailed JSON files (recursively to handle nested run_* directories)
    pair_path = Path(pair_dir)
    json_files = list(pair_path.glob("**/*_detailed_*.json"))

    if not json_files:
        # Also check for pipeline_results JSON
        json_files = list(pair_path.glob("**/pipeline_results_*.json"))

    if not json_files:
        # Fallback to any JSON
        json_files = list(pair_path.glob("**/*.json"))

    if not json_files:
        return None, None

    # Get the most recent file
    latest = max(json_files, key=os.path.getmtime)

    try:
        with open(latest, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Try to get best hypothesis
        best = data.get("best_hypothesis", {})
        if not best:
            # Maybe it's in a different structure
            best = data.get("results", {}).get("best_hypothesis", {})

        prediction = best.get("final_prediction", "")
        fitness_score = best.get("fitness_score", 0)

        # Normalize prediction
        if prediction:
            prediction = prediction.lower().strip()

        # Require valid FINAL_PREDICTION - no fallback
        if not prediction or prediction not in ["well-based", "random"]:
            print(f"  Error: Missing or invalid FINAL_PREDICTION: '{prediction}'")
            return None, fitness_score

        return prediction, fitness_score

    except Exception as e:
        print(f"  Warning: Could not parse {latest}: {e}")
        return None, None


def generate_report(output_dir: str, ground_truth: list) -> dict:
    """
    Generate classification results and summary from existing pipeline outputs.

    Args:
        output_dir: Directory containing all gene pair results
        ground_truth: List of (gene_a, gene_b, label) tuples

    Returns:
        Summary dictionary with metrics
    """
    results = []

    for gene_a, gene_b, label in ground_truth:
        pair_name = f"{gene_a}_{gene_b}"
        pair_dir = os.path.join(output_dir, pair_name)

        prediction, fitness_score = (None, None)
        if os.path.exists(pair_dir):
            prediction, fitness_score = extract_prediction(pair_dir)

        results.append({
            "gene_a": gene_a,
            "gene_b": gene_b,
            "ground_truth": label,
            "prediction": prediction,
            "correct": prediction == label if prediction else None,
            "fitness_score": fitness_score
        })

    # Write CSV
    csv_path = os.path.join(output_dir, "classification_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["gene_a", "gene_b", "ground_truth", "prediction", "correct", "fitness_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to: {csv_path}")

    # Calculate metrics
    valid = [r for r in results if r["prediction"] is not None]

    if not valid:
        summary = {
            "total_pairs": len(ground_truth),
            "evaluated": 0,
            "pending": len(ground_truth),
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "confusion_matrix": {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        }
    else:
        # True Positive: ground_truth=well-based, prediction=well-based
        tp = sum(1 for r in valid if r["ground_truth"] == "well-based" and r["prediction"] == "well-based")
        # False Positive: ground_truth=random, prediction=well-based
        fp = sum(1 for r in valid if r["ground_truth"] == "random" and r["prediction"] == "well-based")
        # True Negative: ground_truth=random, prediction=random
        tn = sum(1 for r in valid if r["ground_truth"] == "random" and r["prediction"] == "random")
        # False Negative: ground_truth=well-based, prediction=random
        fn = sum(1 for r in valid if r["ground_truth"] == "well-based" and r["prediction"] == "random")

        accuracy = (tp + tn) / len(valid) if valid else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        summary = {
            "total_pairs": len(ground_truth),
            "evaluated": len(valid),
            "pending": len(ground_truth) - len(valid),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn
            },
            "positive_count": sum(1 for r in valid if r["ground_truth"] == "well-based"),
            "negative_count": sum(1 for r in valid if r["ground_truth"] == "random")
        }

    # Write summary JSON
    summary_path = os.path.join(output_dir, "classification_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to: {summary_path}")

    return summary


def print_summary(summary: dict):
    """Print summary metrics to console."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total pairs:     {summary['total_pairs']}")
    print(f"Evaluated:       {summary['evaluated']}")
    print(f"Pending:         {summary['pending']}")
    print("-" * 60)
    print(f"Accuracy:        {summary['accuracy']:.2%}")
    print(f"Precision:       {summary['precision']:.2%}")
    print(f"Recall:          {summary['recall']:.2%}")
    print(f"F1 Score:        {summary['f1_score']:.2%}")
    print("-" * 60)
    cm = summary['confusion_matrix']
    print("Confusion Matrix:")
    print(f"  True Positive (well-based correct):  {cm['tp']}")
    print(f"  False Positive (random as well-based): {cm['fp']}")
    print(f"  True Negative (random correct):       {cm['tn']}")
    print(f"  False Negative (well-based as random): {cm['fn']}")
    print("=" * 60)


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load ground truth
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} gene pairs from ground truth")

    # Report-only mode
    if args.report_only:
        print("\nGenerating report from existing results...")
        summary = generate_report(args.output_dir, ground_truth)
        print_summary(summary)
        return

    # Determine batch range
    start = args.start_index
    end = start + args.count if args.count else len(ground_truth)
    end = min(end, len(ground_truth))

    batch = ground_truth[start:end]
    print(f"\nProcessing pairs {start} to {end-1} ({len(batch)} pairs)")
    print(f"Population size: {args.population_size}, Generations: {args.generations}")

    # Process each pair
    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, (gene_a, gene_b, label) in enumerate(batch):
        pair_name = f"{gene_a}_{gene_b}"
        pair_dir = os.path.join(args.output_dir, pair_name)

        print(f"\n[{start + i + 1}/{len(ground_truth)}] {pair_name} (ground truth: {label})")

        # Check if already exists
        if args.skip_existing and os.path.exists(pair_dir):
            json_files = list(Path(pair_dir).glob("*.json"))
            if json_files:
                print(f"  Skipping (results exist)")
                skip_count += 1
                continue

        # Run pipeline
        print(f"  Running pipeline...")
        success = run_pipeline(
            gene_a, gene_b, pair_dir,
            args.population_size, args.generations
        )

        if success:
            print(f"  Completed successfully")
            success_count += 1
        else:
            print(f"  Failed")
            fail_count += 1

    # Print batch summary
    print(f"\n" + "-" * 60)
    print(f"Batch complete: {success_count} successful, {skip_count} skipped, {fail_count} failed")

    # Generate report
    print("\nGenerating classification report...")
    summary = generate_report(args.output_dir, ground_truth)
    print_summary(summary)


if __name__ == "__main__":
    main()
