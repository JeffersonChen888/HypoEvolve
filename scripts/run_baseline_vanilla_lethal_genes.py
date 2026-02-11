#!/usr/bin/env python3
"""
Vanilla LLM + Literature Baseline for Lethal Genes Classification

This script implements the baseline comparison:
- Single LLM call (vs multi-agent GA pipeline)
- WITH literature search (same context as full pipeline)
- Same gene pairs from ground truth
- Same output format for comparison

Usage:
    python scripts/run_baseline_vanilla_lethal_genes.py DGKZ CHEK1
    python scripts/run_baseline_vanilla_lethal_genes.py --all
    python scripts/run_baseline_vanilla_lethal_genes.py --report-only
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add pipeline to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "pipeline"))

from external_tools.llm_client import llm_generate
from external_tools.web_search import search_literature

# Ground truth file path
GROUND_TRUTH_FILE = project_root / "data" / "lethal_genes" / "Positive and Negative pair curated filtered known.tsv"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def load_ground_truth() -> list:
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


def generate_search_queries(gene_a: str, gene_b: str) -> list:
    """Generate literature search queries for a gene pair (same as full pipeline)."""
    return [
        f"{gene_a} AND {gene_b} AND synthetic lethality",
        f"{gene_a} AND {gene_b} AND cancer",
        f"{gene_a} gene function",
        f"{gene_b} gene function"
    ]


def search_literature_for_gene_pair(gene_a: str, gene_b: str, num_papers: int = 5) -> tuple:
    """
    Perform literature search for a gene pair.

    Returns:
        Tuple of (all_papers, literature_context_string)
    """
    queries = generate_search_queries(gene_a, gene_b)
    all_papers = []
    literature_context = ""

    for i, query in enumerate(queries):
        logging.info(f"Literature search {i+1}/4: {query}")
        papers = search_literature(query, max_results=num_papers)

        # Handle None or empty results
        if not papers:
            papers = []

        # Filter duplicates
        new_papers = [p for p in papers if p and p.get('title') and not any(
            existing['title'].lower() == p['title'].lower() for existing in all_papers
        )]

        if new_papers:
            all_papers.extend(new_papers)

            # Build context string
            literature_context += f"\n\n--- Search {i+1}: {query} ---\n\n"
            for j, paper in enumerate(new_papers[:3], 1):
                title = paper.get('title', 'Untitled')
                abstract = paper.get('abstract') or ''
                abstract = abstract[:300] if abstract else ''
                literature_context += f"{j}. {title}\n   {abstract}...\n\n"

    logging.info(f"Found {len(all_papers)} papers for {gene_a}-{gene_b}")
    return all_papers, literature_context


def generate_vanilla_hypothesis(gene_a: str, gene_b: str, literature_context: str) -> str:
    """
    Generate hypothesis using a single LLM call (vanilla baseline).

    This is the key difference from the full pipeline:
    - Single LLM call instead of multi-agent generation
    - No reflection/fitness scoring
    - No evolution/GA
    - SIMPLE prompt without detailed analysis guidance
    """
    # Simple prompt (matching drug_repurposing vanilla baseline style)
    # Does NOT include the detailed analysis framework or "be skeptical" criteria
    prompt = f"""You are a computational biology expert analyzing synthetic lethality. Given the literature context below, assess whether the gene pair {gene_a} and {gene_b} could represent a valid synthetic lethal interaction.

LITERATURE CONTEXT:
{literature_context if literature_context else "No literature found. Use your general knowledge."}

GENE PAIR:
Gene A: {gene_a}
Gene B: {gene_b}

Provide:
TITLE: [Concise title for your hypothesis]
SUMMARY: [One sentence summary]
HYPOTHESIS: [2-3 sentences explaining the proposed mechanism]
RATIONALE: [1-2 paragraphs of scientific reasoning]
FINAL_PREDICTION: [well-based OR random]

- "well-based": Evidence supports biologically plausible synthetic lethal interaction
- "random": Likely artifact, no meaningful biological relationship between genes
"""

    logging.info(f"Generating vanilla hypothesis for {gene_a}-{gene_b}")
    response = llm_generate(prompt, max_tokens=2000, temperature=0.7)
    return response


def parse_vanilla_response(response: str, gene_a: str, gene_b: str) -> dict:
    """Parse vanilla LLM response into structured hypothesis."""
    hyp = {
        "id": f"vanilla-001-{uuid.uuid4().hex[:8]}",
        "origin": "vanilla_baseline",
        "generation_method": "single_llm_call",
        "elo_score": 1200,
        "fitness_score": None,
        "gene_a": gene_a,
        "gene_b": gene_b
    }

    # Extract fields from response
    title_match = re.search(r'TITLE:\s*([^\n]+)', response, re.IGNORECASE)
    summary_match = re.search(r'SUMMARY:\s*([^\n]+)', response, re.IGNORECASE)
    hypothesis_match = re.search(r'HYPOTHESIS:\s*(.+?)(?=RATIONALE:|FINAL_PREDICTION:|$)',
                                  response, re.IGNORECASE | re.DOTALL)
    rationale_match = re.search(r'RATIONALE:\s*(.+?)(?=FINAL_PREDICTION:|$)',
                                 response, re.IGNORECASE | re.DOTALL)

    # Normalize unicode hyphens before matching FINAL_PREDICTION
    normalized_response = response.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '-')
    prediction_match = re.search(r'FINAL_PREDICTION:\s*(well-based|random)', normalized_response, re.IGNORECASE)

    hyp['title'] = title_match.group(1).strip() if title_match else f"Hypothesis: {gene_a}-{gene_b}"
    hyp['summary'] = summary_match.group(1).strip() if summary_match else ""
    hyp['hypothesis_statement'] = hypothesis_match.group(1).strip() if hypothesis_match else ""
    hyp['rationale'] = rationale_match.group(1).strip() if rationale_match else ""
    hyp['final_prediction'] = prediction_match.group(1).strip().lower() if prediction_match else None

    return hyp


def run_vanilla_baseline(gene_a: str, gene_b: str, output_dir: str = None) -> dict:
    """
    Run the vanilla baseline for a single gene pair.

    Args:
        gene_a: First gene name
        gene_b: Second gene name
        output_dir: Output directory (default: output/baselines/vanilla_lethal_genes/{gene_a}_{gene_b})

    Returns:
        Dictionary with results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pair_name = f"{gene_a}_{gene_b}"

    # Set up output directory
    if output_dir is None:
        output_dir = project_root / "output" / "baselines" / "vanilla_lethal_genes" / pair_name
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging to file
    log_file = output_dir / f"vanilla_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logging.getLogger().addHandler(file_handler)

    logging.info(f"Starting vanilla baseline for: {gene_a} - {gene_b}")
    logging.info(f"Output directory: {output_dir}")

    try:
        # Step 1: Literature search
        papers, literature_context = search_literature_for_gene_pair(gene_a, gene_b)

        # Step 2: Generate hypothesis with single LLM call
        response = generate_vanilla_hypothesis(gene_a, gene_b, literature_context)

        # Step 3: Parse response
        hypothesis = parse_vanilla_response(response, gene_a, gene_b)

        # Compile results
        results = {
            "baseline_type": "vanilla_llm_literature",
            "gene_a": gene_a,
            "gene_b": gene_b,
            "timestamp": timestamp,
            "model": os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            "num_papers_found": len(papers),
            "literature_queries": generate_search_queries(gene_a, gene_b),
            "hypothesis": hypothesis,
            "raw_response": response
        }

        # Save results
        results_file = output_dir / f"vanilla_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logging.info(f"Results saved to: {results_file}")
        logging.info(f"Prediction: {hypothesis.get('final_prediction', 'None')}")

        return results

    except Exception as e:
        logging.error(f"Error running vanilla baseline: {e}")
        raise
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


def extract_prediction(pair_dir: str) -> tuple:
    """
    Extract FINAL_PREDICTION from vanilla results JSON.

    Args:
        pair_dir: Directory containing vanilla results

    Returns:
        Tuple of (prediction, None) - fitness_score is always None for vanilla
    """
    pair_path = Path(pair_dir)
    json_files = list(pair_path.glob("vanilla_results_*.json"))

    if not json_files:
        return None, None

    # Get the most recent file
    latest = max(json_files, key=os.path.getmtime)

    try:
        with open(latest, 'r', encoding='utf-8') as f:
            data = json.load(f)

        hypothesis = data.get("hypothesis", {})
        prediction = hypothesis.get("final_prediction", "")

        # Normalize prediction
        if prediction:
            prediction = prediction.lower().strip()

        if not prediction or prediction not in ["well-based", "random"]:
            logging.warning(f"Invalid prediction in {latest}: '{prediction}'")
            return None, None

        return prediction, None

    except Exception as e:
        logging.warning(f"Could not parse {latest}: {e}")
        return None, None


def generate_report(output_dir: str, ground_truth: list) -> dict:
    """
    Generate classification results and summary from vanilla baseline outputs.

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

        prediction, _ = (None, None)
        if os.path.exists(pair_dir):
            prediction, _ = extract_prediction(pair_dir)

        results.append({
            "gene_a": gene_a,
            "gene_b": gene_b,
            "ground_truth": label,
            "prediction": prediction,
            "correct": prediction == label if prediction else None
        })

    # Write CSV
    csv_path = os.path.join(output_dir, "classification_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["gene_a", "gene_b", "ground_truth", "prediction", "correct"]
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

        # Prediction distribution
        well_based_count = sum(1 for r in valid if r["prediction"] == "well-based")
        random_count = sum(1 for r in valid if r["prediction"] == "random")

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
            "ground_truth_distribution": {
                "well_based": sum(1 for r in valid if r["ground_truth"] == "well-based"),
                "random": sum(1 for r in valid if r["ground_truth"] == "random")
            },
            "prediction_distribution": {
                "well_based": well_based_count,
                "random": random_count
            }
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
    print("VANILLA BASELINE CLASSIFICATION SUMMARY")
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
    print(f"  True Positive (well-based correct):   {cm['tp']}")
    print(f"  False Positive (random as well-based): {cm['fp']}")
    print(f"  True Negative (random correct):        {cm['tn']}")
    print(f"  False Negative (well-based as random): {cm['fn']}")
    print("-" * 60)
    if "prediction_distribution" in summary:
        pd = summary["prediction_distribution"]
        print(f"Predictions: {pd['well_based']} well-based, {pd['random']} random")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run vanilla LLM + literature baseline for lethal genes classification"
    )
    parser.add_argument(
        "gene_a",
        nargs="?",
        help="First gene name (e.g., 'DGKZ')"
    )
    parser.add_argument(
        "gene_b",
        nargs="?",
        help="Second gene name (e.g., 'CHEK1')"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run baseline for all 51 gene pairs from ground truth"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate report from existing results"
    )
    parser.add_argument(
        "--output-dir",
        help="Custom output directory"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip gene pairs that already have results"
    )

    args = parser.parse_args()

    # Default output directory
    base_output_dir = project_root / "output" / "baselines" / "vanilla_lethal_genes"

    # Load ground truth
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} gene pairs from ground truth")

    if args.report_only:
        print("\nGenerating report from existing results...")
        summary = generate_report(str(base_output_dir), ground_truth)
        print_summary(summary)
        return

    if args.all:
        # Run for all gene pairs
        base_output_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        skip_count = 0
        fail_count = 0

        for i, (gene_a, gene_b, label) in enumerate(ground_truth):
            pair_name = f"{gene_a}_{gene_b}"
            pair_dir = base_output_dir / pair_name

            print(f"\n[{i+1}/{len(ground_truth)}] {pair_name} (ground truth: {label})")

            # Check if already exists
            if args.skip_existing and pair_dir.exists():
                json_files = list(pair_dir.glob("vanilla_results_*.json"))
                if json_files:
                    print(f"  Skipping (results exist)")
                    skip_count += 1
                    continue

            try:
                result = run_vanilla_baseline(gene_a, gene_b, str(pair_dir))
                prediction = result['hypothesis'].get('final_prediction', 'None')
                correct = "correct" if prediction == label else "WRONG"
                print(f"  Prediction: {prediction} ({correct})")
                success_count += 1
            except Exception as e:
                logging.error(f"Failed for {pair_name}: {e}")
                fail_count += 1

        print(f"\n" + "-" * 60)
        print(f"Batch complete: {success_count} successful, {skip_count} skipped, {fail_count} failed")

        # Generate report
        print("\nGenerating classification report...")
        summary = generate_report(str(base_output_dir), ground_truth)
        print_summary(summary)

    elif args.gene_a and args.gene_b:
        # Run for single gene pair
        output_dir = args.output_dir if args.output_dir else None
        result = run_vanilla_baseline(args.gene_a, args.gene_b, output_dir)
        print(f"\nPrediction: {result['hypothesis'].get('final_prediction', 'None')}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
