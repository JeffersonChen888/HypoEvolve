#!/usr/bin/env python3
"""
Extract top 3 hypotheses from each gene pair in a run.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def extract_top_hypotheses(run_dir: Path, top_n: int = 3):
    """Extract top N hypotheses from each gene pair."""

    results = {}

    # Find all gene pair directories
    for gene_dir in sorted(run_dir.iterdir()):
        if not gene_dir.is_dir():
            continue

        gene_pair = gene_dir.name

        # Find the results JSON file
        json_files = list(gene_dir.glob("pipeline3_results_*.json"))
        if not json_files:
            print(f"Warning: No results JSON found for {gene_pair}", file=sys.stderr)
            continue

        json_file = json_files[0]

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Get all hypotheses and sort by fitness
            all_hypotheses = data.get('all_hypotheses', [])
            if not all_hypotheses:
                print(f"Warning: No hypotheses found for {gene_pair}", file=sys.stderr)
                continue

            # Sort by fitness score (descending)
            sorted_hyps = sorted(
                all_hypotheses,
                key=lambda h: h.get('fitness_score', 0),
                reverse=True
            )

            # Take top N
            top_hypotheses = sorted_hyps[:top_n]

            # Store results - complete hypotheses without truncation
            results[gene_pair] = {
                'total_hypotheses': len(all_hypotheses),
                'top_3_hypotheses': [
                    {
                        'rank': i + 1,
                        'id': hyp.get('id', 'unknown'),
                        'generation': hyp.get('generation', -1),
                        'fitness_score': hyp.get('fitness_score', 0),
                        'title': hyp.get('title', ''),
                        'hypothesis': hyp.get('hypothesis', ''),
                        'description': hyp.get('description', ''),  # Complete description
                        'primary_hypothesis': hyp.get('primary_hypothesis', {}),
                        'rival_hypothesis': hyp.get('rival_hypothesis', {}),
                        'reviews': hyp.get('reviews', []),  # Include reviews if available
                        'parent_ids': hyp.get('parent_ids', []),  # Include lineage
                        'evolution_strategy': hyp.get('evolution_strategy', None),
                    }
                    for i, hyp in enumerate(top_hypotheses)
                ]
            }

            print(f"✓ {gene_pair}: extracted {len(top_hypotheses)} top hypotheses (max_fitness={top_hypotheses[0]['fitness_score']:.2f})")

        except Exception as e:
            print(f"Error processing {gene_pair}: {e}", file=sys.stderr)
            continue

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_top_hypotheses.py <run_directory>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])

    if not run_dir.exists():
        print(f"Error: Directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting top 3 hypotheses from {run_dir}")
    print("=" * 80)

    results = extract_top_hypotheses(run_dir, top_n=3)

    # Save to JSON
    output_file = run_dir / "top_3_hypotheses_per_pair.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"✓ Saved top 3 hypotheses for {len(results)} gene pairs to:")
    print(f"  {output_file}")
    print()
    print("Summary:")
    for gene_pair, data in results.items():
        top_fitness = data['top_3_hypotheses'][0]['fitness_score']
        print(f"  {gene_pair}: {data['total_hypotheses']} total, top={top_fitness:.2f}")


if __name__ == "__main__":
    main()
