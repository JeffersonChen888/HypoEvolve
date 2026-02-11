#!/usr/bin/env python3
"""
Extract Generation Comparison from Existing Batch Data

This script extracts Gen 0 vs Final generation statistics from existing
batch_constrained runs to demonstrate the value of GA evolution.

Output:
- Per-cancer-type statistics (Gen 0 mean/max vs Final mean/max)
- Learning curve data (fitness per generation)
- Overall summary statistics

Usage:
    python scripts/extract_generation_comparison.py
    python scripts/extract_generation_comparison.py --input-dir output/drug_repurposing/batch_constrained
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Project root
project_root = Path(__file__).parent.parent


def find_result_files(input_dir: Path) -> list:
    """Find all pipeline result JSON files in the input directory."""
    result_files = list(input_dir.glob("*/*/pipeline_results_*.json"))
    logging.info(f"Found {len(result_files)} result files")
    return result_files


def extract_generation_data(json_file: Path) -> dict:
    """Extract generation statistics from a single result file."""
    with open(json_file) as f:
        data = json.load(f)

    ga_results = data.get('genetic_algorithm_results', {})
    gen_history = data.get('generation_history', [])

    # Extract cancer type from research goal
    research_goal = ga_results.get('research_goal', '')
    cancer_type = research_goal.replace('Find drug repurposing candidates for ', '')

    # Get Gen 0 and Final stats
    gen0_data = None
    final_data = None

    if gen_history:
        gen0_data = gen_history[0] if len(gen_history) > 0 else None
        final_data = gen_history[-1] if len(gen_history) > 0 else None

    # Extract final population hypotheses
    final_population = data.get('final_population', [])

    return {
        'cancer_type': cancer_type,
        'file_path': str(json_file),
        'generations_completed': ga_results.get('generations_completed', 0),

        # Gen 0 (initial) statistics
        'gen0_mean_fitness': gen0_data.get('mean_fitness') if gen0_data else None,
        'gen0_max_fitness': gen0_data.get('max_fitness') if gen0_data else None,
        'gen0_min_fitness': gen0_data.get('min_fitness') if gen0_data else None,

        # Final generation statistics
        'final_mean_fitness': final_data.get('mean_fitness') if final_data else None,
        'final_max_fitness': final_data.get('max_fitness') if final_data else None,
        'final_min_fitness': final_data.get('min_fitness') if final_data else None,

        # Improvement
        'fitness_improvement_percent': ga_results.get('fitness_improvement_percent', 0),

        # Full generation history for learning curve
        'generation_history': [
            {
                'generation': g.get('generation'),
                'mean_fitness': g.get('mean_fitness'),
                'max_fitness': g.get('max_fitness'),
                'min_fitness': g.get('min_fitness'),
                'fitness_std': g.get('fitness_std', 0)
            }
            for g in gen_history
        ],

        # Final population drugs for compliance check
        'final_drugs': [h.get('final_drug') for h in final_population if h.get('final_drug')]
    }


def compute_summary_statistics(cancer_data: list) -> dict:
    """Compute summary statistics across all cancer types."""
    # Filter out None values
    gen0_means = [d['gen0_mean_fitness'] for d in cancer_data if d['gen0_mean_fitness'] is not None]
    final_means = [d['final_mean_fitness'] for d in cancer_data if d['final_mean_fitness'] is not None]
    improvements = [d['fitness_improvement_percent'] for d in cancer_data if d['fitness_improvement_percent']]

    summary = {
        'num_cancer_types': len(cancer_data),
        'avg_gen0_mean_fitness': mean(gen0_means) if gen0_means else None,
        'avg_final_mean_fitness': mean(final_means) if final_means else None,
        'avg_improvement_percent': mean(improvements) if improvements else None,
        'std_improvement_percent': stdev(improvements) if len(improvements) > 1 else None,
        'min_improvement_percent': min(improvements) if improvements else None,
        'max_improvement_percent': max(improvements) if improvements else None,
    }

    # Count cancer types with improvement
    improved_count = sum(1 for d in cancer_data
                         if d['gen0_mean_fitness'] and d['final_mean_fitness']
                         and d['final_mean_fitness'] > d['gen0_mean_fitness'])
    summary['num_improved'] = improved_count
    summary['improvement_rate'] = improved_count / len(cancer_data) if cancer_data else 0

    return summary


def compute_learning_curve(cancer_data: list) -> list:
    """Compute average fitness per generation across all cancer types."""
    # Find max generations
    max_gens = max(len(d['generation_history']) for d in cancer_data)

    learning_curve = []
    for gen_idx in range(max_gens):
        gen_means = []
        gen_maxes = []
        gen_mins = []

        for cancer in cancer_data:
            history = cancer['generation_history']
            if gen_idx < len(history):
                if history[gen_idx]['mean_fitness'] is not None:
                    gen_means.append(history[gen_idx]['mean_fitness'])
                if history[gen_idx]['max_fitness'] is not None:
                    gen_maxes.append(history[gen_idx]['max_fitness'])
                if history[gen_idx]['min_fitness'] is not None:
                    gen_mins.append(history[gen_idx]['min_fitness'])

        learning_curve.append({
            'generation': gen_idx,
            'avg_mean_fitness': mean(gen_means) if gen_means else None,
            'std_mean_fitness': stdev(gen_means) if len(gen_means) > 1 else 0,
            'avg_max_fitness': mean(gen_maxes) if gen_maxes else None,
            'avg_min_fitness': mean(gen_mins) if gen_mins else None,
            'num_samples': len(gen_means)
        })

    return learning_curve


def main():
    parser = argparse.ArgumentParser(
        description="Extract Gen 0 vs Final comparison from batch_constrained data"
    )
    parser.add_argument(
        "--input-dir",
        default="output/drug_repurposing/batch_constrained",
        help="Input directory with batch_constrained results"
    )
    parser.add_argument(
        "--output-dir",
        default="output/baselines/generation_comparison",
        help="Output directory for comparison results"
    )

    args = parser.parse_args()

    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all result files
    result_files = find_result_files(input_dir)

    if not result_files:
        logging.error(f"No result files found in {input_dir}")
        sys.exit(1)

    # Extract data from each file
    cancer_data = []
    for json_file in result_files:
        try:
            data = extract_generation_data(json_file)
            cancer_data.append(data)
            gen0 = data['gen0_mean_fitness']
            final = data['final_mean_fitness']
            imp = data['fitness_improvement_percent']
            gen0_str = f"{gen0:.1f}" if gen0 is not None else "N/A"
            final_str = f"{final:.1f}" if final is not None else "N/A"
            imp_str = f"{imp:.1f}" if imp is not None else "0.0"
            logging.info(f"Extracted: {data['cancer_type']} - "
                        f"Gen0={gen0_str}, Final={final_str}, Improvement={imp_str}%")
        except Exception as e:
            logging.error(f"Failed to process {json_file}: {e}")

    # Compute summary statistics
    summary = compute_summary_statistics(cancer_data)

    # Compute learning curve
    learning_curve = compute_learning_curve(cancer_data)

    # Prepare output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        'extraction_timestamp': timestamp,
        'input_directory': str(input_dir),
        'summary_statistics': summary,
        'learning_curve': learning_curve,
        'per_cancer_data': cancer_data
    }

    # Save full output
    output_file = output_dir / f"generation_comparison_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logging.info(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("GENERATION COMPARISON SUMMARY")
    print("="*60)
    print(f"Cancer types analyzed: {summary['num_cancer_types']}")
    print(f"\nGen 0 (Initial) Statistics:")
    print(f"  Average mean fitness: {summary['avg_gen0_mean_fitness']:.2f}")
    print(f"\nFinal Generation Statistics:")
    print(f"  Average mean fitness: {summary['avg_final_mean_fitness']:.2f}")
    print(f"\nImprovement (Evolution Value):")
    print(f"  Average improvement: {summary['avg_improvement_percent']:.2f}%")
    print(f"  Std deviation: {summary['std_improvement_percent']:.2f}%" if summary['std_improvement_percent'] else "")
    print(f"  Range: {summary['min_improvement_percent']:.2f}% - {summary['max_improvement_percent']:.2f}%")
    print(f"  Cancer types improved: {summary['num_improved']}/{summary['num_cancer_types']} ({summary['improvement_rate']*100:.1f}%)")

    print("\nLearning Curve (Avg Mean Fitness per Generation):")
    for lc in learning_curve:
        print(f"  Gen {lc['generation']}: {lc['avg_mean_fitness']:.2f} ± {lc['std_mean_fitness']:.2f}")

    print("="*60)

    # Save CSV summary for easy import
    csv_file = output_dir / f"generation_comparison_{timestamp}.csv"
    with open(csv_file, 'w') as f:
        f.write("cancer_type,gen0_mean_fitness,final_mean_fitness,improvement_percent\n")
        for d in cancer_data:
            f.write(f"{d['cancer_type']},{d['gen0_mean_fitness']},{d['final_mean_fitness']},{d['fitness_improvement_percent']}\n")

    logging.info(f"CSV summary saved to: {csv_file}")


if __name__ == "__main__":
    main()
