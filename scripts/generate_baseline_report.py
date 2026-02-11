#!/usr/bin/env python3
"""
Generate Baseline Comparison Report for Paper

This script combines:
1. Vanilla baseline results
2. Gen 0 vs Final comparison (from extract_generation_comparison.py)
3. DepMap validation results

And generates:
- Comparison tables (Vanilla vs Gen 0 vs Final)
- Learning curve data
- Statistical significance tests
- Figures ready for paper

Usage:
    python scripts/generate_baseline_report.py
    python scripts/generate_baseline_report.py --vanilla-dir output/baselines/vanilla
    python scripts/generate_baseline_report.py --run-depmap-validation
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Project root
project_root = Path(__file__).parent.parent


def load_vanilla_results(vanilla_dir: Path) -> dict:
    """Load vanilla baseline results for all cancer types."""
    results = {}

    for cancer_dir in vanilla_dir.iterdir():
        if not cancer_dir.is_dir():
            continue

        # Find the most recent result file
        result_files = list(cancer_dir.glob("vanilla_results_*.json"))
        if not result_files:
            continue

        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)

        with open(latest_file) as f:
            data = json.load(f)

        cancer_type = data.get('cancer_type', cancer_dir.name.replace('_', ' '))
        results[cancer_type] = data

    logging.info(f"Loaded vanilla results for {len(results)} cancer types")
    return results


def load_generation_comparison(comparison_dir: Path) -> dict:
    """Load generation comparison data."""
    comparison_files = list(comparison_dir.glob("generation_comparison_*.json"))
    if not comparison_files:
        logging.warning(f"No generation comparison files found in {comparison_dir}")
        return {}

    latest_file = max(comparison_files, key=lambda x: x.stat().st_mtime)

    with open(latest_file) as f:
        data = json.load(f)

    logging.info(f"Loaded generation comparison from {latest_file}")
    return data


def load_depmap_validation(depmap_dir: Path) -> dict:
    """Load DepMap validation results."""
    validation_files = list(depmap_dir.glob("validation_summary_*.json"))
    if not validation_files:
        # Try the extracted results directory
        validation_files = list(depmap_dir.glob("**/validation_summary_*.json"))

    if not validation_files:
        logging.warning(f"No DepMap validation files found")
        return {}

    latest_file = max(validation_files, key=lambda x: x.stat().st_mtime)

    with open(latest_file) as f:
        data = json.load(f)

    logging.info(f"Loaded DepMap validation from {latest_file}")
    return data


def run_depmap_validation_for_vanilla(vanilla_dir: Path) -> dict:
    """Run DepMap validation on vanilla baseline results."""
    logging.info("Running DepMap validation on vanilla baseline...")

    # Find all vanilla result files
    result_files = list(vanilla_dir.glob("*/vanilla_results_*.json"))

    all_hypotheses = []
    for result_file in result_files:
        with open(result_file) as f:
            data = json.load(f)

        for hyp in data.get('hypotheses', []):
            if hyp.get('final_drug') and hyp.get('cancer_type'):
                all_hypotheses.append({
                    'final_drug': hyp['final_drug'],
                    'cancer_type': hyp['cancer_type'],
                    'title': hyp.get('title', ''),
                    'source': 'vanilla_baseline'
                })

    if not all_hypotheses:
        logging.warning("No hypotheses found for DepMap validation")
        return {}

    # Save to temp file for validation script
    temp_file = vanilla_dir / "temp_hypotheses_for_validation.json"
    with open(temp_file, 'w') as f:
        json.dump(all_hypotheses, f)

    # Run validation
    try:
        result = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "depmap_validation.py"),
             str(temp_file), "--output-dir", str(vanilla_dir)],
            capture_output=True,
            text=True,
            cwd=str(project_root)
        )

        if result.returncode != 0:
            logging.error(f"DepMap validation failed: {result.stderr}")
            return {}

        # Load the generated validation results
        validation_files = list(vanilla_dir.glob("validation_summary_*.json"))
        if validation_files:
            latest = max(validation_files, key=lambda x: x.stat().st_mtime)
            with open(latest) as f:
                return json.load(f)

    except Exception as e:
        logging.error(f"Failed to run DepMap validation: {e}")

    return {}


def compute_statistical_significance(gen0_scores: list, final_scores: list) -> dict:
    """Compute statistical significance using paired tests."""
    try:
        from scipy import stats
    except ImportError:
        logging.warning("scipy not installed, skipping statistical tests")
        return {'note': 'scipy not installed'}

    if len(gen0_scores) != len(final_scores):
        logging.warning("Mismatched sample sizes for paired test")
        return {'error': 'mismatched sample sizes'}

    if len(gen0_scores) < 3:
        logging.warning("Not enough samples for statistical test")
        return {'error': 'insufficient samples'}

    # Paired t-test
    t_stat, p_value_ttest = stats.ttest_rel(final_scores, gen0_scores)

    # Wilcoxon signed-rank test (non-parametric alternative)
    try:
        w_stat, p_value_wilcoxon = stats.wilcoxon(final_scores, gen0_scores)
    except ValueError:
        w_stat, p_value_wilcoxon = None, None

    return {
        'paired_ttest': {
            't_statistic': float(t_stat) if t_stat is not None else None,
            'p_value': float(p_value_ttest) if p_value_ttest is not None else None,
            'significant_005': bool(p_value_ttest < 0.05) if p_value_ttest is not None else None,
            'significant_001': bool(p_value_ttest < 0.01) if p_value_ttest is not None else None
        },
        'wilcoxon': {
            'w_statistic': float(w_stat) if w_stat is not None else None,
            'p_value': float(p_value_wilcoxon) if p_value_wilcoxon is not None else None,
            'significant_005': bool(p_value_wilcoxon < 0.05) if p_value_wilcoxon is not None else None
        } if w_stat is not None else None,
        'n_samples': len(gen0_scores)
    }


def generate_comparison_table(vanilla_results: dict, gen_comparison: dict) -> list:
    """Generate comparison table: Vanilla vs Gen 0 vs Final."""
    table = []

    # Build lookup for generation data
    gen_lookup = {}
    for cancer_data in gen_comparison.get('per_cancer_data', []):
        gen_lookup[cancer_data['cancer_type']] = cancer_data

    # Combine all cancer types
    all_cancers = set(vanilla_results.keys()) | set(gen_lookup.keys())

    for cancer_type in sorted(all_cancers):
        row = {'cancer_type': cancer_type}

        # Vanilla data
        vanilla = vanilla_results.get(cancer_type, {})
        row['vanilla_num_hypotheses'] = vanilla.get('num_hypotheses', 0)
        row['vanilla_drug_compliance'] = vanilla.get('drug_compliance_rate', 0)

        # Generation comparison data
        gen = gen_lookup.get(cancer_type, {})
        row['gen0_mean_fitness'] = gen.get('gen0_mean_fitness')
        row['final_mean_fitness'] = gen.get('final_mean_fitness')
        row['improvement_percent'] = gen.get('fitness_improvement_percent')

        table.append(row)

    return table


def generate_report(vanilla_results: dict, gen_comparison: dict,
                    depmap_validation: dict, output_dir: Path):
    """Generate the final comparison report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Comparison table
    comparison_table = generate_comparison_table(vanilla_results, gen_comparison)

    # Extract paired scores for statistical tests
    gen0_scores = []
    final_scores = []
    for cancer_data in gen_comparison.get('per_cancer_data', []):
        if cancer_data['gen0_mean_fitness'] and cancer_data['final_mean_fitness']:
            gen0_scores.append(cancer_data['gen0_mean_fitness'])
            final_scores.append(cancer_data['final_mean_fitness'])

    # Statistical significance
    stat_tests = compute_statistical_significance(gen0_scores, final_scores)

    # Summary statistics
    summary = gen_comparison.get('summary_statistics', {})
    learning_curve = gen_comparison.get('learning_curve', [])

    # Full report
    report = {
        'report_timestamp': timestamp,
        'summary': {
            'num_cancer_types': len(comparison_table),
            'vanilla_baseline': {
                'total_hypotheses': sum(r.get('vanilla_num_hypotheses', 0) for r in comparison_table),
                'avg_drug_compliance': mean([r['vanilla_drug_compliance'] for r in comparison_table
                                            if r.get('vanilla_drug_compliance')]) if any(r.get('vanilla_drug_compliance') for r in comparison_table) else None
            },
            'gen0_vs_final': {
                'avg_gen0_fitness': summary.get('avg_gen0_mean_fitness'),
                'avg_final_fitness': summary.get('avg_final_mean_fitness'),
                'avg_improvement': summary.get('avg_improvement_percent'),
                'improvement_rate': summary.get('improvement_rate')
            }
        },
        'statistical_significance': stat_tests,
        'learning_curve': learning_curve,
        'comparison_table': comparison_table
    }

    # Save JSON report
    report_file = output_dir / f"baseline_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    logging.info(f"Report saved to: {report_file}")

    # Generate markdown summary for paper
    md_report = generate_markdown_report(report, vanilla_results, gen_comparison)
    md_file = output_dir / f"baseline_report_{timestamp}.md"
    with open(md_file, 'w') as f:
        f.write(md_report)

    logging.info(f"Markdown report saved to: {md_file}")

    # Generate CSV for tables
    csv_file = output_dir / f"comparison_table_{timestamp}.csv"
    with open(csv_file, 'w') as f:
        headers = ['cancer_type', 'vanilla_num_hyp', 'vanilla_drug_compliance',
                   'gen0_mean_fitness', 'final_mean_fitness', 'improvement_pct']
        f.write(','.join(headers) + '\n')
        for row in comparison_table:
            values = [
                row['cancer_type'],
                str(row.get('vanilla_num_hypotheses', '')),
                f"{row.get('vanilla_drug_compliance', 0)*100:.1f}" if row.get('vanilla_drug_compliance') else '',
                f"{row.get('gen0_mean_fitness', ''):.2f}" if row.get('gen0_mean_fitness') else '',
                f"{row.get('final_mean_fitness', ''):.2f}" if row.get('final_mean_fitness') else '',
                f"{row.get('improvement_percent', ''):.1f}" if row.get('improvement_percent') else ''
            ]
            f.write(','.join(values) + '\n')

    logging.info(f"CSV table saved to: {csv_file}")

    return report


def generate_markdown_report(report: dict, vanilla_results: dict, gen_comparison: dict) -> str:
    """Generate markdown report for paper."""
    md = """# Baseline Comparison Report

## Summary

This report compares three conditions:
1. **Vanilla LLM + Literature**: Single LLM call with literature context (baseline)
2. **Gen 0**: Initial population from multi-agent structured generation (before evolution)
3. **Final**: Population after GA evolution (full pipeline)

## Key Findings

"""
    summary = report.get('summary', {})
    gen_summary = summary.get('gen0_vs_final', {})

    avg_gen0 = gen_summary.get('avg_gen0_fitness')
    avg_final = gen_summary.get('avg_final_fitness')
    avg_improve = gen_summary.get('avg_improvement')
    improve_rate = gen_summary.get('improvement_rate')

    md += f"""### Evolution Value (Gen 0 vs Final)
- Average Gen 0 mean fitness: {f'{avg_gen0:.2f}' if avg_gen0 is not None else 'N/A'}
- Average Final mean fitness: {f'{avg_final:.2f}' if avg_final is not None else 'N/A'}
- Average improvement: {f'{avg_improve:.2f}%' if avg_improve is not None else 'N/A'}
- Cancer types that improved: {f'{improve_rate*100:.1f}%' if improve_rate is not None else 'N/A'}

"""

    # Statistical significance
    stat = report.get('statistical_significance', {})
    if 'paired_ttest' in stat:
        ttest = stat['paired_ttest']
        md += f"""### Statistical Significance
- Paired t-test p-value: {ttest['p_value']:.4e}
- Significant at p<0.05: {'Yes' if ttest['significant_005'] else 'No'}
- Significant at p<0.01: {'Yes' if ttest['significant_001'] else 'No'}
- Sample size (N): {stat.get('n_samples', 'N/A')}

"""

    # Learning curve table
    md += """## Learning Curve

| Generation | Avg Mean Fitness | Std Dev |
|------------|------------------|---------|
"""
    for lc in report.get('learning_curve', []):
        md += f"| {lc['generation']} | {lc['avg_mean_fitness']:.2f} | {lc['std_mean_fitness']:.2f} |\n"

    # Comparison table
    md += """
## Per-Cancer-Type Comparison

| Cancer Type | Gen 0 | Final | Improvement |
|-------------|-------|-------|-------------|
"""
    for row in report.get('comparison_table', [])[:20]:  # First 20 for readability
        gen0 = f"{row.get('gen0_mean_fitness', 0):.1f}" if row.get('gen0_mean_fitness') else "N/A"
        final = f"{row.get('final_mean_fitness', 0):.1f}" if row.get('final_mean_fitness') else "N/A"
        imp = f"{row.get('improvement_percent', 0):.1f}%" if row.get('improvement_percent') else "N/A"
        md += f"| {row['cancer_type'][:30]} | {gen0} | {final} | {imp} |\n"

    if len(report.get('comparison_table', [])) > 20:
        md += f"\n*... and {len(report['comparison_table']) - 20} more cancer types*\n"

    return md


def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline comparison report for paper"
    )
    parser.add_argument(
        "--vanilla-dir",
        default="output/baselines/vanilla",
        help="Directory with vanilla baseline results"
    )
    parser.add_argument(
        "--comparison-dir",
        default="output/baselines/generation_comparison",
        help="Directory with generation comparison results"
    )
    parser.add_argument(
        "--output-dir",
        default="output/baselines/report",
        help="Output directory for report"
    )
    parser.add_argument(
        "--run-depmap-validation",
        action="store_true",
        help="Run DepMap validation on vanilla results"
    )

    args = parser.parse_args()

    vanilla_dir = project_root / args.vanilla_dir
    comparison_dir = project_root / args.comparison_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    vanilla_results = load_vanilla_results(vanilla_dir) if vanilla_dir.exists() else {}
    gen_comparison = load_generation_comparison(comparison_dir) if comparison_dir.exists() else {}

    # Run DepMap validation if requested
    depmap_validation = {}
    if args.run_depmap_validation and vanilla_results:
        depmap_validation = run_depmap_validation_for_vanilla(vanilla_dir)

    if not vanilla_results and not gen_comparison:
        logging.error("No data found. Run vanilla baseline and/or extract_generation_comparison first.")
        print("\nTo generate data, run:")
        print("  1. python scripts/run_baseline_vanilla.py --all")
        print("  2. python scripts/extract_generation_comparison.py")
        sys.exit(1)

    # Generate report
    report = generate_report(vanilla_results, gen_comparison, depmap_validation, output_dir)

    # Print summary
    print("\n" + "="*70)
    print("BASELINE COMPARISON REPORT GENERATED")
    print("="*70)

    if gen_comparison:
        summary = gen_comparison.get('summary_statistics', {})
        print(f"\nEvolution Value (from existing data):")
        print(f"  - Gen 0 avg fitness: {summary.get('avg_gen0_mean_fitness', 'N/A'):.2f}")
        print(f"  - Final avg fitness: {summary.get('avg_final_mean_fitness', 'N/A'):.2f}")
        print(f"  - Avg improvement: {summary.get('avg_improvement_percent', 'N/A'):.2f}%")

    stat = report.get('statistical_significance', {})
    if 'paired_ttest' in stat:
        print(f"\nStatistical Significance:")
        print(f"  - p-value: {stat['paired_ttest']['p_value']:.4e}")
        print(f"  - Significant: {'Yes' if stat['paired_ttest']['significant_005'] else 'No'}")

    print(f"\nOutput files in: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
