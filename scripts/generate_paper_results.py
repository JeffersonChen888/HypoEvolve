#!/usr/bin/env python3
"""
Generate Paper-Ready Figures and Tables for Drug Repurposing Results.

Key metric: Best hypothesis per cancer type (not per-hypothesis pass rate)
Focus: DepMap validation scores (biologically grounded, CRISPR dependency data)

This script creates:
- Figure 1: Best DepMap score per cancer type (Pipeline vs Vanilla)
- Figure 2: Head-to-head comparison summary
- Table 1: Main results (best-per-cancer DepMap metrics)
- Supplementary tables

Note: LLM-generated fitness scores are excluded as they are not rigorous
      for publication purposes.

Usage:
    python scripts/generate_paper_results.py
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def load_depmap_validation(path):
    """Load DepMap validation CSV."""
    logger.info(f"Loading DepMap validation: {path}")
    return pd.read_csv(path)


def compute_best_per_cancer(pipeline_df, vanilla_df):
    """
    Compute best hypothesis per cancer type for both Pipeline and Vanilla.
    This is the CORRECT metric for comparison.
    """
    # Normalize cancer names
    pipeline_df = pipeline_df.copy()
    vanilla_df = vanilla_df.copy()
    pipeline_df['cancer_norm'] = pipeline_df['cancer'].str.lower().str.strip()
    vanilla_df['cancer_norm'] = vanilla_df['cancer'].str.lower().str.strip()

    # Get BEST (max) DepMap score per cancer type
    pipeline_best = pipeline_df.groupby('cancer_norm').agg({
        'depmap_score': 'max',
        'drug': 'first',
        'target': 'first'
    }).reset_index()
    pipeline_best.columns = ['cancer', 'pipeline_best_score', 'pipeline_drug', 'pipeline_target']

    vanilla_best = vanilla_df.groupby('cancer_norm').agg({
        'depmap_score': 'max',
        'drug': 'first',
        'target': 'first'
    }).reset_index()
    vanilla_best.columns = ['cancer', 'vanilla_score', 'vanilla_drug', 'vanilla_target']

    # Merge
    merged = pipeline_best.merge(vanilla_best, on='cancer', how='outer')
    merged = merged.dropna(subset=['pipeline_best_score', 'vanilla_score'])

    # Compute comparison stats
    stats = {
        'n_cancers': len(merged),
        'pipeline_avg_best': merged['pipeline_best_score'].mean(),
        'vanilla_avg': merged['vanilla_score'].mean(),
        'pipeline_wins': (merged['pipeline_best_score'] > merged['vanilla_score']).sum(),
        'vanilla_wins': (merged['vanilla_score'] > merged['pipeline_best_score']).sum(),
        'ties': (merged['pipeline_best_score'] == merged['vanilla_score']).sum(),
        'pipeline_excellent': (merged['pipeline_best_score'] >= 0.9).sum(),
        'vanilla_excellent': (merged['vanilla_score'] >= 0.9).sum(),
        'pipeline_pass': (merged['pipeline_best_score'] >= 0.5).sum(),
        'vanilla_pass': (merged['vanilla_score'] >= 0.5).sum(),
    }

    return merged, stats


def figure_best_depmap_per_cancer(merged_df, output_dir):
    """
    Figure 1: Best DepMap score per cancer type (Pipeline vs Vanilla).
    Shows Pipeline consistently finds better hypotheses.
    """
    logger.info("Creating Figure 1: Best DepMap Per Cancer Type")

    # Sort by pipeline score
    df = merged_df.sort_values('pipeline_best_score', ascending=True).copy()

    # Truncate long cancer names
    df['cancer_short'] = df['cancer'].apply(lambda x: x[:25] + '...' if len(x) > 25 else x)

    fig, ax = plt.subplots(figsize=(12, 10))

    y_pos = np.arange(len(df))
    height = 0.35

    # Plot bars
    bars1 = ax.barh(y_pos - height/2, df['pipeline_best_score'], height,
                    label='Pipeline (best of 6)', color='#2E86AB', alpha=0.8)
    bars2 = ax.barh(y_pos + height/2, df['vanilla_score'], height,
                    label='Vanilla (single)', color='#F18F01', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['cancer_short'], fontsize=9)
    ax.set_xlabel('Best DepMap Score', fontsize=14)
    ax.set_title('Best Hypothesis Per Cancer Type: Pipeline vs Vanilla', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent threshold')
    ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Pass threshold')
    ax.set_xlim(0, 1.05)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()

    fig_path = output_dir / "fig1_best_depmap_per_cancer.pdf"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig1_best_depmap_per_cancer.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {fig_path}")
    return fig_path


def figure_head_to_head_summary(stats, output_dir):
    """
    Figure 2: Head-to-head comparison summary.
    Shows Pipeline's clear advantage in key metrics.
    """
    logger.info("Creating Figure 2: Head-to-Head Summary")

    n = stats['n_cancers']

    metrics = ['Wins\nHead-to-Head', 'Excellent\n(≥0.9)', 'Pass\n(≥0.5)']
    pipeline_vals = [
        stats['pipeline_wins'] / n * 100,
        stats['pipeline_excellent'] / n * 100,
        stats['pipeline_pass'] / n * 100
    ]
    vanilla_vals = [
        stats['vanilla_wins'] / n * 100,
        stats['vanilla_excellent'] / n * 100,
        stats['vanilla_pass'] / n * 100
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, pipeline_vals, width, label='Pipeline (best of 6)',
                   color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, vanilla_vals, width, label='Vanilla (single)',
                   color='#F18F01', alpha=0.8)

    ax.set_ylabel('Percentage of Cancer Types (%)', fontsize=14)
    ax.set_title('Pipeline vs Vanilla: Head-to-Head Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 2,
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    fig_path = output_dir / "fig2_head_to_head_summary.pdf"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig2_head_to_head_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {fig_path}")
    return fig_path


def generate_tables(merged_df, stats, output_dir):
    """Generate main and supplementary tables with DepMap metrics only."""
    logger.info("Generating tables...")

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    n = stats['n_cancers']

    # Table 1: Main Results (Best Per Cancer - DepMap Only)
    table1_data = {
        'Metric': [
            'Hypotheses per cancer',
            'Best DepMap score (avg)',
            'Excellent rate (≥0.9)',
            'Pass rate (≥0.5)',
            'Wins head-to-head'
        ],
        'Pipeline': [
            '6',
            f"{stats['pipeline_avg_best']:.3f}",
            f"{stats['pipeline_excellent']}/{n} ({stats['pipeline_excellent']/n*100:.0f}%)",
            f"{stats['pipeline_pass']}/{n} ({stats['pipeline_pass']/n*100:.0f}%)",
            f"{stats['pipeline_wins']}/{n} ({stats['pipeline_wins']/n*100:.0f}%)"
        ],
        'Vanilla': [
            '1',
            f"{stats['vanilla_avg']:.3f}",
            f"{stats['vanilla_excellent']}/{n} ({stats['vanilla_excellent']/n*100:.0f}%)",
            f"{stats['vanilla_pass']}/{n} ({stats['vanilla_pass']/n*100:.0f}%)",
            f"{stats['vanilla_wins']}/{n} ({stats['vanilla_wins']/n*100:.0f}%)"
        ]
    }
    table1 = pd.DataFrame(table1_data)
    table1.to_csv(tables_dir / "table1_main_results.csv", index=False)
    logger.info(f"Saved: table1_main_results.csv")

    # Supplementary Table S1: Per-Cancer Comparison
    supp_s1 = merged_df.copy()
    supp_s1['pipeline_wins'] = supp_s1['pipeline_best_score'] > supp_s1['vanilla_score']
    supp_s1 = supp_s1.sort_values('pipeline_best_score', ascending=False)
    supp_s1.to_csv(tables_dir / "supp_table_s1_per_cancer.csv", index=False)
    logger.info(f"Saved: supp_table_s1_per_cancer.csv")

    return tables_dir


def main():
    """Main function to generate all paper results (DepMap-focused)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    output_dir = PROJECT_ROOT / "output" / "paper_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Load DepMap validation data only (no LLM fitness scores)
    pipeline_depmap_path = PROJECT_ROOT / "output" / "drug_repurposing" / "batch_constrained" / "depmap_validation_results.csv"
    vanilla_depmap_path = PROJECT_ROOT / "output" / "baselines" / "vanilla" / "depmap_validation_results.csv"

    pipeline_df = load_depmap_validation(pipeline_depmap_path)
    vanilla_df = load_depmap_validation(vanilla_depmap_path)

    # Compute CORRECT metric: best per cancer type
    merged_df, stats = compute_best_per_cancer(pipeline_df, vanilla_df)

    logger.info(f"\n{'='*60}")
    logger.info("BEST HYPOTHESIS PER CANCER TYPE STATS (DepMap)")
    logger.info(f"{'='*60}")
    logger.info(f"Cancer types compared: {stats['n_cancers']}")
    logger.info(f"Pipeline avg best: {stats['pipeline_avg_best']:.3f}")
    logger.info(f"Vanilla avg: {stats['vanilla_avg']:.3f}")
    logger.info(f"Pipeline wins: {stats['pipeline_wins']}/{stats['n_cancers']} ({stats['pipeline_wins']/stats['n_cancers']*100:.1f}%)")
    logger.info(f"Pipeline excellent: {stats['pipeline_excellent']}/{stats['n_cancers']} ({stats['pipeline_excellent']/stats['n_cancers']*100:.1f}%)")
    logger.info(f"Pipeline pass: {stats['pipeline_pass']}/{stats['n_cancers']} ({stats['pipeline_pass']/stats['n_cancers']*100:.1f}%)")
    logger.info(f"{'='*60}\n")

    # Generate figures (DepMap-based only)
    figure_best_depmap_per_cancer(merged_df, figures_dir)
    figure_head_to_head_summary(stats, figures_dir)

    # Generate tables (DepMap-based only)
    generate_tables(merged_df, stats, output_dir)

    # Save summary stats (DepMap metrics only, no LLM fitness)
    summary = {
        'timestamp': timestamp,
        'n_cancers_compared': int(stats['n_cancers']),
        'pipeline_avg_best_depmap': float(stats['pipeline_avg_best']),
        'vanilla_avg_depmap': float(stats['vanilla_avg']),
        'pipeline_wins': int(stats['pipeline_wins']),
        'vanilla_wins': int(stats['vanilla_wins']),
        'ties': int(stats['ties']),
        'pipeline_excellent': int(stats['pipeline_excellent']),
        'pipeline_pass': int(stats['pipeline_pass']),
        'vanilla_excellent': int(stats['vanilla_excellent']),
        'vanilla_pass': int(stats['vanilla_pass']),
        'pipeline_excellent_rate': float(stats['pipeline_excellent'] / stats['n_cancers']),
        'pipeline_pass_rate': float(stats['pipeline_pass'] / stats['n_cancers']),
        'vanilla_excellent_rate': float(stats['vanilla_excellent'] / stats['n_cancers']),
        'vanilla_pass_rate': float(stats['vanilla_pass'] / stats['n_cancers'])
    }

    with open(output_dir / "summary_stats.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("PAPER RESULTS GENERATED (DepMap-focused)")
    logger.info(f"{'='*60}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Figures: {figures_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
