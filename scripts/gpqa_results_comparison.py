"""
GPQA Results Comparison and Visualization

Compares baseline GPQA results with the framework's GA-enhanced results.
Generates comprehensive visualizations for analysis.
"""

import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_baseline_results(baseline_path: str) -> dict:
    """Load baseline GPQA results."""
    with open(baseline_path, 'r') as f:
        return json.load(f)

def load_framework_results(results_dir: str) -> list:
    """Load all batch results from the framework."""
    all_questions = []
    batch_files = sorted(glob.glob(f"{results_dir}/batch_*.json"))
    
    for batch_file in batch_files:
        with open(batch_file, 'r') as f:
            data = json.load(f)
            all_questions.extend(data['questions'])
    
    return all_questions

def extract_metrics(questions: list, source: str = 'baseline') -> pd.DataFrame:
    """Extract metrics from questions into a DataFrame."""
    records = []
    
    for q in questions:
        record = {
            'question_index': q['question_index'],
            'correct_answer': q['correct_answer'],
        }
        
        if source == 'baseline':
            # Baseline file structure
            bl = q.get('baseline', {})
            if bl:
                record.update({
                    'baseline_predicted': bl.get('predicted'),
                    'baseline_correct': bl.get('correct'),
                    'baseline_time': bl.get('time_seconds'),
                    'baseline_tokens': bl.get('total_tokens'),
                    'baseline_cost': bl.get('cost_usd'),
                })
        else:
            # Framework file structure - has baseline, ga3, ga5
            bl = q.get('baseline', {})
            if bl:
                record.update({
                    'fw_baseline_predicted': bl.get('predicted'),
                    'fw_baseline_correct': bl.get('correct'),
                    'fw_baseline_time': bl.get('time_seconds'),
                    'fw_baseline_tokens': bl.get('total_tokens'),
                    'fw_baseline_cost': bl.get('cost_usd'),
                    'fw_baseline_generations': bl.get('generations', 0),
                })
            
            ga3 = q.get('ga3', {})
            if ga3:
                record.update({
                    'ga3_predicted': ga3.get('predicted'),
                    'ga3_correct': ga3.get('correct'),
                    'ga3_time': ga3.get('time_seconds'),
                    'ga3_tokens': ga3.get('total_tokens'),
                    'ga3_cost': ga3.get('cost_usd'),
                    'ga3_generations': ga3.get('generations', 3),
                })
            
            ga5 = q.get('ga5', {})
            if ga5:
                record.update({
                    'ga5_predicted': ga5.get('predicted'),
                    'ga5_correct': ga5.get('correct'),
                    'ga5_time': ga5.get('time_seconds'),
                    'ga5_tokens': ga5.get('total_tokens'),
                    'ga5_cost': ga5.get('cost_usd'),
                    'ga5_generations': ga5.get('generations', 5),
                })
        
        records.append(record)
    
    return pd.DataFrame(records)

def create_accuracy_comparison(baseline_df: pd.DataFrame, framework_df: pd.DataFrame, output_dir: str):
    """Create accuracy comparison visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate accuracies
    baseline_acc = baseline_df['baseline_correct'].sum() / len(baseline_df) * 100
    
    # Framework accuracies
    fw_baseline_acc = framework_df['fw_baseline_correct'].sum() / len(framework_df) * 100
    ga3_acc = framework_df['ga3_correct'].sum() / len(framework_df) * 100
    ga5_acc = framework_df['ga5_correct'].sum() / len(framework_df) * 100
    
    # Bar chart for accuracy comparison
    methods = ['Baseline\n(Direct)', 'Framework\nBaseline', 'GA-3gen', 'GA-5gen']
    accuracies = [baseline_acc, fw_baseline_acc, ga3_acc, ga5_acc]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    bars = axes[0].bar(methods, accuracies, color=colors, edgecolor='white', linewidth=2)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('GPQA Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add horizontal line for reference
    axes[0].axhline(y=baseline_acc, color='#3498db', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Pie chart showing correct/incorrect distribution for best method
    best_method = 'GA-5gen'
    best_correct = framework_df['ga5_correct'].sum()
    best_incorrect = len(framework_df) - best_correct
    
    wedges, texts, autotexts = axes[1].pie(
        [best_correct, best_incorrect],
        labels=['Correct', 'Incorrect'],
        autopct='%1.1f%%',
        colors=['#2ecc71', '#e74c3c'],
        explode=(0.05, 0),
        startangle=90
    )
    axes[1].set_title(f'{best_method} Results Distribution\n({len(framework_df)} questions)', 
                      fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'baseline': baseline_acc,
        'fw_baseline': fw_baseline_acc,
        'ga3': ga3_acc,
        'ga5': ga5_acc
    }

def create_cost_comparison(baseline_df: pd.DataFrame, framework_df: pd.DataFrame, output_dir: str):
    """Create cost and token usage comparison visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Cost comparison
    baseline_cost = baseline_df['baseline_cost'].sum()
    fw_baseline_cost = framework_df['fw_baseline_cost'].sum()
    ga3_cost = framework_df['ga3_cost'].sum()
    ga5_cost = framework_df['ga5_cost'].sum()
    
    methods = ['Baseline\n(Direct)', 'Framework\nBaseline', 'GA-3gen', 'GA-5gen']
    costs = [baseline_cost, fw_baseline_cost, ga3_cost, ga5_cost]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    bars = axes[0].bar(methods, costs, color=colors, edgecolor='white', linewidth=2)
    axes[0].set_ylabel('Total Cost (USD)', fontsize=12)
    axes[0].set_title('API Cost Comparison', fontsize=14, fontweight='bold')
    
    for bar, cost in zip(bars, costs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'${cost:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Token usage comparison
    baseline_tokens = baseline_df['baseline_tokens'].sum()
    fw_baseline_tokens = framework_df['fw_baseline_tokens'].sum()
    ga3_tokens = framework_df['ga3_tokens'].sum()
    ga5_tokens = framework_df['ga5_tokens'].sum()
    
    tokens = [baseline_tokens/1e6, fw_baseline_tokens/1e6, ga3_tokens/1e6, ga5_tokens/1e6]
    
    bars2 = axes[1].bar(methods, tokens, color=colors, edgecolor='white', linewidth=2)
    axes[1].set_ylabel('Total Tokens (Millions)', fontsize=12)
    axes[1].set_title('Token Usage Comparison', fontsize=14, fontweight='bold')
    
    for bar, tok in zip(bars2, tokens):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{tok:.2f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cost_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'costs': dict(zip(methods, costs)),
        'tokens': dict(zip(methods, tokens))
    }

def create_time_comparison(baseline_df: pd.DataFrame, framework_df: pd.DataFrame, output_dir: str):
    """Create time comparison visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Total time comparison
    baseline_time = baseline_df['baseline_time'].sum()
    fw_baseline_time = framework_df['fw_baseline_time'].sum()
    ga3_time = framework_df['ga3_time'].sum()
    ga5_time = framework_df['ga5_time'].sum()
    
    methods = ['Baseline\n(Direct)', 'Framework\nBaseline', 'GA-3gen', 'GA-5gen']
    times = [baseline_time/3600, fw_baseline_time/3600, ga3_time/3600, ga5_time/3600]  # Convert to hours
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    bars = axes[0].bar(methods, times, color=colors, edgecolor='white', linewidth=2)
    axes[0].set_ylabel('Total Time (Hours)', fontsize=12)
    axes[0].set_title('Total Processing Time', fontsize=14, fontweight='bold')
    
    for bar, t in zip(bars, times):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{t:.2f}h', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Average time per question
    avg_times = [
        baseline_df['baseline_time'].mean(),
        framework_df['fw_baseline_time'].mean(),
        framework_df['ga3_time'].mean(),
        framework_df['ga5_time'].mean()
    ]
    
    bars2 = axes[1].bar(methods, avg_times, color=colors, edgecolor='white', linewidth=2)
    axes[1].set_ylabel('Average Time per Question (seconds)', fontsize=12)
    axes[1].set_title('Average Processing Time per Question', fontsize=14, fontweight='bold')
    
    for bar, t in zip(bars2, avg_times):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    f'{t:.0f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_efficiency_analysis(baseline_df: pd.DataFrame, framework_df: pd.DataFrame, output_dir: str):
    """Create efficiency analysis (accuracy per dollar, per token)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy per dollar
    baseline_acc = baseline_df['baseline_correct'].sum() / len(baseline_df) * 100
    baseline_cost = baseline_df['baseline_cost'].sum()
    
    fw_baseline_acc = framework_df['fw_baseline_correct'].sum() / len(framework_df) * 100
    fw_baseline_cost = framework_df['fw_baseline_cost'].sum()
    
    ga3_acc = framework_df['ga3_correct'].sum() / len(framework_df) * 100
    ga3_cost = framework_df['ga3_cost'].sum()
    
    ga5_acc = framework_df['ga5_correct'].sum() / len(framework_df) * 100
    ga5_cost = framework_df['ga5_cost'].sum()
    
    methods = ['Baseline\n(Direct)', 'Framework\nBaseline', 'GA-3gen', 'GA-5gen']
    acc_per_dollar = [
        baseline_acc / baseline_cost if baseline_cost > 0 else 0,
        fw_baseline_acc / fw_baseline_cost if fw_baseline_cost > 0 else 0,
        ga3_acc / ga3_cost if ga3_cost > 0 else 0,
        ga5_acc / ga5_cost if ga5_cost > 0 else 0
    ]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    bars = axes[0].bar(methods, acc_per_dollar, color=colors, edgecolor='white', linewidth=2)
    axes[0].set_ylabel('Accuracy Points per Dollar', fontsize=12)
    axes[0].set_title('Cost Efficiency\n(Higher is Better)', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, acc_per_dollar):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Accuracy gain vs cost ratio (compared to baseline)
    acc_gains = [
        0,  # Baseline reference
        fw_baseline_acc - baseline_acc,
        ga3_acc - baseline_acc,
        ga5_acc - baseline_acc
    ]
    cost_ratios = [
        1,  # Baseline reference
        fw_baseline_cost / baseline_cost if baseline_cost > 0 else 0,
        ga3_cost / baseline_cost if baseline_cost > 0 else 0,
        ga5_cost / baseline_cost if baseline_cost > 0 else 0
    ]
    
    colors_scatter = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    for i, (method, gain, ratio) in enumerate(zip(methods, acc_gains, cost_ratios)):
        axes[1].scatter(ratio, gain, s=200, c=colors_scatter[i], edgecolors='white', 
                       linewidth=2, label=method.replace('\n', ' '), zorder=5)
    
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    axes[1].axvline(x=1, color='gray', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Cost Ratio (vs Baseline)', fontsize=12)
    axes[1].set_ylabel('Accuracy Gain (% points vs Baseline)', fontsize=12)
    axes[1].set_title('Accuracy Gain vs Cost Trade-off', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].set_xlim(0, max(cost_ratios) * 1.2)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/efficiency_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_question_level_analysis(baseline_df: pd.DataFrame, framework_df: pd.DataFrame, output_dir: str):
    """Create question-level comparison heatmap."""
    # Merge dataframes on question_index
    merged = baseline_df[['question_index', 'baseline_correct']].merge(
        framework_df[['question_index', 'fw_baseline_correct', 'ga3_correct', 'ga5_correct']],
        on='question_index',
        how='inner'
    )
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create a matrix of correct/incorrect for each method
    methods = ['Baseline (Direct)', 'Framework Baseline', 'GA-3gen', 'GA-5gen']
    correct_cols = ['baseline_correct', 'fw_baseline_correct', 'ga3_correct', 'ga5_correct']
    
    matrix = merged[correct_cols].values.T.astype(float)
    
    # Create heatmap
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel('Question Index', fontsize=12)
    ax.set_title('Question-Level Correctness Comparison\n(Green=Correct, Red=Incorrect)', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Correct (1) / Incorrect (0)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/question_level_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return merged

def create_improvement_analysis(merged_df: pd.DataFrame, output_dir: str):
    """Analyze where GA improved or degraded performance."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Calculate improvement categories for GA-5gen vs Baseline
    categories = []
    for _, row in merged_df.iterrows():
        bl = row['baseline_correct']
        ga5 = row['ga5_correct']
        
        if bl and ga5:
            categories.append('Both Correct')
        elif not bl and ga5:
            categories.append('GA5 Fixed')
        elif bl and not ga5:
            categories.append('GA5 Broke')
        else:
            categories.append('Both Wrong')
    
    merged_df = merged_df.copy()
    merged_df['category'] = categories
    
    # Pie chart of categories
    cat_counts = merged_df['category'].value_counts()
    colors_pie = ['#2ecc71', '#9b59b6', '#e74c3c', '#95a5a6']
    
    wedges, texts, autotexts = axes[0, 0].pie(
        cat_counts.values,
        labels=cat_counts.index,
        autopct='%1.1f%%',
        colors=colors_pie[:len(cat_counts)],
        explode=[0.05 if 'Fixed' in cat else 0 for cat in cat_counts.index],
        startangle=90
    )
    axes[0, 0].set_title('GA-5gen vs Baseline Comparison', fontsize=14, fontweight='bold')
    
    # Summary statistics bar chart
    summary_data = {
        'Questions where GA5 helped': len(merged_df[merged_df['category'] == 'GA5 Fixed']),
        'Questions where GA5 hurt': len(merged_df[merged_df['category'] == 'GA5 Broke']),
        'Both methods correct': len(merged_df[merged_df['category'] == 'Both Correct']),
        'Both methods wrong': len(merged_df[merged_df['category'] == 'Both Wrong']),
    }
    
    bars = axes[0, 1].barh(list(summary_data.keys()), list(summary_data.values()),
                           color=['#2ecc71', '#e74c3c', '#3498db', '#95a5a6'])
    axes[0, 1].set_xlabel('Number of Questions', fontsize=12)
    axes[0, 1].set_title('Impact Summary', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, summary_data.values()):
        axes[0, 1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                       str(val), ha='left', va='center', fontsize=11, fontweight='bold')
    
    # Same analysis for GA-3gen
    categories_ga3 = []
    for _, row in merged_df.iterrows():
        bl = row['baseline_correct']
        ga3 = row['ga3_correct']
        
        if bl and ga3:
            categories_ga3.append('Both Correct')
        elif not bl and ga3:
            categories_ga3.append('GA3 Fixed')
        elif bl and not ga3:
            categories_ga3.append('GA3 Broke')
        else:
            categories_ga3.append('Both Wrong')
    
    merged_df['category_ga3'] = categories_ga3
    cat_counts_ga3 = merged_df['category_ga3'].value_counts()
    
    wedges, texts, autotexts = axes[1, 0].pie(
        cat_counts_ga3.values,
        labels=cat_counts_ga3.index,
        autopct='%1.1f%%',
        colors=colors_pie[:len(cat_counts_ga3)],
        explode=[0.05 if 'Fixed' in cat else 0 for cat in cat_counts_ga3.index],
        startangle=90
    )
    axes[1, 0].set_title('GA-3gen vs Baseline Comparison', fontsize=14, fontweight='bold')
    
    # Net improvement summary
    net_improvement_ga3 = len(merged_df[merged_df['category_ga3'] == 'GA3 Fixed']) - \
                          len(merged_df[merged_df['category_ga3'] == 'GA3 Broke'])
    net_improvement_ga5 = len(merged_df[merged_df['category'] == 'GA5 Fixed']) - \
                          len(merged_df[merged_df['category'] == 'GA5 Broke'])
    
    methods = ['GA-3gen', 'GA-5gen']
    net_improvements = [net_improvement_ga3, net_improvement_ga5]
    colors_net = ['#2ecc71' if x >= 0 else '#e74c3c' for x in net_improvements]
    
    bars = axes[1, 1].bar(methods, net_improvements, color=colors_net, edgecolor='white', linewidth=2)
    axes[1, 1].set_ylabel('Net Questions Improved', fontsize=12)
    axes[1, 1].set_title('Net Improvement\n(Fixed - Broke)', fontsize=14, fontweight='bold')
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    for bar, val in zip(bars, net_improvements):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'+{val}' if val >= 0 else str(val), 
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/improvement_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return merged_df

def create_summary_dashboard(accuracies: dict, baseline_df: pd.DataFrame, 
                            framework_df: pd.DataFrame, output_dir: str):
    """Create a summary dashboard with key metrics."""
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Main accuracy comparison
    ax1 = fig.add_subplot(gs[0, :2])
    methods = ['Baseline (Direct)', 'Framework Baseline', 'GA-3gen', 'GA-5gen']
    accs = [accuracies['baseline'], accuracies['fw_baseline'], 
            accuracies['ga3'], accuracies['ga5']]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    bars = ax1.bar(methods, accs, color=colors, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('GPQA Diamond Accuracy Comparison', fontsize=16, fontweight='bold')
    ax1.set_ylim(0, 100)
    
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Key metrics table
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    # Calculate key metrics
    total_questions = len(framework_df)
    baseline_correct = baseline_df['baseline_correct'].sum()
    ga5_correct = framework_df['ga5_correct'].sum()
    improvement = ga5_correct - baseline_correct
    
    metrics_text = f"""
    Key Metrics Summary
    ─────────────────────
    Total Questions: {total_questions}
    
    Baseline Correct: {baseline_correct}
    GA-5gen Correct: {ga5_correct}
    
    Net Improvement: {'+' if improvement >= 0 else ''}{improvement}
    
    Accuracy Gain: {(ga5_correct/total_questions - baseline_correct/total_questions)*100:.1f}%
    """
    
    ax2.text(0.1, 0.5, metrics_text, transform=ax2.transAxes, fontsize=12,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Cost comparison
    ax3 = fig.add_subplot(gs[1, 0])
    costs = [
        baseline_df['baseline_cost'].sum(),
        framework_df['fw_baseline_cost'].sum(),
        framework_df['ga3_cost'].sum(),
        framework_df['ga5_cost'].sum()
    ]
    
    bars = ax3.bar(methods, costs, color=colors, edgecolor='white', linewidth=2)
    ax3.set_ylabel('Cost (USD)', fontsize=10)
    ax3.set_title('Total Cost', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, cost in zip(bars, costs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'${cost:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Time comparison  
    ax4 = fig.add_subplot(gs[1, 1])
    times = [
        baseline_df['baseline_time'].sum()/3600,
        framework_df['fw_baseline_time'].sum()/3600,
        framework_df['ga3_time'].sum()/3600,
        framework_df['ga5_time'].sum()/3600
    ]
    
    bars = ax4.bar(methods, times, color=colors, edgecolor='white', linewidth=2)
    ax4.set_ylabel('Time (Hours)', fontsize=10)
    ax4.set_title('Total Processing Time', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, t in zip(bars, times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{t:.1f}h', ha='center', va='bottom', fontsize=8)
    
    # Accuracy per dollar
    ax5 = fig.add_subplot(gs[1, 2])
    acc_per_dollar = [accs[i] / costs[i] if costs[i] > 0 else 0 for i in range(4)]
    
    bars = ax5.bar(methods, acc_per_dollar, color=colors, edgecolor='white', linewidth=2)
    ax5.set_ylabel('Accuracy/Dollar', fontsize=10)
    ax5.set_title('Cost Efficiency', fontsize=12, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, acc_per_dollar):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('GPQA Results Analysis Dashboard', fontsize=18, fontweight='bold', y=1.02)
    plt.savefig(f'{output_dir}/summary_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_report(accuracies: dict, baseline_df: pd.DataFrame, 
                   framework_df: pd.DataFrame, merged_df: pd.DataFrame,
                   output_dir: str):
    """Generate a text report with key findings."""
    
    # Calculate metrics
    total_questions = len(merged_df)
    
    # Improvement analysis
    ga5_fixed = len(merged_df[merged_df['category'] == 'GA5 Fixed'])
    ga5_broke = len(merged_df[merged_df['category'] == 'GA5 Broke'])
    both_correct = len(merged_df[merged_df['category'] == 'Both Correct'])
    both_wrong = len(merged_df[merged_df['category'] == 'Both Wrong'])
    
    report = f"""
# GPQA Results Comparison Report

## Overview
- **Total Questions Analyzed**: {total_questions}
- **Baseline Source**: gpqa_baseline.json
- **Framework Source**: gpqa_unified/results/batch_*.json

## Accuracy Comparison

| Method | Accuracy | Correct | Incorrect |
|--------|----------|---------|-----------|
| Baseline (Direct) | {accuracies['baseline']:.1f}% | {baseline_df['baseline_correct'].sum()} | {len(baseline_df) - baseline_df['baseline_correct'].sum()} |
| Framework Baseline | {accuracies['fw_baseline']:.1f}% | {framework_df['fw_baseline_correct'].sum()} | {len(framework_df) - framework_df['fw_baseline_correct'].sum()} |
| GA-3gen | {accuracies['ga3']:.1f}% | {framework_df['ga3_correct'].sum()} | {len(framework_df) - framework_df['ga3_correct'].sum()} |
| GA-5gen | {accuracies['ga5']:.1f}% | {framework_df['ga5_correct'].sum()} | {len(framework_df) - framework_df['ga5_correct'].sum()} |

## GA-5gen Impact Analysis

- **Questions GA-5gen Fixed** (Baseline wrong, GA5 correct): {ga5_fixed}
- **Questions GA-5gen Broke** (Baseline correct, GA5 wrong): {ga5_broke}
- **Both Methods Correct**: {both_correct}
- **Both Methods Wrong**: {both_wrong}
- **Net Improvement**: {ga5_fixed - ga5_broke:+d} questions

## Cost Analysis

| Method | Total Cost (USD) | Cost per Question | Tokens Used |
|--------|-----------------|-------------------|-------------|
| Baseline (Direct) | ${baseline_df['baseline_cost'].sum():.2f} | ${baseline_df['baseline_cost'].mean():.4f} | {baseline_df['baseline_tokens'].sum():,} |
| Framework Baseline | ${framework_df['fw_baseline_cost'].sum():.2f} | ${framework_df['fw_baseline_cost'].mean():.4f} | {framework_df['fw_baseline_tokens'].sum():,} |
| GA-3gen | ${framework_df['ga3_cost'].sum():.2f} | ${framework_df['ga3_cost'].mean():.4f} | {framework_df['ga3_tokens'].sum():,} |
| GA-5gen | ${framework_df['ga5_cost'].sum():.2f} | ${framework_df['ga5_cost'].mean():.4f} | {framework_df['ga5_tokens'].sum():,} |

## Time Analysis

| Method | Total Time | Avg Time per Question |
|--------|------------|----------------------|
| Baseline (Direct) | {baseline_df['baseline_time'].sum()/3600:.2f} hours | {baseline_df['baseline_time'].mean():.1f} seconds |
| Framework Baseline | {framework_df['fw_baseline_time'].sum()/3600:.2f} hours | {framework_df['fw_baseline_time'].mean():.1f} seconds |
| GA-3gen | {framework_df['ga3_time'].sum()/3600:.2f} hours | {framework_df['ga3_time'].mean():.1f} seconds |
| GA-5gen | {framework_df['ga5_time'].sum()/3600:.2f} hours | {framework_df['ga5_time'].mean():.1f} seconds |

## Key Findings

1. **Accuracy**: GA-5gen achieved {accuracies['ga5']:.1f}% accuracy compared to {accuracies['baseline']:.1f}% for the baseline.
2. **Net Improvement**: GA-5gen correctly answered {ga5_fixed - ga5_broke:+d} more questions than the baseline.
3. **Cost Trade-off**: GA-5gen cost ${framework_df['ga5_cost'].sum():.2f} vs ${baseline_df['baseline_cost'].sum():.2f} for baseline ({framework_df['ga5_cost'].sum()/baseline_df['baseline_cost'].sum():.1f}x more expensive).
4. **Time Trade-off**: GA-5gen took {framework_df['ga5_time'].sum()/3600:.1f} hours vs {baseline_df['baseline_time'].sum()/3600:.2f} hours for baseline.

## Generated Visualizations

1. `accuracy_comparison.png` - Overall accuracy comparison
2. `cost_comparison.png` - Cost and token usage analysis
3. `time_comparison.png` - Processing time analysis
4. `efficiency_analysis.png` - Accuracy per dollar analysis
5. `question_level_comparison.png` - Per-question correctness heatmap
6. `improvement_analysis.png` - Where GA helped/hurt analysis
7. `summary_dashboard.png` - Combined dashboard view
"""
    
    with open(f'{output_dir}/comparison_report.md', 'w') as f:
        f.write(report)
    
    return report

def main():
    # Paths
    base_dir = Path('/Users/jeffersonchen/programming/MixLab/DeepScientists/AI_Coscientist_Rep')
    baseline_path = base_dir / 'output/gpqa/gpqa_baseline.json'
    framework_dir = base_dir / 'output/gpqa_unified/results'
    output_dir = base_dir / 'output/gpqa_analysis'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading baseline results...")
    baseline_data = load_baseline_results(str(baseline_path))
    baseline_df = extract_metrics(baseline_data['questions'], source='baseline')
    print(f"  Loaded {len(baseline_df)} baseline questions")
    
    print("Loading framework results...")
    framework_questions = load_framework_results(str(framework_dir))
    framework_df = extract_metrics(framework_questions, source='framework')
    print(f"  Loaded {len(framework_df)} framework questions")
    
    print("\nGenerating visualizations...")
    
    print("  - Accuracy comparison")
    accuracies = create_accuracy_comparison(baseline_df, framework_df, str(output_dir))
    
    print("  - Cost comparison")
    create_cost_comparison(baseline_df, framework_df, str(output_dir))
    
    print("  - Time comparison")
    create_time_comparison(baseline_df, framework_df, str(output_dir))
    
    print("  - Efficiency analysis")
    create_efficiency_analysis(baseline_df, framework_df, str(output_dir))
    
    print("  - Question-level analysis")
    merged_df = create_question_level_analysis(baseline_df, framework_df, str(output_dir))
    
    print("  - Improvement analysis")
    merged_df = create_improvement_analysis(merged_df, str(output_dir))
    
    print("  - Summary dashboard")
    create_summary_dashboard(accuracies, baseline_df, framework_df, str(output_dir))
    
    print("\nGenerating report...")
    report = generate_report(accuracies, baseline_df, framework_df, merged_df, str(output_dir))
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    for f in output_dir.glob('*'):
        print(f"  - {f.name}")
    
    # Print quick summary
    print("\n" + "-"*60)
    print("QUICK SUMMARY:")
    print("-"*60)
    print(f"Baseline Accuracy: {accuracies['baseline']:.1f}%")
    print(f"GA-5gen Accuracy:  {accuracies['ga5']:.1f}%")
    print(f"Improvement:       {accuracies['ga5'] - accuracies['baseline']:+.1f}%")

if __name__ == '__main__':
    main()
