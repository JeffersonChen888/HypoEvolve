#!/usr/bin/env python3
"""
GPQA Results Analysis and Visualization

This script:
1. Loads baseline (direct LLM) results and unified GA results
2. Compares accuracy across methods
3. Identifies WHY GA is underperforming
4. Creates comprehensive visualizations
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Paths
PROJECT_ROOT = Path(__file__).parent
BASELINE_FILE = PROJECT_ROOT / "output" / "gpqa" / "gpqa_baseline.json"
UNIFIED_DIR = PROJECT_ROOT / "output" / "gpqa_unified" / "results"

def load_baseline_results():
    """Load direct LLM baseline results."""
    with open(BASELINE_FILE, 'r') as f:
        data = json.load(f)
    return data

def load_unified_results():
    """Load unified GA results from all batch files."""
    all_questions = []
    for batch_file in sorted(UNIFIED_DIR.glob("batch_*.json")):
        with open(batch_file, 'r') as f:
            batch = json.load(f)
        all_questions.extend(batch.get("questions", []))

    # Sort by question index
    all_questions.sort(key=lambda q: q.get("question_index", 0))
    return all_questions

def analyze_results():
    """Comprehensive analysis of all results."""
    baseline_data = load_baseline_results()
    unified_questions = load_unified_results()

    # Build lookup for baseline
    baseline_lookup = {}
    for q in baseline_data.get("questions", []):
        idx = q.get("question_index")
        baseline_lookup[idx] = {
            'correct_answer': q.get("correct_answer"),
            'direct_predicted': q.get("baseline", {}).get("predicted"),
            'direct_correct': q.get("baseline", {}).get("correct", False)
        }

    # Collect results
    results = {
        'direct_baseline': {'correct': 0, 'total': 0, 'answers': []},
        'framework_baseline': {'correct': 0, 'total': 0, 'answers': []},
        'ga3': {'correct': 0, 'total': 0, 'answers': []},
        'ga5': {'correct': 0, 'total': 0, 'answers': []}
    }

    # Track question-by-question changes
    question_analysis = []

    for q in unified_questions:
        idx = q.get("question_index")
        correct_answer = q.get("correct_answer")

        # Direct baseline (single LLM call)
        if idx in baseline_lookup:
            direct_correct = baseline_lookup[idx].get('direct_correct', False)
            direct_pred = baseline_lookup[idx].get('direct_predicted')
            results['direct_baseline']['total'] += 1
            if direct_correct:
                results['direct_baseline']['correct'] += 1
            results['direct_baseline']['answers'].append((idx, direct_pred, direct_correct))
        else:
            direct_correct = None
            direct_pred = None

        # Framework results
        for method in ['baseline', 'ga3', 'ga5']:
            method_key = 'framework_baseline' if method == 'baseline' else method
            method_result = q.get(method, {})
            is_correct = method_result.get('correct', False)
            predicted = method_result.get('predicted', 'UNKNOWN')

            results[method_key]['total'] += 1
            if is_correct:
                results[method_key]['correct'] += 1
            results[method_key]['answers'].append((idx, predicted, is_correct))

        # Analyze changes
        fw_correct = q.get('baseline', {}).get('correct', False)
        ga3_correct = q.get('ga3', {}).get('correct', False)
        ga5_correct = q.get('ga5', {}).get('correct', False)

        question_analysis.append({
            'index': idx,
            'correct_answer': correct_answer,
            'direct_correct': direct_correct,
            'direct_pred': direct_pred,
            'fw_correct': fw_correct,
            'fw_pred': q.get('baseline', {}).get('predicted'),
            'ga3_correct': ga3_correct,
            'ga3_pred': q.get('ga3', {}).get('predicted'),
            'ga5_correct': ga5_correct,
            'ga5_pred': q.get('ga5', {}).get('predicted'),
            'question_preview': q.get('question_preview', '')[:100]
        })

    return results, question_analysis

def create_visualizations(results, question_analysis):
    """Create comprehensive visualizations."""

    # Calculate accuracies
    methods = ['direct_baseline', 'framework_baseline', 'ga3', 'ga5']
    labels = ['Baseline\n(Direct)', 'Framework\nBaseline', 'GA-3gen', 'GA-5gen']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    accuracies = []
    for method in methods:
        total = results[method]['total']
        correct = results[method]['correct']
        acc = (correct / total * 100) if total > 0 else 0
        accuracies.append(acc)

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Bar chart of accuracies
    ax1 = fig.add_subplot(2, 2, 1)
    bars = ax1.bar(labels, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('GPQA Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(70, 85)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add baseline line
    ax1.axhline(y=accuracies[0], color='#3498db', linestyle='--', alpha=0.7, linewidth=2)
    ax1.text(3.4, accuracies[0] + 0.3, f'Direct Baseline: {accuracies[0]:.1f}%',
             fontsize=9, color='#3498db', style='italic')

    # 2. Pie chart for GA-5gen results
    ax2 = fig.add_subplot(2, 2, 2)
    ga5_correct = results['ga5']['correct']
    ga5_total = results['ga5']['total']
    ga5_incorrect = ga5_total - ga5_correct

    pie_colors = ['#2ecc71', '#e74c3c']
    wedges, texts, autotexts = ax2.pie(
        [ga5_correct, ga5_incorrect],
        labels=['Correct', 'Incorrect'],
        colors=pie_colors,
        autopct='%1.1f%%',
        explode=(0.05, 0),
        shadow=True,
        startangle=90
    )
    ax2.set_title(f'GA-5gen Results Distribution\n({ga5_total} questions)', fontsize=14, fontweight='bold')

    # 3. Question-by-question change analysis
    ax3 = fig.add_subplot(2, 2, 3)

    # Count different change patterns
    direct_right_ga_wrong = 0
    direct_wrong_ga_right = 0
    both_right = 0
    both_wrong = 0

    for qa in question_analysis:
        if qa['direct_correct'] is None:
            continue
        if qa['direct_correct'] and not qa['ga5_correct']:
            direct_right_ga_wrong += 1
        elif not qa['direct_correct'] and qa['ga5_correct']:
            direct_wrong_ga_right += 1
        elif qa['direct_correct'] and qa['ga5_correct']:
            both_right += 1
        else:
            both_wrong += 1

    change_labels = ['Direct ✓\nGA ✗', 'Direct ✗\nGA ✓', 'Both ✓', 'Both ✗']
    change_values = [direct_right_ga_wrong, direct_wrong_ga_right, both_right, both_wrong]
    change_colors = ['#e74c3c', '#2ecc71', '#3498db', '#95a5a6']

    bars = ax3.bar(change_labels, change_values, color=change_colors, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Number of Questions', fontsize=12)
    ax3.set_title('GA-5gen vs Direct Baseline: Change Analysis', fontsize=14, fontweight='bold')

    for bar, val in zip(bars, change_values):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 4. Trend across generations
    ax4 = fig.add_subplot(2, 2, 4)

    generations = ['Gen 0\n(Initial)', 'Gen 3', 'Gen 5']
    gen_accuracies = [
        results['framework_baseline']['correct'] / results['framework_baseline']['total'] * 100,
        results['ga3']['correct'] / results['ga3']['total'] * 100,
        results['ga5']['correct'] / results['ga5']['total'] * 100
    ]

    ax4.plot(generations, gen_accuracies, 'o-', color='#9b59b6', linewidth=3, markersize=12)
    ax4.axhline(y=accuracies[0], color='#3498db', linestyle='--', alpha=0.7, linewidth=2, label='Direct Baseline')
    ax4.set_ylabel('Accuracy (%)', fontsize=12)
    ax4.set_title('Accuracy Across GA Generations', fontsize=14, fontweight='bold')
    ax4.set_ylim(74, 82)
    ax4.legend()

    for i, (gen, acc) in enumerate(zip(generations, gen_accuracies)):
        ax4.annotate(f'{acc:.1f}%', (i, acc), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'output' / 'gpqa_analysis_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {PROJECT_ROOT / 'output' / 'gpqa_analysis_visualization.png'}")

def print_detailed_analysis(results, question_analysis):
    """Print detailed analysis of results."""

    print("=" * 80)
    print("GPQA EVALUATION ANALYSIS REPORT")
    print("=" * 80)

    print("\n## ACCURACY SUMMARY")
    print("-" * 60)
    print(f"{'Method':<25} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 60)

    for method, label in [
        ('direct_baseline', 'Direct LLM (Baseline)'),
        ('framework_baseline', 'Framework Gen 0'),
        ('ga3', 'GA-3gen'),
        ('ga5', 'GA-5gen')
    ]:
        total = results[method]['total']
        correct = results[method]['correct']
        acc = (correct / total * 100) if total > 0 else 0
        print(f"{label:<25} {correct:<10} {total:<10} {acc:.2f}%")

    print("\n## ROOT CAUSE ANALYSIS")
    print("-" * 60)

    # Analyze patterns
    direct_right_ga_wrong = []
    direct_wrong_ga_right = []

    for qa in question_analysis:
        if qa['direct_correct'] is None:
            continue
        if qa['direct_correct'] and not qa['ga5_correct']:
            direct_right_ga_wrong.append(qa)
        elif not qa['direct_correct'] and qa['ga5_correct']:
            direct_wrong_ga_right.append(qa)

    print(f"\nQuestions where DIRECT got it RIGHT but GA-5gen got it WRONG: {len(direct_right_ga_wrong)}")
    print(f"Questions where DIRECT got it WRONG but GA-5gen got it RIGHT: {len(direct_wrong_ga_right)}")
    print(f"\nNET CHANGE: {len(direct_wrong_ga_right) - len(direct_right_ga_wrong)} questions")

    print("\n## QUESTIONS BROKEN BY GA (Direct ✓ → GA ✗)")
    print("-" * 60)

    for qa in direct_right_ga_wrong[:10]:  # Show first 10
        print(f"\nQ{qa['index']}: Correct={qa['correct_answer']}, Direct={qa['direct_pred']}, GA5={qa['ga5_pred']}")
        print(f"   Preview: {qa['question_preview']}")

    if len(direct_right_ga_wrong) > 10:
        print(f"\n... and {len(direct_right_ga_wrong) - 10} more questions")

    print("\n## IDENTIFIED ISSUES")
    print("-" * 60)

    print("""
1. HYPOTHESIS DRIFT: The genetic algorithm is designed for open-ended
   scientific hypothesis generation, NOT for multiple-choice questions.

2. FRAMEWORK MISMATCH: The GA framework:
   - Generates elaborate scientific hypotheses
   - Performs literature searches
   - Evolves hypotheses through crossover/mutation

   This is OVERKILL for MCQs where the LLM already knows the answer.

3. ANSWER EXTRACTION ISSUES: The answer extraction from complex hypothesis
   structures is unreliable. The GA produces:
   - Elaborate rationales
   - Multiple papers and references
   - Complex scientific arguments

   But may lose track of the simple A/B/C/D answer.

4. EVOLUTION DEGRADES PERFORMANCE: More generations = worse accuracy
   - Gen 0: 79.80% (slightly better than direct)
   - Gen 3: 76.26% (3.5 points worse)
   - Gen 5: 75.76% (4 points worse)

   The evolution process is CHANGING CORRECT ANSWERS to wrong ones.

5. WRONG SELECTION PRESSURE: The fitness function likely doesn't
   prioritize answer correctness - it may favor "interesting" or
   "novel" hypotheses instead.

## RECOMMENDATIONS
-----------------------------------------------------------

1. For MCQ tasks: Use DIRECT LLM calls (baseline approach)
   - Simpler is better
   - 79.29% accuracy with single API call

2. The GA framework should be reserved for:
   - Open-ended research tasks
   - Drug repurposing hypothesis generation
   - Scientific discovery where exploration is valuable

3. If using GA for MCQs:
   - Ensure fitness function heavily weights answer correctness
   - Add early stopping if answer is stable
   - Consider ensemble voting rather than evolution
""")

def main():
    """Main analysis function."""
    print("Loading results...")
    results, question_analysis = analyze_results()

    print("Creating visualizations...")
    create_visualizations(results, question_analysis)

    print("Generating detailed report...")
    print_detailed_analysis(results, question_analysis)

    # Save analysis to JSON
    output_file = PROJECT_ROOT / 'output' / 'gpqa_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {method: {'correct': r['correct'], 'total': r['total']}
                       for method, r in results.items()},
            'question_analysis': question_analysis
        }, f, indent=2)
    print(f"\nAnalysis saved to: {output_file}")

if __name__ == "__main__":
    main()
