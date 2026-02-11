#!/usr/bin/env python3
"""
Calculate actual API costs from pipeline logs by extracting TOKEN_USAGE entries.

Usage:
    python pipeline3/scripts/calculate_actual_cost.py <run_directory>

Example:
    python pipeline3/scripts/calculate_actual_cost.py pipeline3/output/lethal_genes_2/run_20251114_091032
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

# Pricing per 1M tokens (as of Nov 2025)
PRICING = {
    'gpt-5': {'input': 1.25, 'output': 10.00},
    'gpt-5-mini': {'input': 0.10, 'output': 0.50},
    'gpt-4o': {'input': 2.50, 'output': 10.00},
    'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
    'o3': {'input': 10.00, 'output': 40.00},
    'o3-mini': {'input': 1.10, 'output': 4.40},
    'o1': {'input': 10.00, 'output': 40.00},
    'o1-mini': {'input': 1.10, 'output': 4.40},
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate actual API costs from pipeline logs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "run_directory",
        type=str,
        help="Path to the run directory containing subdirectories with log files"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model detection (default: auto-detect from logs)"
    )

    return parser.parse_args()


def extract_token_usage_from_log(log_file: Path) -> list:
    """
    Extract all TOKEN_USAGE entries from a log file.

    Expected format in logs:
    TOKEN_USAGE: input=1234, output=5678, total=6912, cost=$0.012345

    Returns:
        List of tuples: [(input_tokens, output_tokens, cost), ...]
    """
    token_usage = []

    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if 'TOKEN_USAGE:' in line:
                    # Extract input, output, and cost
                    input_match = re.search(r'input=(\d+)', line)
                    output_match = re.search(r'output=(\d+)', line)
                    cost_match = re.search(r'cost=\$?([\d.]+)', line)

                    if input_match and output_match:
                        input_tokens = int(input_match.group(1))
                        output_tokens = int(output_match.group(1))
                        cost = float(cost_match.group(1)) if cost_match else 0.0

                        token_usage.append((input_tokens, output_tokens, cost))

    except Exception as e:
        print(f"Warning: Could not read {log_file}: {e}", file=sys.stderr)

    return token_usage


def detect_model_from_logs(log_file: Path) -> str:
    """Detect which model was used from log file."""
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

            # Look for model mentions
            if 'GPT-5' in content or 'gpt-5' in content:
                return 'gpt-5'
            elif 'GPT-4O' in content or 'gpt-4o' in content:
                if 'mini' in content.lower():
                    return 'gpt-4o-mini'
                return 'gpt-4o'
            elif 'o3-mini' in content or 'O3-MINI' in content:
                return 'o3-mini'
            elif 'o3' in content or 'O3' in content:
                return 'o3'
            elif 'o1-mini' in content or 'O1-MINI' in content:
                return 'o1-mini'
            elif 'o1' in content or 'O1' in content:
                return 'o1'
    except Exception:
        pass

    return 'gpt-4o'  # Default fallback


def main():
    """Main function."""
    args = parse_arguments()

    run_dir = Path(args.run_directory)

    if not run_dir.exists():
        print(f"Error: Directory not found: {run_dir}", file=sys.stderr)
        return 1

    # Find all log files
    log_files = list(run_dir.rglob("*.log"))

    if not log_files:
        print(f"Error: No log files found in {run_dir}", file=sys.stderr)
        print("Note: Log files should have .log extension", file=sys.stderr)
        return 1

    print("=" * 80)
    print(f"ACTUAL API COST CALCULATION")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Log files found: {len(log_files)}")
    print()

    # Extract token usage from all logs
    total_input = 0
    total_output = 0
    total_logged_cost = 0.0
    api_calls = 0

    per_gene_pair = defaultdict(lambda: {'input': 0, 'output': 0, 'calls': 0, 'cost': 0.0})

    for log_file in log_files:
        # Determine which gene pair this log belongs to
        gene_pair = log_file.parent.name if log_file.parent != run_dir else "unknown"

        token_usage = extract_token_usage_from_log(log_file)

        for input_tok, output_tok, cost in token_usage:
            total_input += input_tok
            total_output += output_tok
            total_logged_cost += cost
            api_calls += 1

            per_gene_pair[gene_pair]['input'] += input_tok
            per_gene_pair[gene_pair]['output'] += output_tok
            per_gene_pair[gene_pair]['calls'] += 1
            per_gene_pair[gene_pair]['cost'] += cost

    if api_calls == 0:
        print("Error: No TOKEN_USAGE entries found in log files", file=sys.stderr)
        print("Make sure your logs contain lines like:", file=sys.stderr)
        print("  TOKEN_USAGE: input=1234, output=5678, total=6912, cost=$0.012345", file=sys.stderr)
        return 1

    # Detect or use specified model
    if args.model:
        model = args.model
    else:
        # Try to detect from first log file
        model = detect_model_from_logs(log_files[0])

    print(f"Detected/specified model: {model}")
    print()

    # Print per-gene-pair breakdown
    print("--- Breakdown by Gene Pair ---")
    for gene_pair in sorted(per_gene_pair.keys()):
        data = per_gene_pair[gene_pair]
        print(f"{gene_pair}:")
        print(f"  API calls: {data['calls']}")
        print(f"  Input tokens:  {data['input']:,}")
        print(f"  Output tokens: {data['output']:,}")
        print(f"  Logged cost: ${data['cost']:.4f}")
        print()

    # Print total usage
    print("--- Total Token Usage ---")
    print(f"Total API calls: {api_calls}")
    print(f"Input tokens:  {total_input:,}")
    print(f"Output tokens: {total_output:,}")
    print(f"Total tokens:  {total_input + total_output:,}")
    print()

    # Calculate cost using current pricing
    if model in PRICING:
        pricing = PRICING[model]
        calculated_input_cost = (total_input * pricing['input']) / 1_000_000
        calculated_output_cost = (total_output * pricing['output']) / 1_000_000
        calculated_total_cost = calculated_input_cost + calculated_output_cost

        print("--- Actual Costs ---")
        print(f"Model: {model}")
        print(f"Pricing: ${pricing['input']}/1M input, ${pricing['output']}/1M output")
        print()
        print(f"Input cost:  {total_input:,} × ${pricing['input']}/1M = ${calculated_input_cost:.4f}")
        print(f"Output cost: {total_output:,} × ${pricing['output']}/1M = ${calculated_output_cost:.4f}")
        print(f"{'='*60}")
        print(f"TOTAL COST: ${calculated_total_cost:.4f}")
        print(f"{'='*60}")
        print()
        print(f"Logged cost in files: ${total_logged_cost:.4f}")

        # Check if there's a discrepancy
        if abs(calculated_total_cost - total_logged_cost) > 0.01:
            diff = calculated_total_cost - total_logged_cost
            diff_pct = (diff / calculated_total_cost) * 100 if calculated_total_cost > 0 else 0
            print(f"Discrepancy: ${diff:+.4f} ({diff_pct:+.1f}%)")
            print("Note: Discrepancy may be due to pricing changes or model detection errors")
    else:
        print(f"Warning: Unknown model '{model}' - cannot calculate accurate cost")
        print(f"Available models: {', '.join(PRICING.keys())}")
        print(f"\nLogged cost from files: ${total_logged_cost:.4f}")

    print()
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
