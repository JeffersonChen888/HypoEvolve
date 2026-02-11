#!/usr/bin/env python3
"""
Re-extract fitness scores from existing pipeline logs and update JSON output files.
This script fixes the issue where GPT-5 scores weren't properly extracted due to regex mismatch.
"""

import json
import re
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple


def extract_scores_from_review(review_text: str) -> Dict[str, int]:
    """Extract scores from reflection review text."""
    scores = {
        "novelty": 5,
        "biological_relevance": 5,
        "mechanistic_clarity": 5,
        "rival_quality": 5,
        "tractability": 5,
        "clinical_relevance": 5
    }

    # Updated patterns that match "Score: 9" format
    patterns = {
        "novelty": r"Novelty.*?Score:?\s*(\d+)",
        "biological_relevance": r"Biological [Rr]elevance.*?Score:?\s*(\d+)",
        "mechanistic_clarity": r"Mechanistic [Cc]larity.*?Score:?\s*(\d+)",
        "rival_quality": r"Rival [Qq]uality.*?Score:?\s*(\d+)",
        "tractability": r"Tractability.*?Score:?\s*(\d+)",
        "clinical_relevance": r"Clinical [Rr]elevance.*?Score:?\s*(\d+)"
    }

    for score_name, pattern in patterns.items():
        match = re.search(pattern, review_text, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                scores[score_name] = int(match.group(1))
            except (IndexError, ValueError) as e:
                print(f"  Warning: Failed to parse {score_name} score: {e}")

    return scores


def calculate_fitness_score(scores: Dict[str, int]) -> float:
    """Calculate fitness score from individual scores."""
    total_score = sum(scores.values())
    max_score = len(scores) * 10  # 6 criteria * 10 points each = 60

    if max_score > 0:
        fitness_score = (total_score / max_score) * 100
    else:
        fitness_score = 50.0

    return round(fitness_score, 2)


def parse_log_file(log_path: str) -> Dict[str, Tuple[Dict[str, int], float]]:
    """
    Parse log file to extract scores for each hypothesis.

    Returns:
        Dict mapping hypothesis_id to (scores_dict, fitness_score)
    """
    hypothesis_scores = {}

    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()

    # Find all fitness evaluation blocks
    # Pattern: "Performing lethal genes fitness evaluation for X x Y" followed by response
    eval_pattern = r"Performing lethal genes fitness evaluation for (\w+) x (\w+).*?GPT-5 RESPONSE START=+\s*(.*?)\s*GPT-5 RESPONSE END=+"

    for match in re.finditer(eval_pattern, log_content, re.DOTALL):
        gene_a, gene_b, review_text = match.groups()

        # Extract hypothesis ID from the next "fitness evaluation complete" line
        # Look for pattern like "fitness evaluation complete for hyp_pair-1_abc123: score = 50.00"
        complete_pattern = rf"fitness evaluation complete for (hyp_[^:]+): score = [\d.]+"
        complete_match = re.search(complete_pattern, log_content[match.end():match.end()+500])

        if complete_match:
            hypothesis_id = complete_match.group(1)
            scores = extract_scores_from_review(review_text)
            fitness = calculate_fitness_score(scores)

            hypothesis_scores[hypothesis_id] = (scores, fitness)
            print(f"  {hypothesis_id} ({gene_a}-{gene_b}): {scores} -> fitness={fitness:.2f}")

    return hypothesis_scores


def update_json_file(json_path: str, hypothesis_scores: Dict[str, Tuple[Dict[str, int], float]]) -> None:
    """Update JSON file with corrected fitness scores."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Update hypotheses in sorted list
    if "hypotheses_sorted_by_score" in data:
        for hyp in data["hypotheses_sorted_by_score"]:
            hyp_id = hyp.get("id")
            if hyp_id in hypothesis_scores:
                scores, fitness = hypothesis_scores[hyp_id]
                hyp["fitness_score"] = fitness
                print(f"  Updated {hyp_id}: {hyp.get('gene_a')}-{hyp.get('gene_b')} -> {fitness:.2f}")

        # Re-sort by fitness score
        data["hypotheses_sorted_by_score"].sort(key=lambda h: h.get("fitness_score", 0), reverse=True)
        print(f"\n  Re-sorted {len(data['hypotheses_sorted_by_score'])} hypotheses by fitness score")

    # Save updated JSON
    backup_path = json_path + ".backup"
    os.rename(json_path, backup_path)
    print(f"  Created backup: {backup_path}")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"  Updated: {json_path}")


def update_genealogy_file(json_path: str, hypothesis_scores: Dict[str, Tuple[Dict[str, int], float]]) -> None:
    """Update genealogy JSON file with corrected fitness scores."""
    if not os.path.exists(json_path):
        print(f"  Genealogy file not found: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Update hypotheses in each generation
    total_updated = 0
    if "generations" in data:
        for gen in data["generations"]:
            for hyp in gen.get("hypotheses", []):
                hyp_id = hyp.get("id")
                if hyp_id in hypothesis_scores:
                    scores, fitness = hypothesis_scores[hyp_id]
                    hyp["fitness_score"] = fitness
                    total_updated += 1

            # Recalculate generation fitness stats
            hyps = gen.get("hypotheses", [])
            if hyps:
                fitness_values = [h["fitness_score"] for h in hyps]
                gen["fitness_stats"] = {
                    "min": min(fitness_values),
                    "max": max(fitness_values),
                    "mean": sum(fitness_values) / len(fitness_values)
                }

    print(f"  Updated {total_updated} hypotheses in genealogy")

    # Save updated genealogy
    backup_path = json_path + ".backup"
    os.rename(json_path, backup_path)
    print(f"  Created backup: {backup_path}")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"  Updated: {json_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python reextract_scores_from_logs.py <run_folder>")
        print("Example: python reextract_scores_from_logs.py pipeline3/output/lethal_genes/run_20251106_181721")
        sys.exit(1)

    run_folder = sys.argv[1]

    if not os.path.isdir(run_folder):
        print(f"Error: Directory not found: {run_folder}")
        sys.exit(1)

    print(f"Re-extracting scores from: {run_folder}\n")

    # Find log file
    log_files = list(Path(run_folder).glob("pipeline3_*.log"))
    if not log_files:
        print(f"Error: No pipeline log file found in {run_folder}")
        sys.exit(1)

    log_path = str(log_files[0])
    print(f"Parsing log file: {log_path}\n")

    # Extract scores from log
    hypothesis_scores = parse_log_file(log_path)
    print(f"\nExtracted scores for {len(hypothesis_scores)} hypotheses\n")

    # Find and update JSON files
    json_files = list(Path(run_folder).glob("all_hypotheses_sorted_*.json"))
    if json_files:
        print("Updating sorted hypotheses JSON:")
        update_json_file(str(json_files[0]), hypothesis_scores)
    else:
        # Check parent directory (old location before fix)
        parent_dir = Path(run_folder).parent
        json_files = list(parent_dir.glob("all_hypotheses_sorted_*.json"))
        if json_files:
            print("Updating sorted hypotheses JSON (parent directory):")
            update_json_file(str(json_files[0]), hypothesis_scores)

    print()

    # Update genealogy file
    genealogy_files = list(Path(run_folder).glob("genealogy_*.json"))
    if genealogy_files:
        print("Updating genealogy JSON:")
        update_genealogy_file(str(genealogy_files[0]), hypothesis_scores)

    print("\n✓ Score re-extraction complete!")
    print("\nBackup files created with .backup extension")
    print("Compare old vs new fitness scores to verify extraction.")


if __name__ == "__main__":
    main()
