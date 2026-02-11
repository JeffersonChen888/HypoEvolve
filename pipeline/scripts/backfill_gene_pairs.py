#!/usr/bin/env python3
"""
Backfill gene_a and gene_b fields in evolved hypotheses from existing results.
Evolved hypotheses should inherit gene pairs from their parent hypotheses.
"""

import json
import sys
import os
from typing import Dict, List


def build_hypothesis_map(hypotheses: List[Dict]) -> Dict[str, Dict]:
    """Build a map of hypothesis_id -> hypothesis for quick lookup."""
    return {h.get("id"): h for h in hypotheses if h.get("id")}


def find_root_parent(hyp_id: str, hyp_map: Dict[str, Dict]) -> Dict:
    """
    Trace back to the root parent (Generation 0) to find gene_a and gene_b.

    Args:
        hyp_id: The hypothesis ID to trace
        hyp_map: Map of hypothesis_id -> hypothesis

    Returns:
        The root parent hypothesis (Generation 0) with gene_a and gene_b
    """
    visited = set()
    current_id = hyp_id

    while current_id and current_id not in visited:
        visited.add(current_id)
        current_hyp = hyp_map.get(current_id)

        if not current_hyp:
            return None

        # If this hypothesis has gene_a and gene_b, we found it
        if current_hyp.get("gene_a") and current_hyp.get("gene_b"):
            return current_hyp

        # Try parent_ids first (list of parents)
        parent_ids = current_hyp.get("parent_ids", [])
        if parent_ids:
            # Use first parent
            current_id = parent_ids[0]
            continue

        # Fallback to single parent_id field
        parent_id = current_hyp.get("parent_id")
        if parent_id:
            current_id = parent_id
            continue

        # No more parents
        break

    return None


def backfill_gene_pairs(hypotheses: List[Dict]) -> int:
    """
    Backfill gene_a and gene_b for evolved hypotheses by tracing to parent.

    Returns:
        Number of hypotheses updated
    """
    hyp_map = build_hypothesis_map(hypotheses)
    updated_count = 0

    for hyp in hypotheses:
        # Skip if already has gene_a and gene_b
        if hyp.get("gene_a") and hyp.get("gene_b"):
            continue

        hyp_id = hyp.get("id")
        if not hyp_id:
            continue

        # Find root parent with gene pairs
        root_parent = find_root_parent(hyp_id, hyp_map)

        if root_parent:
            gene_a = root_parent.get("gene_a")
            gene_b = root_parent.get("gene_b")

            if gene_a and gene_b:
                hyp["gene_a"] = gene_a
                hyp["gene_b"] = gene_b
                updated_count += 1
                print(f"  Updated {hyp_id}: {gene_a} x {gene_b} (from parent {root_parent.get('id')})")

    return updated_count


def main():
    if len(sys.argv) < 2:
        print("Usage: python backfill_gene_pairs.py <json_file>")
        print("Example: python backfill_gene_pairs.py pipeline3/output/lethal_genes/all_hypotheses_sorted_*.json")
        sys.exit(1)

    json_file = sys.argv[1]

    if not os.path.exists(json_file):
        print(f"Error: File not found: {json_file}")
        sys.exit(1)

    print(f"Backfilling gene pairs in: {json_file}\n")

    # Load JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process hypotheses
    if "hypotheses_sorted_by_score" in data:
        hypotheses = data["hypotheses_sorted_by_score"]
    elif "hypotheses" in data:
        hypotheses = data["hypotheses"]
    else:
        print("Error: No hypotheses found in JSON file")
        sys.exit(1)

    print(f"Found {len(hypotheses)} hypotheses")
    print("Backfilling gene pairs from parent hypotheses...\n")

    updated_count = backfill_gene_pairs(hypotheses)

    print(f"\n✓ Updated {updated_count} hypotheses with gene pairs")

    if updated_count > 0:
        # Create backup
        backup_file = json_file + ".pre-backfill.backup"
        os.rename(json_file, backup_file)
        print(f"\nCreated backup: {backup_file}")

        # Save updated JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Updated: {json_file}")
    else:
        print("\nNo updates needed - all hypotheses already have gene pairs")


if __name__ == "__main__":
    main()
