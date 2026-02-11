#!/usr/bin/env python3
"""
Extract predictions from pipeline results and baseline, unmask gene names,
and optionally fetch OpenTargets association scores.

Usage:
    python extract_predictions.py <run_dir> [--fetch-scores]

Arguments:
    run_dir        Path to the pipeline run directory
    --fetch-scores Optional: Fetch OpenTargets T2D association scores (requires internet)

Example:
    python extract_predictions.py output/t2d_target/run_20260206_104118
    python extract_predictions.py output/t2d_target/run_20260206_104118 --fetch-scores
"""

import json
import re
import sys
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class GenePrediction:
    """Represents a single gene prediction with metadata."""
    masked_id: str
    real_gene: str
    rank: int
    source: str  # "framework" or "baseline"
    hypothesis_id: Optional[str] = None
    fitness_score: Optional[float] = None
    confidence: Optional[str] = None
    rationale: Optional[str] = None
    opentargets_score: Optional[float] = None
    ensembl_id: Optional[str] = None


class OpenTargetsClient:
    """Client for querying OpenTargets API."""

    GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

    # T2D disease IDs (try multiple for broader coverage)
    T2D_DISEASE_IDS = [
        "MONDO_0005148",  # Primary: Type 2 diabetes mellitus (MONDO)
        "EFO_0001360",    # Type 2 diabetes mellitus (EFO)
        "MONDO_0005015",  # Diabetes mellitus (broader)
    ]
    T2D_PRIMARY_ID = "MONDO_0005148"

    def __init__(self, rate_limit_delay: float = 0.5):
        self.rate_limit_delay = rate_limit_delay
        self._ensembl_cache: Dict[str, str] = {}

    def _query_graphql(self, query: str, variables: dict) -> Optional[dict]:
        """Execute a GraphQL query against OpenTargets."""
        try:
            response = requests.post(
                self.GRAPHQL_URL,
                json={"query": query, "variables": variables},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"  OpenTargets API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"  OpenTargets API exception: {e}")
            return None

    def get_ensembl_id(self, gene_symbol: str) -> Optional[str]:
        """Look up Ensembl ID for a gene symbol."""
        if gene_symbol in self._ensembl_cache:
            return self._ensembl_cache[gene_symbol]

        query = """
        query SearchGene($queryString: String!) {
            search(queryString: $queryString, entityNames: ["target"], page: {size: 5, index: 0}) {
                hits {
                    id
                    name
                    entity
                }
            }
        }
        """

        result = self._query_graphql(query, {"queryString": gene_symbol})
        time.sleep(self.rate_limit_delay)

        if result and "data" in result and result["data"]["search"]["hits"]:
            for hit in result["data"]["search"]["hits"]:
                if hit["entity"] == "target":
                    # Check if the name matches
                    if hit["name"].upper() == gene_symbol.upper():
                        self._ensembl_cache[gene_symbol] = hit["id"]
                        return hit["id"]
            # If no exact match, use the first target hit
            for hit in result["data"]["search"]["hits"]:
                if hit["entity"] == "target":
                    self._ensembl_cache[gene_symbol] = hit["id"]
                    return hit["id"]

        return None

    def get_t2d_association_score(self, ensembl_id: str) -> Optional[float]:
        """Get the T2D association score for a gene (by Ensembl ID)."""
        query = """
        query TargetDiseaseAssociation($ensemblId: String!) {
            target(ensemblId: $ensemblId) {
                id
                approvedSymbol
                associatedDiseases(page: {size: 200, index: 0}) {
                    rows {
                        disease {
                            id
                            name
                        }
                        score
                    }
                }
            }
        }
        """

        result = self._query_graphql(query, {"ensemblId": ensembl_id})
        time.sleep(self.rate_limit_delay)

        if result and "data" in result and result["data"]["target"]:
            target_data = result["data"]["target"]
            associations = target_data.get("associatedDiseases", {}).get("rows", [])

            # Find T2D association - check all known T2D disease IDs
            best_score = None
            for assoc in associations:
                disease_id = assoc.get("disease", {}).get("id", "")
                disease_name = assoc.get("disease", {}).get("name", "").lower()
                score = assoc.get("score")

                # Check if this is a T2D-related disease
                is_t2d = (
                    disease_id in self.T2D_DISEASE_IDS or
                    "type 2 diabetes" in disease_name or
                    "type ii diabetes" in disease_name or
                    disease_id == "MONDO_0005148" or
                    disease_id == "EFO_0001360"
                )

                if is_t2d and score is not None:
                    # Return the highest score if multiple T2D associations found
                    if best_score is None or score > best_score:
                        best_score = score

            return best_score

        return None


def load_gene_mapping(run_dir: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load gene mapping and create both directions."""
    mapping_file = run_dir / "gene_mapping.json"

    if not mapping_file.exists():
        raise FileNotFoundError(f"Gene mapping not found: {mapping_file}")

    with open(mapping_file) as f:
        mapping = json.load(f)

    real_to_masked = mapping.get("real_to_masked", {})
    masked_to_real = {v: k for k, v in real_to_masked.items()}

    return real_to_masked, masked_to_real


def extract_ranked_targets_from_description(description: str) -> List[str]:
    """Extract ranked target gene IDs from hypothesis description text."""
    targets = []

    # Pattern to match: "1. G17025 | ..." or "1. G17025:" etc.
    pattern = r'^\d+\.\s*(G\d{5})\s*[\|:]'

    for line in description.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            targets.append(match.group(1))

    return targets


def extract_framework_predictions(results: dict, masked_to_real: Dict[str, str]) -> List[GenePrediction]:
    """Extract all gene predictions from framework hypotheses."""
    predictions = []
    seen_genes = set()

    # Get all hypotheses from final_population
    hypotheses = results.get("final_population", [])

    # Also include best_hypothesis if not already in population
    best = results.get("best_hypothesis", {})
    if best and best.get("id"):
        # Check if it's already in population
        pop_ids = {h.get("id") for h in hypotheses}
        if best.get("id") not in pop_ids:
            hypotheses = [best] + hypotheses

    for hyp in hypotheses:
        hyp_id = hyp.get("id", "unknown")
        fitness = hyp.get("fitness_score")

        # Method 1: Extract from ranked_targets field (if populated)
        ranked = hyp.get("ranked_targets", [])
        if ranked:
            for rank, gene_id in enumerate(ranked, 1):
                if gene_id not in seen_genes:
                    real_gene = masked_to_real.get(gene_id, f"UNKNOWN:{gene_id}")
                    predictions.append(GenePrediction(
                        masked_id=gene_id,
                        real_gene=real_gene,
                        rank=rank,
                        source="framework",
                        hypothesis_id=hyp_id,
                        fitness_score=fitness
                    ))
                    seen_genes.add(gene_id)

        # Method 2: Extract from description text (RANKED_TARGETS section)
        description = hyp.get("description", "")
        if description:
            desc_targets = extract_ranked_targets_from_description(description)
            for rank, gene_id in enumerate(desc_targets, 1):
                if gene_id not in seen_genes:
                    real_gene = masked_to_real.get(gene_id, f"UNKNOWN:{gene_id}")
                    predictions.append(GenePrediction(
                        masked_id=gene_id,
                        real_gene=real_gene,
                        rank=rank,
                        source="framework",
                        hypothesis_id=hyp_id,
                        fitness_score=fitness
                    ))
                    seen_genes.add(gene_id)

        # Method 3: Use target_gene_masked (primary target)
        primary = hyp.get("target_gene_masked", "")
        if primary and primary not in seen_genes:
            real_gene = hyp.get("target_gene_real") or masked_to_real.get(primary, f"UNKNOWN:{primary}")
            predictions.append(GenePrediction(
                masked_id=primary,
                real_gene=real_gene,
                rank=1,  # Primary target
                source="framework",
                hypothesis_id=hyp_id,
                fitness_score=fitness
            ))
            seen_genes.add(primary)

    return predictions


def extract_baseline_predictions(results: dict, masked_to_real: Dict[str, str]) -> List[GenePrediction]:
    """Extract baseline LLM predictions."""
    predictions = []

    # Try multiple locations for baseline predictions
    baseline = results.get("baseline_predictions", [])
    if not baseline:
        # Check in full_evaluation
        full_eval = results.get("full_evaluation", {})
        baseline = full_eval.get("baseline_predictions", [])

    for item in baseline:
        gene_id = item.get("target_gene_masked", "")
        if not gene_id:
            continue

        real_gene = masked_to_real.get(gene_id, f"UNKNOWN:{gene_id}")

        predictions.append(GenePrediction(
            masked_id=gene_id,
            real_gene=real_gene,
            rank=item.get("rank", 0),
            source="baseline",
            confidence=item.get("confidence"),
            rationale=item.get("rationale")
        ))

    return predictions


def load_ground_truth_cache(run_dir: Path) -> Dict[str, float]:
    """
    Try to load OpenTargets scores from cached ground truth or evaluation data.
    Returns a dict mapping gene symbols to their T2D association scores.
    """
    scores: Dict[str, float] = {}

    # Try evaluation_results.json
    eval_file = run_dir / "evaluation_results.json"
    if eval_file.exists():
        try:
            with open(eval_file) as f:
                data = json.load(f)
            # Extract scores from ground truth hits
            for tier in ["tier1", "tier2", "tier3"]:
                tier_data = data.get(tier, {})
                for gene_info in tier_data.get("hit_genes", []):
                    if isinstance(gene_info, dict):
                        gene = gene_info.get("gene", "")
                        score = gene_info.get("score")
                        if gene and score:
                            scores[gene] = score
        except Exception as e:
            print(f"  Warning: Could not load evaluation_results.json: {e}")

    # Try candidates.json which may have druggability scores
    candidates_file = run_dir / "candidates.json"
    if candidates_file.exists():
        try:
            with open(candidates_file) as f:
                candidates = json.load(f)
            for gene, info in candidates.items():
                if isinstance(info, dict) and "opentargets_score" in info:
                    scores[gene] = info["opentargets_score"]
        except Exception as e:
            print(f"  Warning: Could not load candidates.json: {e}")

    return scores


def fetch_opentargets_scores(predictions: List[GenePrediction], client: OpenTargetsClient) -> None:
    """Fetch OpenTargets association scores for all predictions (in-place update)."""
    unique_genes = {p.real_gene for p in predictions if not p.real_gene.startswith("UNKNOWN:")}

    print(f"\nFetching OpenTargets scores for {len(unique_genes)} unique genes...")

    gene_to_score: Dict[str, Tuple[Optional[str], Optional[float]]] = {}

    for i, gene in enumerate(unique_genes, 1):
        print(f"  [{i}/{len(unique_genes)}] Looking up {gene}...", end=" ")

        # Get Ensembl ID
        ensembl_id = client.get_ensembl_id(gene)
        if not ensembl_id:
            print("No Ensembl ID found")
            gene_to_score[gene] = (None, None)
            continue

        # Get T2D association score
        score = client.get_t2d_association_score(ensembl_id)
        gene_to_score[gene] = (ensembl_id, score)

        if score is not None:
            print(f"Ensembl={ensembl_id}, Score={score:.4f}")
        else:
            print(f"Ensembl={ensembl_id}, No T2D association")

    # Update predictions with scores
    for pred in predictions:
        if pred.real_gene in gene_to_score:
            pred.ensembl_id, pred.opentargets_score = gene_to_score[pred.real_gene]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Parse arguments
    fetch_scores = "--fetch-scores" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if not args:
        print(__doc__)
        sys.exit(1)

    run_dir = Path(args[0])
    if not run_dir.exists():
        # Try relative to pipeline directory
        base_dir = Path(__file__).parent.parent
        run_dir = base_dir / sys.argv[1]

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    print(f"Processing run: {run_dir}")

    # Find the pipeline results file
    results_files = list(run_dir.glob("pipeline_results_*.json"))
    if not results_files:
        print("Error: No pipeline_results_*.json file found")
        sys.exit(1)

    results_file = max(results_files, key=lambda f: f.stat().st_mtime)
    print(f"Using results file: {results_file.name}")

    # Load data
    with open(results_file) as f:
        results = json.load(f)

    real_to_masked, masked_to_real = load_gene_mapping(run_dir)
    print(f"Loaded gene mapping: {len(masked_to_real)} genes")

    # Extract predictions
    framework_preds = extract_framework_predictions(results, masked_to_real)
    baseline_preds = extract_baseline_predictions(results, masked_to_real)

    print(f"Framework predictions: {len(framework_preds)} genes")
    print(f"Baseline predictions: {len(baseline_preds)} genes")

    # Try to load OpenTargets scores from cache first
    all_predictions = framework_preds + baseline_preds
    cached_scores = load_ground_truth_cache(run_dir)

    if cached_scores:
        print(f"\nLoaded {len(cached_scores)} cached OpenTargets scores")
        # Apply cached scores to predictions
        for pred in all_predictions:
            if pred.real_gene in cached_scores:
                pred.opentargets_score = cached_scores[pred.real_gene]
    else:
        print("\nNo cached OpenTargets scores found")

    # Fetch remaining scores from API if requested
    if fetch_scores:
        # Only fetch for predictions that don't have scores yet
        unfetched = [p for p in all_predictions if p.opentargets_score is None and not p.real_gene.startswith("UNKNOWN:")]
        if unfetched:
            print(f"\n--fetch-scores enabled, querying OpenTargets API for {len(unfetched)} genes...")
            client = OpenTargetsClient(rate_limit_delay=0.3)
            fetch_opentargets_scores(unfetched, client)
        else:
            print("\nAll scores already loaded from cache")
    else:
        n_missing = sum(1 for p in all_predictions if p.opentargets_score is None)
        if n_missing > 0:
            print(f"\n{n_missing} genes still missing scores (use --fetch-scores to query API)")

    # Prepare output
    output = {
        "extraction_timestamp": datetime.now().isoformat(),
        "run_directory": str(run_dir),
        "pipeline_results_file": results_file.name,
        "framework_predictions": [asdict(p) for p in framework_preds],
        "baseline_predictions": [asdict(p) for p in baseline_preds],
        "summary": {
            "n_framework_genes": len(framework_preds),
            "n_baseline_genes": len(baseline_preds),
            "n_with_opentargets_score": sum(1 for p in all_predictions if p.opentargets_score is not None),
            "framework_genes_with_t2d_association": [
                {"gene": p.real_gene, "score": p.opentargets_score}
                for p in framework_preds if p.opentargets_score is not None
            ],
            "baseline_genes_with_t2d_association": [
                {"gene": p.real_gene, "score": p.opentargets_score}
                for p in baseline_preds if p.opentargets_score is not None
            ]
        }
    }

    # Save output
    output_file = run_dir / "extracted_predictions.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)

    print("\nFRAMEWORK PREDICTIONS:")
    print("-" * 50)
    for p in sorted(framework_preds, key=lambda x: (x.hypothesis_id or "", x.rank)):
        score_str = f"{p.opentargets_score:.4f}" if p.opentargets_score else "N/A"
        print(f"  {p.masked_id} -> {p.real_gene:15} | OT Score: {score_str:8} | Hyp: {p.hypothesis_id}")

    print("\nBASELINE PREDICTIONS:")
    print("-" * 50)
    for p in sorted(baseline_preds, key=lambda x: x.rank):
        score_str = f"{p.opentargets_score:.4f}" if p.opentargets_score else "N/A"
        print(f"  {p.masked_id} -> {p.real_gene:15} | OT Score: {score_str:8} | Rank: {p.rank}")

    return output


if __name__ == "__main__":
    main()
