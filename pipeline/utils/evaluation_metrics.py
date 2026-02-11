"""
T2D Evaluation Metrics Module

This module provides comprehensive evaluation metrics for T2D drug target predictions:
1. OpenTargets association score correlation
2. Baseline LLM comparison
3. @K recall, precision, MRR, MAP metrics

NO TRACTABILITY FEATURES - all druggability features are safe, non-identifying.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
import numpy as np

# Try to import scipy for correlation metrics
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available - correlation metrics will be limited")


class T2DEvaluationMetrics:
    """
    Comprehensive evaluation metrics for T2D drug target predictions.

    Metrics include:
    1. OpenTargets association score vs framework fitness score correlation
    2. Baseline LLM comparison (single-call prediction)
    3. @K recall, precision, MRR, MAP for known drug targets
    """

    def __init__(self, cache_dir: str = None):
        """
        Initialize the evaluation metrics.

        Args:
            cache_dir: Directory for caching OpenTargets data
        """
        self.cache_dir = cache_dir
        self.opentargets_scores: Dict[str, float] = {}
        self.ground_truth_genes: Set[str] = set()
        self.tier_genes: Dict[str, Set[str]] = {}

    # =========================================================================
    # METRIC 1: OpenTargets Association Score Correlation
    # =========================================================================

    def load_opentargets_association_scores(self, gene_list: List[str]) -> Dict[str, float]:
        """
        Load association scores from OpenTargets for T2D.

        The association score represents the overall evidence strength for
        a gene-disease association in OpenTargets.

        Args:
            gene_list: List of gene symbols to query

        Returns:
            Dict mapping gene symbol to association score (0-1)
        """
        logging.info(f"Loading OpenTargets association scores for {len(gene_list)} genes...")

        scores = {}
        
        try:
            # Use GroundTruthLoader to get scores (prioritizes TSV)
            from utils.ground_truth_loader import GroundTruthLoader
            loader = GroundTruthLoader()
            all_targets, _ = loader.load_all_ground_truth()
            
            # Create score map
            scores = {t.gene_symbol.upper(): t.evidence_score for t in all_targets}
            
            logging.info(f"Loaded {len(scores)} scores from GroundTruthLoader")
            
            # Filter to only requested genes
            gene_set = set(g.upper() for g in gene_list)
            filtered_scores = {
                g: scores[g] for g in gene_set if g in scores
            }
            
            logging.info(f"Matched {len(filtered_scores)} genes from query list")
            scores = filtered_scores
                
        except Exception as e:
            logging.warning(f"Failed to fetch OpenTargets scores via loader: {e}")
            # Fallback to empty if loader fails
            scores = {}

        self.opentargets_scores = scores
        logging.info(f"Loaded association scores for {len(scores)} genes")

        return scores

    def _get_ensembl_ids(self, gene_symbols: List[str]) -> Dict[str, str]:
        """
        Map gene symbols to Ensembl IDs using MyGene.info.

        Args:
            gene_symbols: List of gene symbols

        Returns:
            Dict mapping symbol to Ensembl ID
        """
        import requests

        mapping = {}

        try:
            response = requests.post(
                "https://mygene.info/v3/query",
                data={
                    "q": ",".join(gene_symbols),
                    "scopes": "symbol",
                    "fields": "ensembl.gene",
                    "species": "human"
                },
                timeout=30
            )

            if response.status_code == 200:
                results = response.json()
                for result in results:
                    if isinstance(result, dict) and 'query' in result:
                        symbol = result['query']
                        ensembl = result.get('ensembl', {})
                        if isinstance(ensembl, dict):
                            ensembl_id = ensembl.get('gene')
                        elif isinstance(ensembl, list) and len(ensembl) > 0:
                            ensembl_id = ensembl[0].get('gene')
                        else:
                            ensembl_id = None

                        if ensembl_id:
                            mapping[symbol] = ensembl_id

        except Exception as e:
            logging.warning(f"Failed to get Ensembl IDs: {e}")

        return mapping

    def compute_score_correlation(self,
                                   predictions: List[Dict],
                                   gene_mapping: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Compute correlation between framework fitness scores and OpenTargets association scores.

        This measures whether our framework's ranking aligns with established
        evidence in OpenTargets.

        Args:
            predictions: List of hypothesis dicts with fitness_score and target_gene_masked
            gene_mapping: Dict mapping masked ID to real gene symbol

        Returns:
            Dict with correlation statistics
        """
        if not SCIPY_AVAILABLE:
            return {"error": "scipy not available for correlation computation"}

        if not self.opentargets_scores:
            logging.warning("No OpenTargets scores loaded. Call load_opentargets_association_scores first.")
            return {"error": "No OpenTargets scores available"}

        # Build paired scores
        framework_scores = []
        opentargets_scores = []
        gene_pairs = []

        for pred in predictions:
            masked_id = pred.get('target_gene_masked')
            fitness = pred.get('fitness_score', 0)

            if not masked_id or fitness is None:
                continue

            # Get real gene name
            real_gene = None
            if gene_mapping:
                real_gene = gene_mapping.get(masked_id)
            elif 'target_gene_real' in pred:
                real_gene = pred['target_gene_real']

            if not real_gene:
                continue

            # Get OpenTargets score
            ot_score = self.opentargets_scores.get(real_gene)
            if ot_score is not None:
                framework_scores.append(fitness)
                opentargets_scores.append(ot_score)
                gene_pairs.append((masked_id, real_gene))

        if len(framework_scores) < 3:
            return {
                "error": "Insufficient paired scores for correlation",
                "n_pairs": len(framework_scores)
            }

        # Compute correlations
        framework_arr = np.array(framework_scores)
        opentargets_arr = np.array(opentargets_scores)

        pearson_r, pearson_p = stats.pearsonr(framework_arr, opentargets_arr)
        spearman_r, spearman_p = stats.spearmanr(framework_arr, opentargets_arr)
        kendall_tau, kendall_p = stats.kendalltau(framework_arr, opentargets_arr)

        return {
            "n_genes_evaluated": len(framework_scores),
            "pearson_correlation": round(pearson_r, 4),
            "pearson_pvalue": pearson_p,
            "spearman_correlation": round(spearman_r, 4),
            "spearman_pvalue": spearman_p,
            "kendall_tau": round(kendall_tau, 4),
            "kendall_pvalue": kendall_p,
            "framework_score_range": (min(framework_scores), max(framework_scores)),
            "opentargets_score_range": (min(opentargets_scores), max(opentargets_scores)),
            "gene_pairs_used": gene_pairs
        }

    # =========================================================================
    # METRIC 2: Baseline LLM Comparison
    # =========================================================================

    def run_baseline_llm(self,
                         analysis_context: str,
                         literature_context: str = None,  # DEPRECATED - not used
                         n_predictions: int = 5) -> List[Dict]:
        """
        Run baseline LLM prediction (single call, no GA framework).

        This provides a fair comparison to see if the GA framework improves
        over simple LLM prediction using DATA ONLY.

        IMPORTANT: Baseline receives ONLY the data analysis, NOT literature context.
        This ensures fair comparison: baseline = LLM + data only.
        The literature_context parameter is kept for backward compatibility but ignored.

        Args:
            analysis_context: The same analysis context provided to framework
            literature_context: DEPRECATED - ignored for fair comparison
            n_predictions: Number of genes to predict (for multi-gene ranking)

        Returns:
            List of baseline predictions
        """
        from external_tools.llm_client import llm_generate

        logging.info(f"Running baseline LLM prediction for {n_predictions} genes (DATA ONLY, no literature)...")

        # Baseline prompt - DATA ONLY, no literature context
        # This ensures fair comparison: baseline tests what LLM can do with just the data
        baseline_prompt = f"""You are a computational biologist analyzing gene expression data.

=== GENE EXPRESSION ANALYSIS (MASKED) ===
{analysis_context}

=== TASK ===
Based ONLY on the multi-omics data above, identify the TOP {n_predictions} genes
most likely to be causally involved in the disease phenotype.

Rank them by strength of DATA EVIDENCE.

CRITICAL CONSTRAINTS:
- Select ONLY gene IDs from the analysis (e.g., G00042, G00015)
- Base reasoning ONLY on: priority scores, fold-change, p-values, pathway enrichment
- Do NOT use external knowledge about known disease genes
- Do NOT guess real gene names

=== OUTPUT FORMAT ===
Provide your top {n_predictions} predictions in this exact format:

RANKED_PREDICTIONS:
1. [Gene ID] - [Rationale citing specific DATA EVIDENCE]
2. [Gene ID] - [Rationale citing specific DATA EVIDENCE]
3. [Gene ID] - [Rationale citing specific DATA EVIDENCE]
{f'4. [Gene ID] - [Rationale citing specific DATA EVIDENCE]' if n_predictions >= 4 else ''}
{f'5. [Gene ID] - [Rationale citing specific DATA EVIDENCE]' if n_predictions >= 5 else ''}

CONFIDENCE: [HIGH/MEDIUM/LOW]
"""

        response = llm_generate(baseline_prompt)

        # Parse baseline predictions
        predictions = self._parse_baseline_predictions(response, n_predictions)

        logging.info(f"Baseline LLM predicted {len(predictions)} genes")

        return predictions

    def _parse_baseline_predictions(self, response: str, expected_n: int) -> List[Dict]:
        """
        Parse baseline LLM response into structured predictions.

        Args:
            response: Raw LLM response
            expected_n: Expected number of predictions

        Returns:
            List of prediction dicts with rank, gene_id, rationale
        """
        predictions = []

        # Find RANKED_PREDICTIONS section
        pattern = r'(\d+)\.\s*(G\d{5})\s*[-:]\s*(.+?)(?=\n\d+\.|CONFIDENCE:|$)'
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            rank = int(match[0])
            gene_id = match[1]
            rationale = match[2].strip()

            predictions.append({
                "rank": rank,
                "target_gene_masked": gene_id,
                "rationale": rationale,
                "source": "baseline_llm"
            })

        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', response, re.IGNORECASE)
        confidence = conf_match.group(1).upper() if conf_match else "MEDIUM"

        for pred in predictions:
            pred["confidence"] = confidence

        return predictions[:expected_n]

        return results

    def compare_to_baseline(self,
                            framework_predictions: List[Dict],
                            baseline_predictions: List[Dict],
                            ground_truth: Set[str],
                            gene_mapping: Dict[str, str],
                            tier_genes: Dict[str, Set[str]] = None) -> Dict[str, Any]:
        """
        Compare framework predictions to baseline LLM predictions.
        
        Args:
            framework_predictions: Predictions from GA framework
            baseline_predictions: Predictions from baseline LLM
            ground_truth: Set of real gene symbols that are known T2D targets
            gene_mapping: Dict mapping masked ID to real gene symbol
            tier_genes: Optional dict with tier info for more detailed analysis
            
        Returns:
            Comparison metrics dict
        """
        def count_hits(predictions, ground_truth, gene_mapping):
            """Count how many predictions hit ground truth."""
            hits = []
            for pred in predictions:
                masked_id = pred.get('target_gene_masked')
                real_gene = gene_mapping.get(masked_id)
                if real_gene and real_gene.upper() in {g.upper() for g in ground_truth}:
                    hits.append(real_gene)
            return hits

        # Get framework predictions (sorted by fitness if available)
        framework_sorted = sorted(
            framework_predictions,
            key=lambda x: x.get('fitness_score', 0),
            reverse=True
        )
        
        # Build rank maps (MaskedID -> Rank 1-based)
        framework_ranks = {}
        for i, pred in enumerate(framework_sorted):
            mid = pred.get('target_gene_masked')
            if mid:
                framework_ranks[mid] = i + 1
                
        baseline_ranks = {}
        for i, pred in enumerate(baseline_predictions):
            mid = pred.get('target_gene_masked')
            if mid:
                baseline_ranks[mid] = pred.get('rank', i + 1)

        # Compare at different K values
        results = {}
        for k in [3, 5, 10, 20]:
            framework_top_k = framework_sorted[:k]
            baseline_top_k = baseline_predictions[:k]

            framework_hits = count_hits(framework_top_k, ground_truth, gene_mapping)
            baseline_hits = count_hits(baseline_top_k, ground_truth, gene_mapping)

            results[f"@{k}"] = {
                "framework_hits": len(framework_hits),
                "framework_hit_genes": framework_hits,
                "baseline_hits": len(baseline_hits),
                "baseline_hit_genes": baseline_hits,
                "framework_advantage": len(framework_hits) - len(baseline_hits)
            }

        # Overall summary
        framework_all_hits = count_hits(framework_predictions, ground_truth, gene_mapping)
        baseline_all_hits = count_hits(baseline_predictions, ground_truth, gene_mapping)

        results["summary"] = {
            "framework_total_hits": len(framework_all_hits),
            "baseline_total_hits": len(baseline_all_hits),
            "framework_predictions_evaluated": len(framework_predictions),
            "baseline_predictions_evaluated": len(baseline_predictions),
            "framework_outperforms": len(framework_all_hits) > len(baseline_all_hits),
            "hit_improvement": len(framework_all_hits) - len(baseline_all_hits)
        }
        
        # ---------------------------------------------------------------------
        # DETAILED RANK COMPARISON (New Feature)
        # ---------------------------------------------------------------------
        rank_comparisons = []
        better_rank_count = 0
        worse_rank_count = 0
        
        # Find all validated genes found by EITHER system
        all_found_hits = set(framework_all_hits) | set(baseline_all_hits)
        
        for real_gene in all_found_hits:
            # Find masked ID
            masked_id = None
            for mid, gene in gene_mapping.items():
                if gene == real_gene:
                    masked_id = mid
                    break
            
            if not masked_id:
                continue
                
            f_rank = framework_ranks.get(masked_id, 999) # 999 = not found/ranked low
            b_rank = baseline_ranks.get(masked_id, 999)
            
            # Determine Tier
            tier = "Unknown"
            if tier_genes:
                for t_name, t_set in tier_genes.items():
                    if t_name != 'all' and real_gene in t_set:
                        tier = t_name
                        break
            
            rank_diff = b_rank - f_rank # Positive means Framework is better (lower rank)
            
            if f_rank < b_rank:
                better_rank_count += 1
                status = "WIN"
            elif f_rank > b_rank:
                worse_rank_count += 1
                status = "LOSS"
            else:
                status = "TIE"
                
            rank_comparisons.append({
                "gene": real_gene,
                "tier": tier,
                "framework_rank": f_rank if f_rank != 999 else ">" + str(len(framework_sorted)),
                "baseline_rank": b_rank if b_rank != 999 else ">" + str(len(baseline_predictions)),
                "improvement": rank_diff,
                "status": status
            })
            
        results["rank_comparisons"] = sorted(rank_comparisons, key=lambda x: x['improvement'], reverse=True)
        results["framework_wins"] = better_rank_count
        results["framework_losses"] = worse_rank_count
        results["win_rate"] = better_rank_count / len(rank_comparisons) if rank_comparisons else 0

        return results

    # =========================================================================
    # METRIC 3: @K Metrics (Recall, Precision, MRR, MAP)
    # =========================================================================

    def set_ground_truth(self,
                         all_targets: List[str],
                         tier_genes: Dict[str, Set[str]] = None):
        """
        Set ground truth genes for evaluation.

        Args:
            all_targets: List of all known T2D drug target gene symbols
            tier_genes: Optional dict with tier1, tier2, tier3 gene sets
        """
        self.ground_truth_genes = set(g.upper() for g in all_targets)
        if tier_genes:
            self.tier_genes = {
                tier: set(g.upper() for g in genes)
                for tier, genes in tier_genes.items()
            }

        logging.info(f"Set ground truth: {len(self.ground_truth_genes)} total targets")
        if tier_genes:
            for tier, genes in self.tier_genes.items():
                logging.info(f"  {tier}: {len(genes)} genes")

    def compute_at_k_metrics(self,
                              ranked_predictions: List[Dict],
                              gene_mapping: Dict[str, str],
                              k_values: List[int] = None) -> Dict[str, Any]:
        """
        Compute @K metrics: Recall@K, Precision@K, MRR, MAP.

        Args:
            ranked_predictions: Predictions sorted by rank (best first)
            gene_mapping: Dict mapping masked ID to real gene symbol
            k_values: List of K values to evaluate (default: [1, 3, 5, 10, 20])

        Returns:
            Dict with @K metrics
        """
        if not self.ground_truth_genes:
            return {"error": "No ground truth set. Call set_ground_truth first."}

        if k_values is None:
            k_values = [1, 3, 5, 10, 20]

        # Map predictions to real gene names
        real_predictions = []
        for pred in ranked_predictions:
            masked_id = pred.get('target_gene_masked')
            if masked_id:
                real_gene = gene_mapping.get(masked_id)
                if real_gene:
                    real_predictions.append({
                        **pred,
                        "real_gene": real_gene.upper()
                    })

        if not real_predictions:
            return {"error": "No predictions could be mapped to real genes"}

        results = {
            "n_predictions": len(real_predictions),
            "n_ground_truth": len(self.ground_truth_genes),
            "metrics_by_k": {}
        }

        # Compute metrics at each K
        for k in k_values:
            if k > len(real_predictions):
                continue

            top_k = real_predictions[:k]
            top_k_genes = [p["real_gene"] for p in top_k]

            # Hits in top K
            hits = [g for g in top_k_genes if g in self.ground_truth_genes]

            # Precision@K = hits / k
            precision = len(hits) / k

            # Recall@K = hits / total_ground_truth
            recall = len(hits) / len(self.ground_truth_genes) if self.ground_truth_genes else 0

            # F1@K
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            results["metrics_by_k"][f"@{k}"] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "hits": len(hits),
                "hit_genes": hits
            }

        # Compute MRR (Mean Reciprocal Rank)
        first_hit_rank = None
        for i, pred in enumerate(real_predictions, 1):
            if pred["real_gene"] in self.ground_truth_genes:
                first_hit_rank = i
                break

        results["mrr"] = 1.0 / first_hit_rank if first_hit_rank else 0.0
        results["first_hit_rank"] = first_hit_rank

        # Compute MAP (Mean Average Precision)
        map_score = self._compute_map(real_predictions)
        results["map"] = round(map_score, 4)

        # Tier-specific metrics (if available)
        if self.tier_genes:
            results["tier_metrics"] = {}
            for tier_name, tier_set in self.tier_genes.items():
                tier_hits = [p["real_gene"] for p in real_predictions if p["real_gene"] in tier_set]
                results["tier_metrics"][tier_name] = {
                    "hits": len(tier_hits),
                    "total_in_tier": len(tier_set),
                    "recall": round(len(tier_hits) / len(tier_set), 4) if tier_set else 0,
                    "hit_genes": tier_hits[:10]  # Limit for display
                }

        return results

    def _compute_map(self, ranked_predictions: List[Dict]) -> float:
        """
        Compute Mean Average Precision.

        Args:
            ranked_predictions: Predictions with real_gene field, sorted by rank

        Returns:
            MAP score
        """
        if not ranked_predictions or not self.ground_truth_genes:
            return 0.0

        precisions = []
        n_relevant_seen = 0

        for i, pred in enumerate(ranked_predictions, 1):
            if pred["real_gene"] in self.ground_truth_genes:
                n_relevant_seen += 1
                precision_at_i = n_relevant_seen / i
                precisions.append(precision_at_i)

        if not precisions:
            return 0.0

        return sum(precisions) / len(self.ground_truth_genes)

    # =========================================================================
    # COMPREHENSIVE EVALUATION REPORT
    # =========================================================================

    def generate_evaluation_report(self,
                                    correlation_results: Dict,
                                    baseline_comparison: Dict,
                                    at_k_metrics: Dict) -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            correlation_results: Results from compute_score_correlation()
            baseline_comparison: Results from compare_to_baseline()
            at_k_metrics: Results from compute_at_k_metrics()

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("T2D DRUG TARGET PREDICTION - COMPREHENSIVE EVALUATION REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Section 1: OpenTargets Correlation
        lines.append("-" * 70)
        lines.append("1. OPENTARGETS ASSOCIATION SCORE CORRELATION")
        lines.append("-" * 70)

        if "error" in correlation_results:
            lines.append(f"  Error: {correlation_results['error']}")
        else:
            lines.append(f"  Genes evaluated: {correlation_results.get('n_genes_evaluated', 0)}")
            lines.append(f"  Spearman correlation: {correlation_results.get('spearman_correlation', 'N/A')}")
            lines.append(f"  Spearman p-value: {correlation_results.get('spearman_pvalue', 'N/A'):.4e}")
            lines.append(f"  Pearson correlation: {correlation_results.get('pearson_correlation', 'N/A')}")
            lines.append(f"  Kendall tau: {correlation_results.get('kendall_tau', 'N/A')}")
            lines.append("")
            lines.append("  Interpretation:")
            spearman = correlation_results.get('spearman_correlation', 0)
            if spearman > 0.5:
                lines.append("    Strong positive correlation - framework ranking aligns well with OpenTargets evidence")
            elif spearman > 0.3:
                lines.append("    Moderate correlation - framework shows reasonable alignment with established evidence")
            elif spearman > 0:
                lines.append("    Weak correlation - framework ranking differs from OpenTargets evidence")
            else:
                lines.append("    No/negative correlation - framework may be identifying novel targets")

        lines.append("")

        # Section 2: Baseline Comparison
        lines.append("-" * 70)
        lines.append("2. BASELINE LLM COMPARISON")
        lines.append("-" * 70)

        if "error" in baseline_comparison:
            lines.append(f"  Error: {baseline_comparison['error']}")
        else:
            summary = baseline_comparison.get("summary", {})
            lines.append(f"  Framework total hits: {summary.get('framework_total_hits', 0)}")
            lines.append(f"  Baseline total hits: {summary.get('baseline_total_hits', 0)}")
            lines.append(f"  Hit improvement: {summary.get('hit_improvement', 0)}")
            lines.append(f"  Framework outperforms: {summary.get('framework_outperforms', False)}")
            
            if "rank_comparisons" in baseline_comparison:
                lines.append("")
                lines.append("  Detailed Rank Comparison (Validated Targets):")
                lines.append(f"    {'Gene':<10} {'Tier':<10} {'FwRank':<8} {'BaseRank':<8} {'Diff':<6} {'Status'}")
                lines.append("    " + "-" * 60)
                for rc in baseline_comparison["rank_comparisons"]:
                    lines.append(f"    {rc['gene']:<10} {rc['tier']:<10} {str(rc['framework_rank']):<8} {str(rc['baseline_rank']):<8} {rc['improvement']:<+6d} {rc['status']}")
                
                lines.append("")
                lines.append(f"    Win Rate: {baseline_comparison.get('win_rate', 0):.1%} ({baseline_comparison.get('framework_wins', 0)} wins, {baseline_comparison.get('framework_losses', 0)} losses)")

            lines.append("")
            lines.append("  Performance at different K:")
            for k_key in ["@3", "@5", "@10"]:
                if k_key in baseline_comparison:
                    k_data = baseline_comparison[k_key]
                    lines.append(f"    {k_key}: Framework={k_data.get('framework_hits', 0)} hits, "
                               f"Baseline={k_data.get('baseline_hits', 0)} hits, "
                               f"Advantage={k_data.get('framework_advantage', 0)}")

        lines.append("")

        # Section 3: @K Metrics
        lines.append("-" * 70)
        lines.append("3. @K METRICS (RECALL, PRECISION, MRR, MAP)")
        lines.append("-" * 70)

        if "error" in at_k_metrics:
            lines.append(f"  Error: {at_k_metrics['error']}")
        else:
            lines.append(f"  Predictions evaluated: {at_k_metrics.get('n_predictions', 0)}")
            lines.append(f"  Ground truth size: {at_k_metrics.get('n_ground_truth', 0)}")
            lines.append(f"  MRR (Mean Reciprocal Rank): {at_k_metrics.get('mrr', 0):.4f}")
            lines.append(f"  MAP (Mean Average Precision): {at_k_metrics.get('map', 0):.4f}")
            lines.append(f"  First hit at rank: {at_k_metrics.get('first_hit_rank', 'None')}")
            lines.append("")
            lines.append("  Metrics by K:")
            for k_key, k_data in at_k_metrics.get("metrics_by_k", {}).items():
                lines.append(f"    {k_key}: P={k_data.get('precision', 0):.3f}, "
                           f"R={k_data.get('recall', 0):.3f}, "
                           f"F1={k_data.get('f1', 0):.3f}, "
                           f"Hits={k_data.get('hits', 0)}")

            # Tier metrics
            if "tier_metrics" in at_k_metrics:
                lines.append("")
                lines.append("  Performance by Target Tier:")
                for tier, tier_data in at_k_metrics["tier_metrics"].items():
                    lines.append(f"    {tier.upper()}: {tier_data.get('hits', 0)}/{tier_data.get('total_in_tier', 0)} "
                               f"(Recall={tier_data.get('recall', 0):.3f})")
                    if tier_data.get('hit_genes'):
                        lines.append(f"      Identified: {', '.join(tier_data['hit_genes'][:5])}")

        lines.append("")
        lines.append("=" * 70)
        lines.append("END OF EVALUATION REPORT")
        lines.append("=" * 70)

        return "\n".join(lines)

    def run_full_evaluation(self,
                            framework_predictions: List[Dict],
                            analysis_context: str,
                            literature_context: str,
                            gene_mapping: Dict[str, str],
                            ground_truth_genes: List[str],
                            tier_genes: Dict[str, Set[str]] = None,
                            run_baseline: bool = True) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.

        Args:
            framework_predictions: Predictions from GA framework (sorted by fitness)
            analysis_context: Analysis context string (for baseline comparison)
            literature_context: Literature context string (for baseline comparison)
            gene_mapping: Dict mapping masked ID to real gene symbol
            ground_truth_genes: List of known T2D drug target gene symbols
            tier_genes: Optional tier classification of ground truth
            run_baseline: Whether to run baseline LLM comparison

        Returns:
            Complete evaluation results dict
        """
        logging.info("Running full T2D evaluation pipeline...")

        results = {
            "n_predictions": len(framework_predictions),
            "n_ground_truth": len(ground_truth_genes)
        }

        # Set ground truth
        self.set_ground_truth(ground_truth_genes, tier_genes)

        # Get real gene names for OpenTargets lookup
        real_genes = [gene_mapping.get(p.get('target_gene_masked', ''))
                      for p in framework_predictions
                      if gene_mapping.get(p.get('target_gene_masked', ''))]

        # 1. OpenTargets correlation
        logging.info("Computing OpenTargets correlation...")
        if real_genes:
            self.load_opentargets_association_scores(real_genes)
        correlation_results = self.compute_score_correlation(framework_predictions, gene_mapping)
        results["opentargets_correlation"] = correlation_results

        # 2. Baseline comparison
        baseline_comparison = {}
        if run_baseline:
            logging.info("Running baseline LLM comparison...")
            baseline_predictions = self.run_baseline_llm(
                analysis_context, literature_context, n_predictions=5
            )
            baseline_comparison = self.compare_to_baseline(
                framework_predictions, baseline_predictions,
                self.ground_truth_genes, gene_mapping, tier_genes
            )
            results["baseline_predictions"] = baseline_predictions
        results["baseline_comparison"] = baseline_comparison

        # 3. @K metrics
        logging.info("Computing @K metrics...")
        # Sort predictions by fitness score (best first)
        sorted_predictions = sorted(
            framework_predictions,
            key=lambda x: x.get('fitness_score', 0),
            reverse=True
        )
        at_k_metrics = self.compute_at_k_metrics(sorted_predictions, gene_mapping)
        results["at_k_metrics"] = at_k_metrics

        # 4. Generate report
        report = self.generate_evaluation_report(
            correlation_results, baseline_comparison, at_k_metrics
        )
        results["evaluation_report"] = report

        logging.info("Full evaluation complete")

        return results
