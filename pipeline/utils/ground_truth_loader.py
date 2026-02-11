"""
Ground Truth Loader

Loads T2D drug targets from OpenTargets API dynamically.
NO HARDCODED GENE LISTS - everything comes from the API.

Tier System:
- Tier 1: Genes with APPROVED drugs for T2D
- Tier 2: Genes with GWAS associations for T2D
- Tier 3: Genes in clinical trials for T2D
"""

import requests
import pandas as pd
import logging
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class DrugTarget:
    """Represents a drug target with its evidence."""
    gene_symbol: str
    ensembl_id: str
    tier: int
    evidence_type: str
    evidence_score: float
    drug_names: Optional[List[str]] = None  # Only for evaluation, not shown to LLM


class GroundTruthLoader:
    """
    Loads T2D drug targets from OpenTargets API.

    This class fetches REAL drug targets from OpenTargets, which will be
    used to evaluate the pipeline's predictions. The gene names are
    NEVER shown to the LLM during hypothesis generation.
    """

    # T2D disease ID in OpenTargets (MONDO ontology)
    T2D_DISEASE_ID = "MONDO_0005148"

    # Alternative disease IDs for broader coverage
    RELATED_DISEASE_IDS = [
        "EFO_0001360",   # Type 2 diabetes mellitus (EFO)
        "MONDO_0005015", # Diabetes mellitus
    ]

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the ground truth loader.

        Args:
            cache_dir: Optional directory to cache API responses
        """
        self.graphql_endpoint = "https://api.platform.opentargets.org/api/v4/graphql"
        self.cache_dir = cache_dir

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.2  # 200ms between requests

        # Caches
        self._approved_drugs_cache: Optional[Dict] = None
        self._gwas_cache: Optional[Dict] = None
        self._clinical_trials_cache: Optional[Dict] = None

    def _rate_limit(self):
        """Enforce rate limiting between API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _execute_query(self, query: str, variables: Dict) -> Dict:
        """Execute a GraphQL query against OpenTargets API."""
        self._rate_limit()

        try:
            response = requests.post(
                self.graphql_endpoint,
                json={"query": query, "variables": variables},
                headers={"Content-Type": "application/json"},
                timeout=60
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed: {response.status_code}")
                return {}

        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {}

    def load_tier1_approved_drugs(self, disease_id: str = None) -> List[DrugTarget]:
        """
        Load Tier 1: Genes with APPROVED drugs for T2D.

        These are the highest confidence targets - drugs have been
        approved specifically for treating T2D.

        Args:
            disease_id: Disease ID to query (defaults to T2D)

        Returns:
            List of DrugTarget objects for Tier 1
        """
        if disease_id is None:
            disease_id = self.T2D_DISEASE_ID

        logger.info(f"Loading Tier 1 targets (approved drugs) for {disease_id}...")

        # Use simplified knownDrugs query without optional null parameters
        query = """
        query knownDrugsQuery($efoId: String!, $size: Int!) {
            disease(efoId: $efoId) {
                id
                name
                knownDrugs(size: $size) {
                    count
                    rows {
                        drug {
                            id
                            name
                            maximumClinicalTrialPhase
                            isApproved
                        }
                        target {
                            id
                            approvedSymbol
                        }
                        phase
                        status
                    }
                }
            }
        }
        """

        variables = {"efoId": disease_id, "size": 500}
        result = self._execute_query(query, variables)

        targets = []
        seen_genes = set()

        disease_data = result.get("data", {}).get("disease", {})
        if not disease_data:
            logger.warning(f"No data found for disease {disease_id}")
            return targets

        rows = disease_data.get("knownDrugs", {}).get("rows", [])

        for row in rows:
            drug = row.get("drug", {})
            target = row.get("target", {})

            # Only include APPROVED drugs
            if not drug.get("isApproved", False):
                continue

            gene_symbol = target.get("approvedSymbol")
            ensembl_id = target.get("id")

            if not gene_symbol or gene_symbol in seen_genes:
                continue

            seen_genes.add(gene_symbol)

            targets.append(DrugTarget(
                gene_symbol=gene_symbol,
                ensembl_id=ensembl_id,
                tier=1,
                evidence_type="approved_drug",
                evidence_score=1.0,
                drug_names=[drug.get("name")]
            ))

        logger.info(f"Found {len(targets)} Tier 1 targets with approved drugs")
        return targets

    def load_tier2_gwas(self, disease_id: str = None) -> List[DrugTarget]:
        """
        Load Tier 2: Genes with GWAS associations for T2D.

        These are genes with genetic evidence linking them to T2D risk.

        Args:
            disease_id: Disease ID to query (defaults to T2D)

        Returns:
            List of DrugTarget objects for Tier 2
        """
        if disease_id is None:
            disease_id = self.T2D_DISEASE_ID

        logger.info(f"Loading Tier 2 targets (GWAS) for {disease_id}...")

        # Use associatedTargets with aggregationFilters for genetic evidence
        query = """
        query diseaseAssociationsQuery($efoId: String!, $index: Int!, $size: Int!) {
            disease(efoId: $efoId) {
                id
                name
                associatedTargets(
                    page: {index: $index, size: $size}
                    orderByScore: "genetic_association"
                ) {
                    count
                    rows {
                        target {
                            id
                            approvedSymbol
                        }
                        score
                        datatypeScores {
                            id
                            score
                        }
                    }
                }
            }
        }
        """

        variables = {"efoId": disease_id, "size": 500, "index": 0}
        result = self._execute_query(query, variables)

        targets = []
        seen_genes = set()

        disease_data = result.get("data", {}).get("disease", {})
        if not disease_data:
            logger.warning(f"No data found for disease {disease_id}")
            return targets

        rows = disease_data.get("associatedTargets", {}).get("rows", [])

        for row in rows:
            target = row.get("target", {})
            score = row.get("score", 0)

            gene_symbol = target.get("approvedSymbol")
            ensembl_id = target.get("id")

            if not gene_symbol or gene_symbol in seen_genes:
                continue

            # Filter for minimum score (genetic evidence)
            if score < 0.1:
                continue

            seen_genes.add(gene_symbol)

            targets.append(DrugTarget(
                gene_symbol=gene_symbol,
                ensembl_id=ensembl_id,
                tier=2,
                evidence_type="gwas",
                evidence_score=score
            ))

        logger.info(f"Found {len(targets)} Tier 2 targets with GWAS evidence")
        return targets

    def load_tier3_clinical_trials(self, disease_id: str = None) -> List[DrugTarget]:
        """
        Load Tier 3: Genes in clinical trials for T2D.

        These are genes being targeted by drugs in clinical development.

        Args:
            disease_id: Disease ID to query (defaults to T2D)

        Returns:
            List of DrugTarget objects for Tier 3
        """
        if disease_id is None:
            disease_id = self.T2D_DISEASE_ID

        logger.info(f"Loading Tier 3 targets (clinical trials) for {disease_id}...")

        # Same query as Tier 1, but filter for non-approved drugs in trials
        query = """
        query clinicalTrialsQuery($efoId: String!, $size: Int!) {
            disease(efoId: $efoId) {
                id
                name
                knownDrugs(size: $size) {
                    count
                    rows {
                        drug {
                            id
                            name
                            maximumClinicalTrialPhase
                            isApproved
                        }
                        target {
                            id
                            approvedSymbol
                        }
                        phase
                    }
                }
            }
        }
        """

        variables = {"efoId": disease_id, "size": 500}
        result = self._execute_query(query, variables)

        targets = []
        seen_genes = set()

        disease_data = result.get("data", {}).get("disease", {})
        if not disease_data:
            return targets

        rows = disease_data.get("knownDrugs", {}).get("rows", [])

        for row in rows:
            drug = row.get("drug", {})
            target = row.get("target", {})

            # Only include drugs in trials (not approved)
            if drug.get("isApproved", False):
                continue

            phase = drug.get("maximumClinicalTrialPhase", 0)
            if phase < 1:
                continue

            gene_symbol = target.get("approvedSymbol")
            ensembl_id = target.get("id")

            if not gene_symbol or gene_symbol in seen_genes:
                continue

            seen_genes.add(gene_symbol)

            targets.append(DrugTarget(
                gene_symbol=gene_symbol,
                ensembl_id=ensembl_id,
                tier=3,
                evidence_type=f"clinical_trial_phase_{phase}",
                evidence_score=phase / 4.0,  # Normalize to 0-1
                drug_names=[drug.get("name")]
            ))

        logger.info(f"Found {len(targets)} Tier 3 targets in clinical trials")
        return targets

        return targets

    def load_from_tsv(self, file_path: str) -> Tuple[List[DrugTarget], Dict[str, Set[str]]]:
        """
        Load ground truth from OpenTargets TSV export.
        
        Args:
            file_path: Path to the TSV file
            
        Returns:
            Tuple of (list of all targets, dict of gene sets by tier)
        """
        import os
        if not os.path.exists(file_path):
            logger.warning(f"TSV file not found: {file_path}")
            return [], {}
            
        logger.info(f"Loading ground truth from TSV: {file_path}")
        try:
            df = pd.read_csv(file_path, sep='\t')
        except Exception as e:
            logger.error(f"Failed to read TSV file: {e}")
            return [], {}
        
        all_targets = []
        tier_genes = {
            'tier1': set(),
            'tier2': set(),
            'tier3': set(),
            'all': set()
        }
        
        for _, row in df.iterrows():
            symbol = row.get('symbol')
            if not isinstance(symbol, str):
                continue
                
            global_score = row.get('globalScore', 0.0)
            max_phase = row.get('maxClinicalTrialPhase')
            
            # Determine Tier
            tier = 2 # Default to Tier 2 (Associated)
            evidence_type = "Association Score"
            
            # Check for Tier 1 (Approved) - phase == 1.0 (normalized)
            try:
                phase_val = float(max_phase) if max_phase != 'No data' else 0.0
            except (ValueError, TypeError):
                phase_val = 0.0
                
            if phase_val == 1.0:
                tier = 1
                evidence_type = "Approved Drug"
            # Check for Tier 3 (Clinical Trials) - 0 < phase < 1.0
            elif 0 < phase_val < 1.0:
                tier = 3
                evidence_type = "Clinical Trials"
            
            # Create DrugTarget object
            target = DrugTarget(
                gene_symbol=symbol,
                ensembl_id=f"ENSG_TSV_{symbol}", # Placeholder
                tier=tier,
                evidence_type=evidence_type,
                evidence_score=global_score
            )
            
            all_targets.append(target)
            tier_key = f"tier{tier}"
            tier_genes[tier_key].add(symbol)
            tier_genes['all'].add(symbol)
            
        logger.info(f"Loaded from TSV: {len(tier_genes['tier1'])} Tier 1, "
                   f"{len(tier_genes['tier2'])} Tier 2, {len(tier_genes['tier3'])} Tier 3")
                   
        return all_targets, tier_genes

    def load_all_ground_truth(
        self,
        include_related_diseases: bool = True
    ) -> Tuple[List[DrugTarget], Dict[str, Set[str]]]:
        """
        Load all ground truth targets across all tiers.
        Prioritizes local TSV file if available, otherwise falls back to API.

        Args:
            include_related_diseases: Whether to include related disease IDs (only for API)

        Returns:
            Tuple of (list of all targets, dict of gene sets by tier)
        """
        import os
        
        # Check for TSV file in standard location
        # Assume we are in pipeline/utils/ground_truth_loader.py
        # Data dir is ../../data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pipeline_root = os.path.dirname(os.path.dirname(current_dir))
        tsv_path = os.path.join(pipeline_root, "data", "OT-MONDO_0005148-associated-targets-2026_2_10-v25_12.tsv")
        
        # Try finding it relative to current working directory if pipeline_root fails
        if not os.path.exists(tsv_path):
             tsv_path = "data/OT-MONDO_0005148-associated-targets-2026_2_10-v25_12.tsv"
        
        if os.path.exists(tsv_path):
            return self.load_from_tsv(tsv_path)
            
        logger.info("TSV file not found, falling back to OpenTargets API...")

        all_targets = []
        tier_genes = {
            'tier1': set(),
            'tier2': set(),
            'tier3': set(),
            'all': set()
        }

        disease_ids = [self.T2D_DISEASE_ID]
        # ... rest of API loading logic ...

        for disease_id in disease_ids:
            # Load each tier
            try:
                tier1 = self.load_tier1_approved_drugs(disease_id)
                tier2 = self.load_tier2_gwas(disease_id)
                tier3 = self.load_tier3_clinical_trials(disease_id)

                # Add to collections (avoid duplicates)
                for target in tier1:
                    if target.gene_symbol not in tier_genes['tier1']:
                        all_targets.append(target)
                        tier_genes['tier1'].add(target.gene_symbol)
                        tier_genes['all'].add(target.gene_symbol)

                for target in tier2:
                    if target.gene_symbol not in tier_genes['tier2']:
                        # Don't add if already in tier1
                        if target.gene_symbol not in tier_genes['tier1']:
                            all_targets.append(target)
                        tier_genes['tier2'].add(target.gene_symbol)
                        tier_genes['all'].add(target.gene_symbol)

                for target in tier3:
                    if target.gene_symbol not in tier_genes['tier3']:
                        # Don't add if already in tier1 or tier2
                        if target.gene_symbol not in tier_genes['tier1'] and \
                           target.gene_symbol not in tier_genes['tier2']:
                            all_targets.append(target)
                        tier_genes['tier3'].add(target.gene_symbol)
                        tier_genes['all'].add(target.gene_symbol)

            except Exception as e:
                logger.error(f"Error loading targets for {disease_id}: {e}")
                continue

        logger.info(f"Total ground truth: {len(tier_genes['tier1'])} Tier 1, "
                   f"{len(tier_genes['tier2'])} Tier 2, {len(tier_genes['tier3'])} Tier 3")

        return all_targets, tier_genes

    def create_evaluation_dataframe(
        self,
        all_targets: List[DrugTarget]
    ) -> pd.DataFrame:
        """
        Create a DataFrame for evaluation purposes.

        This DataFrame contains the REAL gene names and is used ONLY
        for evaluation AFTER the pipeline has made its predictions.
        It should NEVER be shown to the LLM.

        Args:
            all_targets: List of DrugTarget objects

        Returns:
            DataFrame with columns: gene_symbol, ensembl_id, tier, evidence_type, score
        """
        data = []
        for target in all_targets:
            data.append({
                'gene_symbol': target.gene_symbol,
                'ensembl_id': target.ensembl_id,
                'tier': target.tier,
                'evidence_type': target.evidence_type,
                'evidence_score': target.evidence_score
            })

        df = pd.DataFrame(data)

        # Sort by tier, then score
        df = df.sort_values(['tier', 'evidence_score'], ascending=[True, False])

        return df

    def evaluate_predictions(
        self,
        predicted_genes: List[str],
        tier_genes: Dict[str, Set[str]],
        gene_mapping_reverse: Dict[str, str]
    ) -> Dict[str, any]:
        """
        Evaluate predicted genes against ground truth.

        Args:
            predicted_genes: List of MASKED gene IDs from pipeline
            tier_genes: Dictionary of gene sets by tier (from load_all_ground_truth)
            gene_mapping_reverse: Mapping from masked IDs back to real names

        Returns:
            Dictionary with evaluation metrics
        """
        # Unmask predicted genes
        unmasked_predictions = []
        for masked_id in predicted_genes:
            real_name = gene_mapping_reverse.get(masked_id)
            if real_name:
                unmasked_predictions.append(real_name)

        predicted_set = set(unmasked_predictions)

        # Calculate metrics for each tier
        results = {
            'predictions': unmasked_predictions,
            'n_predictions': len(predicted_set),
            'tiers': {}
        }

        for tier_name, tier_set in tier_genes.items():
            if tier_name == 'all':
                continue

            overlap = predicted_set & tier_set
            precision = len(overlap) / len(predicted_set) if predicted_set else 0
            recall = len(overlap) / len(tier_set) if tier_set else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results['tiers'][tier_name] = {
                'ground_truth_size': len(tier_set),
                'overlap': list(overlap),
                'overlap_count': len(overlap),
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        # Overall metrics
        all_ground_truth = tier_genes['all']
        overall_overlap = predicted_set & all_ground_truth
        results['overall'] = {
            'ground_truth_size': len(all_ground_truth),
            'overlap': list(overall_overlap),
            'overlap_count': len(overall_overlap),
            'precision': len(overall_overlap) / len(predicted_set) if predicted_set else 0,
            'recall': len(overall_overlap) / len(all_ground_truth) if all_ground_truth else 0
        }

        # Tier 1 specific (most important)
        tier1_overlap = predicted_set & tier_genes['tier1']
        results['tier1_hits'] = list(tier1_overlap)
        results['tier1_hit_rate'] = len(tier1_overlap) / len(tier_genes['tier1']) if tier_genes['tier1'] else 0

        return results

    def generate_evaluation_report(self, eval_results: Dict) -> str:
        """
        Generate a human-readable evaluation report.

        Args:
            eval_results: Results from evaluate_predictions()

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("T2D DRUG TARGET PREDICTION EVALUATION")
        lines.append("=" * 60)
        lines.append("")

        # Summary
        lines.append(f"Total Predictions: {eval_results['n_predictions']}")
        lines.append(f"Ground Truth Size: {eval_results['overall']['ground_truth_size']}")
        lines.append(f"Overall Overlap: {eval_results['overall']['overlap_count']}")
        lines.append("")

        # Tier 1 Results (most important)
        lines.append("-" * 40)
        lines.append("TIER 1 (Approved Drug Targets):")
        lines.append("-" * 40)
        tier1 = eval_results['tiers'].get('tier1', {})
        lines.append(f"  Hits: {tier1.get('overlap_count', 0)} / {tier1.get('ground_truth_size', 0)}")
        lines.append(f"  Precision: {tier1.get('precision', 0):.1%}")
        lines.append(f"  Recall: {tier1.get('recall', 0):.1%}")
        lines.append(f"  F1 Score: {tier1.get('f1', 0):.3f}")
        if eval_results.get('tier1_hits'):
            lines.append(f"  Identified targets: {', '.join(eval_results['tier1_hits'])}")
        lines.append("")

        # Tier 2 Results
        lines.append("-" * 40)
        lines.append("TIER 2 (GWAS Genes):")
        lines.append("-" * 40)
        tier2 = eval_results['tiers'].get('tier2', {})
        lines.append(f"  Hits: {tier2.get('overlap_count', 0)} / {tier2.get('ground_truth_size', 0)}")
        lines.append(f"  Precision: {tier2.get('precision', 0):.1%}")
        lines.append(f"  Recall: {tier2.get('recall', 0):.1%}")
        lines.append("")

        # Tier 3 Results
        lines.append("-" * 40)
        lines.append("TIER 3 (Clinical Trial Targets):")
        lines.append("-" * 40)
        tier3 = eval_results['tiers'].get('tier3', {})
        lines.append(f"  Hits: {tier3.get('overlap_count', 0)} / {tier3.get('ground_truth_size', 0)}")
        lines.append(f"  Precision: {tier3.get('precision', 0):.1%}")
        lines.append(f"  Recall: {tier3.get('recall', 0):.1%}")
        lines.append("")

        # Top predictions
        lines.append("-" * 40)
        lines.append("Top Predictions:")
        lines.append("-" * 40)
        for i, gene in enumerate(eval_results['predictions'][:10], 1):
            in_tier1 = gene in eval_results['tiers'].get('tier1', {}).get('overlap', [])
            in_tier2 = gene in eval_results['tiers'].get('tier2', {}).get('overlap', [])
            in_tier3 = gene in eval_results['tiers'].get('tier3', {}).get('overlap', [])

            status = ""
            if in_tier1:
                status = "★ TIER 1"
            elif in_tier2:
                status = "● TIER 2"
            elif in_tier3:
                status = "○ TIER 3"
            else:
                status = "  Novel"

            lines.append(f"  {i:2d}. {gene:<15} {status}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


def main():
    """Test the ground truth loader."""
    logging.basicConfig(level=logging.INFO)

    loader = GroundTruthLoader()

    print("Loading T2D ground truth from OpenTargets...")
    print("=" * 60)

    # Load all targets
    all_targets, tier_genes = loader.load_all_ground_truth()

    # Create evaluation DataFrame
    eval_df = loader.create_evaluation_dataframe(all_targets)

    print("\nGround Truth Summary:")
    print(f"  Tier 1 (Approved): {len(tier_genes['tier1'])} genes")
    print(f"  Tier 2 (GWAS): {len(tier_genes['tier2'])} genes")
    print(f"  Tier 3 (Clinical): {len(tier_genes['tier3'])} genes")
    print(f"  Total unique: {len(tier_genes['all'])} genes")

    print("\nSample Tier 1 targets:")
    tier1_df = eval_df[eval_df['tier'] == 1].head(10)
    print(tier1_df.to_string(index=False))

    # Test evaluation with dummy predictions
    print("\n" + "=" * 60)
    print("Testing evaluation with dummy predictions...")

    dummy_predictions = ["G00001", "G00002", "G00003"]
    dummy_mapping_reverse = {
        "G00001": list(tier_genes['tier1'])[0] if tier_genes['tier1'] else "UNKNOWN",
        "G00002": "FAKE_GENE",
        "G00003": list(tier_genes['tier2'])[0] if tier_genes['tier2'] else "UNKNOWN"
    }

    eval_results = loader.evaluate_predictions(
        dummy_predictions, tier_genes, dummy_mapping_reverse
    )

    print(loader.generate_evaluation_report(eval_results))


if __name__ == "__main__":
    main()
