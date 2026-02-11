#!/usr/bin/env python3
"""
Validate JSON results from drug repurposing pipeline against DepMap.

This script:
1. Reads pipeline_results_*.json files
2. Extracts drug-cancer pairs from hypotheses
3. Validates against DepMap CRISPR dependency data
4. Reports validation results

Usage:
    # Validate single run
    python scripts/validate_json_results.py pipeline/output/drug_repurposing/run_20260101_150014

    # Validate batch run
    python scripts/validate_json_results.py pipeline/output/drug_repurposing/batch_20260101
"""

import json
import sys
from pathlib import Path
import pandas as pd
import logging

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from depmap_validation import DepMapValidator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class JSONResultsValidator:
    """Validates JSON results from drug repurposing pipeline."""

    def __init__(self, depmap_data_dir="data/depmap_data", approved_drugs_file="pipeline/data/approved_drugs_depmap.json"):
        """Initialize validator."""
        self.depmap_validator = DepMapValidator(depmap_data_dir)
        self.approved_drugs_file = Path(approved_drugs_file)
        self.drug_targets = self._load_drug_targets()

    def _load_drug_targets(self):
        """Load drug targets from approved_drugs_depmap.json."""
        if not self.approved_drugs_file.exists():
            logger.warning(f"Approved drugs file not found: {self.approved_drugs_file}")
            return {}

        with open(self.approved_drugs_file) as f:
            data = json.load(f)

        # Build mapping: drug_name -> target_genes
        drug_targets = {}
        for drug in data.get('details', []):
            name = drug['name'].upper()
            targets = drug.get('depmap_targets', [])
            drug_targets[name] = targets

        logger.info(f"Loaded {len(drug_targets)} drugs with target information")
        return drug_targets

    def find_json_files(self, input_path):
        """Find all pipeline_results_*.json or vanilla_results_*.json files in the input path."""
        path = Path(input_path)

        if path.is_file() and path.suffix == '.json':
            if path.name.startswith('pipeline_results') or path.name.startswith('vanilla_results'):
                return [path]

        # Search recursively for JSON files (both pipeline and vanilla results)
        json_files = list(path.glob('**/pipeline_results_*.json'))
        json_files.extend(path.glob('**/vanilla_results_*.json'))
        logger.info(f"Found {len(json_files)} JSON result files in {input_path}")
        return json_files

    def extract_drug_cancer_pairs(self, json_file):
        """Extract drug-cancer pairs from a JSON results file."""
        with open(json_file) as f:
            hypotheses = json.load(f)

        if not isinstance(hypotheses, list):
            # Support both pipeline results (final_population) and vanilla results (hypotheses)
            hypotheses = hypotheses.get('final_population', hypotheses.get('hypotheses', []))

        pairs = []
        for hyp in hypotheses:
            drug = hyp.get('final_drug')
            cancer = hyp.get('cancer_type')
            fitness = hyp.get('fitness_score', 0)
            hyp_id = hyp.get('id', 'unknown')

            if drug and cancer:
                pairs.append({
                    'drug': drug.upper().strip(),
                    'cancer': cancer,
                    'fitness_score': fitness,
                    'hypothesis_id': hyp_id,
                    'title': hyp.get('title', 'N/A')[:80]
                })

        logger.info(f"Extracted {len(pairs)} drug-cancer pairs from {json_file.name}")
        return pairs

    def validate_pair(self, drug, cancer):
        """Validate a single drug-cancer pair against DepMap."""
        # Get target genes
        targets = self.drug_targets.get(drug)
        if not targets:
            logger.warning(f"No targets found for drug: {drug}")
            return None

        # Get cancer cell lines
        cancer_cell_lines = self.depmap_validator.get_cancer_cell_lines(cancer)
        if not cancer_cell_lines:
            logger.warning(f"No cell lines found for cancer: {cancer}")
            return None

        # Calculate dependency scores for each target
        gene_scores = []
        for gene in targets:
            gene_col = self.depmap_validator.find_gene_column(gene)
            if gene_col:
                try:
                    existing_lines = [ach for ach in cancer_cell_lines
                                    if ach in self.depmap_validator.dependency_df.index]

                    if existing_lines:
                        scores = []
                        for cell_line in existing_lines:
                            score = self.depmap_validator.dependency_df.loc[cell_line, gene_col]
                            if pd.notna(score):
                                scores.append(score)

                        if scores:
                            import numpy as np
                            median_score = np.median(scores)
                            gene_scores.append({
                                'gene': gene,
                                'median_score': median_score,
                                'num_cell_lines': len(scores)
                            })
                except Exception as e:
                    logger.debug(f"Error processing {gene}: {e}")
            else:
                logger.debug(f"Gene {gene} not found in DepMap data")

        if not gene_scores:
            return None

        # Find best target (highest dependency score)
        best_target = max(gene_scores, key=lambda x: x['median_score'])

        return {
            'targets': targets,
            'best_target': best_target['gene'],
            'depmap_score': best_target['median_score'],
            'num_cell_lines': best_target['num_cell_lines'],
            'all_gene_scores': gene_scores
        }

    def validate_results(self, input_path):
        """Validate all JSON results in the input path."""
        json_files = self.find_json_files(input_path)

        if not json_files:
            logger.error(f"No JSON result files found in {input_path}")
            return

        all_results = []

        for json_file in json_files:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing: {json_file}")
            logger.info(f"{'='*80}")

            pairs = self.extract_drug_cancer_pairs(json_file)

            for pair in pairs:
                drug = pair['drug']
                cancer = pair['cancer']

                logger.info(f"\n--- Validating: {drug} for {cancer} ---")
                logger.info(f"    Hypothesis: {pair['title']}")
                fitness = pair['fitness_score']
                logger.info(f"    Fitness Score: {fitness:.2f}" if fitness is not None else "    Fitness Score: N/A (vanilla baseline)")

                validation = self.validate_pair(drug, cancer)

                if validation:
                    score = validation['depmap_score']

                    # Interpret score
                    if score >= 0.9:
                        status = "✅ EXCELLENT"
                    elif score >= 0.5:
                        status = "✅ GOOD"
                    elif score >= 0.3:
                        status = "⚠️ MODERATE"
                    else:
                        status = "❌ WEAK"

                    logger.info(f"    Target: {validation['best_target']}")
                    logger.info(f"    DepMap Score: {score:.4f} ({validation['num_cell_lines']} cell lines)")
                    logger.info(f"    Validation: {status}")

                    all_results.append({
                        'file': json_file.name,
                        'drug': drug,
                        'cancer': cancer,
                        'fitness_score': pair['fitness_score'],
                        'target': validation['best_target'],
                        'depmap_score': score,
                        'num_cell_lines': validation['num_cell_lines'],
                        'status': status
                    })
                else:
                    logger.warning(f"    ⚠️ Could not validate (missing data)")

        # Summary
        if all_results:
            logger.info(f"\n{'='*80}")
            logger.info(f"VALIDATION SUMMARY")
            logger.info(f"{'='*80}")
            logger.info(f"Total validated: {len(all_results)}")

            df = pd.DataFrame(all_results)

            # Count by status
            excellent = len(df[df['depmap_score'] >= 0.9])
            good = len(df[(df['depmap_score'] >= 0.5) & (df['depmap_score'] < 0.9)])
            moderate = len(df[(df['depmap_score'] >= 0.3) & (df['depmap_score'] < 0.5)])
            weak = len(df[df['depmap_score'] < 0.3])

            logger.info(f"  ✅ Excellent (≥0.9): {excellent}")
            logger.info(f"  ✅ Good (0.5-0.9): {good}")
            logger.info(f"  ⚠️ Moderate (0.3-0.5): {moderate}")
            logger.info(f"  ❌ Weak (<0.3): {weak}")

            # Save to CSV
            output_file = Path(input_path) / "depmap_validation_results.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"\nResults saved to: {output_file}")

            return df
        else:
            logger.warning("No validation results generated")
            return None


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate JSON results against DepMap")
    parser.add_argument('input_path', help='Path to JSON file or directory containing JSON files')
    parser.add_argument('--depmap-data', default='data/depmap_data', help='DepMap data directory')
    parser.add_argument('--approved-drugs', default='pipeline/data/approved_drugs_depmap.json',
                       help='Approved drugs JSON file')

    args = parser.parse_args()

    validator = JSONResultsValidator(args.depmap_data, args.approved_drugs)
    validator.validate_results(args.input_path)


if __name__ == '__main__':
    main()
