"""
Leakage Tester

Tests whether the druggability features allow an LLM to identify gene identity.
If leakage is detected (>5% identification accuracy), the pipeline should
fall back to Option C (expression-only mode).

Key Tests:
1. Single feature uniqueness - Does any single feature uniquely identify genes?
2. Combination uniqueness - Do feature combinations uniquely identify genes?
3. LLM leakage test - Can an LLM guess gene identity from features?
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter
import json

logger = logging.getLogger(__name__)


class LeakageTester:
    """
    Tests for information leakage in druggability features.

    The goal is to ensure that the features provided to the LLM
    do not allow it to identify which real gene corresponds to
    which masked ID.
    """

    def __init__(self, llm_client=None):
        """
        Initialize the leakage tester.

        Args:
            llm_client: Optional LLM client for LLM-based leakage testing
        """
        self.llm_client = llm_client
        self.leakage_threshold = 0.05  # 5% - if LLM can identify >5%, it's leaking

    def test_single_feature_uniqueness(
        self,
        feature_df: pd.DataFrame,
        feature_columns: List[str]
    ) -> Dict[str, float]:
        """
        Test if any single feature uniquely identifies genes.

        A feature is considered safe if its values are shared by many genes.

        Args:
            feature_df: DataFrame with features (must have 'masked_id' column)
            feature_columns: List of feature column names to test

        Returns:
            Dictionary mapping feature names to uniqueness scores
            (lower is better, 0 = perfect, 1 = all unique)
        """
        results = {}
        n_genes = len(feature_df)

        for col in feature_columns:
            if col not in feature_df.columns:
                logger.warning(f"Feature column {col} not found in DataFrame")
                continue

            # Count unique values
            value_counts = feature_df[col].value_counts()
            n_unique = len(value_counts)

            # Calculate uniqueness score
            # If all genes have unique values, score = 1 (bad)
            # If all genes share the same value, score = 0 (good but useless)
            # We want something in between - values shared by multiple genes
            uniqueness_score = n_unique / n_genes

            # Also check if any value appears only once (potentially identifying)
            singleton_count = sum(1 for count in value_counts.values if count == 1)
            singleton_ratio = singleton_count / n_genes

            results[col] = {
                'uniqueness_score': uniqueness_score,
                'n_unique_values': n_unique,
                'singleton_ratio': singleton_ratio,
                'is_safe': uniqueness_score < 0.5 and singleton_ratio < 0.1
            }

            logger.info(f"Feature '{col}': uniqueness={uniqueness_score:.3f}, "
                       f"singletons={singleton_ratio:.3f}, safe={results[col]['is_safe']}")

        return results

    def test_combination_uniqueness(
        self,
        feature_df: pd.DataFrame,
        feature_columns: List[str],
        max_combination_size: int = 3
    ) -> Dict[str, float]:
        """
        Test if combinations of features uniquely identify genes.

        Even if individual features are safe, their combinations might
        uniquely identify genes.

        Args:
            feature_df: DataFrame with features
            feature_columns: List of feature columns to test
            max_combination_size: Maximum number of features to combine

        Returns:
            Dictionary with combination test results
        """
        from itertools import combinations

        results = {}
        n_genes = len(feature_df)

        # Filter to available columns
        available_cols = [col for col in feature_columns if col in feature_df.columns]

        for size in range(2, min(max_combination_size + 1, len(available_cols) + 1)):
            for combo in combinations(available_cols, size):
                combo_name = " + ".join(combo)

                # Create combined feature
                combined = feature_df[list(combo)].apply(
                    lambda row: tuple(row.values), axis=1
                )

                # Count unique combinations
                value_counts = combined.value_counts()
                n_unique = len(value_counts)
                uniqueness_score = n_unique / n_genes

                # Check for singletons
                singleton_count = sum(1 for count in value_counts.values if count == 1)
                singleton_ratio = singleton_count / n_genes

                results[combo_name] = {
                    'uniqueness_score': uniqueness_score,
                    'n_unique_combinations': n_unique,
                    'singleton_ratio': singleton_ratio,
                    'is_safe': uniqueness_score < 0.8 and singleton_ratio < 0.2
                }

                if not results[combo_name]['is_safe']:
                    logger.warning(f"Combination '{combo_name}' may leak identity: "
                                 f"uniqueness={uniqueness_score:.3f}, "
                                 f"singletons={singleton_ratio:.3f}")

        return results

    def test_llm_leakage(
        self,
        feature_df: pd.DataFrame,
        real_gene_names: List[str],
        gene_mapping: Dict[str, str],
        n_samples: int = 20
    ) -> Dict[str, any]:
        """
        Test if an LLM can identify gene identity from features.

        This is the most important test - it directly measures whether
        the LLM can "cheat" by recognizing genes from their features.

        Args:
            feature_df: DataFrame with masked features
            real_gene_names: List of real gene names (ground truth)
            gene_mapping: Mapping from real names to masked IDs
            n_samples: Number of genes to test

        Returns:
            Dictionary with LLM leakage test results
        """
        if self.llm_client is None:
            logger.warning("No LLM client provided, skipping LLM leakage test")
            return {
                'skipped': True,
                'reason': 'No LLM client provided'
            }

        # Sample genes to test
        sample_size = min(n_samples, len(real_gene_names))
        sample_indices = np.random.choice(len(real_gene_names), sample_size, replace=False)

        correct_guesses = 0
        test_results = []

        for idx in sample_indices:
            real_name = real_gene_names[idx]
            masked_id = gene_mapping.get(real_name)

            if masked_id is None:
                continue

            # Get features for this gene
            gene_features = feature_df[feature_df['masked_id'] == masked_id]
            if len(gene_features) == 0:
                continue

            features = gene_features.iloc[0].to_dict()
            del features['masked_id']

            # Ask LLM to guess the gene
            prompt = self._create_leakage_test_prompt(features, real_gene_names[:100])

            try:
                response = self.llm_client.generate(prompt, max_tokens=100)
                guessed_gene = self._parse_gene_guess(response)

                is_correct = guessed_gene.upper() == real_name.upper()
                if is_correct:
                    correct_guesses += 1

                test_results.append({
                    'real_gene': real_name,
                    'guessed_gene': guessed_gene,
                    'correct': is_correct,
                    'features': features
                })

            except Exception as e:
                logger.error(f"Error in LLM leakage test: {e}")
                continue

        # Calculate accuracy
        accuracy = correct_guesses / len(test_results) if test_results else 0
        is_safe = accuracy < self.leakage_threshold

        result = {
            'skipped': False,
            'accuracy': accuracy,
            'correct_guesses': correct_guesses,
            'total_tests': len(test_results),
            'is_safe': is_safe,
            'threshold': self.leakage_threshold,
            'details': test_results
        }

        if is_safe:
            logger.info(f"LLM leakage test PASSED: {accuracy:.1%} accuracy "
                       f"(threshold: {self.leakage_threshold:.1%})")
        else:
            logger.warning(f"LLM leakage test FAILED: {accuracy:.1%} accuracy "
                          f"(threshold: {self.leakage_threshold:.1%})")

        return result

    def _create_leakage_test_prompt(
        self,
        features: Dict[str, str],
        candidate_genes: List[str]
    ) -> str:
        """Create a prompt to test if LLM can identify a gene from its features."""

        feature_str = "\n".join([f"- {k}: {v}" for k, v in features.items()])
        gene_list = ", ".join(candidate_genes[:50])

        prompt = f"""I have a gene with the following characteristics:

{feature_str}

Based on these features alone, can you identify which gene this might be?

Candidate genes: {gene_list}

Please respond with ONLY the gene symbol you think this is, nothing else.
If you cannot determine the gene, respond with "UNKNOWN".
"""
        return prompt

    def _parse_gene_guess(self, response: str) -> str:
        """Parse the LLM's gene guess from its response."""
        # Clean up response
        guess = response.strip().upper()

        # Remove common prefixes/suffixes
        for prefix in ["THE GENE IS", "I THINK IT'S", "MY GUESS IS", "ANSWER:"]:
            if guess.startswith(prefix):
                guess = guess[len(prefix):].strip()

        # Take first word if multiple
        guess = guess.split()[0] if guess.split() else "UNKNOWN"

        # Remove punctuation
        guess = ''.join(c for c in guess if c.isalnum())

        return guess

    def run_all_tests(
        self,
        feature_df: pd.DataFrame,
        feature_columns: List[str],
        real_gene_names: Optional[List[str]] = None,
        gene_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, any]:
        """
        Run all leakage tests and return comprehensive results.

        Args:
            feature_df: DataFrame with features
            feature_columns: List of feature columns to test
            real_gene_names: Optional list of real gene names for LLM test
            gene_mapping: Optional mapping for LLM test

        Returns:
            Dictionary with all test results and overall safety assessment
        """
        results = {
            'single_feature_tests': {},
            'combination_tests': {},
            'llm_test': {},
            'overall_safe': True,
            'recommendations': []
        }

        # Test 1: Single feature uniqueness
        logger.info("Running single feature uniqueness tests...")
        results['single_feature_tests'] = self.test_single_feature_uniqueness(
            feature_df, feature_columns
        )

        # Check if any single feature is unsafe
        for feature, test_result in results['single_feature_tests'].items():
            if not test_result['is_safe']:
                results['overall_safe'] = False
                results['recommendations'].append(
                    f"Feature '{feature}' may leak identity. Consider removing or binning."
                )

        # Test 2: Combination uniqueness
        logger.info("Running combination uniqueness tests...")
        results['combination_tests'] = self.test_combination_uniqueness(
            feature_df, feature_columns
        )

        # Check if any combination is unsafe
        for combo, test_result in results['combination_tests'].items():
            if not test_result['is_safe']:
                results['overall_safe'] = False
                results['recommendations'].append(
                    f"Feature combination '{combo}' may leak identity."
                )

        # Test 3: LLM leakage (if data provided)
        if real_gene_names and gene_mapping and self.llm_client:
            logger.info("Running LLM leakage test...")
            results['llm_test'] = self.test_llm_leakage(
                feature_df, real_gene_names, gene_mapping
            )

            if not results['llm_test'].get('skipped', True):
                if not results['llm_test']['is_safe']:
                    results['overall_safe'] = False
                    results['recommendations'].append(
                        f"LLM can identify genes with {results['llm_test']['accuracy']:.1%} accuracy. "
                        "Consider using Option C (expression-only mode)."
                    )
        else:
            results['llm_test'] = {'skipped': True, 'reason': 'Missing data or LLM client'}

        # Summary
        if results['overall_safe']:
            logger.info("All leakage tests PASSED - safe to use Option A")
        else:
            logger.warning("Leakage tests FAILED - recommend Option C (expression-only)")

        return results

    def generate_report(self, test_results: Dict) -> str:
        """
        Generate a human-readable report of leakage test results.

        Args:
            test_results: Results from run_all_tests()

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("LEAKAGE TEST REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Overall status
        status = "PASSED ✓" if test_results['overall_safe'] else "FAILED ✗"
        lines.append(f"Overall Status: {status}")
        lines.append("")

        # Single feature tests
        lines.append("-" * 40)
        lines.append("Single Feature Tests:")
        lines.append("-" * 40)
        for feature, result in test_results['single_feature_tests'].items():
            status = "✓" if result['is_safe'] else "✗"
            lines.append(f"  {feature}: {status}")
            lines.append(f"    - Uniqueness: {result['uniqueness_score']:.3f}")
            lines.append(f"    - Singletons: {result['singleton_ratio']:.3f}")
        lines.append("")

        # Combination tests
        lines.append("-" * 40)
        lines.append("Combination Tests:")
        lines.append("-" * 40)
        unsafe_combos = [k for k, v in test_results['combination_tests'].items()
                        if not v['is_safe']]
        if unsafe_combos:
            for combo in unsafe_combos:
                result = test_results['combination_tests'][combo]
                lines.append(f"  {combo}: ✗")
                lines.append(f"    - Uniqueness: {result['uniqueness_score']:.3f}")
        else:
            lines.append("  All combinations safe ✓")
        lines.append("")

        # LLM test
        lines.append("-" * 40)
        lines.append("LLM Leakage Test:")
        lines.append("-" * 40)
        llm_result = test_results['llm_test']
        if llm_result.get('skipped'):
            lines.append(f"  Skipped: {llm_result.get('reason', 'Unknown')}")
        else:
            status = "✓" if llm_result['is_safe'] else "✗"
            lines.append(f"  Status: {status}")
            lines.append(f"  Accuracy: {llm_result['accuracy']:.1%}")
            lines.append(f"  Threshold: {llm_result['threshold']:.1%}")
        lines.append("")

        # Recommendations
        if test_results['recommendations']:
            lines.append("-" * 40)
            lines.append("Recommendations:")
            lines.append("-" * 40)
            for rec in test_results['recommendations']:
                lines.append(f"  • {rec}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


def main():
    """Test the leakage tester with sample data."""
    logging.basicConfig(level=logging.INFO)

    # Create sample feature data using IDG standard families
    sample_data = {
        'masked_id': [f'G{i:05d}' for i in range(1, 21)],
        'idg_family': ['Kinase'] * 5 + ['GPCR'] * 5 + ['Enzyme'] * 5 + ['TF'] * 5,
        'subcellular_location': ['Membrane'] * 8 + ['Cytoplasm'] * 6 + ['Nucleus'] * 6,
    }

    feature_df = pd.DataFrame(sample_data)
    feature_columns = ['idg_family', 'subcellular_location']

    # Run tests
    tester = LeakageTester()
    results = tester.run_all_tests(feature_df, feature_columns)

    # Print report
    print(tester.generate_report(results))


if __name__ == "__main__":
    main()
