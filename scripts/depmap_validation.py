#!/usr/bin/env python3
"""
DepMap Validation Script for Drug Repurposing Pipeline Results

This script validates drug-cancer pairs from the AI co-scientist pipeline
against DepMap gene dependency data, following the methodology from the
original AI co-scientist paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
import json
import re
import logging
import argparse
from pathlib import Path
import sys
import os

# Import from local scripts directory
from tcga_cancer_types import get_cancer_abbreviation, CANCER_TYPE_ALIASES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DepMapValidator:
    """Validates drug-cancer pairs against DepMap dependency data."""
    
    def __init__(self, depmap_data_dir="depmap_data"):
        """Initialize with DepMap data directory."""
        self.depmap_data_dir = Path(depmap_data_dir)
        self.dependency_df = None
        self.model_df = None
        self.load_depmap_data()
    
    def load_depmap_data(self):
        """Load DepMap dependency and model data."""
        logger.info("Loading DepMap data...")
        
        # Load CRISPR gene dependency data
        crispr_file = self.depmap_data_dir / "CRISPRGeneDependency.csv"
        if not crispr_file.exists():
            raise FileNotFoundError(f"CRISPR dependency file not found: {crispr_file}")
        
        logger.info("Loading CRISPR gene dependency data...")
        self.dependency_df = pd.read_csv(crispr_file, index_col=0)
        logger.info(f"Loaded dependency data: {self.dependency_df.shape}")
        
        # Load model metadata
        model_file = self.depmap_data_dir / "Model.csv" 
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        logger.info("Loading model metadata...")
        self.model_df = pd.read_csv(model_file)
        logger.info(f"Loaded model data: {self.model_df.shape}")
    
    def get_open_targets_genes(self, drug_name):
        """
        Get target genes for a drug from the approved_drugs_depmap.json file.
        This file contains ChEMBL-sourced drug-target mappings for all 62 approved drugs.

        Args:
            drug_name (str): Name of the drug

        Returns:
            list: List of gene symbols
        """
        # Load drug targets from the curated JSON file (ChEMBL-sourced)
        if not hasattr(self, '_drug_targets_cache'):
            self._drug_targets_cache = {}
            # Try multiple possible locations for the JSON file
            possible_paths = [
                Path(__file__).parent.parent / "pipeline" / "data" / "approved_drugs_depmap.json",
                Path(__file__).parent / "approved_drugs_depmap.json",
                Path("pipeline/data/approved_drugs_depmap.json"),
            ]

            for json_path in possible_paths:
                if json_path.exists():
                    try:
                        with open(json_path) as f:
                            data = json.load(f)
                        for drug_info in data.get('details', []):
                            name = drug_info.get('name', '').upper()
                            targets = drug_info.get('depmap_targets', [])
                            # Remove duplicates while preserving order
                            unique_targets = list(dict.fromkeys(targets))
                            self._drug_targets_cache[name] = unique_targets
                        logger.info(f"Loaded {len(self._drug_targets_cache)} drug-target mappings from {json_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load drug targets from {json_path}: {e}")

            if not self._drug_targets_cache:
                logger.error("Could not load drug-target mappings from approved_drugs_depmap.json")

        # Normalize drug name for lookup
        drug_key = drug_name.upper().strip()

        if drug_key in self._drug_targets_cache:
            targets = self._drug_targets_cache[drug_key]
            logger.info(f"Found {len(targets)} targets for {drug_name}: {targets}")
            return targets
        else:
            logger.warning(f"No targets found for {drug_name}")
            return []
    
    def get_cancer_cell_lines(self, cancer_type):
        """
        Get ACH-IDs for a specific cancer type.

        Args:
            cancer_type (str): Cancer type name or abbreviation

        Returns:
            list: List of ModelIDs (ACH-IDs)
        """
        # Strip abbreviations in parentheses that LLM sometimes adds
        # e.g., "Acute myeloid leukemia (AML)" → "Acute myeloid leukemia"
        # Handle both regular hyphen (-) and en-dash (‑)
        cancer_type_clean = re.sub(r'\s*\([A-Z0-9‑\-]+\)\s*$', '', cancer_type).strip()
        if cancer_type_clean != cancer_type:
            logger.debug(f"Stripped abbreviation: '{cancer_type}' → '{cancer_type_clean}'")
            cancer_type = cancer_type_clean

        # Normalize en-dash (‑) to hyphen (-) for consistent matching
        cancer_type = cancer_type.replace('‑', '-')

        # Normalize alternative cancer type names that LLM sometimes uses
        alternative_names = {
            'Malignant pleural mesothelioma': 'Mesothelioma',
            'Pancreatic ductal adenocarcinoma': 'Pancreatic adenocarcinoma',
            'Soft tissue sarcoma': 'Sarcoma',
        }
        if cancer_type in alternative_names:
            cancer_type = alternative_names[cancer_type]

        # Try to map cancer type to TCGA abbreviation
        tcga_abbrev = get_cancer_abbreviation(cancer_type)
        
        # DepMap uses different naming conventions than TCGA
        # Maps TCGA cancer type names to DepMap OncotreePrimaryDisease/OncotreeSubtype names
        depmap_mapping = {
            # Original mappings
            'Breast invasive carcinoma': ['Invasive Breast Carcinoma', 'Breast Neoplasm'],
            'Lung adenocarcinoma': ['Non-Small Cell Lung Cancer', 'Lung Neuroendocrine Tumor'],
            'Colon adenocarcinoma': ['Colorectal Adenocarcinoma', 'Colon Adenocarcinoma'],
            'Acute Myeloid Leukemia': ['Acute Myeloid Leukemia'],
            'Pancreatic adenocarcinoma': ['Pancreatic Adenocarcinoma', 'Pancreatic Ductal Adenocarcinoma'],
            'Ovarian serous cystadenocarcinoma': ['Ovarian Serous Cystadenocarcinoma', 'Ovarian Epithelial Tumor'],
            # Added mappings for missing TCGA cancer types
            'Brain Lower Grade Glioma': ['Diffuse Glioma', 'Astrocytoma', 'Oligodendroglioma', 'Anaplastic Astrocytoma'],
            'Cervical squamous cell carcinoma and endocervical adenocarcinoma': ['Cervical Squamous Cell Carcinoma', 'Cervical Adenocarcinoma', 'Endocervical Adenocarcinoma', 'Mixed Cervical Carcinoma'],
            'Bladder Urothelial Carcinoma': ['Bladder Urothelial Carcinoma'],
            'Adrenocortical carcinoma': ['Adrenocortical Carcinoma'],
            'Cholangiocarcinoma': ['Cholangiocarcinoma', 'Intrahepatic Cholangiocarcinoma', 'Extrahepatic Cholangiocarcinoma'],
            # Additional common TCGA types
            'Glioblastoma multiforme': ['Glioblastoma', 'Diffuse Glioma', 'Gliosarcoma'],
            'Liver hepatocellular carcinoma': ['Hepatocellular Carcinoma'],
            'Lung squamous cell carcinoma': ['Lung Squamous Cell Carcinoma', 'Non-Small Cell Lung Cancer'],
            'Skin Cutaneous Melanoma': ['Cutaneous Melanoma', 'Melanoma'],
            'Stomach adenocarcinoma': ['Stomach Adenocarcinoma', 'Gastric Adenocarcinoma'],
            'Prostate adenocarcinoma': ['Prostate Adenocarcinoma'],
            'Kidney renal clear cell carcinoma': ['Clear Cell Renal Cell Carcinoma', 'Renal Cell Carcinoma'],
            'Head and Neck squamous cell carcinoma': ['Head and Neck Squamous Cell Carcinoma'],
            'Uterine Corpus Endometrial Carcinoma': ['Endometrial Carcinoma', 'Uterine Endometrioid Carcinoma'],
            'Thyroid carcinoma': ['Thyroid Cancer', 'Papillary Thyroid Cancer', 'Anaplastic Thyroid Cancer'],
            'Rectum adenocarcinoma': ['Colorectal Adenocarcinoma', 'Rectal Adenocarcinoma'],
            'Esophageal carcinoma': ['Esophageal Adenocarcinoma', 'Esophageal Squamous Cell Carcinoma'],
            'Sarcoma': ['Soft Tissue Sarcoma', 'Sarcoma'],
            'Mesothelioma': ['Mesothelioma', 'Pleural Mesothelioma'],
            'Kidney renal papillary cell carcinoma': ['Papillary Renal Cell Carcinoma', 'Renal Cell Carcinoma'],
            'Lymphoid Neoplasm Diffuse Large B-cell Lymphoma': ['Diffuse Large B-Cell Lymphoma, NOS', 'Mature B-Cell Neoplasms'],
            'Chronic Myelogenous Leukemia': ['Chronic Myeloid Leukemia, BCR-ABL1+', 'Myeloproliferative Neoplasms'],
            'Testicular Germ Cell Tumors': ['Non-Seminomatous Germ Cell Tumor', 'Germ Cell Tumor', 'Mixed Germ Cell Tumor'],
            # Note: Kidney Chromophobe has no cell lines in DepMap Q2 2024
        }
        
        # Get DepMap-specific search terms first
        if cancer_type in depmap_mapping:
            search_patterns = depmap_mapping[cancer_type]
        else:
            # Fallback to original logic
            search_patterns = [
                cancer_type,
                tcga_abbrev if tcga_abbrev != "UNKNOWN" else None
            ]
            
            # Add aliases from our mapping
            if cancer_type.lower() in CANCER_TYPE_ALIASES:
                alias = CANCER_TYPE_ALIASES[cancer_type.lower()]
                if isinstance(alias, list):
                    search_patterns.extend(alias)
                else:
                    search_patterns.append(alias)
        
        # Filter search patterns to remove None values
        search_patterns = [p for p in search_patterns if p is not None]
        
        cancer_ach_ids = []
        
        for pattern in search_patterns:
            # Search in both OncotreePrimaryDisease AND OncotreeSubtype columns
            # Escape regex special characters to treat pattern as literal string
            escaped_pattern = re.escape(pattern)
            
            # Search primary disease column
            mask_primary = self.model_df['OncotreePrimaryDisease'].str.contains(
                escaped_pattern, case=False, na=False, regex=True
            )
            
            # Search subtype column
            mask_subtype = self.model_df['OncotreeSubtype'].str.contains(
                escaped_pattern, case=False, na=False, regex=True
            )
            
            # Combine both searches
            combined_mask = mask_primary | mask_subtype
            matching_ids = self.model_df[combined_mask]['ModelID'].tolist()
            cancer_ach_ids.extend(matching_ids)
        
        # Remove duplicates while preserving order
        cancer_ach_ids = list(dict.fromkeys(cancer_ach_ids))
        
        logger.info(f"Found {len(cancer_ach_ids)} cell lines for {cancer_type}")
        return cancer_ach_ids
    
    def find_gene_column(self, gene_symbol):
        """
        Find the column name for a gene in the dependency dataframe.
        DepMap format: "GENE_SYMBOL (EntrezID)"
        
        Args:
            gene_symbol (str): Gene symbol to search for
            
        Returns:
            str or None: Column name if found, None otherwise
        """
        # Look for columns that start with the gene symbol followed by space and parentheses
        pattern = f"^{re.escape(gene_symbol)} \\("
        matching_cols = [col for col in self.dependency_df.columns 
                        if re.match(pattern, col)]
        
        if matching_cols:
            return matching_cols[0]  # Return first match
        
        return None
    
    def validate_drug_cancer_pairs(self, input_pairs, min_ai_score=4):
        """
        Validate drug-cancer pairs against DepMap data.
        
        Args:
            input_pairs (list): List of (drug, cancer, ai_score) tuples
            min_ai_score (int): Minimum AI score to consider
            
        Returns:
            list: List of (drug, cancer, ai_score, max_depmap_score) tuples
        """
        validation_data = []
        candidates = []
        
        logger.info(f"Validating {len(input_pairs)} drug-cancer pairs...")
        
        for i, (drug, cancer, ai_score) in enumerate(input_pairs):
            logger.info(f"Processing {i+1}/{len(input_pairs)}: {drug} vs {cancer} (AI score: {ai_score})")
            
            # Filter by AI score
            if ai_score < min_ai_score:
                logger.debug(f"Skipping {drug}-{cancer}: AI score {ai_score} < {min_ai_score}")
                continue
            
            # Get target genes for drug
            target_genes = self.get_open_targets_genes(drug)
            if not target_genes:
                logger.warning(f"No target genes found for {drug}")
                continue
            
            # Get ACH-IDs for cancer type
            cancer_ach_ids = self.get_cancer_cell_lines(cancer)
            if not cancer_ach_ids:
                logger.warning(f"No cell lines found for cancer type: {cancer}")
                continue
            
            # Calculate DepMap scores for each target gene
            gene_scores = []
            for gene in target_genes:
                gene_col = self.find_gene_column(gene)
                if gene_col:
                    # Get scores across all cell lines for this cancer
                    try:
                        # Filter for existing cell lines in the data
                        existing_cell_lines = [ach_id for ach_id in cancer_ach_ids 
                                             if ach_id in self.dependency_df.index]
                        
                        if existing_cell_lines:
                            # Step 1: Get all scores for this target gene across cell lines
                            target_scores = []
                            for cell_line in existing_cell_lines:
                                score = self.dependency_df.loc[cell_line, gene_col]
                                if pd.notna(score):  # Only add non-NaN scores
                                    target_scores.append(score)
                            
                            # Step 2: Calculate median score for this gene (representative for this gene)
                            if len(target_scores) > 0:
                                median_score = np.median(target_scores)
                                gene_scores.append(median_score)  # Keep original sign
                                logger.debug(f"Gene {gene}: median DepMap score = {median_score:.3f} (from {len(target_scores)} cell lines)")
                    
                    except Exception as e:
                        logger.error(f"Error processing {gene} for {cancer}: {e}")
                        continue
                else:
                    logger.debug(f"Gene {gene} not found in dependency data")
            
            # Step 3: Aggregate across genes (take max = most negative = best target gene)
            if gene_scores:
                max_depmap_score = max(gene_scores)
                validation_data.append((drug, cancer, ai_score, max_depmap_score))
                logger.info(f"Added validation point: {drug}-{cancer}, AI={ai_score}, DepMap={max_depmap_score:.3f}")
                
                # Optional: Filter for candidates (more negative = higher dependency)
                if max_depmap_score <= -0.5:  # High dependency threshold
                    candidates.append((drug, cancer, max_depmap_score))
                    logger.info(f"High-priority candidate: {drug} for {cancer} (DepMap={max_depmap_score:.3f})")
        
        logger.info(f"Validation complete: {len(validation_data)} validated pairs, {len(candidates)} high-priority candidates")
        return validation_data, candidates
    
    def create_validation_plot(self, validation_data, output_file="depmap_validation.png"):
        """
        Create validation plot matching the AI co-scientist paper style.
        
        Args:
            validation_data (list): List of (drug, cancer, ai_score, depmap_score) tuples
            output_file (str): Output file name for the plot
        """
        if not validation_data:
            logger.warning("No validation data to plot")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(validation_data, 
                         columns=['drug', 'cancer', 'ai_score', 'depmap_score'])
        
        logger.info(f"Creating validation plot with {len(df)} data points...")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get unique AI scores and prepare data for box plot
        unique_scores = sorted(df['ai_score'].unique(), reverse=True)
        box_data = []
        positions = []
        labels = []
        
        for i, score in enumerate(unique_scores):
            score_data = df[df['ai_score']==score]['depmap_score'].values
            if len(score_data) > 0:
                box_data.append(score_data)
                positions.append(i + 1)
                labels.append(str(int(score)))
        
        # Create box plot with exact styling from paper
        if box_data:
            box_plot = ax.boxplot(box_data, 
                                 positions=positions,
                                 widths=0.6,
                                 patch_artist=True,
                                 medianprops=dict(color='orange', linewidth=2),
                                 boxprops=dict(facecolor='white', color='black'),
                                 whiskerprops=dict(color='black'),
                                 capprops=dict(color='black'),
                                 flierprops=dict(marker='o', markerfacecolor='black', 
                                               markersize=4, markeredgecolor='black'))
            
            # Add statistical significance annotations (simplified)
            for i, (pos, score) in enumerate(zip(positions, unique_scores)):
                score_data = df[df['ai_score']==score]['depmap_score']
                if len(score_data) > 0:
                    # Simple annotation - could add actual statistical test
                    ax.text(pos, max(score_data) + 0.05, f'n={len(score_data)}', 
                           ha='center', va='bottom', fontsize=10)
        
        # Formatting to match paper
        ax.set_xlabel('AI co-scientist score', fontsize=12)
        ax.set_ylabel('DepMap Score', fontsize=12)
        ax.set_xlim(0.5, len(positions) + 0.5)
        ax.set_ylim(0, max(df['depmap_score']) * 1.2)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        
        plt.title('Drug-Cancer Pair Validation: AI Score vs DepMap Dependency', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Validation plot saved to {output_file}")
        plt.show()
        
        # Print summary statistics
        self.print_validation_summary(df)
    
    def print_validation_summary(self, df):
        """Print summary statistics for validation results."""
        logger.info("\n=== Validation Summary ===")
        logger.info(f"Total validated pairs: {len(df)}")
        logger.info(f"AI score range: {df['ai_score'].min()} - {df['ai_score'].max()}")
        logger.info(f"DepMap score range: {df['depmap_score'].min():.3f} - {df['depmap_score'].max():.3f}")
        logger.info(f"Mean DepMap score: {df['depmap_score'].mean():.3f}")
        logger.info(f"Median DepMap score: {df['depmap_score'].median():.3f}")
        
        # Correlation analysis
        if len(df) > 1:
            corr_coef, p_value = stats.pearsonr(df['ai_score'], df['depmap_score'])
            logger.info(f"Correlation (AI vs DepMap): r={corr_coef:.3f}, p={p_value:.3f}")

def parse_pipeline_results(results_file):
    """
    Parse drug repurposing pipeline results from CSV file created by extract_drug_results().
    Expected format from pipeline2/utils/result_extractor.py:
    - drug_name: Name of the proposed drug
    - rating: Rating/score (1-5) from reflection agent
    - cancer_type: Extracted cancer type 
    - hypothesis_id: ID of hypothesis (optional)
    - hypothesis_title: Title of hypothesis (optional)
    
    Args:
        results_file (str): Path to CSV results file
        
    Returns:
        list: List of (drug, cancer, ai_score) tuples
    """
    logger.info(f"Parsing pipeline results from {results_file}")
    input_pairs = []
    
    try:
        if not results_file.endswith('.csv'):
            logger.error(f"Expected CSV file from pipeline result extractor, got: {results_file}")
            return []
            
        df = pd.read_csv(results_file)
        logger.info(f"Loaded CSV with columns: {list(df.columns)}")
        
        # Check for expected columns from result_extractor.py
        required_columns = ['drug_name', 'rating', 'cancer_type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns in CSV: {missing_columns}")
            logger.info(f"Available columns: {list(df.columns)}")
            return []
        
        # Parse each row
        for idx, row in df.iterrows():
            drug_name = row['drug_name']
            cancer_type = row['cancer_type'] 
            rating = row['rating']
            
            # Skip rows with missing essential data
            if pd.isna(drug_name) or not drug_name.strip():
                logger.warning(f"Row {idx}: Missing drug name, skipping")
                continue
                
            if pd.isna(cancer_type) or not cancer_type.strip():
                logger.warning(f"Row {idx}: Missing cancer type for {drug_name}, skipping")
                continue
            
            # Handle missing ratings - could be None/NaN from result extractor
            if pd.isna(rating):
                logger.warning(f"Row {idx}: Missing rating for {drug_name}-{cancer_type}, using default score 3")
                rating = 3  # Default middle score
            else:
                try:
                    rating = float(rating)
                except (ValueError, TypeError):
                    logger.warning(f"Row {idx}: Invalid rating '{rating}' for {drug_name}-{cancer_type}, using default score 3")
                    rating = 3
            
            input_pairs.append((drug_name.strip(), cancer_type.strip(), rating))
            logger.debug(f"Parsed: {drug_name} -> {cancer_type} (score: {rating})")
            
    except Exception as e:
        logger.error(f"Error parsing pipeline results file: {e}")
        return []
    
    logger.info(f"Successfully parsed {len(input_pairs)} drug-cancer pairs")
    return input_pairs

def main():
    parser = argparse.ArgumentParser(
        description="Validate drug repurposing results against DepMap data",
        epilog="""
Example usage:
  # First, run the pipeline with drug extraction:
  cd pipeline2 && python main.py "drug repurposing for acute myeloid leukemia" --extract-drugs --output-csv results.csv
  
  # Then validate against DepMap:
  python scripts/depmap_validation.py results.csv
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results_file", 
                       help="Path to CSV file from pipeline --extract-drugs (drug_name, rating, cancer_type columns)")
    parser.add_argument("--depmap-dir", default="depmap_data", 
                       help="Directory containing DepMap data files")
    parser.add_argument("--min-score", type=int, default=4,
                       help="Minimum AI score to include in validation")
    parser.add_argument("--output", default="depmap_validation.png",
                       help="Output file for validation plot")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if results file exists
    if not Path(args.results_file).exists():
        logger.error(f"Results file not found: {args.results_file}")
        return 1
    
    # Parse pipeline results
    input_pairs = parse_pipeline_results(args.results_file)
    if not input_pairs:
        logger.error("No valid drug-cancer pairs found in results file")
        return 1
    
    # Initialize validator
    try:
        validator = DepMapValidator(args.depmap_dir)
    except Exception as e:
        logger.error(f"Error initializing DepMap validator: {e}")
        return 1
    
    # Perform validation
    validation_data, candidates = validator.validate_drug_cancer_pairs(
        input_pairs, min_ai_score=args.min_score
    )
    
    if validation_data:
        # Create validation plot
        validator.create_validation_plot(validation_data, args.output)
        
        # Save results
        results_df = pd.DataFrame(validation_data, 
                                 columns=['drug', 'cancer', 'ai_score', 'depmap_score'])
        results_file = args.output.replace('.png', '_results.csv')
        results_df.to_csv(results_file, index=False)
        logger.info(f"Validation results saved to {results_file}")
        
        if candidates:
            candidates_df = pd.DataFrame(candidates, 
                                       columns=['drug', 'cancer', 'depmap_score'])
            candidates_file = args.output.replace('.png', '_candidates.csv')
            candidates_df.to_csv(candidates_file, index=False)
            logger.info(f"High-priority candidates saved to {candidates_file}")
    
    else:
        logger.warning("No drug-cancer pairs could be validated")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())