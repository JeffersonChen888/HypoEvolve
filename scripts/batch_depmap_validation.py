#!/usr/bin/env python3
"""
Batch DepMap Validation Script for Multiple Cancer Type CSV Files

This script processes multiple CSV files (one per cancer type) from depmap_trajectories,
filters drug-cancer pairs with scores >= 4, and performs DepMap validation analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging
import sys
import os
from typing import List, Dict, Tuple
import json

# Import from local scripts directory
from depmap_validation import DepMapValidator
from tcga_cancer_types import TCGA_CANCER_TYPES, get_cancer_full_name

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchDepMapValidator:
    """Batch processor for DepMap validation across multiple cancer types."""
    
    def __init__(self, depmap_data_dir="depmap_data", min_score=4):
        """Initialize batch validator."""
        self.depmap_data_dir = Path(depmap_data_dir)
        self.min_score = min_score
        self.validator = DepMapValidator(depmap_data_dir)
        self.results_summary = []
        
    def find_csv_files(self, input_dir: str) -> List[Path]:
        """Find all CSV files in the input directory."""
        input_path = Path(input_dir)
        csv_files = list(input_path.glob("**/*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files in {input_dir}")
        
        for file in csv_files:
            logger.info(f"  - {file}")
            
        return csv_files
    
    def load_and_filter_csv(self, csv_file: Path) -> pd.DataFrame:
        """Load CSV and filter for high-scoring drug-cancer pairs."""
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(df)} rows from {csv_file.name}")
            
            # Check required columns
            required_cols = ['drug_name', 'rating', 'cancer_type']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns {missing_cols} in {csv_file.name}")
                return pd.DataFrame()
            
            # Filter for high scores and non-null ratings
            initial_count = len(df)
            df = df.dropna(subset=['rating'])
            df = df[df['rating'] >= self.min_score]
            
            logger.info(f"Filtered to {len(df)} high-scoring pairs (>= {self.min_score}) from {initial_count} total")
            
            # Add source file information
            df['source_file'] = csv_file.name
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")
            return pd.DataFrame()
    
    def process_all_files(self, input_dir: str, output_dir: str) -> Dict:
        """Process all CSV files and perform DepMap validation."""
        csv_files = self.find_csv_files(input_dir)
        
        if not csv_files:
            logger.error(f"No CSV files found in {input_dir}")
            return {}
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        cancer_type_summary = {}
        
        for csv_file in csv_files:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {csv_file.name}")
            logger.info(f"{'='*60}")
            
            # Load and filter data
            df = self.load_and_filter_csv(csv_file)
            
            if df.empty:
                logger.warning(f"No valid high-scoring pairs found in {csv_file.name}")
                continue
            
            # Extract cancer type from filename or data
            cancer_type = self.extract_cancer_type(csv_file, df)
            
            # Validate with DepMap
            try:
                input_pairs = []
                for _, row in df.iterrows():
                    # Use filename-derived cancer type instead of malformed CSV cancer_type field
                    input_pairs.append((
                        row['drug_name'],
                        cancer_type,  # Use TCGA-mapped cancer type from filename
                        row['rating']
                    ))
                
                logger.info(f"Validating {len(input_pairs)} drug-cancer pairs for {cancer_type}")
                validation_results, candidates = self.validator.validate_drug_cancer_pairs(
                    input_pairs, min_ai_score=self.min_score
                )
                
                # Convert tuples to dictionaries and add metadata
                result_dicts = []
                for result_tuple in validation_results:
                    drug, cancer, ai_score, depmap_score = result_tuple
                    result_dict = {
                        'drug': drug,
                        'cancer_type': cancer,
                        'ai_score': ai_score,
                        'depmap_score': depmap_score,
                        'source_file': csv_file.name,
                        'cancer_type_inferred': cancer_type
                    }
                    result_dicts.append(result_dict)
                
                all_results.extend(result_dicts)
                
                # Summary statistics for this cancer type
                valid_results = [r for r in result_dicts if r['depmap_score'] is not None]
                cancer_type_summary[cancer_type] = {
                    'source_file': csv_file.name,
                    'total_pairs': len(input_pairs),
                    'validated_pairs': len(valid_results),
                    'avg_depmap_score': np.mean([r['depmap_score'] for r in valid_results]) if valid_results else None,
                    'min_depmap_score': np.min([r['depmap_score'] for r in valid_results]) if valid_results else None,
                    'max_depmap_score': np.max([r['depmap_score'] for r in valid_results]) if valid_results else None
                }
                
                logger.info(f"Completed validation for {cancer_type}: {len(valid_results)}/{len(input_pairs)} pairs validated")
                
            except Exception as e:
                logger.error(f"Error during DepMap validation for {csv_file.name}: {e}")
                continue
        
        # Save consolidated results
        self.save_results(all_results, cancer_type_summary, output_path)
        
        # Generate plots
        self.generate_plots(all_results, cancer_type_summary, output_path)
        
        return {
            'total_results': len(all_results),
            'cancer_types': len(cancer_type_summary),
            'summary': cancer_type_summary
        }
    
    def extract_cancer_type(self, csv_file: Path, df: pd.DataFrame) -> str:
        """Extract cancer type from filename using TCGA mapping."""
        # Extract filename without extension
        filename = csv_file.stem.upper()
        
        # Try direct TCGA abbreviation match first
        if filename in TCGA_CANCER_TYPES:
            full_name = TCGA_CANCER_TYPES[filename]
            logger.info(f"Mapped filename '{csv_file.name}' -> {filename} -> {full_name}")
            return full_name
        
        # Try partial matching for common abbreviations
        for tcga_abbrev, full_name in TCGA_CANCER_TYPES.items():
            if tcga_abbrev in filename:
                logger.info(f"Partial match filename '{csv_file.name}' -> {tcga_abbrev} -> {full_name}")
                return full_name
        
        # Fallback to filename if no TCGA mapping found
        fallback_name = filename.replace('_', ' ').title()
        logger.warning(f"No TCGA mapping found for filename '{csv_file.name}', using fallback: {fallback_name}")
        return fallback_name
    
    def save_results(self, all_results: List[Dict], cancer_summary: Dict, output_path: Path):
        """Save consolidated results to files."""
        # Save detailed results
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_file = output_path / "consolidated_depmap_validation.csv"
            results_df.to_csv(results_file, index=False)
            logger.info(f"Saved detailed results to {results_file}")
        
        # Save summary statistics
        summary_df = pd.DataFrame.from_dict(cancer_summary, orient='index')
        summary_file = output_path / "cancer_type_summary.csv"
        summary_df.to_csv(summary_file, index=True)
        logger.info(f"Saved summary statistics to {summary_file}")
        
        # Save JSON summary
        json_file = output_path / "validation_summary.json"
        with open(json_file, 'w') as f:
            json.dump(cancer_summary, f, indent=2, default=str)
        logger.info(f"Saved JSON summary to {json_file}")
    
    def generate_plots(self, all_results: List[Dict], cancer_summary: Dict, output_path: Path):
        """Generate single box plot for all drug-cancer pairs."""
        if not all_results:
            logger.warning("No results to plot")
            return
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(all_results)
        valid_results = df[df['depmap_score'].notna()]
        
        if valid_results.empty:
            logger.warning("No valid DepMap scores to plot")
            return
        
        # Prepare data for box plot - group by AI score
        validation_data = []
        for _, row in valid_results.iterrows():
            validation_data.append([
                row['drug'], 
                row['cancer_type'], 
                row['ai_score'], 
                row['depmap_score']
            ])
        
        self.create_validation_plot(validation_data, output_path)
        
    def create_validation_plot(self, validation_data, output_path):
        """Create box plot matching the paper's style."""
        from scipy import stats
        
        # Convert to DataFrame
        df = pd.DataFrame(validation_data, 
                         columns=['drug', 'cancer', 'ai_score', 'depmap_score'])
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get unique AI scores and prepare data for each score
        unique_scores = sorted(df['ai_score'].unique(), reverse=True)  # [5, 4, 3, 2, 1]
        box_data = []
        positions = []
        labels = []
        
        for i, score in enumerate(unique_scores):
            score_data = df[df['ai_score'] == score]['depmap_score'].values
            if len(score_data) > 0:
                box_data.append(score_data)
                positions.append(i + 1)
                labels.append(str(int(score)))
        
        # Box plot with exact styling from paper
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
            
            # Add sample size annotations
            for i, (pos, score) in enumerate(zip(positions, unique_scores)):
                score_data = df[df['ai_score'] == score]['depmap_score']
                if len(score_data) > 0:
                    ax.text(pos, 1.05, f'n={len(score_data)}', ha='center', va='bottom', fontsize=10)
        
        # Formatting to match paper
        ax.set_xlabel('AI co-scientist score', fontsize=12)
        ax.set_ylabel('DepMap Score', fontsize=12)
        ax.set_xlim(0.5, len(positions) + 0.5)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        
        plt.tight_layout()
        plt.savefig(output_path / 'depmap_validation_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated box plot: {output_path / 'depmap_validation_boxplot.png'}")

def main():
    """Main function for batch DepMap validation."""
    parser = argparse.ArgumentParser(
        description="Batch DepMap validation for multiple cancer type CSV files",
        epilog="""
Example usage:
  # Process all CSV files in depmap_trajectories with default settings
  python batch_depmap_validation.py depmap_trajectories --output-dir depmap_batch_results
  
  # Custom minimum score and DepMap data location
  python batch_depmap_validation.py depmap_trajectories \\
      --output-dir results \\
      --depmap-dir ../depmap_data \\
      --min-score 4 \\
      --verbose
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input_dir", 
                       help="Directory containing CSV files (one per cancer type)")
    parser.add_argument("--output-dir", default="depmap_batch_results",
                       help="Output directory for results and plots")
    parser.add_argument("--depmap-dir", default="depmap_data",
                       help="Directory containing DepMap data files")
    parser.add_argument("--min-score", type=int, default=4,
                       help="Minimum AI score for drug-cancer pairs to include")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directory
    if not Path(args.input_dir).exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1
    
    # Initialize batch validator
    try:
        validator = BatchDepMapValidator(args.depmap_dir, args.min_score)
    except Exception as e:
        logger.error(f"Failed to initialize DepMap validator: {e}")
        return 1
    
    # Process all files
    logger.info(f"Starting batch DepMap validation...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Minimum score threshold: {args.min_score}")
    
    try:
        results = validator.process_all_files(args.input_dir, args.output_dir)
        
        logger.info(f"\n{'='*60}")
        logger.info("BATCH VALIDATION COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(f"Total validated drug-cancer pairs: {results['total_results']}")
        logger.info(f"Cancer types processed: {results['cancer_types']}")
        logger.info(f"Results saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Batch validation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())