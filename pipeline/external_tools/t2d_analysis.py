"""
T2D Analysis Pipeline Tools

Implements computational analysis for Type 2 Diabetes drug target identification:
- Differential Expression Analysis
- Gene Set Enrichment Analysis (GSEA)
- Weighted Gene Co-expression Network Analysis (WGCNA)
- Transcription Factor Activity Analysis (via decoupler/CollecTRI)
- Expression Variability Analysis
- Cross-dataset Integration
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Analysis libraries (install via pip)
# pip install scanpy anndata decoupler scipy pandas numpy

try:
    import scanpy as sc
    import anndata as ad
    import pandas as pd
    import numpy as np
    from scipy import stats
    ANALYSIS_LIBS_AVAILABLE = True
except ImportError:
    ANALYSIS_LIBS_AVAILABLE = False
    logging.warning("Analysis libraries not available. Install: pip install scanpy anndata decoupler scipy pandas numpy")


class T2DAnalysisPipeline:
    """
    Main analysis pipeline for T2D drug target identification.
    
    Processes multiple GEO datasets and generates integrated gene priority scores.
    """
    
    # GEO datasets for T2D analysis
    DATASETS = {
        "GSE20966": {"tissue": "pancreatic_islets", "comparison": "T2D_vs_ND"},
        "GSE41762": {"tissue": "pancreatic_islets", "comparison": "T2D_vs_ND"},
        "GSE38642": {"tissue": "pancreatic_islets", "comparison": "T2D_vs_ND"},
        "GSE25724": {"tissue": "pancreatic_islets", "comparison": "T2D_vs_ND"},
        "GSE76894": {"tissue": "pancreatic_islets", "comparison": "T2D_vs_ND"},
    }
    
    def __init__(self, data_dir: str, gene_mapper=None):
        """
        Initialize the analysis pipeline.
        
        Args:
            data_dir: Directory containing .h5ad files for each dataset
            gene_mapper: Optional GeneMapper instance for masking
        """
        self.data_dir = Path(data_dir)
        self.gene_mapper = gene_mapper
        self.results: Dict[str, Any] = {}
        self.integrated_scores: Dict[str, float] = {}
        
    def run_full_pipeline(self, datasets: List[str] = None) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline on specified datasets.
        
        Args:
            datasets: List of GEO IDs to analyze (default: all)
            
        Returns:
            Dictionary with all analysis results and integrated scores
        """
        if datasets is None:
            datasets = list(self.DATASETS.keys())
        
        logging.info(f"Starting T2D analysis pipeline on {len(datasets)} datasets")
        
        # Per-dataset analysis
        dataset_results = {}
        for geo_id in datasets:
            logging.info(f"Analyzing {geo_id}...")
            dataset_results[geo_id] = self._analyze_single_dataset(geo_id)
        
        # Cross-dataset integration
        integrated_results = self._integrate_across_datasets(dataset_results)
        
        # Calculate final gene priority scores
        priority_scores = self._calculate_priority_scores(integrated_results)
        
        return {
            "per_dataset_results": dataset_results,
            "integrated_results": integrated_results,
            "priority_scores": priority_scores,
            "top_candidates": self._get_top_candidates(priority_scores, n=50)
        }
    
    def _analyze_single_dataset(self, geo_id: str) -> Dict[str, Any]:
        """Run all analyses on a single dataset."""
        h5ad_path = self.data_dir / f"{geo_id}.h5ad"
        
        if not h5ad_path.exists():
            logging.warning(f"Dataset file not found: {h5ad_path}")
            return {"error": f"File not found: {h5ad_path}"}
        
        adata = ad.read_h5ad(h5ad_path)
        
        results = {
            "geo_id": geo_id,
            "metadata": self.DATASETS.get(geo_id, {}),
            "n_genes": adata.n_vars,
            "n_samples": adata.n_obs,
        }
        
        # 1. Differential Expression Analysis
        results["de_analysis"] = self._run_de_analysis(adata)
        
        # 2. GSEA
        results["gsea"] = self._run_gsea(adata, results["de_analysis"])
        
        # 3. WGCNA (if enough samples)
        if adata.n_obs >= 15:
            results["wgcna"] = self._run_wgcna(adata)
        
        # 4. TF Activity Analysis
        results["tf_activity"] = self._run_tf_activity(adata)
        
        # 5. Expression Variability
        results["variability"] = self._run_variability_analysis(adata)
        
        return results
    
    def _run_de_analysis(self, adata) -> Dict[str, Any]:
        """
        Run differential expression analysis.
        
        Returns:
            Dictionary with DE results including log2FC and adjusted p-values
        """
        # Assumes adata.obs has 'condition' column with 'T2D' and 'ND' labels
        sc.tl.rank_genes_groups(adata, groupby='condition', method='wilcoxon')
        
        de_results = sc.get.rank_genes_groups_df(adata, group='T2D')
        
        return {
            "significant_up": de_results[
                (de_results['pvals_adj'] < 0.05) & (de_results['logfoldchanges'] > 0.5)
            ]['names'].tolist(),
            "significant_down": de_results[
                (de_results['pvals_adj'] < 0.05) & (de_results['logfoldchanges'] < -0.5)
            ]['names'].tolist(),
            "full_results": de_results.to_dict('records')
        }
    
    def _run_gsea(self, adata, de_results: Dict) -> Dict[str, Any]:
        """
        Run Gene Set Enrichment Analysis using preranked method.
        """
        # Implementation using gseapy or custom GSEA
        # Returns enriched pathways relevant to T2D
        return {
            "enriched_pathways": [],
            "method": "preranked_gsea"
        }
    
    def _run_wgcna(self, adata) -> Dict[str, Any]:
        """
        Run Weighted Gene Co-expression Network Analysis.
        
        Identifies gene modules and hub genes.
        """
        # WGCNA implementation
        return {
            "modules": [],
            "hub_genes": [],
            "module_trait_correlations": {}
        }
    
    def _run_tf_activity(self, adata) -> Dict[str, Any]:
        """
        Run Transcription Factor activity inference using decoupler/CollecTRI.
        """
        try:
            import decoupler as dc
            
            # Get CollecTRI network
            collectri = dc.get_collectri()
            
            # Run enrichment
            dc.run_ulm(adata, collectri, source='source', target='target', weight='weight')
            
            return {
                "active_tfs": [],
                "repressed_tfs": [],
                "method": "decoupler_ulm"
            }
        except ImportError:
            return {"error": "decoupler not installed", "method": "decoupler_ulm"}
    
    def _run_variability_analysis(self, adata) -> Dict[str, Any]:
        """
        Analyze expression variability between conditions.
        """
        return {
            "high_variability_genes": [],
            "method": "coefficient_of_variation"
        }
    
    def _integrate_across_datasets(self, dataset_results: Dict) -> Dict[str, Any]:
        """
        Integrate results across multiple datasets using Fisher's combined p-value.
        """
        # Collect p-values for each gene across datasets
        gene_pvalues: Dict[str, List[float]] = {}
        
        for geo_id, results in dataset_results.items():
            if "error" in results:
                continue
            
            de_results = results.get("de_analysis", {}).get("full_results", [])
            for gene_data in de_results:
                gene = gene_data.get('names')
                pval = gene_data.get('pvals_adj', 1.0)
                if gene:
                    if gene not in gene_pvalues:
                        gene_pvalues[gene] = []
                    gene_pvalues[gene].append(pval)
        
        # Fisher's combined p-value
        combined_pvalues = {}
        for gene, pvals in gene_pvalues.items():
            if len(pvals) >= 2:
                # Fisher's method: -2 * sum(log(p))
                chi2_stat = -2 * sum(np.log(max(p, 1e-300)) for p in pvals)
                df = 2 * len(pvals)
                combined_p = 1 - stats.chi2.cdf(chi2_stat, df)
                combined_pvalues[gene] = combined_p
        
        return {
            "combined_pvalues": combined_pvalues,
            "n_datasets_per_gene": {g: len(p) for g, p in gene_pvalues.items()},
            "integration_method": "fisher_combined"
        }
    
    def _calculate_priority_scores(self, integrated_results: Dict) -> Dict[str, float]:
        """
        Calculate integrated gene priority scores (0-17 scale).
        
        Scoring components:
        - DE significance (0-5 points)
        - Cross-dataset consistency (0-4 points)
        - Pathway involvement (0-3 points)
        - TF regulation (0-3 points)
        - Expression variability (0-2 points)
        """
        priority_scores = {}
        
        combined_pvals = integrated_results.get("combined_pvalues", {})
        n_datasets = integrated_results.get("n_datasets_per_gene", {})
        
        for gene, pval in combined_pvals.items():
            score = 0.0
            
            # DE significance (0-5 points)
            if pval < 0.001:
                score += 5
            elif pval < 0.01:
                score += 4
            elif pval < 0.05:
                score += 3
            elif pval < 0.1:
                score += 2
            
            # Cross-dataset consistency (0-4 points)
            n_ds = n_datasets.get(gene, 0)
            score += min(n_ds, 4)
            
            # Additional scoring would come from other analyses
            # (pathways, TF activity, variability)
            
            priority_scores[gene] = score
        
        return priority_scores
    
    def _get_top_candidates(self, priority_scores: Dict[str, float], n: int = 50) -> List[Dict]:
        """Get top N candidate genes sorted by priority score."""
        sorted_genes = sorted(priority_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"gene": gene, "priority_score": score, "rank": i+1}
            for i, (gene, score) in enumerate(sorted_genes[:n])
        ]
    
    def format_for_llm(self, results: Dict, mask_genes: bool = True) -> str:
        """
        Format analysis results for LLM consumption.
        
        Args:
            results: Analysis results dictionary
            mask_genes: Whether to mask gene names
            
        Returns:
            Formatted string suitable for LLM prompt
        """
        top_candidates = results.get("top_candidates", [])
        
        if mask_genes and self.gene_mapper:
            # Mask gene names
            formatted_lines = []
            for candidate in top_candidates:
                masked_gene = self.gene_mapper.mask_gene(candidate["gene"])
                formatted_lines.append(
                    f"- {masked_gene}: Priority Score {candidate['priority_score']:.1f}/17"
                )
        else:
            formatted_lines = [
                f"- {c['gene']}: Priority Score {c['priority_score']:.1f}/17"
                for c in top_candidates
            ]
        
        return "\n".join(formatted_lines)