import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import warnings
from utils.gene_masking import GeneMapper
warnings.filterwarnings('ignore')

# =============================================================================
# MASKED ANALYSIS PIPELINE CLASS
# =============================================================================

class MaskedAnalysisPipeline:
    """
    Pipeline that runs all analyses with consistent gene masking.
    Outputs LLM-friendly summaries without gene name leakage.
    
    Supports multi-dataset analysis with cross-dataset integration.
    """
    
    def __init__(self, mapper: GeneMapper, datasets_info: Optional[Dict] = None):
        """
        Initialize the analysis pipeline.
        
        Args:
            mapper: GeneMapper instance (should be unified across all datasets)
            datasets_info: Optional dict with dataset metadata
        """
        self.mapper = mapper
        self.datasets_info = datasets_info or {}
        
        # Storage for full results (for later unmasking) - organized by dataset
        self.full_results = {}  # {dataset_name: {'de': df, 'gsea': df, ...}}
        
        # Storage for LLM-ready summaries - organized by dataset
        self.llm_summaries = {}  # {dataset_name: {'de': {...}, 'gsea': {...}, ...}}
        
        # Storage for cross-dataset results
        self.cross_dataset_results = {}
        
        # Storage for integrated results
        self.integrated_results = {}
    
    # =========================================================================
    # DIFFERENTIAL EXPRESSION ANALYSIS
    # =========================================================================
    
    def run_differential_expression(self, adata: ad.AnnData, group_col: str,
                                     control_name: str, treatment_name: str,
                                     dataset_name: str, n_genes: int = 100) -> Dict:
        """
        Run differential expression analysis using Wilcoxon rank-sum test.
        
        Args:
            adata: AnnData object with gene expression data
            group_col: Column name in adata.obs containing group labels
            control_name: Label for control group
            treatment_name: Label for treatment/disease group
            dataset_name: Identifier for this dataset
            n_genes: Number of top genes to include in LLM summary
        
        Returns:
            Dict with masked DE results and summary statistics
        """
        # Initialize storage for this dataset
        if dataset_name not in self.full_results:
            self.full_results[dataset_name] = {}
        if dataset_name not in self.llm_summaries:
            self.llm_summaries[dataset_name] = {}
        
        # Subset to groups of interest
        adata_sub = adata[
            adata.obs[group_col].isin([control_name, treatment_name])
        ].copy()
        
        # Log transform if needed
        if adata_sub.X.max() > 50:
            sc.pp.log1p(adata_sub)
        
        # Run DE analysis
        sc.tl.rank_genes_groups(
            adata_sub,
            groupby=group_col,
            groups=[treatment_name],
            reference=control_name,
            method='wilcoxon',
            pts=True
        )
        
        # Extract results
        df = sc.get.rank_genes_groups_df(adata_sub, group=treatment_name)
        df = df.rename(columns={
            'names': 'gene', 'pvals': 'pvalue', 'pvals_adj': 'padj',
            'logfoldchanges': 'log2FC', 'scores': 'score'
        })
        
        # Remove rows with NaN log2FC
        n_before = len(df)
        df = df.dropna(subset=['log2FC'])
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"  [{dataset_name}] Dropped {n_dropped} genes with NaN log2FC")
        
        # Mark significant genes
        df['significant'] = (df['padj'] < 0.05) & (np.abs(df['log2FC']) > 0.5)
        df = df.sort_values('padj')
        
        # Store full results
        self.full_results[dataset_name]['de'] = df.copy()
        
        # Create masked version for LLM
        df_top = df.head(n_genes).copy()
        df_masked = df_top.copy()
        df_masked['gene'] = self.mapper.mask(df_top['gene'].tolist())
        
        self.llm_summaries[dataset_name]['de'] = {
            'table': df_masked[['gene', 'log2FC', 'padj', 'score', 'significant']].round(4),
            'summary': {
                'dataset': dataset_name,
                'total_genes_tested': len(df),
                'total_significant': int(df['significant'].sum()),
                'upregulated': int(((df['significant']) & (df['log2FC'] > 0)).sum()),
                'downregulated': int(((df['significant']) & (df['log2FC'] < 0)).sum()),
                'top_effect_sizes': df_top['log2FC'].abs().describe().to_dict()
            }
        }
        
        print(f"  [{dataset_name}] DE complete: {df['significant'].sum()} significant genes")
        return self.llm_summaries[dataset_name]['de']
    
    # =========================================================================
    # GENE SET ENRICHMENT ANALYSIS
    # =========================================================================
    
    def run_gsea(self, adata: ad.AnnData, dataset_name: str,
                 gene_sets: List[str] = ['KEGG_2021_Human', 'GO_Biological_Process_2023']) -> Dict:
        """
        Run Gene Set Enrichment Analysis with prerank method.
        
        Args:
            adata: AnnData object (must have run DE first for this dataset)
            dataset_name: Identifier matching the DE analysis
            gene_sets: Gene set libraries to use
        
        Returns:
            Dict with pathway enrichment results and masked leading edge genes
        """
        import gseapy as gp
        
        if dataset_name not in self.full_results or 'de' not in self.full_results[dataset_name]:
            raise ValueError(f"Must run differential expression first for dataset '{dataset_name}'")
        
        de_df = self.full_results[dataset_name]['de']
        
        # Prepare ranked gene list
        rnk = de_df[['gene', 'score']].set_index('gene')['score']
        rnk = rnk.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Run GSEA prerank
        pre_res = gp.prerank(
            rnk=rnk,
            gene_sets=gene_sets,
            min_size=15,
            max_size=500,
            permutation_num=1000,
            seed=42,
            verbose=False
        )
        
        results = pre_res.res2d.copy()
        self.full_results[dataset_name]['gsea'] = results
        
        # Create LLM-friendly version with masked leading edge genes
        llm_gsea = results[['Term', 'NES', 'FDR q-val', 'FWER p-val', 'Tag %', 'Gene %', 'Lead_genes']].copy()
        llm_gsea = llm_gsea.rename(columns={'FDR q-val': 'FDR', 'FWER p-val': 'FWER'})
        llm_gsea = llm_gsea.sort_values('FDR')
        
        # Mask the leading edge genes
        def mask_gene_list(gene_str):
            if pd.isna(gene_str) or gene_str == '':
                return ''
            genes = [g.strip() for g in str(gene_str).split(';')]
            masked = self.mapper.mask(genes)
            return ';'.join(masked)
        
        llm_gsea['Lead_genes_masked'] = llm_gsea['Lead_genes'].apply(mask_gene_list)
        llm_gsea = llm_gsea.drop(columns=['Lead_genes'])  # Remove real gene names
        
        self.llm_summaries[dataset_name]['gsea'] = {
            'table': llm_gsea.head(50).round(4),
            'summary': {
                'dataset': dataset_name,
                'total_pathways_tested': len(results),
                'significant_pathways_fdr25': int((results['FDR q-val'] < 0.25).sum()),
                'significant_pathways_fdr05': int((results['FDR q-val'] < 0.05).sum()),
                'top_upregulated': llm_gsea[llm_gsea['NES'] > 0].head(10)['Term'].tolist(),
                'top_downregulated': llm_gsea[llm_gsea['NES'] < 0].head(10)['Term'].tolist()
            }
        }
        
        print(f"  [{dataset_name}] GSEA complete: {(results['FDR q-val'] < 0.05).sum()} significant pathways (FDR < 0.05)")
        return self.llm_summaries[dataset_name]['gsea']
    
    # =========================================================================
    # WGCNA CO-EXPRESSION ANALYSIS
    # =========================================================================
    
    def run_wgcna(self, adata: ad.AnnData, group_col: str,
                  control_name: str, treatment_name: str,
                  dataset_name: str, min_module_size: int = 30,
                  merge_cut_height: float = 0.25, soft_power: Optional[int] = None,
                  save: bool = False, output_path: str = '') -> Dict:
        """
        Run Weighted Gene Co-expression Network Analysis.
        
        Args:
            adata: AnnData object with gene expression data
            group_col: Column name in adata.obs containing group labels
            control_name: Label for control group
            treatment_name: Label for treatment/disease group
            dataset_name: Identifier for this dataset
            min_module_size: Minimum genes per module
            merge_cut_height: Threshold for merging similar modules
            soft_power: Soft thresholding power (auto-detected if None)
            save: Whether to save WGCNA object
            output_path: Path for saving outputs
        
        Returns:
            Dict with module information and masked hub genes
        """
        import PyWGCNA
        
        # Initialize storage
        if dataset_name not in self.full_results:
            self.full_results[dataset_name] = {}
        if dataset_name not in self.llm_summaries:
            self.llm_summaries[dataset_name] = {}
        
        # Subset to groups of interest
        mask = adata.obs[group_col].isin([control_name, treatment_name])
        adata_sub = adata[mask].copy()
        
        # Prepare expression matrix (samples x genes)
        X = adata_sub.X if not hasattr(adata_sub.X, 'toarray') else adata_sub.X.toarray()
        expr_df = pd.DataFrame(X, index=adata_sub.obs_names, columns=adata_sub.var_names)
        
        # Ensure all data is numeric (fixes "datExpr must contain numeric data" error)
        expr_df = expr_df.apply(pd.to_numeric, errors='coerce')
        
        # Replace inf values with NaN, then fill NaN with 0
        expr_df = expr_df.replace([np.inf, -np.inf], np.nan)
        expr_df = expr_df.fillna(0)
        
        # Ensure float64 dtype
        expr_df = expr_df.astype(np.float64)
        
        # Filter to top variable genes
        gene_var = expr_df.var()
        top_genes = gene_var.nlargest(min(5000, len(gene_var))).index
        expr_df = expr_df[top_genes]
        print(f"  [{dataset_name}] Using top {len(top_genes)} variable genes for WGCNA")
        
        # Initialize and run WGCNA
        wgcna = PyWGCNA.WGCNA(
            name=f'{dataset_name}_analysis',
            species='human',
            geneExp=expr_df,
            outputPath=output_path,
            save=save
        )
        
        if soft_power is not None:
            wgcna.power = soft_power
        
        print(f"  [{dataset_name}] Running WGCNA pipeline...")
        wgcna.runWGCNA()
        
        soft_power_used = wgcna.power
        print(f"  [{dataset_name}] Soft power used: {soft_power_used}")
        
        # Extract module assignments
        module_df = wgcna.datExpr.var[['moduleColors']].copy()
        module_df = module_df.reset_index().rename(columns={'index': 'gene', 'moduleColors': 'module'})
        
        # Get module eigengenes
        module_eigengenes = wgcna.datME.copy()
        
        # Prepare trait data
        traits = pd.DataFrame({
            'condition': (adata_sub.obs[group_col] == treatment_name).astype(float).values
        }, index=adata_sub.obs_names)
        
        # Align indices
        common_samples = module_eigengenes.index.intersection(traits.index)
        module_eigengenes = module_eigengenes.loc[common_samples]
        traits = traits.loc[common_samples]
        
        # Calculate module-trait correlations
        module_trait_cor = pd.DataFrame(index=module_eigengenes.columns)
        module_trait_pval = pd.DataFrame(index=module_eigengenes.columns)
        
        for trait_col in traits.columns:
            cors = []
            pvals = []
            for me_col in module_eigengenes.columns:
                r, p = stats.pearsonr(module_eigengenes[me_col], traits[trait_col])
                cors.append(r)
                pvals.append(p)
            module_trait_cor[trait_col] = cors
            module_trait_pval[trait_col] = pvals
        
        # Get hub genes per module
        hub_genes = {}
        module_sizes = module_df['module'].value_counts().to_dict()
        
        for module in module_df['module'].unique():
            if module == 'grey':
                continue
            try:
                hub_df = wgcna.top_n_hub_genes(module, n=10)
                hub_genes[module] = hub_df.index.tolist() if hasattr(hub_df, 'index') else hub_df['gene_id'].tolist()
            except:
                module_genes = module_df[module_df['module'] == module]['gene'].tolist()
                hub_genes[module] = module_genes[:10]
        
        n_modules = len([m for m in module_sizes.keys() if m != 'grey'])
        
        # Store full results
        self.full_results[dataset_name]['wgcna'] = {
            'wgcna_object': wgcna,
            'module_assignments': module_df,
            'module_eigengenes': module_eigengenes,
            'module_trait_correlation': module_trait_cor,
            'module_trait_pvalue': module_trait_pval,
            'hub_genes': hub_genes,
            'module_sizes': module_sizes,
            'soft_power': soft_power_used,
            'n_modules': n_modules
        }
        
        # Create LLM summary
        self.llm_summaries[dataset_name]['wgcna'] = self._summarize_wgcna_for_llm(
            self.full_results[dataset_name]['wgcna'],
            dataset_name
        )
        
        print(f"  [{dataset_name}] WGCNA complete: {n_modules} modules detected")
        return self.llm_summaries[dataset_name]['wgcna']
    
    def _summarize_wgcna_for_llm(self, wgcna_results: Dict, dataset_name: str,
                                  top_modules: int = 5, top_hubs: int = 10) -> Dict:
        """Create LLM-friendly WGCNA summary with masked gene names."""
        cor_df = wgcna_results['module_trait_correlation']
        pval_df = wgcna_results['module_trait_pvalue']
        module_sizes = wgcna_results['module_sizes']
        
        summary_data = []
        sorted_modules = cor_df.abs().sort_values('condition', ascending=False).index
        
        for module_name in sorted_modules:
            clean_name = module_name.replace('ME', '')
            
            # Skip unassigned grey module (but keep lightgrey, dimgrey, etc.)
            if clean_name.lower() == 'grey':
                continue
            
            if len(summary_data) >= top_modules:
                break
            
            correlation = cor_df.loc[module_name, 'condition']
            p_value = pval_df.loc[module_name, 'condition']
            size = module_sizes.get(clean_name, 0)
            
            # Get hub genes with hierarchy
            raw_hubs = wgcna_results['hub_genes'].get(clean_name, [])[:top_hubs]
            masked_hubs_detailed = []
            
            for i, gene_real in enumerate(raw_hubs):
                gene_masked = self.mapper.mask(gene_real)
                role = "Hub Component"
                if i == 0:
                    role = "Primary Driver"
                elif i < 3:
                    role = "Major Regulator"
                
                masked_hubs_detailed.append({
                    "gene_id": gene_masked,
                    "rank": i + 1,
                    "role": role
                })
            
            # Confidence assessment
            confidence = "High"
            if p_value > 0.05:
                confidence = "Low"
            elif abs(correlation) < 0.2:
                confidence = "Moderate"
            
            module_summary = {
                "module_name": clean_name,
                "stats": {
                    "correlation_to_disease": round(correlation, 4),
                    "p_value": float(p_value),
                    "statistical_significance": confidence,
                    "module_size": int(size)
                },
                "interpretation": f"This network is {'UP' if correlation > 0 else 'DOWN'}-regulated in disease.",
                "top_drivers": masked_hubs_detailed
            }
            
            summary_data.append(module_summary)
        
        return {
            "analysis_method": "WGCNA",
            "dataset": dataset_name,
            "total_modules_detected": wgcna_results['n_modules'],
            "soft_power_used": wgcna_results['soft_power'],
            "top_prioritized_modules": summary_data
        }
    
    # =========================================================================
    # TRANSCRIPTION FACTOR ACTIVITY ANALYSIS
    # =========================================================================

    def _load_collectri_with_retry(self, organism: str = 'human',
                                    dataset_name: str = '',
                                    max_retries: int = 3,
                                    cache_dir: str = None) -> pd.DataFrame:
        """
        Load CollecTRI network with retry logic and local caching.

        Args:
            organism: 'human', 'mouse', or 'rat'
            dataset_name: For logging purposes
            max_retries: Number of retry attempts
            cache_dir: Directory to cache the downloaded file

        Returns:
            CollecTRI DataFrame or None if failed
        """
        import decoupler as dc
        import time
        import os

        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache')
        os.makedirs(cache_dir, exist_ok=True)

        cache_file = os.path.join(cache_dir, f'collectri_{organism}.csv')

        # Try to load from cache first
        if os.path.exists(cache_file):
            try:
                print(f"  [{dataset_name}] Loading CollecTRI from cache: {cache_file}")
                collectri = pd.read_csv(cache_file)
                return collectri
            except Exception as e:
                print(f"  [{dataset_name}] Cache load failed: {e}, will download fresh")

        # Try to download with retries
        for attempt in range(max_retries):
            try:
                print(f"  [{dataset_name}] Downloading CollecTRI (attempt {attempt + 1}/{max_retries})...")
                collectri = dc.op.collectri(organism=organism)

                # Cache for future use
                try:
                    collectri.to_csv(cache_file, index=False)
                    print(f"  [{dataset_name}] Cached CollecTRI to: {cache_file}")
                except Exception as e:
                    print(f"  [{dataset_name}] Warning: Could not cache CollecTRI: {e}")

                return collectri

            except Exception as e:
                print(f"  [{dataset_name}] Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # 10s, 20s, 30s
                    print(f"  [{dataset_name}] Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

        print(f"  [{dataset_name}] ERROR: All CollecTRI download attempts failed")
        return None

    def run_tf_activity(self, adata: ad.AnnData, group_col: str,
                        control_name: str, treatment_name: str,
                        dataset_name: str, organism: str = 'human') -> Dict:
        """
        Run TF activity inference using decoupler with CollecTRI regulons.
        
        Args:
            adata: AnnData object with gene expression data
            group_col: Column name in adata.obs containing group labels
            control_name: Label for control group
            treatment_name: Label for treatment/disease group
            dataset_name: Identifier for this dataset
            organism: 'human', 'mouse', or 'rat'
        
        Returns:
            Dict with TF activities and masked target genes
        """
        import decoupler as dc
        from statsmodels.stats.multitest import multipletests
        
        # Initialize storage
        if dataset_name not in self.full_results:
            self.full_results[dataset_name] = {}
        if dataset_name not in self.llm_summaries:
            self.llm_summaries[dataset_name] = {}
        
        # Subset data
        mask = adata.obs[group_col].isin([control_name, treatment_name])
        adata_sub = adata[mask].copy()
        
        # Get CollecTRI network with retry logic and local caching
        print(f"  [{dataset_name}] Loading CollecTRI gene regulatory network...")
        collectri = self._load_collectri_with_retry(organism=organism, dataset_name=dataset_name)
        if collectri is None:
            print(f"  [{dataset_name}] WARNING: Could not load CollecTRI. Skipping TF analysis.")
            self.full_results[dataset_name]['tf'] = {'error': 'CollecTRI download failed'}
            self.llm_summaries[dataset_name]['tf'] = {'n_significant': 0, 'significant_tfs': [], 'error': True}
            return {'error': 'CollecTRI download failed'}
        print(f"  [{dataset_name}] Loaded {collectri['source'].nunique()} TFs")
        
        # Prepare expression matrix
        X = adata_sub.X if not hasattr(adata_sub.X, 'toarray') else adata_sub.X.toarray()
        expr_df = pd.DataFrame(X, index=adata_sub.obs_names, columns=adata_sub.var_names)

        # Clean NaN and Inf values (required by decoupler)
        expr_df = expr_df.apply(pd.to_numeric, errors='coerce')
        expr_df = expr_df.replace([np.inf, -np.inf], np.nan)
        expr_df = expr_df.fillna(0)
        expr_df = expr_df.astype(np.float64)

        print(f"  [{dataset_name}] Expression matrix shape: {expr_df.shape}, NaN count: {expr_df.isna().sum().sum()}")

        # Run TF activity inference
        print(f"  [{dataset_name}] Inferring TF activities with ULM...")
        tf_acts, tf_padj = dc.mt.ulm(data=expr_df, net=collectri)
        
        # Calculate differential TF activity
        ctrl_mask = adata_sub.obs[group_col] == control_name
        treat_mask = adata_sub.obs[group_col] == treatment_name
        
        ctrl_samples = adata_sub.obs_names[ctrl_mask]
        treat_samples = adata_sub.obs_names[treat_mask]
        
        diff_results = []
        for tf in tf_acts.columns:
            ctrl_vals = tf_acts.loc[ctrl_samples, tf]
            treat_vals = tf_acts.loc[treat_samples, tf]
            
            if ctrl_vals.isna().all() or treat_vals.isna().all():
                continue
            
            t_stat, pval = stats.ttest_ind(
                treat_vals.dropna(),
                ctrl_vals.dropna(),
                equal_var=False
            )
            mean_diff = treat_vals.mean() - ctrl_vals.mean()
            
            diff_results.append({
                'tf': tf,
                'mean_control': ctrl_vals.mean(),
                'mean_treatment': treat_vals.mean(),
                'activity_change': mean_diff,
                't_statistic': t_stat,
                'pvalue': pval
            })
        
        diff_df = pd.DataFrame(diff_results)
        
        # FDR correction
        valid_pvals = diff_df['pvalue'].fillna(1)
        _, diff_df['padj'], _, _ = multipletests(valid_pvals, method='fdr_bh')
        
        diff_df['significant'] = (diff_df['padj'] < 0.05) & (abs(diff_df['activity_change']) > 0.5)
        diff_df = diff_df.sort_values('pvalue')
        
        # Get TF targets
        tf_targets = collectri.groupby('source')['target'].apply(list).to_dict()
        
        # Store full results
        self.full_results[dataset_name]['tf'] = {
            'tf_activities': tf_acts,
            'tf_padj': tf_padj,
            'differential_tf': diff_df,
            'tf_targets': tf_targets,
            'collectri_net': collectri,
            'n_tfs': len(tf_acts.columns),
            'n_significant': int(diff_df['significant'].sum())
        }
        
        # Create LLM summary
        self.llm_summaries[dataset_name]['tf'] = self._summarize_tf_for_llm(
            self.full_results[dataset_name]['tf'],
            dataset_name
        )
        
        print(f"  [{dataset_name}] TF analysis complete: {diff_df['significant'].sum()} significant TFs")
        return self.llm_summaries[dataset_name]['tf']
    
    def _summarize_tf_for_llm(self, tf_results: Dict, dataset_name: str,
                               top_tfs: int = 20) -> Dict:
        """Create LLM-friendly TF summary with masked target genes."""
        diff_df = tf_results['differential_tf'].copy()
        tf_targets = tf_results['tf_targets']
        collectri_net = tf_results.get('collectri_net', None)
        
        if len(diff_df) == 0:
            return {
                'dataset': dataset_name,
                'n_tfs_analyzed': 0,
                'n_significant': 0,
                'top_activated_tfs': [],
                'top_repressed_tfs': [],
                'all_differential_tfs': []
            }
        
        # Get dataset genes for filtering
        dataset_genes = set(self.mapper.real_to_masked.keys())
        
        # Mask target genes - only include genes in dataset
        tf_targets_masked = {}
        tf_target_info = {}
        
        for tf, targets in tf_targets.items():
            masked = []
            for gene in targets:
                if gene in dataset_genes:
                    masked_gene = self.mapper.mask([gene])[0]
                    masked.append(masked_gene)
                if len(masked) >= 15:
                    break
            
            tf_targets_masked[tf] = masked
            
            if collectri_net is not None:
                tf_net = collectri_net[collectri_net['source'] == tf]
                tf_net_in_data = tf_net[tf_net['target'].isin(dataset_genes)]
                n_activating = (tf_net_in_data['mor'] > 0).sum() if 'mor' in tf_net_in_data.columns else 0
                n_repressing = (tf_net_in_data['mor'] < 0).sum() if 'mor' in tf_net_in_data.columns else 0
                tf_target_info[tf] = {
                    'n_targets_total': len(tf_net),
                    'n_targets_in_data': len(tf_net_in_data),
                    'n_activating': int(n_activating),
                    'n_repressing': int(n_repressing)
                }
        
        def enrich_tf_row(row):
            tf = row['tf']
            targets = tf_targets_masked.get(tf, [])
            info = tf_target_info.get(tf, {})
            
            return pd.Series({
                'tf': tf,
                'activity_change': round(row['activity_change'], 4) if pd.notna(row['activity_change']) else None,
                'pvalue': row.get('pvalue', None),
                'padj': row.get('padj', None),
                'significant': row.get('significant', False),
                'direction': 'UP' if row['activity_change'] > 0 else 'DOWN',
                'n_targets_in_data': info.get('n_targets_in_data', len(targets)),
                'target_coverage': f"{info.get('n_targets_in_data', 0)}/{info.get('n_targets_total', 0)}",
                'targets_masked': ';'.join(targets[:10]) if targets else ''
            })
        
        enriched_df = diff_df.apply(enrich_tf_row, axis=1)
        enriched_df['abs_change'] = enriched_df['activity_change'].abs()
        enriched_df = enriched_df.sort_values('abs_change', ascending=False)
        
        top_up = enriched_df[enriched_df['direction'] == 'UP'].head(top_tfs // 2)
        top_down = enriched_df[enriched_df['direction'] == 'DOWN'].head(top_tfs // 2)
        
        cols_to_keep = ['tf', 'activity_change', 'pvalue', 'padj', 'significant',
                        'direction', 'n_targets_in_data', 'target_coverage', 'targets_masked']
        
        return {
            'analysis_method': 'decoupler_ulm_collectri',
            'dataset': dataset_name,
            'n_tfs_analyzed': tf_results['n_tfs'],
            'n_significant': tf_results['n_significant'],
            'significant_tfs': enriched_df[enriched_df['significant'] == True]['tf'].tolist()[:20],
            'top_activated_tfs': top_up[cols_to_keep].to_dict('records'),
            'top_repressed_tfs': top_down[cols_to_keep].to_dict('records'),
            'all_differential_tfs': enriched_df.sort_values('pvalue')[cols_to_keep].head(50).to_dict('records'),
            'interpretation_guide': {
                'activity_change': 'Positive = TF activated in treatment, Negative = TF repressed',
                'targets_masked': 'Gene IDs masked; only includes genes present in dataset',
                'target_coverage': 'Genes in dataset / Total known targets for TF'
            }
        }
    
    # =========================================================================
    # EXPRESSION VARIABILITY ANALYSIS
    # =========================================================================
    
    def run_expression_variability(self, adata: ad.AnnData, group_col: str,
                                    control_name: str, treatment_name: str,
                                    dataset_name: str) -> Dict:
        """
        Analyze expression variability between conditions.
        
        Genes with high variance in disease but low variance in control
        may indicate dysregulation and loss of homeostatic control.
        
        Args:
            adata: AnnData object
            group_col: Column with condition labels
            control_name: Control group label
            treatment_name: Treatment group label
            dataset_name: Identifier for this dataset
        
        Returns:
            Dict with variability metrics for top variable genes
        """
        if dataset_name not in self.llm_summaries:
            self.llm_summaries[dataset_name] = {}
        
        # Separate conditions
        control_mask = adata.obs[group_col] == control_name
        treatment_mask = adata.obs[group_col] == treatment_name
        
        if hasattr(adata.X, 'toarray'):
            X_ctrl = adata.X[control_mask].toarray()
            X_treat = adata.X[treatment_mask].toarray()
        else:
            X_ctrl = adata.X[control_mask]
            X_treat = adata.X[treatment_mask]
        
        genes = adata.var_names.tolist()
        
        results = []
        for i, gene in enumerate(genes):
            ctrl_expr = X_ctrl[:, i]
            treat_expr = X_treat[:, i]
            
            cv_ctrl = np.std(ctrl_expr) / (np.mean(ctrl_expr) + 1e-10)
            cv_treat = np.std(treat_expr) / (np.mean(treat_expr) + 1e-10)
            var_ratio = (np.var(treat_expr) + 1e-10) / (np.var(ctrl_expr) + 1e-10)
            fano_ctrl = np.var(ctrl_expr) / (np.mean(ctrl_expr) + 1e-10)
            fano_treat = np.var(treat_expr) / (np.mean(treat_expr) + 1e-10)
            
            results.append({
                'gene': gene,
                'cv_control': cv_ctrl,
                'cv_treatment': cv_treat,
                'cv_change': cv_treat - cv_ctrl,
                'variance_ratio': var_ratio,
                'fano_control': fano_ctrl,
                'fano_treatment': fano_treat,
                'fano_change': fano_treat - fano_ctrl
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('variance_ratio', ascending=False)
        df['gene_masked'] = self.mapper.mask(df['gene'].tolist())
        
        self.llm_summaries[dataset_name]['variability'] = {
            'description': 'Genes with increased expression variability in disease state',
            'interpretation': 'High variance ratio suggests dysregulation; may indicate loss of homeostatic control',
            'dataset': dataset_name,
            'top_variable_genes': df.head(50)[['gene_masked', 'cv_control', 'cv_treatment',
                                               'variance_ratio', 'fano_change']].to_dict('records')
        }
        
        print(f"  [{dataset_name}] Variability analysis complete")
        return self.llm_summaries[dataset_name]['variability']
    
    # =========================================================================
    # CROSS-DATASET INTEGRATION: DIFFERENTIAL EXPRESSION
    # =========================================================================
    
    def combine_de_across_datasets(self, dataset_names: List[str]) -> Dict:
        """
        Combine differential expression results across multiple datasets.
        Identifies genes with consistent direction of change.
        
        Args:
            dataset_names: List of dataset names (must have run DE for each)
        
        Returns:
            Dict with cross-dataset consistency metrics
        """
        # Collect DE results
        de_results = {}
        for name in dataset_names:
            if name not in self.full_results or 'de' not in self.full_results[name]:
                raise ValueError(f"DE results not found for dataset '{name}'")
            de_results[name] = self.full_results[name]['de']
        
        # Collect all genes across datasets
        all_genes = set()
        for de_df in de_results.values():
            all_genes.update(de_df['gene'].tolist())
        
        consistency_results = []
        
        for gene in all_genes:
            gene_data = {
                'gene': gene,
                'n_datasets_present': 0,
                'n_datasets_significant': 0,
                'n_datasets_upregulated': 0,
                'n_datasets_downregulated': 0,
                'log2fc_values': [],
                'pvalue_values': [],
                'padj_values': [],
                'datasets_present': [],
                'direction_consistent': False
            }
            
            for name, de_df in de_results.items():
                gene_rows = de_df[de_df['gene'] == gene]
                if len(gene_rows) > 0:
                    row = gene_rows.iloc[0]
                    gene_data['n_datasets_present'] += 1
                    gene_data['datasets_present'].append(name)
                    
                    log2fc = row['log2FC']
                    pval = row['pvalue']
                    padj = row['padj']
                    
                    gene_data['log2fc_values'].append(log2fc)
                    gene_data['pvalue_values'].append(pval)
                    gene_data['padj_values'].append(padj)
                    
                    if padj < 0.05 and abs(log2fc) > 0.5:
                        gene_data['n_datasets_significant'] += 1
                    
                    if log2fc > 0:
                        gene_data['n_datasets_upregulated'] += 1
                    else:
                        gene_data['n_datasets_downregulated'] += 1
            
            # Calculate consistency metrics
            if gene_data['n_datasets_present'] >= 2:
                gene_data['direction_consistent'] = (
                    gene_data['n_datasets_upregulated'] == gene_data['n_datasets_present'] or
                    gene_data['n_datasets_downregulated'] == gene_data['n_datasets_present']
                )
                
                gene_data['mean_log2fc'] = np.mean(gene_data['log2fc_values'])
                gene_data['std_log2fc'] = np.std(gene_data['log2fc_values'])
                
                # Fisher's combined p-value
                valid_pvals = [p for p in gene_data['pvalue_values'] if p > 0 and p < 1]
                if len(valid_pvals) >= 2:
                    chi2_stat = -2 * sum(np.log(p) for p in valid_pvals)
                    gene_data['combined_pvalue'] = 1 - stats.chi2.cdf(chi2_stat, 2 * len(valid_pvals))
                else:
                    gene_data['combined_pvalue'] = min(gene_data['pvalue_values']) if gene_data['pvalue_values'] else 1
                
                consistency_results.append(gene_data)
        
        df = pd.DataFrame(consistency_results)
        
        # Filter for consistent genes
        df_consistent = df[
            (df['direction_consistent'] == True) &
            (df['n_datasets_present'] >= 2)
        ].copy()
        
        # Calculate consistency score
        df_consistent['consistency_score'] = (
            df_consistent['n_datasets_present'] * 2 +
            df_consistent['n_datasets_significant'] * 3 +
            np.abs(df_consistent['mean_log2fc'])
        )
        df_consistent = df_consistent.sort_values('consistency_score', ascending=False)
        
        # Mask gene names
        df_consistent['gene_masked'] = self.mapper.mask(df_consistent['gene'].tolist())
        
        # Store results
        self.cross_dataset_results['de'] = df_consistent
        
        result = {
            'description': 'Genes with consistent differential expression across multiple datasets',
            'n_datasets_analyzed': len(dataset_names),
            'dataset_names': dataset_names,
            'total_genes_analyzed': len(all_genes),
            'consistent_genes_count': len(df_consistent),
            'consistent_genes': df_consistent.head(100)[[
                'gene_masked', 'n_datasets_present', 'n_datasets_significant',
                'direction_consistent', 'mean_log2fc', 'std_log2fc',
                'combined_pvalue', 'consistency_score'
            ]].to_dict('records')
        }
        
        print(f"  Cross-dataset DE: {len(df_consistent)} genes with consistent direction across datasets")
        return result
    
    # =========================================================================
    # CROSS-DATASET INTEGRATION: GSEA
    # =========================================================================
    
    def combine_gsea_across_datasets(self, dataset_names: List[str]) -> Dict:
        """
        Combine GSEA results across multiple datasets.
        Identifies pathways enriched in multiple datasets and tracks leading edge gene frequency.
        
        Args:
            dataset_names: List of dataset names (must have run GSEA for each)
        
        Returns:
            Dict with cross-dataset pathway consistency and gene frequency
        """
        pathway_evidence = {}
        leading_edge_gene_counts = {}  # Track how often each gene appears in leading edges
        
        for dataset_name in dataset_names:
            if dataset_name not in self.llm_summaries or 'gsea' not in self.llm_summaries[dataset_name]:
                raise ValueError(f"GSEA results not found for dataset '{dataset_name}'")
            
            gsea_table = self.llm_summaries[dataset_name]['gsea']['table']
            
            for _, row in gsea_table.iterrows():
                pathway_name = row['Term']
                
                if pathway_name not in pathway_evidence:
                    pathway_evidence[pathway_name] = {
                        'n_datasets_significant': 0,
                        'n_datasets_present': 0,
                        'datasets_significant': [],
                        'datasets_present': [],
                        'nes_values': [],
                        'fdr_values': [],
                        'leading_edge_genes': {},
                        'direction_consistent': True
                    }
                
                pathway_evidence[pathway_name]['n_datasets_present'] += 1
                pathway_evidence[pathway_name]['datasets_present'].append(dataset_name)
                pathway_evidence[pathway_name]['nes_values'].append(row['NES'])
                pathway_evidence[pathway_name]['fdr_values'].append(row['FDR'])
                
                if row['FDR'] < 0.25:
                    pathway_evidence[pathway_name]['n_datasets_significant'] += 1
                    pathway_evidence[pathway_name]['datasets_significant'].append(dataset_name)
                    
                    # Track leading edge gene frequency
                    lead_genes = row.get('Lead_genes_masked', '')
                    if lead_genes and not pd.isna(lead_genes):
                        genes = [g.strip() for g in str(lead_genes).split(';') if g.strip()]
                        for gene in genes:
                            # Track per pathway
                            le_genes = pathway_evidence[pathway_name]['leading_edge_genes']
                            le_genes[gene] = le_genes.get(gene, 0) + 1
                            
                            # Track globally
                            leading_edge_gene_counts[gene] = leading_edge_gene_counts.get(gene, 0) + 1
        
        # Check direction consistency
        for pathway, data in pathway_evidence.items():
            if len(data['nes_values']) > 1:
                all_positive = all(nes > 0 for nes in data['nes_values'])
                all_negative = all(nes < 0 for nes in data['nes_values'])
                data['direction_consistent'] = all_positive or all_negative
        
        # Build consistent pathways list
        consistent_pathways = []
        for pathway_name, data in pathway_evidence.items():
            if data['n_datasets_significant'] >= 2:
                # Sort leading edge genes by frequency
                top_le_genes = sorted(
                    data['leading_edge_genes'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:20]
                
                consistent_pathways.append({
                    'pathway': pathway_name,
                    'n_datasets_significant': data['n_datasets_significant'],
                    'n_datasets_present': data['n_datasets_present'],
                    'datasets_significant': data['datasets_significant'],
                    'direction_consistent': data['direction_consistent'],
                    'mean_nes': round(np.mean(data['nes_values']), 3),
                    'mean_fdr': round(np.mean(data['fdr_values']), 4),
                    'top_leading_edge_genes': top_le_genes
                })
        
        # Sort by number of datasets
        consistent_pathways.sort(key=lambda x: (x['n_datasets_significant'], abs(x['mean_nes'])), reverse=True)
        
        # Sort global gene counts
        top_leading_edge_genes = sorted(
            leading_edge_gene_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:100]
        
        # Store results
        self.cross_dataset_results['gsea'] = {
            'consistent_pathways': consistent_pathways,
            'leading_edge_gene_frequency': dict(top_leading_edge_genes)
        }
        
        result = {
            'description': 'Pathways with consistent enrichment across multiple datasets',
            'n_datasets_analyzed': len(dataset_names),
            'dataset_names': dataset_names,
            'n_consistent_pathways': len(consistent_pathways),
            'consistent_pathways': consistent_pathways[:50],
            'top_leading_edge_genes': top_leading_edge_genes[:50],
            'interpretation': 'Genes appearing in multiple pathway leading edges across datasets are strong candidates'
        }
        
        print(f"  Cross-dataset GSEA: {len(consistent_pathways)} pathways significant in 2+ datasets")
        return result
    
    # =========================================================================
    # INTEGRATED GENE PRIORITY TABLE
    # =========================================================================
    
    def create_integrated_gene_table(self, dataset_names: List[str], top_n: int = 100) -> Dict:
        """
        Create integrated gene priority table combining all evidence types.
        This is the PRIMARY output for LLM drug target prioritization.
        
        Args:
            dataset_names: List of dataset names to include
            top_n: Number of top genes to keep in the priority table
        
        Returns:
            Dict with priority-scored gene table
        """
        gene_evidence = {}
        
        # 1. Aggregate DE evidence across datasets
        for dataset_name in dataset_names:
            if dataset_name in self.full_results and 'de' in self.full_results[dataset_name]:
                de_df = self.full_results[dataset_name]['de']
                
                for i, row in de_df.iterrows():
                    gene = row['gene']
                    masked = self.mapper.mask(gene)
                    
                    if masked not in gene_evidence:
                        gene_evidence[masked] = {
                            'de_ranks': [],
                            'de_log2fcs': [],
                            'de_padjs': [],
                            'de_datasets': []
                        }
                    
                    # Store rank (position in sorted list)
                    rank = de_df.index.get_loc(i) + 1 if isinstance(de_df.index, pd.RangeIndex) else i + 1
                    gene_evidence[masked]['de_ranks'].append(rank)
                    gene_evidence[masked]['de_log2fcs'].append(row['log2FC'])
                    gene_evidence[masked]['de_padjs'].append(row['padj'])
                    gene_evidence[masked]['de_datasets'].append(dataset_name)
        
        # 2. Add cross-dataset DE consistency
        if 'de' in self.cross_dataset_results:
            for gene_data in self.cross_dataset_results['de'].to_dict('records'):
                masked = gene_data['gene_masked']
                if masked in gene_evidence:
                    gene_evidence[masked]['cross_dataset_consistent'] = gene_data['direction_consistent']
                    gene_evidence[masked]['n_datasets_de'] = gene_data['n_datasets_present']
                    gene_evidence[masked]['n_datasets_significant'] = gene_data['n_datasets_significant']
                    gene_evidence[masked]['combined_pvalue'] = gene_data['combined_pvalue']
        
        # 3. Add pathway leading edge evidence
        if 'gsea' in self.cross_dataset_results:
            le_freq = self.cross_dataset_results['gsea'].get('leading_edge_gene_frequency', {})
            for masked, count in le_freq.items():
                if masked not in gene_evidence:
                    gene_evidence[masked] = {}
                gene_evidence[masked]['n_pathway_appearances'] = count
        
        # Also count unique pathways per gene
        if 'gsea' in self.cross_dataset_results:
            gene_pathways = {}
            for pathway in self.cross_dataset_results['gsea']['consistent_pathways']:
                for gene, count in pathway['top_leading_edge_genes']:
                    if gene not in gene_pathways:
                        gene_pathways[gene] = set()
                    gene_pathways[gene].add(pathway['pathway'])
            
            for masked, pathways in gene_pathways.items():
                if masked in gene_evidence:
                    gene_evidence[masked]['n_unique_pathways'] = len(pathways)
        
        # 4. Add WGCNA hub gene evidence
        for dataset_name in dataset_names:
            if dataset_name in self.llm_summaries and 'wgcna' in self.llm_summaries[dataset_name]:
                wgcna_summary = self.llm_summaries[dataset_name]['wgcna']
                
                for module in wgcna_summary.get('top_prioritized_modules', []):
                    for driver in module.get('top_drivers', []):
                        masked = driver['gene_id']
                        if masked not in gene_evidence:
                            gene_evidence[masked] = {}
                        
                        if 'wgcna_modules' not in gene_evidence[masked]:
                            gene_evidence[masked]['wgcna_modules'] = []
                            gene_evidence[masked]['wgcna_roles'] = []
                            gene_evidence[masked]['wgcna_ranks'] = []
                        
                        gene_evidence[masked]['wgcna_modules'].append(f"{dataset_name}:{module['module_name']}")
                        gene_evidence[masked]['wgcna_roles'].append(driver['role'])
                        gene_evidence[masked]['wgcna_ranks'].append(driver['rank'])
        
        # 5. Add TF target evidence
        for dataset_name in dataset_names:
            if dataset_name in self.llm_summaries and 'tf' in self.llm_summaries[dataset_name]:
                tf_summary = self.llm_summaries[dataset_name]['tf']
                
                for tf_data in tf_summary.get('all_differential_tfs', [])[:30]:
                    targets = tf_data.get('targets_masked', '').split(';')
                    for target in targets:
                        if target and target in gene_evidence:
                            if 'regulating_tfs' not in gene_evidence[target]:
                                gene_evidence[target]['regulating_tfs'] = []
                            gene_evidence[target]['regulating_tfs'].append(tf_data['tf'])
        
        # 6. Add variability evidence
        for dataset_name in dataset_names:
            if dataset_name in self.llm_summaries and 'variability' in self.llm_summaries[dataset_name]:
                var_data = self.llm_summaries[dataset_name]['variability']
                
                for gene_data in var_data.get('top_variable_genes', [])[:100]:
                    masked = gene_data['gene_masked']
                    if masked in gene_evidence:
                        if 'variance_ratios' not in gene_evidence[masked]:
                            gene_evidence[masked]['variance_ratios'] = []
                        gene_evidence[masked]['variance_ratios'].append(gene_data['variance_ratio'])
        
        # 7. Calculate priority scores
        for masked, evidence in gene_evidence.items():
            score = 0
            evidence_types = 0
            
            # DE evidence (max 4 points)
            if evidence.get('de_ranks'):
                best_rank = min(evidence['de_ranks'])
                if best_rank <= 10:
                    score += 4
                elif best_rank <= 50:
                    score += 3
                elif best_rank <= 200:
                    score += 2
                else:
                    score += 1
                evidence_types += 1
                evidence['best_de_rank'] = best_rank
                evidence['mean_log2fc'] = np.mean(evidence['de_log2fcs'])
            
            # Cross-dataset consistency (max 4 points)
            if evidence.get('cross_dataset_consistent'):
                score += evidence.get('n_datasets_significant', 0) * 1.5
                score = min(score + 4, score)  # Cap contribution
                evidence_types += 1
            
            # Pathway evidence (max 3 points)
            n_pathways = evidence.get('n_unique_pathways', 0)
            if n_pathways > 0:
                score += min(n_pathways * 0.5, 3)
                evidence_types += 1
            
            # WGCNA hub evidence (max 3 points)
            if evidence.get('wgcna_roles'):
                best_role = evidence['wgcna_roles'][0]
                if 'Primary Driver' in evidence['wgcna_roles']:
                    score += 3
                elif 'Major Regulator' in evidence['wgcna_roles']:
                    score += 2
                else:
                    score += 1
                evidence_types += 1
            
            # TF regulation (max 2 points)
            n_tfs = len(evidence.get('regulating_tfs', []))
            if n_tfs > 0:
                score += min(n_tfs * 0.5, 2)
                evidence_types += 1
                evidence['n_regulating_tfs'] = n_tfs
            
            # High variability (max 1 point)
            if evidence.get('variance_ratios'):
                mean_var_ratio = np.mean(evidence['variance_ratios'])
                if mean_var_ratio > 2:
                    score += 1
                evidence['mean_variance_ratio'] = mean_var_ratio
            
            evidence['priority_score'] = round(score, 2)
            evidence['n_evidence_types'] = evidence_types
        
        # Sort and prepare output
        sorted_genes = sorted(
            gene_evidence.items(),
            key=lambda x: (x[1].get('priority_score', 0), x[1].get('n_evidence_types', 0)),
            reverse=True
        )
        
        # Format for output
        top_genes = []
        for masked, evidence in sorted_genes[:top_n]:
            gene_record = {
                'gene_id': masked,
                'priority_score': evidence.get('priority_score', 0),
                'n_evidence_types': evidence.get('n_evidence_types', 0),
                'best_de_rank': evidence.get('best_de_rank'),
                'mean_log2fc': round(evidence.get('mean_log2fc', 0), 3) if evidence.get('mean_log2fc') else None,
                'n_datasets_de': evidence.get('n_datasets_de', len(evidence.get('de_datasets', []))),
                'cross_dataset_consistent': evidence.get('cross_dataset_consistent', False),
                'n_pathways': evidence.get('n_unique_pathways', 0),
                'wgcna_modules': evidence.get('wgcna_modules', []),
                'wgcna_best_role': evidence.get('wgcna_roles', [None])[0],
                'n_regulating_tfs': evidence.get('n_regulating_tfs', 0),
                'mean_variance_ratio': round(evidence.get('mean_variance_ratio', 0), 2) if evidence.get('mean_variance_ratio') else None
            }
            top_genes.append(gene_record)
        
        self.integrated_results = {
            'description': 'Integrated gene priority table based on DATA-ONLY evidence',
            'scoring_components': [
                'Differential expression rank (max 4 pts)',
                'Cross-dataset consistency (max 4 pts)',
                'Pathway membership (max 3 pts)',
                'WGCNA hub status (max 3 pts)',
                'TF regulation (max 2 pts)',
                'Expression variability (max 1 pt)'
            ],
            'max_possible_score': 17,
            'n_datasets': len(dataset_names),
            'dataset_names': dataset_names,
            'top_priority_genes': top_genes
        }
        
        print(f"  Integrated table: {len(top_genes)} genes ranked by multi-evidence support")
        return self.integrated_results
    
    # =========================================================================
    # LLM CONTEXT GENERATION
    # =========================================================================
    
    def create_llm_context(self, dataset_names: List[str],
                           include_de: bool = True,
                           include_gsea: bool = True,
                           include_wgcna: bool = True,
                           include_tf: bool = True,
                           include_integrated: bool = True,
                           max_de_genes: int = 50,
                           max_pathways: int = 30) -> Dict:
        """
        Create comprehensive context for LLM consumption.
        
        Args:
            dataset_names: List of datasets to include
            include_de: Include differential expression
            include_gsea: Include pathway enrichment
            include_wgcna: Include co-expression analysis
            include_tf: Include TF activity
            include_integrated: Include integrated priority table
            max_de_genes: Max DE genes per dataset
            max_pathways: Max pathways per dataset
        
        Returns:
            Dict with all requested analysis summaries
        """
        context = {
            'meta': {
                'n_datasets': len(dataset_names),
                'dataset_names': dataset_names,
                'total_genes_mapped': self.mapper.n_genes,
                'analyses_included': []
            },
            'datasets': {}
        }
        
        # Per-dataset summaries
        for name in dataset_names:
            context['datasets'][name] = {}
            
            if include_de and name in self.llm_summaries and 'de' in self.llm_summaries[name]:
                de_data = self.llm_summaries[name]['de']
                context['datasets'][name]['differential_expression'] = {
                    'summary': de_data['summary'],
                    'top_genes': de_data['table'].head(max_de_genes).to_dict('records')
                }
            
            if include_gsea and name in self.llm_summaries and 'gsea' in self.llm_summaries[name]:
                gsea_data = self.llm_summaries[name]['gsea']
                context['datasets'][name]['pathway_enrichment'] = {
                    'summary': gsea_data['summary'],
                    'top_pathways': gsea_data['table'].head(max_pathways).to_dict('records')
                }
            
            if include_wgcna and name in self.llm_summaries and 'wgcna' in self.llm_summaries[name]:
                context['datasets'][name]['coexpression_modules'] = self.llm_summaries[name]['wgcna']
            
            if include_tf and name in self.llm_summaries and 'tf' in self.llm_summaries[name]:
                context['datasets'][name]['tf_activity'] = self.llm_summaries[name]['tf']
        
        # Cross-dataset results
        if len(dataset_names) > 1:
            context['cross_dataset'] = {}
            
            if 'de' in self.cross_dataset_results:
                context['cross_dataset']['de_consistency'] = {
                    'n_consistent_genes': len(self.cross_dataset_results['de']),
                    'top_consistent_genes': self.cross_dataset_results['de'].head(50)[[
                        'gene_masked', 'n_datasets_present', 'n_datasets_significant',
                        'mean_log2fc', 'combined_pvalue', 'consistency_score'
                    ]].to_dict('records')
                }
                context['meta']['analyses_included'].append('cross_dataset_de')
            
            if 'gsea' in self.cross_dataset_results:
                context['cross_dataset']['gsea_consistency'] = {
                    'n_consistent_pathways': len(self.cross_dataset_results['gsea']['consistent_pathways']),
                    'consistent_pathways': self.cross_dataset_results['gsea']['consistent_pathways'][:30],
                    'top_leading_edge_genes': list(self.cross_dataset_results['gsea']['leading_edge_gene_frequency'].items())[:50]
                }
                context['meta']['analyses_included'].append('cross_dataset_gsea')
        
        # Integrated priority table
        if include_integrated and self.integrated_results:
            context['integrated_priority_table'] = self.integrated_results
            context['meta']['analyses_included'].append('integrated_priority')
        
        return context
    
    def format_for_llm_prompt(self, context: Dict, task_description: str = "",
                               druggability_data: dict = None) -> str:
        """
        Format analysis context as a clean text prompt for LLM.
        
        Args:
            context: Output from create_llm_context()
            task_description: Optional task instructions to append
            druggability_data: Optional dict mapping masked_id -> {idg_family, subcellular_location}
        
        Returns:
            Formatted markdown text for LLM prompt
        """
        lines = []
        
        # Header
        lines.append("# Biological Data Analysis Context")
        lines.append("")
        lines.append(f"**Datasets Analyzed:** {', '.join(context['meta']['dataset_names'])}")
        lines.append(f"**Total Genes Mapped:** {context['meta']['total_genes_mapped']}")
        lines.append("")
        
        # Per-dataset summaries
        for dataset_name, data in context.get('datasets', {}).items():
            lines.append(f"## Dataset: {dataset_name}")
            lines.append("")
            
            if 'differential_expression' in data:
                de = data['differential_expression']
                lines.append("### Differential Expression")
                lines.append(f"- Total Significant: {de['summary']['total_significant']}")
                lines.append(f"- Upregulated: {de['summary']['upregulated']}")
                lines.append(f"- Downregulated: {de['summary']['downregulated']}")
                lines.append("")
                lines.append("| Gene ID | log2FC | padj | Significant |")
                lines.append("|---------|--------|------|-------------|")
                for g in de['top_genes'][:20]:
                    lines.append(f"| {g['gene']} | {g['log2FC']:.3f} | {g['padj']:.2e} | {g['significant']} |")
                lines.append("")
            
            if 'pathway_enrichment' in data:
                gsea = data['pathway_enrichment']
                lines.append("### Pathway Enrichment")
                lines.append(f"- Significant (FDR<0.05): {gsea['summary']['significant_pathways_fdr05']}")
                lines.append("")
                lines.append("| Pathway | NES | FDR | Leading Edge (masked) |")
                lines.append("|---------|-----|-----|----------------------|")
                for p in gsea['top_pathways'][:15]:
                    genes = p.get('Lead_genes_masked', '')[:50]
                    lines.append(f"| {p['Term'][:50]} | {p['NES']:.2f} | {p['FDR']:.3f} | {genes}... |")
                lines.append("")
        
        # Cross-dataset results
        if 'cross_dataset' in context:
            lines.append("## Cross-Dataset Integration")
            lines.append("")
            
            if 'de_consistency' in context['cross_dataset']:
                lines.append("### Consistent DE Genes")
                lines.append("")
                lines.append("| Gene ID | N Datasets | Mean log2FC | Combined P |")
                lines.append("|---------|------------|-------------|------------|")
                for g in context['cross_dataset']['de_consistency']['top_consistent_genes'][:20]:
                    lines.append(f"| {g['gene_masked']} | {g['n_datasets_present']} | {g['mean_log2fc']:.3f} | {g['combined_pvalue']:.2e} |")
                lines.append("")
            
            if 'gsea_consistency' in context['cross_dataset']:
                lines.append("### Consistent Pathways")
                lines.append("")
                for p in context['cross_dataset']['gsea_consistency']['consistent_pathways'][:10]:
                    lines.append(f"- **{p['pathway']}** (NES={p['mean_nes']:.2f}, {p['n_datasets_significant']} datasets)")
                lines.append("")
        
        # Integrated priority table with optional IDG Family column
        if 'integrated_priority_table' in context:
            lines.append("## Integrated Gene Priority Table")
            lines.append("")
            lines.append("Top genes ranked by multi-evidence support:")
            lines.append("")
            
            # Check if druggability data is available
            has_druggability = druggability_data is not None and len(druggability_data) > 0
            
            if has_druggability:
                lines.append("| Gene ID | IDG Family | Priority | Evidence Types | DE Rank | log2FC | Pathways | WGCNA | TFs |")
                lines.append("|---------|------------|----------|----------------|---------|--------|----------|-------|-----|")
            else:
                lines.append("| Gene ID | Priority | Evidence Types | DE Rank | log2FC | Pathways | WGCNA | TFs |")
                lines.append("|---------|----------|----------------|---------|--------|----------|-------|-----|")
            
            for g in context['integrated_priority_table']['top_priority_genes'][:30]:
                gene_id = g['gene_id']
                priority = f"{g['priority_score']:.1f}"
                evidence = g['n_evidence_types']
                de_rank = g['best_de_rank'] or '-'
                log2fc = g['mean_log2fc'] or '-'
                pathways = g['n_pathways']
                wgcna = g['wgcna_best_role'] or '-'
                tfs = g['n_regulating_tfs']
                
                if has_druggability:
                    # Look up IDG family from druggability data
                    drug_info = druggability_data.get(gene_id, {})
                    idg_family = drug_info.get('idg_family', 'Unknown')[:10]  # Truncate for table width
                    lines.append(f"| {gene_id} | {idg_family} | {priority} | {evidence} | "
                               f"{de_rank} | {log2fc} | {pathways} | {wgcna} | {tfs} |")
                else:
                    lines.append(f"| {gene_id} | {priority} | {evidence} | "
                               f"{de_rank} | {log2fc} | {pathways} | {wgcna} | {tfs} |")
            lines.append("")
        
        # Task description
        if task_description:
            lines.append("## Task")
            lines.append("")
            lines.append(task_description)
            lines.append("")
        
        # Important notes
        lines.append("## Important Notes")
        lines.append("")
        lines.append("- Gene IDs are masked (e.g., G00042) to prevent bias from prior knowledge")
        lines.append("- Pathway names and TF names are visible (mechanism knowledge is allowed)")
        lines.append("- Focus on genes with multi-evidence support across analyses")
        lines.append("- Cross-dataset consistency is a strong indicator of true biological signal")
        lines.append("")
        
        return '\n'.join(lines)
    
    # =========================================================================
    # SAVE / LOAD
    # =========================================================================
    
    def save_all(self, output_dir: str, dataset_names: Optional[List[str]] = None,
                 druggability_data: dict = None):
        """
        Save all results and mappings.
        
        Args:
            output_dir: Directory for output files
            dataset_names: Datasets to include (default: all)
            druggability_data: Optional dict mapping masked_id -> {idg_family, subcellular_location}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if dataset_names is None:
            dataset_names = list(self.full_results.keys())
        
        # Save gene mapping (KEEP SECRET from LLM)
        self.mapper.save(output_dir / 'gene_mapping.json')
        
        # Save full results per dataset (KEEP SECRET from LLM)
        for name in dataset_names:
            if name in self.full_results:
                dataset_dir = output_dir / 'full_results' / name
                dataset_dir.mkdir(parents=True, exist_ok=True)
                
                for analysis, result in self.full_results[name].items():
                    if isinstance(result, pd.DataFrame):
                        result.to_csv(dataset_dir / f'{analysis}_results.csv', index=False)
                    elif isinstance(result, dict) and 'module_assignments' in result:
                        # WGCNA results
                        result['module_assignments'].to_csv(dataset_dir / 'wgcna_modules.csv', index=False)
        
        # Save cross-dataset results
        if self.cross_dataset_results:
            cross_dir = output_dir / 'cross_dataset'
            cross_dir.mkdir(parents=True, exist_ok=True)
            
            if 'de' in self.cross_dataset_results:
                self.cross_dataset_results['de'].to_csv(cross_dir / 'de_consistency.csv', index=False)
        
        # Save LLM-ready context
        context = self.create_llm_context(dataset_names)
        with open(output_dir / 'llm_context.json', 'w') as f:
            json.dump(context, f, indent=2, default=str)
        
        # Save formatted prompt (with druggability data if available)
        prompt = self.format_for_llm_prompt(context, druggability_data=druggability_data)
        with open(output_dir / 'llm_prompt.md', 'w') as f:
            f.write(prompt)
        
        # Save integrated results
        if self.integrated_results:
            with open(output_dir / 'integrated_priority_table.json', 'w') as f:
                json.dump(self.integrated_results, f, indent=2, default=str)
        
        print(f"Results saved to {output_dir}")
    
    def unmask_predictions(self, masked_gene_list: List[str]) -> pd.DataFrame:
        """
        After LLM identifies targets, unmask them and return full info.
        
        Args:
            masked_gene_list: List of masked gene IDs from LLM
        
        Returns:
            DataFrame with real gene names and all associated data
        """
        real_genes = self.mapper.unmask(masked_gene_list)
        
        results = []
        for masked, real in zip(masked_gene_list, real_genes):
            gene_info = {
                'masked_id': masked,
                'real_gene': real
            }
            
            # Add DE info from all datasets
            for dataset_name, data in self.full_results.items():
                if 'de' in data:
                    de_df = data['de']
                    gene_rows = de_df[de_df['gene'] == real]
                    if len(gene_rows) > 0:
                        row = gene_rows.iloc[0]
                        gene_info[f'{dataset_name}_log2FC'] = row['log2FC']
                        gene_info[f'{dataset_name}_padj'] = row['padj']
            
            results.append(gene_info)
        
        return pd.DataFrame(results)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_full_pipeline(datasets: Dict[str, ad.AnnData],
                      group_col: str,
                      control_name: str,
                      treatment_name: str,
                      output_dir: str,
                      run_wgcna: bool = True,
                      run_tf: bool = True,
                      top_n_genes: int = 100) -> MaskedAnalysisPipeline:
    """
    Convenience function to run the full pipeline on multiple datasets.
    
    Args:
        datasets: Dict mapping dataset names to AnnData objects
        group_col: Column name with condition labels
        control_name: Control group label
        treatment_name: Treatment group label
        output_dir: Directory for output files
        run_wgcna: Whether to run WGCNA (can be slow)
        run_tf: Whether to run TF analysis
        top_n_genes: Number of top genes to keep in the priority table
    
    Returns:
        MaskedAnalysisPipeline with all results
    """
    print("=" * 60)
    print("MASKED ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Create unified mapper
    print("\n1. Creating unified gene mapper...")
    mapper = GeneMapper(datasets)
    print(f"   Mapped {mapper.n_genes} unique genes across {mapper.n_datasets} datasets")
    
    # Initialize pipeline
    pipeline = MaskedAnalysisPipeline(mapper)
    dataset_names = list(datasets.keys())
    
    # Run per-dataset analyses
    for name, adata in datasets.items():
        print(f"\n2. Analyzing dataset: {name}")
        print("-" * 40)
        
        print("   Running differential expression...")
        pipeline.run_differential_expression(
            adata, group_col, control_name, treatment_name, name
        )
        
        print("   Running GSEA...")
        pipeline.run_gsea(adata, name)
        
        if run_wgcna:
            print("   Running WGCNA...")
            pipeline.run_wgcna(
                adata, group_col, control_name, treatment_name, name
            )
        
        if run_tf:
            print("   Running TF activity analysis...")
            pipeline.run_tf_activity(
                adata, group_col, control_name, treatment_name, name
            )
        
        print("   Running variability analysis...")
        pipeline.run_expression_variability(
            adata, group_col, control_name, treatment_name, name
        )
    
    # Cross-dataset integration
    if len(dataset_names) > 1:
        print(f"\n3. Cross-dataset integration")
        print("-" * 40)
        
        print("   Combining DE results...")
        pipeline.combine_de_across_datasets(dataset_names)
        
        print("   Combining GSEA results...")
        pipeline.combine_gsea_across_datasets(dataset_names)
    
    # Create integrated table
    print(f"\n4. Creating integrated priority table")
    print("-" * 40)
    pipeline.create_integrated_gene_table(dataset_names, top_n=top_n_genes)
    
    # Save results
    print(f"\n5. Saving results to {output_dir}")
    print("-" * 40)
    pipeline.save_all(output_dir, dataset_names)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    return pipeline
