"""
T2D Pipeline Runner - Executes MaskedAnalysisPipeline and formats for LLM consumption.

This module bridges your existing MaskedAnalysisPipeline with the agent framework.
Supports both Option A (druggability-enhanced) and Option C (expression-only) modes.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
import json
import anndata as ad
import pandas as pd

# Import your existing classes
from utils.gene_masking import GeneMapper  # Adjust import path as needed
from utils.masked_analysis_pipeline import MaskedAnalysisPipeline, run_full_pipeline


class T2DPipelineRunner:
    """
    Runs the T2D analysis pipeline and formats results for LLM agents.
    
    This class:
    1. Loads datasets from .h5ad files
    2. Runs the MaskedAnalysisPipeline
    3. Formats output specifically for hypothesis generation
    """
    
    def __init__(self, data_dir: str, output_dir: str, config: Dict = None):
        """
        Initialize the pipeline runner.
        
        Args:
            data_dir: Directory containing .h5ad files
            output_dir: Directory for output files
            config: Optional configuration dict (uses t2d_config defaults otherwise)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.config = config or {}
        
        # Will be populated after running
        self.mapper: Optional[GeneMapper] = None
        self.pipeline: Optional[MaskedAnalysisPipeline] = None
        self.datasets: Dict[str, ad.AnnData] = {}
        self.analysis_complete = False
    
    def load_datasets(self, dataset_names: List[str] = None) -> Dict[str, ad.AnnData]:
        """
        Load .h5ad datasets from data directory.
        
        Args:
            dataset_names: List of dataset names (GEO IDs) to load. 
                          If None, loads all .h5ad files in data_dir.
        
        Returns:
            Dict mapping dataset names to AnnData objects
        """
        from t2d_config import T2D_DATASETS
        
        if dataset_names is None:
            # Find all .h5ad files
            h5ad_files = list(self.data_dir.glob("*.h5ad"))
            dataset_names = [f.stem for f in h5ad_files]
        
        self.datasets = {}
        for name in dataset_names:
            h5ad_path = self.data_dir / f"{name}.h5ad"
            if h5ad_path.exists():
                logging.info(f"Loading dataset: {name}")
                self.datasets[name] = ad.read_h5ad(h5ad_path)
            else:
                logging.warning(f"Dataset file not found: {h5ad_path}")
        
        logging.info(f"Loaded {len(self.datasets)} datasets")
        return self.datasets
    
    def run_analysis(self, 
                     run_wgcna: bool = True,
                     run_tf: bool = True) -> MaskedAnalysisPipeline:
        """
        Run the full analysis pipeline on loaded datasets.
        
        Args:
            run_wgcna: Whether to run WGCNA (can be slow)
            run_tf: Whether to run TF activity analysis
        
        Returns:
            MaskedAnalysisPipeline with all results
        """
        from t2d_config import T2D_DATASETS, T2D_ANALYSIS_PARAMS, GENE_MASKING_SEED
        
        if not self.datasets:
            raise ValueError("No datasets loaded. Call load_datasets() first.")
        
        # Get group column and labels from config
        # Use first dataset's config as reference (assuming consistent across datasets)
        first_dataset = list(self.datasets.keys())[0]
        dataset_config = T2D_DATASETS.get(first_dataset, {})
        
        group_col = dataset_config.get("group_col", "condition")
        control_name = dataset_config.get("control_name", "ND")
        treatment_name = dataset_config.get("treatment_name", "T2D")
        
        # Get top_candidates_for_llm from config
        top_n_genes = T2D_ANALYSIS_PARAMS.get("top_candidates_for_llm", 100)
        
        # Run the full pipeline using your existing function
        self.pipeline = run_full_pipeline(
            datasets=self.datasets,
            group_col=group_col,
            control_name=control_name,
            treatment_name=treatment_name,
            output_dir=str(self.output_dir),
            run_wgcna=run_wgcna,
            run_tf=run_tf,
            top_n_genes=top_n_genes
        )
        
        # Store mapper reference
        self.mapper = self.pipeline.mapper
        self.analysis_complete = True
        
        return self.pipeline
    
    def format_for_generation_agent(self, 
                                     max_genes: int = None,
                                     include_modules: bool = True,
                                     include_pathways: bool = True) -> str:
        """
        Format analysis results for the GenerationAgent.
        
        This creates a custom format optimized for hypothesis generation,
        emphasizing the integrated priority table and key evidence.
        
        Args:
            max_genes: Maximum number of top genes to include (defaults to config's top_candidates_for_llm)
            include_modules: Include WGCNA module summaries
            include_pathways: Include pathway enrichment summaries
        
        Returns:
            Formatted string for LLM prompt
        """
        if max_genes is None:
            from t2d_config import T2D_ANALYSIS_PARAMS
            max_genes = T2D_ANALYSIS_PARAMS.get('top_candidates_for_llm', 50)
        if not self.analysis_complete:
            raise ValueError("Analysis not complete. Call run_analysis() first.")
        
        lines = []
        dataset_names = list(self.datasets.keys())
        
        # Header
        lines.append("=" * 70)
        lines.append("T2D MULTI-OMICS ANALYSIS RESULTS")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Datasets Analyzed: {', '.join(dataset_names)}")
        lines.append(f"Total Unique Genes: {self.mapper.n_genes}")
        lines.append("")
        
        # SECTION 1: Integrated Priority Table (MOST IMPORTANT)
        lines.append("-" * 70)
        lines.append("SECTION 1: INTEGRATED GENE PRIORITY TABLE")
        lines.append("-" * 70)
        lines.append("")
        lines.append("Genes ranked by multi-evidence support (max score: 17 points)")
        lines.append("Scoring: DE rank (4) + Cross-dataset (4) + Pathways (3) + WGCNA (3) + TF (2) + Variability (1)")
        lines.append("")
        
        integrated = self.pipeline.integrated_results
        if integrated and 'top_priority_genes' in integrated:
            # Check if druggability features are available
            has_druggability = hasattr(self, 'druggability_features') and self.druggability_features is not None
            
            if has_druggability:
                # Build a lookup dict for fast access: masked_id -> idg_family
                drug_lookup = {}
                for _, row in self.druggability_features.iterrows():
                    drug_lookup[row['masked_id']] = row.get('idg_family', 'Unknown')
                
                lines.append("| Rank | Gene ID | IDG Family | Score | Evidence | DE Rank | log2FC | Pathways | WGCNA Role | TFs |")
                lines.append("|------|---------|------------|-------|----------|---------|--------|----------|------------|-----|")
            else:
                lines.append("| Rank | Gene ID | Score | Evidence | DE Rank | log2FC | Pathways | WGCNA Role | TFs |")
                lines.append("|------|---------|-------|----------|---------|--------|----------|------------|-----|")
            
            for i, gene in enumerate(integrated['top_priority_genes'][:max_genes], 1):
                wgcna_role = gene.get('wgcna_best_role', '-')
                if wgcna_role and len(wgcna_role) > 12:
                    wgcna_role = wgcna_role[:12]
                
                log2fc = f"{gene['mean_log2fc']:.2f}" if gene.get('mean_log2fc') else "-"
                de_rank = gene.get('best_de_rank', '-')
                
                if has_druggability:
                    idg_family = drug_lookup.get(gene['gene_id'], 'Unknown')[:10]
                    lines.append(
                        f"| {i:4d} | {gene['gene_id']} | {idg_family} | {gene['priority_score']:5.1f} | "
                        f"{gene['n_evidence_types']} types | {de_rank} | {log2fc} | "
                        f"{gene.get('n_pathways', 0)} | {wgcna_role} | {gene.get('n_regulating_tfs', 0)} |"
                    )
                else:
                    lines.append(
                        f"| {i:4d} | {gene['gene_id']} | {gene['priority_score']:5.1f} | "
                        f"{gene['n_evidence_types']} types | {de_rank} | {log2fc} | "
                        f"{gene.get('n_pathways', 0)} | {wgcna_role} | {gene.get('n_regulating_tfs', 0)} |"
                    )
            lines.append("")

        
        # SECTION 2: Cross-Dataset Consistency
        lines.append("-" * 70)
        lines.append("SECTION 2: CROSS-DATASET CONSISTENCY")
        lines.append("-" * 70)
        lines.append("")
        
        if 'de' in self.pipeline.cross_dataset_results:
            de_consistency = self.pipeline.cross_dataset_results['de']
            lines.append(f"Genes with consistent direction across datasets: {len(de_consistency)}")
            lines.append("")
            lines.append("Top consistent genes:")
            lines.append("| Gene ID | Datasets | Significant | Mean log2FC | Combined P |")
            lines.append("|---------|----------|-------------|-------------|------------|")
            
            for _, row in de_consistency.head(20).iterrows():
                lines.append(
                    f"| {row['gene_masked']} | {row['n_datasets_present']} | "
                    f"{row['n_datasets_significant']} | {row['mean_log2fc']:.3f} | "
                    f"{row['combined_pvalue']:.2e} |"
                )
            lines.append("")
        
        # SECTION 3: Pathway Context (for mechanism understanding)
        if include_pathways and 'gsea' in self.pipeline.cross_dataset_results:
            lines.append("-" * 70)
            lines.append("SECTION 3: CONSISTENT PATHWAY ENRICHMENT")
            lines.append("-" * 70)
            lines.append("")
            
            gsea_consistency = self.pipeline.cross_dataset_results['gsea']
            consistent_pathways = gsea_consistency.get('consistent_pathways', [])
            
            lines.append(f"Pathways significant in 2+ datasets: {len(consistent_pathways)}")
            lines.append("")
            
            for pathway in consistent_pathways[:15]:
                direction = "UP" if pathway['mean_nes'] > 0 else "DOWN"
                lines.append(f"• {pathway['pathway']}")
                lines.append(f"  Direction: {direction} | NES: {pathway['mean_nes']:.2f} | "
                           f"Datasets: {pathway['n_datasets_significant']}")
                
                # Top leading edge genes (masked)
                top_le = pathway.get('top_leading_edge_genes', [])[:5]
                if top_le:
                    le_genes = [f"{g[0]}({g[1]})" for g in top_le]
                    lines.append(f"  Key genes: {', '.join(le_genes)}")
                lines.append("")
        
        # SECTION 4: WGCNA Module Context
        if include_modules:
            lines.append("-" * 70)
            lines.append("SECTION 4: CO-EXPRESSION NETWORK MODULES")
            lines.append("-" * 70)
            lines.append("")
            
            for dataset_name in dataset_names:
                if dataset_name in self.pipeline.llm_summaries:
                    wgcna = self.pipeline.llm_summaries[dataset_name].get('wgcna', {})
                    if wgcna:
                        lines.append(f"Dataset: {dataset_name}")
                        lines.append(f"Total modules: {wgcna.get('total_modules_detected', 0)}")
                        lines.append("")
                        
                        for module in wgcna.get('top_prioritized_modules', [])[:3]:
                            stats = module.get('stats', {})
                            lines.append(f"  Module: {module['module_name']}")
                            lines.append(f"  Correlation to T2D: {stats.get('correlation_to_disease', 0):.3f} "
                                       f"(p={stats.get('p_value', 1):.2e})")
                            lines.append(f"  Size: {stats.get('module_size', 0)} genes")
                            
                            # Hub genes
                            drivers = module.get('top_drivers', [])[:5]
                            if drivers:
                                hub_str = ", ".join([f"{d['gene_id']}({d['role']})" for d in drivers])
                                lines.append(f"  Key drivers: {hub_str}")
                            lines.append("")
        
        # SECTION 5: Transcription Factor Context
        lines.append("-" * 70)
        lines.append("SECTION 5: TRANSCRIPTION FACTOR ACTIVITY")
        lines.append("-" * 70)
        lines.append("")
        
        for dataset_name in dataset_names:
            if dataset_name in self.pipeline.llm_summaries:
                tf = self.pipeline.llm_summaries[dataset_name].get('tf', {})
                if tf:
                    lines.append(f"Dataset: {dataset_name}")
                    lines.append(f"Significant TFs: {tf.get('n_significant', 0)}")
                    
                    sig_tfs = tf.get('significant_tfs', [])[:10]
                    if sig_tfs:
                        lines.append(f"Top TFs: {', '.join(sig_tfs)}")
                    lines.append("")
        
        # Important notes
        lines.append("-" * 70)
        lines.append("IMPORTANT NOTES FOR HYPOTHESIS GENERATION")
        lines.append("-" * 70)
        lines.append("")
        lines.append("1. Gene IDs are MASKED (e.g., G00042) - do NOT try to guess real identities")
        lines.append("2. Base your reasoning ONLY on the data patterns shown above")
        lines.append("3. Pathway names and TF names are real - you may use biological knowledge about pathways")
        lines.append("4. Focus on genes with HIGH priority scores and MULTIPLE evidence types")
        lines.append("5. Cross-dataset consistency is a strong indicator of true biological signal")
        lines.append("")
        
        return "\n".join(lines)
    
    def format_for_reflection_agent(self, hypothesis: Dict) -> str:
        """
        Format analysis context for the ReflectionAgent to evaluate a hypothesis.
        
        Args:
            hypothesis: The hypothesis dict to evaluate
        
        Returns:
            Formatted context string
        """
        if not self.analysis_complete:
            raise ValueError("Analysis not complete. Call run_analysis() first.")
        
        lines = []
        
        # Extract the target gene from hypothesis
        target_gene = hypothesis.get('target_gene_masked', 'UNKNOWN')
        
        lines.append("=" * 60)
        lines.append("EVALUATION CONTEXT")
        lines.append("=" * 60)
        lines.append("")
        
        # Get evidence for this specific gene
        lines.append(f"Evidence for target gene: {target_gene}")
        lines.append("-" * 40)
        
        integrated = self.pipeline.integrated_results
        if integrated and 'top_priority_genes' in integrated:
            for gene in integrated['top_priority_genes']:
                if gene['gene_id'] == target_gene:
                    lines.append(f"Priority Score: {gene['priority_score']}/17")
                    lines.append(f"Evidence Types: {gene['n_evidence_types']}")
                    lines.append(f"Best DE Rank: {gene.get('best_de_rank', 'N/A')}")
                    lines.append(f"Mean log2FC: {gene.get('mean_log2fc', 'N/A')}")
                    lines.append(f"N Datasets: {gene.get('n_datasets_de', 'N/A')}")
                    lines.append(f"Cross-dataset Consistent: {gene.get('cross_dataset_consistent', False)}")
                    lines.append(f"N Pathways: {gene.get('n_pathways', 0)}")
                    lines.append(f"WGCNA Role: {gene.get('wgcna_best_role', 'None')}")
                    lines.append(f"Regulating TFs: {gene.get('n_regulating_tfs', 0)}")
                    break
            else:
                lines.append("WARNING: Target gene not found in top priority list!")
        
        lines.append("")
        
        # Add summary of top genes for comparison
        lines.append("Top 10 Priority Genes (for comparison):")
        lines.append("-" * 40)
        if integrated and 'top_priority_genes' in integrated:
            for i, gene in enumerate(integrated['top_priority_genes'][:10], 1):
                lines.append(f"{i}. {gene['gene_id']} - Score: {gene['priority_score']:.1f}")
        
        return "\n".join(lines)
    
    def format_for_evolution_agent(self, 
                                    parent1: Dict = None,
                                    parent2: Dict = None) -> str:
        """
        Format analysis context for the EvolutionAgent.
        
        Args:
            parent1: First parent hypothesis (for crossover)
            parent2: Second parent hypothesis (for crossover)
        
        Returns:
            Formatted context string
        """
        # Reuse generation format but add crossover-specific notes
        base_context = self.format_for_generation_agent()  # uses config default
        
        lines = [base_context]
        lines.append("")
        lines.append("=" * 60)
        lines.append("EVOLUTION CONSTRAINTS")
        lines.append("=" * 60)
        lines.append("")
        lines.append("1. The evolved hypothesis MUST select a target from the priority table above")
        lines.append("2. You may combine mechanisms from parent hypotheses")
        lines.append("3. The child hypothesis should be STRONGER than either parent")
        lines.append("4. Consider genes with multiple evidence types for robustness")
        lines.append("")
        
        return "\n".join(lines)
    
    def get_valid_target_genes(self, top_n: int = None) -> List[str]:
        """
        Get list of valid target gene IDs (masked) for validation.
        
        Args:
            top_n: Number of top priority genes to consider valid (defaults to config's top_candidates_for_llm)
        
        Returns:
            List of masked gene IDs
        """
        if top_n is None:
            from t2d_config import T2D_ANALYSIS_PARAMS
            top_n = T2D_ANALYSIS_PARAMS.get('top_candidates_for_llm', 50)
        
        if not self.analysis_complete:
            return []
        
        integrated = self.pipeline.integrated_results
        if integrated and 'top_priority_genes' in integrated:
            return [g['gene_id'] for g in integrated['top_priority_genes'][:top_n]]
        return []
    
    def unmask_final_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """
        After the pipeline completes, unmask the final predictions.
        
        Args:
            predictions: List of hypothesis dicts with masked gene IDs
        
        Returns:
            List with real gene names added for both primary and ranked targets
        """
        unmasked = []
        for pred in predictions:
            pred_copy = pred.copy()
            
            # Unmask primary target
            masked_id = pred.get('target_gene_masked')
            if masked_id:
                if self.mapper:
                    real_gene = self.mapper.unmask(masked_id)
                    pred_copy['target_gene_real'] = real_gene
                else:
                    pred_copy['target_gene_real'] = masked_id # Fallback if no mapper
            
            # Unmask ranked targets
            if 'ranked_targets' in pred_copy and isinstance(pred_copy['ranked_targets'], list):
                new_ranked = []
                for target in pred_copy['ranked_targets']:
                    target_copy = target.copy()
                    t_masked_id = target.get('gene_id')
                    if t_masked_id:
                        if self.mapper:
                            t_real_gene = self.mapper.unmask(t_masked_id)
                            target_copy['gene_symbol'] = t_real_gene
                        else:
                            target_copy['gene_symbol'] = t_masked_id
                    new_ranked.append(target_copy)
                pred_copy['ranked_targets'] = new_ranked
                
            unmasked.append(pred_copy)
        
        return unmasked
    
    def save_gene_mapping(self, path: str):
        """Save the gene mapping for later unmasking."""
        if self.mapper:
            self.mapper.save(path)
    
    def load_gene_mapping(self, path: str):
        """Load a previously saved gene mapping."""
        self.mapper = GeneMapper.load(path)
    
    def get_top_enriched_pathways(self, n: int = 5) -> List[str]:
        """
        Get top enriched pathway names from GSEA results.

        Args:
            n: Number of top pathways to return

        Returns:
            List of pathway names (without gene details)
        """
        if not self.analysis_complete or self.pipeline is None:
            return []

        pathways = []

        # Try to get from cross-dataset GSEA results first
        if hasattr(self.pipeline, 'cross_dataset_results') and 'gsea' in self.pipeline.cross_dataset_results:
            gsea_consistency = self.pipeline.cross_dataset_results['gsea']
            consistent_pathways = gsea_consistency.get('consistent_pathways', [])

            for pathway in consistent_pathways[:n]:
                pathways.append(pathway['pathway'])

            if pathways:
                return pathways

        # Fallback: Try to get from individual dataset results
        gsea_results = getattr(self.pipeline, 'integrated_results', {}).get('gsea', {})
        
        for dataset_id, dataset_gsea in gsea_results.items():
            if isinstance(dataset_gsea, dict):
                # Get pathway names sorted by significance
                for pathway_name, pathway_data in dataset_gsea.items():
                    if isinstance(pathway_data, dict):
                        pval = pathway_data.get('pvalue', 1.0)
                        if pval < 0.05 and pathway_name not in pathways:
                            pathways.append(pathway_name)
        
        # Return top N unique pathways
        return pathways[:n]

    # =========================================================================
    # DRUGGABILITY INTEGRATION (Option A)
    # =========================================================================

    def run_druggability_extraction(self, top_n_genes: int = 100) -> Optional[pd.DataFrame]:
        """
        Extract druggability features for top priority genes.

        Args:
            top_n_genes: Number of top genes to extract features for

        Returns:
            DataFrame with druggability features (masked IDs) or None if failed
        """
        if not self.analysis_complete:
            raise ValueError("Analysis not complete. Call run_analysis() first.")

        try:
            from utils.druggability_extractor import DruggabilityFeatureExtractor

            logging.info(f"Extracting druggability features for top {top_n_genes} genes...")

            # Get top priority genes (real names for API lookup)
            integrated = self.pipeline.integrated_results
            if not integrated or 'top_priority_genes' not in integrated:
                logging.warning("No integrated results available for druggability extraction")
                return None

            # Get masked IDs and their real gene names
            top_genes = integrated['top_priority_genes'][:top_n_genes]
            gene_list = []
            gene_mapping = {}

            for gene_info in top_genes:
                masked_id = gene_info['gene_id']
                real_name = self.mapper.unmask(masked_id)
                if real_name and real_name != masked_id:
                    gene_list.append(real_name)
                    gene_mapping[real_name] = masked_id

            logging.info(f"Extracting features for {len(gene_list)} genes")

            # Extract features
            extractor = DruggabilityFeatureExtractor()
            feature_df = extractor.create_feature_dataframe(gene_list, gene_mapping)

            # Store for later use
            self.druggability_features = feature_df
            self.druggability_extractor = extractor

            logging.info(f"Druggability extraction complete: {len(feature_df)} genes")
            return feature_df

        except ImportError as e:
            logging.error(f"Could not import druggability extractor: {e}")
            return None
        except Exception as e:
            logging.error(f"Druggability extraction failed: {e}")
            return None

    def run_leakage_test(self) -> Dict[str, Any]:
        """
        Run leakage tests on druggability features.

        Returns:
            Dictionary with leakage test results
        """
        if not hasattr(self, 'druggability_features') or self.druggability_features is None:
            logging.warning("No druggability features available. Run run_druggability_extraction() first.")
            return {'skipped': True, 'reason': 'No druggability features'}

        try:
            from utils.leakage_tester import LeakageTester
            from t2d_config import DRUGGABILITY_CONFIG

            logging.info("Running leakage tests on druggability features...")

            tester = LeakageTester()
            feature_columns = DRUGGABILITY_CONFIG['safe_features']

            results = tester.run_all_tests(
                self.druggability_features,
                feature_columns
            )

            # Generate and log report
            report = tester.generate_report(results)
            logging.info(f"\n{report}")

            # Store results
            self.leakage_test_results = results

            return results

        except ImportError as e:
            logging.error(f"Could not import leakage tester: {e}")
            return {'skipped': True, 'reason': str(e)}
        except Exception as e:
            logging.error(f"Leakage test failed: {e}")
            return {'skipped': True, 'reason': str(e)}

    def format_druggability_for_llm(self, max_genes: int = 50) -> str:
        """
        Format IDG protein family features for LLM consumption.

        Uses IDG standard families (GPCR, Kinase, IC, NR, TF, Enzyme, Transporter, Epigenetic)
        to prevent information leakage from specific protein subfamilies.

        Args:
            max_genes: Maximum number of genes to include

        Returns:
            Formatted string for LLM prompt
        """
        if not hasattr(self, 'druggability_features') or self.druggability_features is None:
            return "No IDG family features available."

        if hasattr(self, 'druggability_extractor'):
            return self.druggability_extractor.format_for_llm(
                self.druggability_features, top_n=max_genes
            )

        # Fallback formatting (uses new IDG column names)
        df = self.druggability_features.head(max_genes)
        lines = ["IDG PROTEIN FAMILIES (standard classification):"]
        lines.append("=" * 50)
        lines.append(f"{'Gene ID':<10} {'IDG Family':<12} {'Location':<12}")
        lines.append("-" * 50)

        for _, row in df.iterrows():
            # Support both old (protein_class) and new (idg_family) column names
            family = row.get('idg_family', row.get('protein_class', 'Unknown'))
            location = row.get('subcellular_location', 'Unknown')
            lines.append(
                f"{row['masked_id']:<10} {family:<12} {location:<12}"
            )

        lines.append("")
        lines.append("NOTE: Family membership alone does NOT indicate disease relevance.")
        lines.append("Prioritize based on DATA EVIDENCE, not family.")

        return "\n".join(lines)

    def get_druggability_for_gene(self, masked_id: str) -> Dict[str, str]:
        """
        Get druggability features for a specific gene.

        Args:
            masked_id: Masked gene ID (e.g., "G00042")

        Returns:
            Dictionary of druggability features
        """
        if not hasattr(self, 'druggability_features') or self.druggability_features is None:
            return {}

        gene_row = self.druggability_features[
            self.druggability_features['masked_id'] == masked_id
        ]

        if len(gene_row) == 0:
            return {}

        return gene_row.iloc[0].to_dict()

    # =========================================================================
    # CANDIDATE SAVING AND EVALUATION
    # =========================================================================

    def save_candidate_genes(self, predictions: List[Dict], output_path: str = None):
        """
        Save candidate genes from predictions for future evaluation.

        Args:
            predictions: List of hypothesis dicts with target_gene_masked
            output_path: Path to save candidates (default: output_dir/candidates.json)
        """
        if output_path is None:
            output_path = self.output_dir / "candidates.json"

        candidates = []
        for pred in predictions:
            masked_id = pred.get('target_gene_masked')
            if masked_id:
                real_name = self.mapper.unmask(masked_id) if self.mapper else None
                candidate = {
                    'masked_id': masked_id,
                    'real_gene': real_name,
                    'fitness_score': pred.get('fitness_score'),
                    'confidence': pred.get('confidence_level'),
                    'title': pred.get('title'),
                    'mechanism': pred.get('mechanism_hypothesis', '')[:500],  # Truncate
                }

                # Add druggability if available
                if hasattr(self, 'druggability_features'):
                    drug_features = self.get_druggability_for_gene(masked_id)
                    candidate['druggability'] = drug_features

                candidates.append(candidate)

        # Save to file
        with open(output_path, 'w') as f:
            json.dump({
                'n_candidates': len(candidates),
                'candidates': candidates,
                'gene_mapping_available': self.mapper is not None
            }, f, indent=2)

        logging.info(f"Saved {len(candidates)} candidate genes to {output_path}")

    def load_ground_truth(self) -> Tuple[List, Dict[str, Set[str]]]:
        """
        Load ground truth T2D drug targets from OpenTargets.

        Returns:
            Tuple of (all_targets list, tier_genes dict)
        """
        try:
            from utils.ground_truth_loader import GroundTruthLoader

            loader = GroundTruthLoader()
            all_targets, tier_genes = loader.load_all_ground_truth()

            logging.info(f"Loaded ground truth: {len(tier_genes['tier1'])} Tier 1, "
                        f"{len(tier_genes['tier2'])} Tier 2, {len(tier_genes['tier3'])} Tier 3")

            self.ground_truth_targets = all_targets
            self.ground_truth_tiers = tier_genes

            return all_targets, tier_genes

        except ImportError as e:
            logging.error(f"Could not import ground truth loader: {e}")
            return [], {}
        except Exception as e:
            logging.error(f"Failed to load ground truth: {e}")
            return [], {}

    def evaluate_predictions(self, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate predictions against OpenTargets ground truth.

        Args:
            predictions: List of hypothesis dicts with target_gene_masked

        Returns:
            Dictionary with evaluation metrics
        """
        # Load ground truth if not already loaded
        if not hasattr(self, 'ground_truth_tiers') or not self.ground_truth_tiers:
            self.load_ground_truth()

        if not hasattr(self, 'ground_truth_tiers') or not self.ground_truth_tiers:
            logging.warning("No ground truth available for evaluation")
            return {'error': 'No ground truth available'}

        try:
            from utils.ground_truth_loader import GroundTruthLoader

            # Build reverse mapping (masked -> real)
            gene_mapping_reverse = {}
            if self.mapper:
                for masked_id in [p.get('target_gene_masked') for p in predictions if p.get('target_gene_masked')]:
                    real_name = self.mapper.unmask(masked_id)
                    if real_name:
                        gene_mapping_reverse[masked_id] = real_name

            # Get predicted masked IDs
            predicted_masked = [p.get('target_gene_masked') for p in predictions if p.get('target_gene_masked')]

            # Evaluate
            loader = GroundTruthLoader()
            results = loader.evaluate_predictions(
                predicted_masked,
                self.ground_truth_tiers,
                gene_mapping_reverse
            )

            # Generate report
            report = loader.generate_evaluation_report(results)
            logging.info(f"\n{report}")

            return results

        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            return {'error': str(e)}

    def format_for_generation_agent_with_druggability(
        self,
        max_genes: int = None,
        include_modules: bool = True,
        include_pathways: bool = True
    ) -> str:
        """
        Format analysis results for GenerationAgent with druggability features.

        This is the Option A version that includes druggability context.

        Args:
            max_genes: Maximum number of top genes to include (defaults to config's top_candidates_for_llm)
            include_modules: Include WGCNA module summaries
            include_pathways: Include pathway enrichment summaries

        Returns:
            Formatted string for LLM prompt
        """
        # Resolve max_genes from config if not specified
        if max_genes is None:
            from t2d_config import T2D_ANALYSIS_PARAMS
            max_genes = T2D_ANALYSIS_PARAMS.get('top_candidates_for_llm', 50)
        
        # Get base format
        base_context = self.format_for_generation_agent(
            max_genes=max_genes,
            include_modules=include_modules,
            include_pathways=include_pathways
        )

        # Add druggability section if available
        if hasattr(self, 'druggability_features') and self.druggability_features is not None:
            druggability_section = self.format_druggability_for_llm(max_genes=max_genes)
            base_context = base_context.replace(
                "IMPORTANT NOTES FOR HYPOTHESIS GENERATION",
                f"{druggability_section}\n\n" + "-" * 70 + "\nIMPORTANT NOTES FOR HYPOTHESIS GENERATION"
            )

        return base_context

    # =========================================================================
    # EVOLUTIONARY OPTIMIZATION
    # =========================================================================

    def run_evolutionary_optimization(self, 
                                     objective: str, 
                                     population_size: int = 20, 
                                     n_generations: int = 10) -> Dict[str, Any]:
        """
        Run the genetic algorithm for T2D target identification.

        Orchestrates the SupervisorAgent to evolve hypotheses based on
        the T2D analysis context.

        Args:
            objective: Research goal/objective string
            population_size: Size of the population
            n_generations: Number of generations to run

        Returns:
            Dict containing final results and statistics
        """
        logging.info("=" * 60)
        logging.info(f"STARTING T2D EVOLUTIONARY OPTIMIZATION")
        logging.info(f"Generations: {n_generations}, Population: {population_size}")
        logging.info("=" * 60)

        # Initialize Supervisor Agent
        # Import here to avoid circular dependency at module level
        from agents.supervisor_agent import SupervisorAgent
        
        supervisor = SupervisorAgent(
            research_goal=objective,
            mode="t2d-target",
            run_folder=str(self.output_dir)
        )
        
        # Attach runner to agents so they can access T2D specific data/methods
        # This is CRITICAL for T2D mode
        supervisor.t2d_runner = self
        supervisor.generation_agent.t2d_runner = self
        supervisor.reflection_agent.t2d_runner = self
        supervisor.evolution_agent.t2d_runner = self
        
        # Override config parameters with arguments
        supervisor.population_size = population_size
        supervisor.num_generations = n_generations
        
        # Check for druggability
        if hasattr(self, 'druggability_features') and self.druggability_features is not None:
             supervisor.generation_agent.use_druggability = True
             logging.info("Enabled druggability-aware generation")

        # Run genetic algorithm
        try:
            results = supervisor.run_genetic_algorithm()
            return results
        except Exception as e:
            logging.error(f"Evolutionary optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    # =========================================================================
    # COMPREHENSIVE EVALUATION
    # =========================================================================

    def run_full_evaluation(self,
                            predictions: List[Dict],
                            analysis_context: str = None,
                            literature_context: str = None,
                            run_baseline: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of T2D predictions using all three metrics:
        1. OpenTargets association score correlation
        2. Baseline LLM comparison
        3. @K recall, precision, MRR, MAP

        Args:
            predictions: List of hypothesis dicts from the framework
            analysis_context: Analysis context string (optional, for baseline)
            literature_context: Literature context string (optional, for baseline)
            run_baseline: Whether to run baseline LLM comparison

        Returns:
            Comprehensive evaluation results dict
        """
        try:
            from utils.evaluation_metrics import T2DEvaluationMetrics

            logging.info("Starting comprehensive T2D evaluation...")

            # Create evaluator instance
            evaluator = T2DEvaluationMetrics()

            # Build gene mapping (masked -> real)
            gene_mapping = {}
            if self.mapper:
                for pred in predictions:
                    masked_id = pred.get('target_gene_masked')
                    if masked_id:
                        real_name = self.mapper.unmask(masked_id)
                        if real_name and real_name != masked_id:
                            gene_mapping[masked_id] = real_name

                    # Also map ranked targets if present
                    for target in pred.get('ranked_targets', []):
                        masked_id = target.get('gene_id')
                        if masked_id and masked_id not in gene_mapping:
                            real_name = self.mapper.unmask(masked_id)
                            if real_name and real_name != masked_id:
                                gene_mapping[masked_id] = real_name

            logging.info(f"Gene mapping built: {len(gene_mapping)} genes")

            # Get ground truth
            if not hasattr(self, 'ground_truth_tiers') or not self.ground_truth_tiers:
                self.load_ground_truth()

            ground_truth_genes = list(self.ground_truth_tiers.get('tier1', set()) |
                                     self.ground_truth_tiers.get('tier2', set()) |
                                     self.ground_truth_tiers.get('tier3', set()))

            # Get analysis context if not provided
            if analysis_context is None:
                analysis_context = self.format_for_generation_agent(max_genes=30)

            if literature_context is None:
                literature_context = "T2D drug target discovery literature context."

            # Expand predictions to include all ranked targets
            expanded_predictions = self._expand_ranked_predictions(predictions)

            # Run full evaluation
            results = evaluator.run_full_evaluation(
                framework_predictions=expanded_predictions,
                analysis_context=analysis_context,
                literature_context=literature_context,
                gene_mapping=gene_mapping,
                ground_truth_genes=ground_truth_genes,
                tier_genes=self.ground_truth_tiers,
                run_baseline=run_baseline
            )

            # Store evaluation results
            self.full_evaluation_results = results

            # Print report
            if 'evaluation_report' in results:
                logging.info(f"\n{results['evaluation_report']}")

            return results

        except ImportError as e:
            logging.error(f"Could not import evaluation metrics: {e}")
            return {'error': str(e)}
        except Exception as e:
            logging.error(f"Full evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def _expand_ranked_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """
        Expand predictions to create individual entries for each ranked target.

        This allows evaluation metrics to assess all ranked genes, not just
        the primary target.

        Args:
            predictions: List of hypothesis dicts with ranked_targets

        Returns:
            Expanded list where each ranked target is a separate entry
        """
        expanded = []

        for pred in predictions:
            ranked_targets = pred.get('ranked_targets', [])

            if ranked_targets:
                # Create entry for each ranked target
                for target in ranked_targets:
                    expanded_pred = {
                        'target_gene_masked': target.get('gene_id'),
                        'rank': target.get('rank', 99),
                        'priority_score': target.get('priority_score'),
                        'fitness_score': pred.get('fitness_score', 0) * (1 - (target.get('rank', 1) - 1) * 0.1),  # Decay by rank
                        'source_hypothesis_id': pred.get('id'),
                        'rationale': target.get('rationale', ''),
                        'druggability': target.get('druggability')
                    }
                    expanded.append(expanded_pred)
            else:
                # Legacy single-target format
                expanded.append({
                    'target_gene_masked': pred.get('target_gene_masked'),
                    'rank': 1,
                    'fitness_score': pred.get('fitness_score', 0),
                    'source_hypothesis_id': pred.get('id')
                })

        # Sort by fitness score (best first)
        expanded.sort(key=lambda x: x.get('fitness_score', 0), reverse=True)

        logging.info(f"Expanded {len(predictions)} hypotheses to {len(expanded)} ranked predictions")

        return expanded

    def get_all_predicted_genes(self, predictions: List[Dict]) -> List[str]:
        """
        Get all unique gene IDs from predictions (including ranked targets).

        Args:
            predictions: List of hypothesis dicts

        Returns:
            List of unique masked gene IDs
        """
        genes = set()

        for pred in predictions:
            # Primary target
            if pred.get('target_gene_masked'):
                genes.add(pred['target_gene_masked'])

            # Ranked targets
            for target in pred.get('ranked_targets', []):
                if target.get('gene_id'):
                    genes.add(target['gene_id'])

        return list(genes)

    def generate_evaluation_summary(self, evaluation_results: Dict) -> str:
        """
        Generate a concise evaluation summary for logging/display.

        Args:
            evaluation_results: Results from run_full_evaluation()

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("T2D EVALUATION SUMMARY")
        lines.append("=" * 60)

        # Correlation summary
        corr = evaluation_results.get('opentargets_correlation', {})
        if 'error' not in corr:
            lines.append(f"OpenTargets Correlation: r={corr.get('spearman_correlation', 'N/A')}")
        else:
            lines.append(f"OpenTargets Correlation: {corr.get('error', 'N/A')}")

        # Baseline comparison summary
        baseline = evaluation_results.get('baseline_comparison', {})
        if baseline and 'summary' in baseline:
            summary = baseline['summary']
            lines.append(f"Framework vs Baseline: {summary.get('hit_improvement', 0):+d} hit advantage")
            lines.append(f"  Framework hits: {summary.get('framework_total_hits', 0)}")
            lines.append(f"  Baseline hits: {summary.get('baseline_total_hits', 0)}")

        # @K metrics summary
        at_k = evaluation_results.get('at_k_metrics', {})
        if 'error' not in at_k:
            lines.append(f"MRR: {at_k.get('mrr', 0):.4f}")
            lines.append(f"MAP: {at_k.get('map', 0):.4f}")
            lines.append(f"First hit at rank: {at_k.get('first_hit_rank', 'None')}")

            # Tier hits
            tier_metrics = at_k.get('tier_metrics', {})
            for tier, tier_data in tier_metrics.items():
                lines.append(f"  {tier.upper()}: {tier_data.get('hits', 0)}/{tier_data.get('total_in_tier', 0)} hits")

        lines.append("=" * 60)

        return "\n".join(lines)