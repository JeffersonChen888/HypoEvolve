"""
Configuration for T2D Drug Target Identification Pipeline
"""
import os

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
# Note: All h5ad files have been preprocessed with gene symbol annotations
# and disease group labels

T2D_DATASETS = {
    "GSE76894": {
        "tissue": "pancreatic_islets",
        "platform": "Affymetrix HG-U133_Plus_2",
        "n_samples": 103,
        "n_genes": 26589,  # probes
        "group_col": "disease",
        "control_name": "Control",
        "treatment_name": "T2D",
        "description": "IMIDIA biobank - Organ donor cohort, 84 ND + 19 T2D"
    },
    "GSE76895": {
        "tissue": "pancreatic_islets",
        "platform": "Affymetrix HG-U133_Plus_2",
        "n_samples": 103,
        "n_genes": 26359,  # probes
        "group_col": "disease",
        "control_name": "Control",
        "treatment_name": "T2D",
        "description": "IMIDIA biobank - Partially pancreatectomized patients, 32 ND + 36 T2D + 15 IGT + 20 T3cD"
    },
    "GSE86468": {
        "tissue": "pancreatic_islets",
        "platform": "Illumina NextSeq 500",
        "n_samples": 24,
        "n_genes": 21909,
        "group_col": "disease",
        "control_name": "Control",
        "treatment_name": "T2D",
        "description": "Single cell transcriptomics bulk islet samples from ND and T2D donors"
    },
    "GSE221156": {
        "tissue": "pancreatic_islets",
        "platform": "10x Genomics scRNA-seq",
        "n_samples": 31,  # 14 T2D + 17 ND donors (after excluding Mixed pooled samples)
        "n_cells": 219351,  # After filtering to T2D + Control
        "n_genes": 36601,
        "group_col": "disease",
        "control_name": "Control",
        "treatment_name": "T2D",
        "description": "Single-cell atlas of pancreatic islets, 219K cells from 31 donors (14 T2D + 17 ND)"
    },
    "GSE221156_pseudobulk": {
        "tissue": "pancreatic_islets",
        "platform": "10x Genomics scRNA-seq (pseudobulk)",
        "n_samples": 31,  # 14 T2D + 17 ND donors
        "n_genes": 36601,
        "group_col": "disease",
        "control_name": "Control",
        "treatment_name": "T2D",
        "description": "Pseudobulk aggregation of GSE221156, mean expression per donor (14 T2D + 17 ND)"
    },
}

# =============================================================================
# DEFAULT DATASETS
# =============================================================================
# All three datasets are high quality human pancreatic islet studies

DEFAULT_T2D_DATASETS = [
    "GSE76894",   # Organ donors (IMIDIA)
    "GSE76895",   # Pancreatectomy patients (IMIDIA)
    "GSE86468",   # RNA-seq bulk islets
    "GSE221156_pseudobulk",  # scRNA-seq pseudobulk (fast, 43 donors)
    # "GSE221156",  # Full scRNA-seq (slow, 307K cells) - uncomment for full analysis
]

# Microarray datasets only
MICROARRAY_DATASETS = [
    "GSE76894",
    "GSE76895",
]

# RNA-seq datasets only
RNASEQ_DATASETS = [
    "GSE86468",
]

# Single-cell RNA-seq datasets
SCRNA_DATASETS = [
    "GSE221156_pseudobulk",  # Use pseudobulk for pipeline (faster)
    # "GSE221156",  # Full scRNA-seq - too slow for pipeline, use for custom analysis
]

ALL_ISLET_DATASETS = [
    "GSE76894",
    "GSE76895",
    "GSE86468",
    "GSE221156_pseudobulk",  # Use pseudobulk instead of full scRNA-seq
]

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

T2D_ANALYSIS_PARAMS = {
    # Differential expression
    "de_method": "wilcoxon",  # 'wilcoxon' or 't-test'
    "de_n_top_genes": 500,
    "de_pval_cutoff": 0.05,
    "de_log2fc_cutoff": 0.5,

    # GSEA
    "gsea_gene_sets": ["KEGG_2021_Human", "GO_Biological_Process_2023"],
    "gsea_min_size": 10,
    "gsea_max_size": 500,

    # WGCNA
    "wgcna_min_module_size": 20,
    "wgcna_merge_cut_height": 0.25,
    "wgcna_soft_power": None,  # Auto-detect if None

    # TF Activity (using decoupler with CollecTRI)
    "tf_organism": "human",
    "tf_min_targets": 5,

    # Expression variability
    "variability_method": "cv",  # coefficient of variation

    # Cross-dataset integration
    "integration_method": "fisher",  # Fisher's combined p-value
    "min_datasets_agreement": 2,  # Gene must be significant in at least N datasets

    # Priority scoring thresholds
    "priority_score_threshold": 8.0,  # Out of 17 max
    "top_candidates_for_llm": 1000,
}

# =============================================================================
# GENETIC ALGORITHM CONFIGURATION
# =============================================================================

GENETIC_ALGORITHM_CONFIG = {
    "population_size": 20,       # Larger population for better diversity
    "num_generations": 10,       # Default generations (benchmark overrides this)
    "selection_ratio": 0.5,      # Top 40% selected as parents
    "elitism_count": 2,          # Keep top 2 unchanged
    "mutation_rate": 0.4,        # 10% chance of random mutation
    "crossover_rate": 0.6,       # 80% chance of crossover
}

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Get the directory where this config file is located
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directory (relative to pipeline folder)
T2D_DATA_DIR = os.getenv("T2D_DATA_DIR", os.path.join(_CONFIG_DIR, "data", "t2d"))
T2D_OUTPUT_DIR = os.getenv("T2D_OUTPUT_DIR", os.path.join(_CONFIG_DIR, "output", "t2d_target"))


# OpenTargets Local Data Path
OPENTARGETS_LOCAL_PATH = os.path.join(os.path.dirname(T2D_DATA_DIR), "OT-MONDO_0005148-associated-targets-2026_2_10-v25_12.tsv")

# =============================================================================
# GENE MASKING CONFIGURATION
# =============================================================================

GENE_MASKING_SEED = 42
GENE_MASK_PREFIX = "G"  # Genes will be masked as G00001, G00002, etc.

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_dataset_path(dataset_id: str) -> str:
    """Get the full path to a dataset h5ad file."""
    return os.path.join(T2D_DATA_DIR, f"{dataset_id}.h5ad")

def get_available_datasets() -> list:
    """Get list of datasets that have h5ad files available."""
    available = []
    for dataset_id in T2D_DATASETS.keys():
        if os.path.exists(get_dataset_path(dataset_id)):
            available.append(dataset_id)
    return available

def get_dataset_info(dataset_id: str) -> dict:
    """Get configuration info for a specific dataset."""
    if dataset_id not in T2D_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_id}. Available: {list(T2D_DATASETS.keys())}")
    return T2D_DATASETS[dataset_id]


# =============================================================================
# DRUGGABILITY CONFIGURATION (Option A)
# =============================================================================

# Feature set for druggability extraction using IDG standard classification
# Reference: https://druggablegenome.net/IDGProteinFamilies
DRUGGABILITY_CONFIG = {
    # API settings (Pharos primary, OpenTargets fallback)
    "pharos_endpoint": "https://pharos-api.ncats.io/graphql",
    "opentargets_endpoint": "https://api.platform.opentargets.org/api/v4/graphql",
    "rate_limit_ms": 100,  # Minimum ms between API requests

    # Safe features that don't leak gene identity (IDG standard)
    # Updated to use idg_family instead of protein_class
    # Removed pathway_role and pathway_position (could leak with family)
    "safe_features": [
        "idg_family",           # IDG family: GPCR, Kinase, IC, NR, TF, Enzyme, Transporter, Epigenetic
        "subcellular_location", # Membrane, Nucleus, Cytoplasm, Secreted (shared by many genes)
        "tractability",         # SmallMol, Ligand, Pocket (coarse-grained, shared by many)
    ],

    # Forbidden features - NEVER use these (leak identity)
    "forbidden_features": [
        "protein_subfamily",         # e.g., "SLC transporter" - too specific
        "pathway_role",              # Removed - too identifying with family
        "pathway_position",          # Removed - too identifying with family
        "tractability_sm",           # Too specific - REMOVED
        "tractability_ab",           # Too specific - REMOVED
        "n_pathways",                # Could narrow down identity - REMOVED
        "has_approved_drug",         # Would identify known drug targets
        "drug_names",                # Obviously identifies the gene
        "disease_association_score", # Too specific to individual genes
        "genetic_associations",      # Would reveal known T2D genes
        "literature_count",          # Unique to each gene
        "tdl",                       # Target Development Level - Tclin reveals drug targets
    ],

    # Leakage testing thresholds
    "leakage_threshold": 0.05,  # Max 5% LLM identification accuracy
    "uniqueness_threshold": 0.5,  # Single feature uniqueness score threshold
    "singleton_threshold": 0.1,   # Max ratio of singleton values
}

# Flag to enable/disable druggability features (Option A vs Option C)
USE_DRUGGABILITY = True  # Set to False for Option C (expression-only)

# Ground truth configuration
GROUND_TRUTH_CONFIG = {
    "t2d_disease_id": "MONDO_0005148",
    # "related_disease_ids": ["EFO_0001360", "MONDO_0005015"],
    "include_related": False,
}