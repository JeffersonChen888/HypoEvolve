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
warnings.filterwarnings('ignore')

# =============================================================================
# GENE MAPPER CLASS
# =============================================================================

class GeneMapper:
    """
    Unified gene name masking system for all analyses.
    Creates consistent masked IDs across multiple datasets.
    
    Supports:
    - Single AnnData object
    - List of gene names
    - Multiple AnnData objects (unified mapping)
    - Dict of dataset_name -> AnnData (unified mapping with tracking)
    """
    
    def __init__(self, datasets, seed: int = 42):
        """
        Initialize with genes from one or more datasets to create global mapping.
        
        Args:
            datasets: One of:
                - AnnData object
                - List of gene names
                - List of AnnData objects
                - Dict[str, AnnData] mapping dataset names to AnnData objects
            seed: Random seed for reproducible shuffling
        """
        self.seed = seed
        self.dataset_gene_sets = {}  # Track which genes come from which dataset
        
        # Collect all unique genes
        all_genes = set()
        
        if isinstance(datasets, ad.AnnData):
            # Single AnnData
            all_genes = set(datasets.var_names)
            self.dataset_gene_sets['dataset'] = all_genes
            
        elif isinstance(datasets, dict):
            # Dict of dataset_name -> AnnData
            for name, adata in datasets.items():
                if isinstance(adata, ad.AnnData):
                    genes = set(adata.var_names)
                    self.dataset_gene_sets[name] = genes
                    all_genes.update(genes)
                else:
                    raise ValueError(f"Dataset '{name}' is not an AnnData object")
                    
        elif isinstance(datasets, list):
            if len(datasets) == 0:
                raise ValueError("Empty dataset list provided")
            
            if isinstance(datasets[0], ad.AnnData):
                # List of AnnData objects
                for i, adata in enumerate(datasets):
                    genes = set(adata.var_names)
                    self.dataset_gene_sets[f'dataset_{i}'] = genes
                    all_genes.update(genes)
            else:
                # List of gene names
                all_genes = set(datasets)
                self.dataset_gene_sets['dataset'] = all_genes
        else:
            raise ValueError("datasets must be AnnData, list of genes, list of AnnData, or dict of AnnData")
        
        # Sort for reproducibility, then shuffle
        all_genes_sorted = sorted(all_genes)
        
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(len(all_genes_sorted))
        
        # Create bidirectional mappings
        self.real_to_masked = {}
        self.masked_to_real = {}
        
        for new_idx, orig_idx in enumerate(shuffled_indices):
            real_name = all_genes_sorted[orig_idx]
            masked_name = f"G{new_idx + 1:05d}"  # G00001, G00002, etc.
            self.real_to_masked[real_name] = masked_name
            self.masked_to_real[masked_name] = real_name
        
        self.n_genes = len(all_genes)
        self.n_datasets = len(self.dataset_gene_sets)
        
        # Calculate overlap statistics
        if self.n_datasets > 1:
            self._calculate_overlap_stats()
    
    def _calculate_overlap_stats(self):
        """Calculate gene overlap statistics across datasets."""
        dataset_names = list(self.dataset_gene_sets.keys())
        
        # Genes in all datasets
        common_genes = set.intersection(*self.dataset_gene_sets.values())
        
        # Genes unique to each dataset
        unique_genes = {}
        for name, genes in self.dataset_gene_sets.items():
            others = set.union(*[g for n, g in self.dataset_gene_sets.items() if n != name])
            unique_genes[name] = genes - others
        
        self.overlap_stats = {
            'n_common_genes': len(common_genes),
            'n_unique_per_dataset': {k: len(v) for k, v in unique_genes.items()},
            'n_genes_per_dataset': {k: len(v) for k, v in self.dataset_gene_sets.items()}
        }
    
    def mask(self, names: Union[str, List[str]]) -> Union[str, List[str]]:
        """Convert real gene names to masked names."""
        if isinstance(names, str):
            return self.real_to_masked.get(names, names)
        return [self.real_to_masked.get(n, n) for n in names]
    
    def unmask(self, names: Union[str, List[str]]) -> Union[str, List[str]]:
        """Convert masked gene names to real names."""
        if isinstance(names, str):
            return self.masked_to_real.get(names, names)
        return [self.masked_to_real.get(n, n) for n in names]
    
    def mask_dataframe(self, df: pd.DataFrame, gene_col: str = 'gene') -> pd.DataFrame:
        """Mask gene column in a dataframe."""
        df = df.copy()
        df[gene_col] = self.mask(df[gene_col].tolist())
        return df
    
    def unmask_dataframe(self, df: pd.DataFrame, gene_col: str = 'gene') -> pd.DataFrame:
        """Unmask gene column in a dataframe."""
        df = df.copy()
        df[gene_col] = self.unmask(df[gene_col].tolist())
        return df
    
    def is_in_dataset(self, gene: str, dataset_name: str) -> bool:
        """Check if a gene exists in a specific dataset."""
        if dataset_name not in self.dataset_gene_sets:
            return False
        # Handle both masked and unmasked input
        real_gene = self.unmask(gene) if gene.startswith('G') else gene
        return real_gene in self.dataset_gene_sets[dataset_name]
    
    def get_dataset_genes(self, dataset_name: str, masked: bool = True) -> List[str]:
        """Get all genes for a specific dataset."""
        if dataset_name not in self.dataset_gene_sets:
            return []
        genes = list(self.dataset_gene_sets[dataset_name])
        if masked:
            return self.mask(genes)
        return genes
    
    def save(self, path: str):
        """Save mapping to JSON."""
        with open(path, 'w') as f:
            json.dump({
                'real_to_masked': self.real_to_masked,
                'masked_to_real': self.masked_to_real,
                'dataset_gene_sets': {k: list(v) for k, v in self.dataset_gene_sets.items()},
                'seed': self.seed
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'GeneMapper':
        """Load mapping from JSON."""
        mapper = cls.__new__(cls)
        with open(path, 'r') as f:
            data = json.load(f)
        mapper.real_to_masked = data['real_to_masked']
        mapper.masked_to_real = data['masked_to_real']
        mapper.dataset_gene_sets = {k: set(v) for k, v in data.get('dataset_gene_sets', {}).items()}
        mapper.seed = data.get('seed', 42)
        mapper.n_genes = len(mapper.real_to_masked)
        mapper.n_datasets = len(mapper.dataset_gene_sets)
        if mapper.n_datasets > 1:
            mapper._calculate_overlap_stats()
        return mapper
    
    def __repr__(self):
        return f"GeneMapper(n_genes={self.n_genes}, n_datasets={self.n_datasets})"

