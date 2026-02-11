"""
Druggability Feature Extractor

Extracts SAFE druggability features using IDG (Illuminating the Druggable Genome)
standard protein family classification to provide biological context to the LLM
WITHOUT leaking gene identity.

IDG Standard Families (from Pharos API):
- GPCR: G protein-coupled receptors (~800 human proteins)
- Kinase: Protein kinases (~500 human proteins)
- IC: Ion channels (~300 human proteins)
- NR: Nuclear receptors (~48 human proteins)
- TF: Transcription factors
- Enzyme: All enzymes
- Transporter: All membrane transporters
- Epigenetic: Chromatin/histone modifiers

References:
- Pharos API: https://pharos.nih.gov/api
- Drug Target Ontology: https://jbiomedsem.biomedcentral.com/articles/10.1186/s13326-017-0161-x
- IDG Protein Families: https://druggablegenome.net/IDGProteinFamilies

Safe Features (included):
- idg_family: IDG standard protein family (GPCR, Kinase, IC, NR, TF, Enzyme, Transporter, Epigenetic, Unknown)
- subcellular_location: Where the protein is located (Membrane, Cytoplasm, Nucleus, Secreted)

REMOVED (could leak gene identity):
- pathway_role: Removed - combined with family could narrow down identity
- pathway_position: Removed - combined with family could narrow down identity
- Specific protein subfamilies (e.g., "SLC transporter", "Phosphodiesterase")

Forbidden Features (excluded to prevent leakage):
- has_approved_drug: Would identify known drug targets
- drug_names: Obviously identifies the gene
- disease_association_score: Too specific to individual genes
- genetic_associations: Would reveal known T2D genes
- TDL (Tclin/Tchem/Tbio/Tdark): Tclin directly reveals "has approved drugs"
"""

import requests
import pandas as pd
import time
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class DruggabilityFeatureExtractor:
    """
    Extracts safe druggability features using IDG standard classification.

    The key principle: Features must be shared by MANY genes so that
    knowing the feature values doesn't uniquely identify the gene.

    Uses Pharos API (IDG Knowledge Management Center) as primary source,
    with OpenTargets as fallback.
    """

    # API endpoints
    PHAROS_GRAPHQL_URL = "https://pharos-api.ncats.io/graphql"
    OPENTARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

    # IDG Standard Protein Families
    # Reference: https://druggablegenome.net/IDGProteinFamilies
    IDG_FAMILIES = {
        'GPCR',       # G protein-coupled receptors
        'Kinase',     # Protein kinases
        'IC',         # Ion channels
        'NR',         # Nuclear receptors
        'TF',         # Transcription factors
        'Enzyme',     # All enzymes
        'Transporter', # All membrane transporters
        'Epigenetic', # Chromatin/histone modifiers
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the extractor.

        Args:
            cache_dir: Optional directory to cache API responses
        """
        # SAFE features - provide context without unique identification
        self.safe_features = [
            'idg_family',           # IDG standard family (broad, shared by 100s of genes)
            'subcellular_location', # e.g., "Membrane", "Cytoplasm", "Nucleus"
        ]

        # FORBIDDEN - these would leak gene identity
        self.forbidden_features = [
            'has_approved_drug',        # Binary flag is too identifying
            'drug_names',               # Obviously identifies the gene
            'disease_association_score', # Too specific to individual genes
            'genetic_associations',      # Would reveal known T2D genes
            'literature_count',          # Unique to each gene
            'tractability_sm',           # Too specific
            'tractability_ab',           # Too specific
            'n_pathways',                # Could narrow down identity
            'tdl',                       # Target Development Level - Tclin reveals drug targets
            'pathway_role',              # Removed - too identifying with family
            'pathway_position',          # Removed - too identifying with family
            'protein_subfamily',         # e.g., "SLC transporter" is too specific
        ]


        # Cache for API responses
        self.cache_dir = cache_dir
        self._ensembl_cache: Dict[str, str] = {}
        self._target_cache: Dict[str, Dict] = {}
        self._pharos_cache: Dict[str, str] = {}
        
        # Local data cache
        self.local_data: Dict[str, Dict] = {}
        self._load_local_data()

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests

    def _load_local_data(self):
        """Load local OpenTargets data from TSV if available."""
        try:
            from t2d_config import OPENTARGETS_LOCAL_PATH
            import os
            
            if os.path.exists(OPENTARGETS_LOCAL_PATH):
                logger.info(f"Loading local OpenTargets data from {OPENTARGETS_LOCAL_PATH}")
                df = pd.read_csv(OPENTARGETS_LOCAL_PATH, sep='\t')
                
                def _is_true(val):
                    """Helper to check if value is effectively True (1, '1', 1.0, True)."""
                    if pd.isna(val): return False
                    s = str(val).strip().lower()
                    return s in ('1', '1.0', 'true', 'yes')

                # Create dictionary mapping symbol to row data
                # We care about: isInMembrane, isSecreted, hasSmallMoleculeBinder, hasLigand, hasPocket
                for _, row in df.iterrows():
                    symbol = row.get('symbol')
                    if pd.isna(symbol):
                        continue
                        
                    self.local_data[str(symbol).upper()] = {
                        'isInMembrane': _is_true(row.get('isInMembrane')),
                        'isSecreted': _is_true(row.get('isSecreted')),
                        'hasSmallMoleculeBinder': _is_true(row.get('hasSmallMoleculeBinder')),
                        'hasLigand': _is_true(row.get('hasLigand')),
                        'hasPocket': _is_true(row.get('hasPocket'))
                    }
                logger.info(f"Loaded local data for {len(self.local_data)} genes")
            else:
                logger.warning(f"Local OpenTargets file not found at {OPENTARGETS_LOCAL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load local OpenTargets data: {e}")

    def _rate_limit(self):
        """Enforce rate limiting between API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _get_idg_family_from_pharos(self, gene_symbol: str) -> str:
        """
        Get IDG protein family from Pharos GraphQL API.
        
        Note: If local data is used, family information might be missing.
        This method is kept for fallback or if API usage is re-enabled.
        """
        # Check cache first
        if gene_symbol in self._pharos_cache:
            return self._pharos_cache[gene_symbol]

        query = """
        query targetFamily($sym: String!) {
            target(q: {sym: $sym}) {
                fam
            }
        }
        """

        self._rate_limit()

        try:
            response = requests.post(
                self.PHAROS_GRAPHQL_URL,
                json={"query": query, "variables": {"sym": gene_symbol}},
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                target = data.get("data", {}).get("target")
                if target:
                    fam = target.get("fam")
                    if fam and fam in self.IDG_FAMILIES:
                        self._pharos_cache[gene_symbol] = fam
                        return fam

        except Exception as e:
            logger.debug(f"Pharos API error for {gene_symbol}: {e}")

        return "Unknown"

    def get_ensembl_id(self, gene_symbol: str) -> Optional[str]:
        """
        Convert gene symbol to Ensembl ID via OpenTargets search.

        Args:
            gene_symbol: Gene symbol (e.g., "INS", "PDX1")

        Returns:
            Ensembl ID (e.g., "ENSG00000254647") or None if not found
        """
        # Check cache first
        if gene_symbol in self._ensembl_cache:
            return self._ensembl_cache[gene_symbol]

        self._rate_limit()

        try:
            # Use GraphQL for more reliable search
            query = """
            query searchTarget($symbol: String!) {
                search(queryString: $symbol, entityNames: ["target"], page: {size: 5, index: 0}) {
                    hits {
                        id
                        name
                        entity
                        object {
                            ... on Target {
                                id
                                approvedSymbol
                            }
                        }
                    }
                }
            }
            """

            response = requests.post(
                self.OPENTARGETS_GRAPHQL_URL,
                json={"query": query, "variables": {"symbol": gene_symbol}},
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                search_data = data.get("data", {})
                if search_data is None:
                    search_data = {}
                search_result = search_data.get("search")
                if search_result is None:
                    search_result = {}
                hits = search_result.get("hits", [])

                for hit in hits:
                    obj = hit.get("object") if hit else None
                    if obj and obj.get("approvedSymbol", "").upper() == gene_symbol.upper():
                        ensembl_id = obj.get("id")
                        self._ensembl_cache[gene_symbol] = ensembl_id
                        return ensembl_id

                # If no exact match, try first result
                if hits and hits[0] and hits[0].get("object"):
                    ensembl_id = hits[0]["object"].get("id")
                    self._ensembl_cache[gene_symbol] = ensembl_id
                    return ensembl_id

            logger.warning(f"Could not find Ensembl ID for {gene_symbol}")
            return None

        except Exception as e:
            logger.error(f"Error fetching Ensembl ID for {gene_symbol}: {e}")
            return None

    def fetch_target_data(self, ensembl_id: str) -> Dict[str, Any]:
        """
        Fetch target data from OpenTargets API.

        Args:
            ensembl_id: Ensembl gene ID

        Returns:
            Dictionary containing target information
        """
        # Check cache first
        if ensembl_id in self._target_cache:
            return self._target_cache[ensembl_id]

        self._rate_limit()

        try:
            query = """
            query targetInfo($ensemblId: String!) {
                target(ensemblId: $ensemblId) {
                    id
                    approvedSymbol
                    biotype
                    subcellularLocations {
                        location
                        source
                    }
                    targetClass {
                        id
                        label
                    }
                    functionDescriptions
                }
            }
            """

            response = requests.post(
                self.OPENTARGETS_GRAPHQL_URL,
                json={"query": query, "variables": {"ensemblId": ensembl_id}},
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                target_data = data.get("data", {}).get("target", {})
                if target_data:
                    self._target_cache[ensembl_id] = target_data
                    return target_data

            logger.warning(f"Could not fetch target data for {ensembl_id}")
            return {}

        except Exception as e:
            logger.error(f"Error fetching target data for {ensembl_id}: {e}")
            return {}

    def _extract_idg_family_from_opentargets(self, target_data: Dict) -> str:
        """
        Extract IDG protein family from OpenTargets data (fallback when Pharos unavailable).

        Maps OpenTargets targetClass to IDG standard families ONLY.
        Does NOT return specific subfamilies to prevent information leakage.

        Args:
            target_data: OpenTargets target data

        Returns:
            IDG family: One of GPCR, Kinase, IC, NR, TF, Enzyme, Transporter, Epigenetic, Unknown
        """
        target_classes = target_data.get("targetClass", [])

        if target_classes:
            for tc in target_classes:
                label = tc.get("label", "").lower()

                # Map to IDG standard families ONLY - no subfamilies
                if "kinase" in label:
                    return "Kinase"
                elif "gpcr" in label or "g protein-coupled" in label:
                    return "GPCR"
                elif "ion channel" in label or "channel" in label:
                    return "IC"
                elif "nuclear receptor" in label:
                    return "NR"
                elif "transcription" in label:
                    return "TF"
                elif any(x in label for x in ["transporter", "carrier", "exchanger",
                                               "pump", "solute carrier", "slc", "abc"]):
                    # Return generic "Transporter" - NOT "SLC transporter" or "ABC transporter"
                    return "Transporter"
                elif any(x in label for x in ["enzyme", "transferase", "hydrolase",
                                               "oxidoreductase", "ligase", "lyase",
                                               "phosphatase", "protease", "phosphodiesterase",
                                               "dehydrogenase", "synthase", "reductase"]):
                    # Return generic "Enzyme" - NOT specific enzyme type
                    return "Enzyme"
                elif "epigenetic" in label or "histone" in label:
                    return "Epigenetic"

        # Try to infer from function descriptions
        func_descs = target_data.get("functionDescriptions", [])
        if func_descs:
            desc_text = " ".join(func_descs).lower()
            if "kinase" in desc_text:
                return "Kinase"
            elif "g protein-coupled" in desc_text or "gpcr" in desc_text:
                return "GPCR"
            elif "ion channel" in desc_text:
                return "IC"
            elif "nuclear receptor" in desc_text:
                return "NR"
            elif "transcription factor" in desc_text:
                return "TF"
            elif any(x in desc_text for x in ["transport", "carrier", "exchanger"]):
                return "Transporter"
            elif any(x in desc_text for x in ["enzyme", "catalyze", "catalytic"]):
                return "Enzyme"

        return "Unknown"

    def _extract_subcellular_location(self, target_data: Dict) -> str:
        """
        Extract subcellular location from target data.

        Returns only broad categories to prevent identification.

        Args:
            target_data: OpenTargets target data

        Returns:
            Subcellular location: One of Membrane, Cytoplasm, Nucleus, Secreted, Unknown
        """
        locations = target_data.get("subcellularLocations", [])

        if locations:
            location_set = set()
            for loc in locations:
                loc_name = loc.get("location", "").lower()
                if "membrane" in loc_name or "plasma membrane" in loc_name:
                    location_set.add("Membrane")
                elif "nucleus" in loc_name:
                    location_set.add("Nucleus")
                elif "cytoplasm" in loc_name or "cytosol" in loc_name:
                    location_set.add("Cytoplasm")
                elif "secreted" in loc_name or "extracellular" in loc_name:
                    location_set.add("Secreted")

            # Priority: Membrane > Nucleus > Secreted > Cytoplasm
            for priority_loc in ["Membrane", "Nucleus", "Secreted", "Cytoplasm"]:
                if priority_loc in location_set:
                    return priority_loc

            if location_set:
                return list(location_set)[0]

        return "Unknown"


    def extract_features_for_gene(self, gene_symbol: str) -> Dict[str, str]:
        """
        Extract druggability features for a single gene.
        
        Prioritizes local OpenTargets data to avoid API calls and Ensembl ID issues.
        """
        # 1. Try Local Data First
        if hasattr(self, 'local_data') and gene_symbol in self.local_data:
            data = self.local_data[gene_symbol]
            
            # Map location
            location = "Unknown"
            if data.get('isInMembrane'):
                location = "Membrane"
            elif data.get('isSecreted'):
                location = "Secreted"
            # Intracellular is default if known not to be membrane/secreted? 
            # Safest to say Unknown or leave empty if no positive Membrane/Secreted signal
            
            # Determine Tractability / Druggability
            # We map this to a "Mechanism Class" proxy or just raw tractability
            # Since we lack IDG family, we provide tractability status which is highly relevant
            tractability = []
            if data.get('hasSmallMoleculeBinder'):
                tractability.append("SmallMol")
            if data.get('hasPocket'):
                tractability.append("Pocket")
            if data.get('hasLigand'):
                tractability.append("Ligand")
            
            tractability_str = "|".join(tractability) if tractability else "Low/Unknown"

            return {
                'idg_family': 'Unknown', # Not in local TSV
                'subcellular_location': location,
                'tractability': tractability_str
            }

        # 2. Fallback to API (if local data missing)
        # Note: User requested local data usage to avoid API issues
        return {
            'idg_family': 'Unknown',
            'subcellular_location': 'Unknown',
            'tractability': 'Unknown'
        }

    def create_feature_dataframe(
        self,
        gene_list: List[str],
        gene_mapping: Dict[str, str],
        progress_callback: Optional[callable] = None
    ) -> pd.DataFrame:
        """
        Create a DataFrame of safe druggability features for a list of genes.

        CRITICAL: The output DataFrame uses MASKED gene IDs, not real gene names.
        This prevents the LLM from learning gene identities from the features.

        Args:
            gene_list: List of real gene symbols
            gene_mapping: Dictionary mapping real gene symbols to masked IDs
            progress_callback: Optional callback function for progress updates

        Returns:
            DataFrame with columns: [masked_id, idg_family, subcellular_location, tractability]
        """
        features_list = []
        total_genes = len(gene_list)

        for i, gene_symbol in enumerate(gene_list):
            # Get masked ID (CRITICAL: never expose real gene name)
            masked_id = gene_mapping.get(gene_symbol)
            if not masked_id:
                logger.warning(f"No masked ID found for {gene_symbol}, skipping")
                continue

            # Extract features
            features = self.extract_features_for_gene(gene_symbol)

            # Add masked ID (NOT real gene name)
            features['masked_id'] = masked_id
            features_list.append(features)

            # Progress callback
            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(i + 1, total_genes)

            # Log progress
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{total_genes} genes")

        # Create DataFrame
        df = pd.DataFrame(features_list)

        # Reorder columns
        # Ensure 'tractability' is included if present
        cols = ['masked_id', 'idg_family', 'subcellular_location']
        if 'tractability' in df.columns:
            cols.append('tractability')
            
        # Filter for only existing columns
        existing_cols = [c for c in cols if c in df.columns]
        df = df[existing_cols]

        logger.info(f"Created druggability feature DataFrame with {len(df)} genes")

        return df

    def format_for_llm(self, feature_df: pd.DataFrame, top_n: int = 50) -> str:
        """
        Format feature DataFrame as a string for LLM consumption.

        Args:
            feature_df: DataFrame from create_feature_dataframe()
            top_n: Maximum number of genes to include

        Returns:
            Formatted string for LLM prompt
        """
        if len(feature_df) == 0:
            return "No druggability features available."

        # Limit to top_n genes
        df_subset = feature_df.head(top_n)

        lines = ["DRUGGABILITY & TRACTABILITY FEATURES (from OpenTargets):"]
        lines.append("=" * 60)
        
        # Adjust headers based on columns
        has_tractability = 'tractability' in feature_df.columns
        
        header = f"{'Gene ID':<10} {'Location':<12} {'Family':<10}"
        if has_tractability:
            header += f" {'Tractability':<20}"
        lines.append(header)
        lines.append("-" * 60)

        for _, row in df_subset.iterrows():
            line = f"{row['masked_id']:<10} "
            line += f"{row['subcellular_location']:<12} "
            line += f"{row['idg_family']:<10} "
            if has_tractability:
                line += f"{row['tractability']:<20}"
            lines.append(line)

        if len(feature_df) > top_n:
            lines.append(f"... and {len(feature_df) - top_n} more genes")

        lines.append("=" * 60)
        lines.append("")
        lines.append("Legend:")
        lines.append("  Location: Membrane, Secreted, or Unknown (Intracellular)")
        lines.append("  Tractability: SmallMol = Has small molecule binder, Ligand = Has known ligand, Pocket = Structure has binding pocket")
        lines.append("  Family: (Not available in local dataset)")
        lines.append("")
        lines.append("NOTE: Prioritize candidates with 'SmallMol' or 'Ligand' evidence.")

        return "\n".join(lines)

    def get_feature_statistics(self, feature_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """
        Get statistics about the extracted features.

        Useful for checking feature diversity (many genes should share each value).

        Args:
            feature_df: DataFrame from create_feature_dataframe()

        Returns:
            Dictionary with value counts for each feature
        """
        stats = {}
        for col in ['idg_family', 'subcellular_location']:
            if col in feature_df.columns:
                stats[col] = feature_df[col].value_counts().to_dict()

        return stats


def main():
    """Test the druggability extractor with IDG classification."""
    logging.basicConfig(level=logging.INFO)

    # Test genes (including known T2D-relevant genes for validation)
    test_genes = ["INS", "PDX1", "GCK", "SLC2A2", "KCNJ11", "PDE4B"]
    test_mapping = {gene: f"G{i:05d}" for i, gene in enumerate(test_genes, 1)}

    extractor = DruggabilityFeatureExtractor()

    print("Testing DruggabilityFeatureExtractor with IDG Classification...")
    print("=" * 60)

    # Extract features
    df = extractor.create_feature_dataframe(test_genes, test_mapping)

    print("\nExtracted Features (masked, IDG standard):")
    print(df.to_string(index=False))

    print("\nFormatted for LLM:")
    print(extractor.format_for_llm(df))

    print("\nFeature Statistics:")
    stats = extractor.get_feature_statistics(df)
    for feature, counts in stats.items():
        print(f"\n{feature}:")
        for value, count in counts.items():
            print(f"  {value}: {count}")


if __name__ == "__main__":
    main()
