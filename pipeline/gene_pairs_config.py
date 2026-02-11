"""
Gene Pairs Configuration for Lethal Genes Mode

This file contains the gene pairs from the CRISPR dual knockout screen
for synthetic lethality analysis.
"""

# Active gene pairs for processing (starting with first 5)
ACTIVE_GENE_PAIRS = [
    ["KDM5C", "AURKA"],
    ["CLIC5", "FMR1"],
    ["MSMB", "PPP1R21"],
    ["NDUFB8", "SMS"],
    ["UHRF1", "PARP1"],
]

# Remaining gene pairs (commented out for now)
REMAINING_GENE_PAIRS = [
    # ["CASP8", "BRD4"],
    # ["TET2", "BRD4"],
    # ["MYC", "MRE11"],
    # ["FBXO44", "LITAF"],
    # ["MYC", "HDAC2"],
    # ["CASP8", "ERBB2"],
    # ["KMT2D", "PLK1"],
    # ["SYNE3", "ZBTB33"],
    # ["MUTYH", "EGFR"],
    # ["RAD52", "EGFR"],
    # ["PIK3CA", "AKT1"],
    # ["NTNG2", "SART1"],
    # ["IGFALS", "TRIM28"],
    # ["TPCN2", "ZFYVE9"],
    # ["CREBBP", "FLT3"],
    # ["MYC", "HDAC1"],
    # ["TNKS2", "CHEK1"],
    # ["PARP6", "EGFR"],
    # ["KLHL31", "OR6K2"],
    # ["CFAP45", "COMMD10"],
    # ["DGKZ", "CHEK1"],
    # ["ATAD1", "CALHM6"],
    # ["ASB17", "VGLL4"],
    # ["UHRF1", "TP53"],
    # ["AP1M2", "YES1"],
    # ["MYC", "SMARCA4"],
    # ["PCDH19", "SPOCK1"],
    # ["FAT1", "PARP2"],
    # ["MYC", "BRD9"],
    # ["ARID1A", "AKT1"],
    # ["CASR", "PIGG"],
    # ["IGF1R", "SPTBN1"],
    # ["ADA", "EGFR"],
    # ["ATP12A", "PACSIN2"],
    # ["MSH2", "BRD4"],
    # ["UHRF1", "SMARCA4"],
    # ["ERCC4", "SMARCA4"],
    # ["MIPOL1", "VASP"],
    # ["MYC", "CDK1"],
    # ["MYC", "TNKS2"],
    # ["CRYGD", "PPP1R1B"],
    # ["MYC", "AKT1"],
    # ["BRD9", "CHEK1"],
    # ["CDR2L", "DHFR"],
    # ["CHEK1", "AKT1"]
]

# Total number of active pairs
NUM_ACTIVE_PAIRS = len(ACTIVE_GENE_PAIRS)
