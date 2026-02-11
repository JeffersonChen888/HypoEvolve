"""
TCGA Cancer Types Configuration

This module contains the 33 TCGA cancer types and their abbreviations 
for the drug repurposing pipeline focus areas.

Note: CNTL (Controls) and MISC (Miscellaneous) are marked with * as they
are not actual cancer types but data categories in TCGA.
"""

# TCGA Cancer Types with abbreviations and full names
TCGA_CANCER_TYPES = {
    "LAML": "Acute Myeloid Leukemia",
    "ACC": "Adrenocortical carcinoma", 
    "BLCA": "Bladder Urothelial Carcinoma",
    "LGG": "Brain Lower Grade Glioma",
    "BRCA": "Breast invasive carcinoma",
    "CESC": "Cervical squamous cell carcinoma and endocervical adenocarcinoma",
    "CHOL": "Cholangiocarcinoma",
    "LCML": "Chronic Myelogenous Leukemia",
    "COAD": "Colon adenocarcinoma",
    "CNTL": "Controls",  # Not a cancer type - data category
    "ESCA": "Esophageal carcinoma",
    "FPPP": "FFPE Pilot Phase II",  # Not a cancer type - data category  
    "GBM": "Glioblastoma multiforme",
    "HNSC": "Head and Neck squamous cell carcinoma",
    "KICH": "Kidney Chromophobe",
    "KIRC": "Kidney renal clear cell carcinoma",
    "KIRP": "Kidney renal papillary cell carcinoma", 
    "LIHC": "Liver hepatocellular carcinoma",
    "LUAD": "Lung adenocarcinoma",
    "LUSC": "Lung squamous cell carcinoma",
    "DLBC": "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma",
    "MESO": "Mesothelioma",
    "MISC": "Miscellaneous",  # Not a cancer type - data category
    "OV": "Ovarian serous cystadenocarcinoma", 
    "PAAD": "Pancreatic adenocarcinoma",
    "PCPG": "Pheochromocytoma and Paraganglioma",
    "PRAD": "Prostate adenocarcinoma",
    "READ": "Rectum adenocarcinoma",
    "SARC": "Sarcoma",
    "SKCM": "Skin Cutaneous Melanoma",
    "STAD": "Stomach adenocarcinoma",
    "TGCT": "Testicular Germ Cell Tumors",
    "THYM": "Thymoma", 
    "THCA": "Thyroid carcinoma",
    "UCS": "Uterine Carcinosarcoma",
    "UCEC": "Uterine Corpus Endometrial Carcinoma",
    "UVM": "Uveal Melanoma"
}

# Cancer types only (excluding controls and miscellaneous categories)
ACTUAL_CANCER_TYPES = {k: v for k, v in TCGA_CANCER_TYPES.items() 
                       if k not in ["CNTL", "MISC", "FPPP"]}

# Common alternative names and synonyms for cancer type matching
CANCER_TYPE_ALIASES = {
    # Leukemia variations
    "AML": "LAML",
    "acute myeloid leukemia": "LAML",
    "CML": "LCML", 
    "chronic myeloid leukemia": "LCML",
    
    # Lung cancer variations  
    "lung cancer": ["LUAD", "LUSC"],
    "lung adenocarcinoma": "LUAD",
    "lung squamous cell carcinoma": "LUSC",
    
    # Kidney cancer variations
    "kidney cancer": ["KICH", "KIRC", "KIRP"],
    "renal cell carcinoma": ["KIRC", "KIRP"],
    "clear cell carcinoma": "KIRC",
    
    # Breast cancer variations
    "breast cancer": "BRCA",
    "invasive breast carcinoma": "BRCA",
    
    # Brain cancer variations
    "brain cancer": ["LGG", "GBM"],
    "glioma": ["LGG", "GBM"],
    "glioblastoma": "GBM",
    
    # Colorectal variations
    "colorectal cancer": ["COAD", "READ"],
    "colon cancer": "COAD",
    "rectal cancer": "READ",
    
    # Skin cancer variations
    "melanoma": ["SKCM", "UVM"],
    "skin cancer": "SKCM",
    "cutaneous melanoma": "SKCM",
    
    # Gynecologic cancers
    "ovarian cancer": "OV", 
    "cervical cancer": "CESC",
    "endometrial cancer": "UCEC",
    "uterine cancer": ["UCS", "UCEC"],
    
    # Other common terms
    "prostate cancer": "PRAD",
    "pancreatic cancer": "PAAD", 
    "liver cancer": "LIHC",
    "stomach cancer": "STAD",
    "gastric cancer": "STAD",
    "bladder cancer": "BLCA",
    "head and neck cancer": "HNSC",
    "thyroid cancer": "THCA",
    "lymphoma": "DLBC"
}

def get_cancer_full_name(abbreviation: str) -> str:
    """Get full cancer name from TCGA abbreviation."""
    return TCGA_CANCER_TYPES.get(abbreviation.upper(), "Unknown cancer type")

def get_cancer_abbreviation(cancer_name: str) -> str:
    """Get TCGA abbreviation from cancer name."""
    cancer_name_lower = cancer_name.lower()
    
    # Check direct aliases first
    if cancer_name_lower in CANCER_TYPE_ALIASES:
        alias_result = CANCER_TYPE_ALIASES[cancer_name_lower]
        if isinstance(alias_result, list):
            return alias_result[0]  # Return first match for multi-matches
        return alias_result
    
    # Check if it matches any full name
    for abbrev, full_name in TCGA_CANCER_TYPES.items():
        if cancer_name_lower in full_name.lower():
            return abbrev
    
    return "UNKNOWN"

def is_valid_tcga_cancer(abbreviation: str) -> bool:
    """Check if abbreviation is a valid TCGA cancer type."""
    return abbreviation.upper() in ACTUAL_CANCER_TYPES

def get_all_cancer_abbreviations() -> list:
    """Get list of all cancer type abbreviations (excluding controls)."""
    return list(ACTUAL_CANCER_TYPES.keys())

def get_all_cancer_names() -> list:
    """Get list of all cancer type full names (excluding controls)."""
    return list(ACTUAL_CANCER_TYPES.values())

def get_cancer_type_mapping() -> dict:
    """Get the complete mapping of abbreviations to full names."""
    return ACTUAL_CANCER_TYPES.copy()