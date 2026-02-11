
import sys
import os
import logging
import pandas as pd

# Add repo root to path
sys.path.append(os.path.join(os.getcwd(), 'pipeline'))

from utils.druggability_extractor import DruggabilityFeatureExtractor

def test_idg_classification():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Expanded test set (50 genes) covering all IDG families and edge cases
    test_genes = {
        # GPCRs
        "ADRB2": "GPCR", "GLP1R": "GPCR", "GIPR": "GPCR", "FFAR1": "GPCR", "MTNR1B": "GPCR",
        "GCGR": "GPCR", "MC4R": "GPCR", "HCRTR1": "GPCR", "DRD2": "GPCR", "OPRM1": "GPCR",

        # Kinases
        "GCK": "Kinase", "INSR": "Kinase", "AKT1": "Kinase", "MAPK1": "Kinase", "PRKAA1": "Kinase",
        "IGF1R": "Kinase", "JAK2": "Kinase", "GSK3B": "Kinase", "MTOR": "Kinase", "BRAF": "Kinase",

        # Ion Channels
        "KCNJ11": "IC", "ABCC8": "IC", "CACNA1C": "IC", "SCN5A": "IC", "KCNQ1": "IC",
        "CACNA1D": "IC", "KCNK16": "IC",

        # Nuclear Receptors
        "PPARG": "NR", "HNF4A": "NR", "NR3C1": "NR", "VDR": "NR", "RORA": "NR",

        # Transporters
        "SLC2A2": "Transporter", "SLC2A4": "Transporter", "SLC30A8": "Transporter", 
        "SLC5A2": "Transporter", "SLC12A3": "Transporter",

        # Transcription Factors
        "PDX1": "TF", "TCF7L2": "TF", "FOXO1": "TF", "HNF1A": "TF", "NEUROD1": "TF",
        "MAFA": "TF",

        # Enzymes
        "PDE4B": "Enzyme", "GAPDH": "Enzyme", "DPP4": "Enzyme", "MGAM": "Enzyme", "PYGM": "Enzyme",

        # Epigenetic
        "HDAC1": "Epigenetic", "SIRT1": "Epigenetic", "KDM6A": "Epigenetic",

        # Others / Unknown
        "INS": "Unknown", "IAPP": "Unknown", "LEP": "Unknown", "ADIPOQ": "Unknown"
    }
    
    extractor = DruggabilityFeatureExtractor()
    print("\n" + "="*60)
    print("VERIFYING IDG FAMILY CLASSIFICATION")
    print("="*60)
    print(f"{'Gene':<10} {'Expected':<12} {'Actual':<12} {'Status':<10}")
    print("-" * 60)
    
    correct = 0
    for gene, expected in test_genes.items():
        features = extractor.extract_features_for_gene(gene)
        actual = features.get('idg_family', 'Error')
        
        status = "✅ PASS" if actual == expected else "❌ FAIL"
        if actual == expected:
            correct += 1
            
        print(f"{gene:<10} {expected:<12} {actual:<12} {status}")
        
    print("-" * 60)
    print(f"Accuracy: {correct}/{len(test_genes)} ({correct/len(test_genes)*100:.1f}%)")

if __name__ == "__main__":
    test_idg_classification()
