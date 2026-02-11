#!/bin/bash
# Batch run for drug-repurposing mode
# Usage: ./run_drug_repurposing_batch.sh --start 0 --count 5

set -e

# Parse arguments
START=0
COUNT=5
POPULATION=6
GENERATIONS=3
BATCH_ID=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --start) START="$2"; shift 2 ;;
        --count) COUNT="$2"; shift 2 ;;
        --population) POPULATION="$2"; shift 2 ;;
        --generations) GENERATIONS="$2"; shift 2 ;;
        --batch-id) BATCH_ID="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Generate batch ID if not provided
if [ -z "$BATCH_ID" ]; then
    BATCH_ID="batch_$(date +%Y%m%d_%H%M%S)"
fi

# All 30 TCGA cancer types (excluding CNTL, MISC, FPPP which are not cancers)
CANCER_TYPES=(
    "Acute Myeloid Leukemia"                                        # 0  LAML
    "Adrenocortical carcinoma"                                      # 1  ACC
    "Bladder Urothelial Carcinoma"                                  # 2  BLCA
    "Brain Lower Grade Glioma"                                      # 3  LGG
    "Breast invasive carcinoma"                                     # 4  BRCA
    "Cervical squamous cell carcinoma and endocervical adenocarcinoma"  # 5  CESC
    "Cholangiocarcinoma"                                            # 6  CHOL
    "Chronic Myelogenous Leukemia"                                  # 7  LCML
    "Colon adenocarcinoma"                                          # 8  COAD
    "Esophageal carcinoma"                                          # 9  ESCA
    "Glioblastoma multiforme"                                       # 10 GBM
    "Head and Neck squamous cell carcinoma"                         # 11 HNSC
    "Kidney Chromophobe"                                            # 12 KICH
    "Kidney renal clear cell carcinoma"                             # 13 KIRC
    "Kidney renal papillary cell carcinoma"                         # 14 KIRP
    "Liver hepatocellular carcinoma"                                # 15 LIHC
    "Lung adenocarcinoma"                                           # 16 LUAD
    "Lung squamous cell carcinoma"                                  # 17 LUSC
    "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma"               # 18 DLBC
    "Mesothelioma"                                                  # 19 MESO
    "Ovarian serous cystadenocarcinoma"                             # 20 OV
    "Pancreatic adenocarcinoma"                                     # 21 PAAD
    "Pheochromocytoma and Paraganglioma"                            # 22 PCPG
    "Prostate adenocarcinoma"                                       # 23 PRAD
    "Rectum adenocarcinoma"                                         # 24 READ
    "Sarcoma"                                                       # 25 SARC
    "Skin Cutaneous Melanoma"                                       # 26 SKCM
    "Stomach adenocarcinoma"                                        # 27 STAD
    "Testicular Germ Cell Tumors"                                   # 28 TGCT
    "Thymoma"                                                       # 29 THYM
    "Thyroid carcinoma"                                             # 30 THCA
    "Uterine Carcinosarcoma"                                        # 31 UCS
    "Uterine Corpus Endometrial Carcinoma"                          # 32 UCEC
    "Uveal Melanoma"                                                # 33 UVM
)

TOTAL=${#CANCER_TYPES[@]}
END=$((START + COUNT))
if [ $END -gt $TOTAL ]; then
    END=$TOTAL
fi

echo "=========================================="
echo "Drug Repurposing Batch: $START to $((END-1)) of $((TOTAL-1))"
echo "Batch ID: $BATCH_ID"
echo "Population: $POPULATION, Generations: $GENERATIONS"
echo "=========================================="

cd "$(dirname "$0")/.."
[ -d "venv" ] && source venv/bin/activate

# Create shared batch output directory
BATCH_DIR="output/drug_repurposing/$BATCH_ID"
mkdir -p "$BATCH_DIR"
echo "Output directory: $BATCH_DIR"
echo ""

for ((i=START; i<END; i++)); do
    CANCER="${CANCER_TYPES[$i]}"
    # Create safe filename from cancer type
    SAFE_NAME=$(echo "$CANCER" | tr ' ' '_' | tr -cd '[:alnum:]_')

    echo "[$i/$((TOTAL-1))] $CANCER"
    echo "  Output: $BATCH_DIR/${SAFE_NAME}/"

    python3 pipeline/main.py \
        "Find drug repurposing candidates for ${CANCER}" \
        --mode drug-repurposing \
        --population-size $POPULATION \
        --generations $GENERATIONS \
        --output-dir "$BATCH_DIR/${SAFE_NAME}" \
        --save-json

    echo ""
done

echo "=========================================="
echo "Batch complete: $BATCH_ID"
echo "Results in: $BATCH_DIR"
echo "Indices: $START to $((END-1))"
echo "=========================================="
