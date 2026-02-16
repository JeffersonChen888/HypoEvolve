# HypoEvolve: Genetic Algorithm for T2D Drug Target Discovery

A genetic algorithm-based framework for discovering novel drug targets for Type 2 Diabetes (T2D) with **gene masking** to prevent LLM prior knowledge bias.

## Overview

HypoEvolve uses large language models (LLMs) to generate and evolve drug target hypotheses through a genetic algorithm (GA). A critical innovation is **gene masking** — gene identifiers are replaced with masked IDs (e.g., `PPARG` → `G00042`) during LLM reasoning, forcing the model to rely solely on provided experimental data patterns rather than memorized associations from pre-training.

### Key Features

- **Gene Masking**: Consistent masked identifiers across all analyses prevent LLM prior knowledge bias
- **Multi-Omics Integration**: Differential expression, WGCNA, TF activity, pathway enrichment
- **Genetic Algorithm**: Population-based hypothesis evolution with crossover and mutation
- **Multi-Model Support**: Compatible with GPT-5, GPT-4o, o1, o3-mini, Gemini, and Ollama local models
- **OpenTargets Validation**: Post-hoc validation against disease association scores

---

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Running the Pipeline](#running-the-pipeline)
4. [Repository Structure](#repository-structure)
5. [Pipeline Agents](#pipeline-agents)
6. [Gene Masking](#gene-masking)
7. [Validation](#validation)
8. [Configuration Reference](#configuration-reference)
9. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.9+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/JeffersonChen888/DSC180A-Capstone.git
cd DSC180A-Capstone

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r pipeline/requirements.txt
```

### Required Python Packages

```
openai>=1.3.0
google-generativeai>=0.8.0
requests>=2.31.0
python-dotenv>=1.0.0
numpy>=1.24.3
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
```

---

## Configuration

### Environment Variables

Create a `.env` file in the `pipeline/` directory:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional - Provider and model
LLM_PROVIDER=openai
LLM_MODEL=gpt-5           # gpt-5, gpt-4o, o3-mini, gemini-2.5-pro, etc.
REASONING_EFFORT=medium    # GPT-5 specific: minimal, low, medium, high

# Optional - Temperature and tokens
TEMPERATURE=0.7
MAX_TOKENS=4000

# Optional - GA parameters
POPULATION_SIZE=6
NUM_GENERATIONS=3
SELECTION_RATIO=0.5
ELITISM_COUNT=2

# Optional - Fitness weights
FITNESS_CORRECTNESS_WEIGHT=1.0
FITNESS_NOVELTY_WEIGHT=1.3
FITNESS_QUALITY_WEIGHT=1.5

# Optional - Web search
TAVILY_API_KEY=...
USE_WEB_SEARCH=True
```

### Data Requirements

| Dataset | Location | Description |
|---------|----------|-------------|
| T2D Expression Data | `pipeline/data/t2d/` | `.h5ad` files for multi-omics analysis |
| OpenTargets TSV | `pipeline/data/` | T2D disease association scores (`OT-MONDO_*.tsv`) |

---

## Running the Pipeline

### Basic Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run T2D target discovery (default mode)
python pipeline/main.py "Identify novel drug targets for Type 2 Diabetes" --mode t2d-target

# Specify T2D data directory
python pipeline/main.py "T2D targets" --mode t2d-target --t2d-data-dir pipeline/data/t2d/

# Custom GA parameters
python pipeline/main.py "T2D targets" --mode t2d-target \
    --population-size 10 \
    --generations 5 \
    --selection-ratio 0.6

# Select a different model
python pipeline/main.py "T2D targets" --mode t2d-target --model gpt-4o
```

### Command Line Arguments

```
usage: main.py [-h] [--mode MODE] [--t2d-data-dir DIR] [--t2d-datasets DATASETS]
               [--skip-wgcna] [--skip-tf] [--population-size N] [--generations N]
               [--selection-ratio R] [--model MODEL] research_goal

positional arguments:
  research_goal          The scientific research goal to explore

optional arguments:
  --mode                 Pipeline mode (use "t2d-target")
  --t2d-data-dir         Directory containing T2D .h5ad files
  --t2d-datasets         Comma-separated GEO IDs to analyze
  --skip-wgcna           Skip WGCNA analysis (faster)
  --skip-tf              Skip TF activity analysis
  --population-size      GA population size (default: 6)
  --generations          Number of GA generations (default: 3)
  --selection-ratio      Parent selection fraction (default: 0.5)
  --model                LLM model to use (default: gpt-5)
```

### Example Workflow

```bash
# 1. Setup environment
source venv/bin/activate
export OPENAI_API_KEY=sk-...

# 2. Run discovery
python pipeline/main.py "Identify T2D targets" --mode t2d-target \
    --t2d-data-dir pipeline/data/t2d/ \
    --population-size 6 \
    --generations 3

# 3. Check output
ls pipeline/output/t2d_target/run_*/

# 4. Validate predictions
python -c "
from pipeline.utils.opentargets_validator import OpenTargetsValidator
v = OpenTargetsValidator('pipeline/data/OT-MONDO_0005148-associated-targets-2026_2_10-v25_12.tsv')
print(v.validate('SLC2A2'))
"
```

---

## Repository Structure

```
DSC180A-Capstone/
├── pipeline/                          # Main pipeline
│   ├── main.py                        # Entry point
│   ├── config.py                      # Configuration and logging
│   ├── prompts.py                     # All LLM prompts
│   ├── t2d_config.py                  # T2D-specific parameters
│   ├── evaluate_t2d_results.py        # T2D result evaluation
│   ├── requirements.txt               # Python dependencies
│   ├── agents/
│   │   ├── generation_agent.py        # Initial hypothesis generation
│   │   ├── reflection_agent.py        # Fitness evaluation
│   │   ├── evolution_agent.py         # Crossover and mutation
│   │   ├── supervisor_agent.py        # GA orchestration
│   │   └── tournament_agent.py        # Elo ranking
│   ├── external_tools/
│   │   ├── llm_client.py              # LLM API wrapper
│   │   ├── web_search.py              # Tavily/web search
│   │   ├── t2d_analysis.py            # Multi-omics analysis engine
│   │   ├── gemini.py                  # Google Gemini support
│   │   └── ollama_client.py           # Local Ollama support
│   ├── utils/
│   │   ├── gene_masking.py            # GeneMapper class
│   │   ├── masked_analysis_pipeline.py # Full masked analysis pipeline
│   │   ├── t2d_pipeline_runner.py     # T2D pipeline runner
│   │   ├── evaluation_metrics.py      # Metrics calculation
│   │   ├── ground_truth_loader.py     # Validation data loading
│   │   ├── druggability_extractor.py  # IDG family extraction
│   │   ├── elo_scoring.py             # Elo rating system
│   │   ├── data_loader.py            # Data loading utilities
│   │   ├── extract_predictions.py     # Prediction extraction
│   │   ├── result_extractor.py        # Result extraction
│   │   ├── leakage_tester.py          # Leakage detection
│   │   ├── clustering.py              # Hypothesis clustering
│   │   ├── analytics.py               # Analytics utilities
│   │   └── prompt_loader.py           # Prompt loading
│   ├── persistence/
│   │   └── context_memory.py          # Persistent context memory
│   ├── scripts/
│   │   ├── calculate_actual_cost.py   # API cost calculation
│   │   ├── extract_top_hypotheses.py  # Top hypothesis extraction
│   │   └── ...
│   ├── data/
│   │   ├── t2d/                       # T2D expression data (.h5ad)
│   │   └── OT-MONDO_*.tsv            # OpenTargets associations
│   └── output/                        # Run outputs
│       └── t2d_target/
│
├── scripts/                           # Evaluation & benchmark scripts
│   ├── comprehensive_ga_evaluation.py
│   ├── benchmark_iterations.py
│   ├── benchmark_models.py
│   ├── generate_paper_results.py
│   ├── generate_baseline_report.py
│   └── ...
│
├── data/                              # Shared datasets
├── output/                            # Top-level outputs
├── GPT4o_Mini_New_Baseline/           # Baseline comparison results
├── HypoEvolve/                        # LaTeX report source
├── external_tools.py                  # Standalone external tools
├── CLAUDE.md                          # Development guidelines
├── .gitignore
└── README.md                          # This file
```

---

## Pipeline Agents

The T2D target discovery pipeline uses 5 specialized agents:

| Agent | File | Purpose |
|-------|------|---------|
| **Generation** | `agents/generation_agent.py` | Creates initial population of T2D drug target hypotheses from multi-omics data |
| **Reflection** | `agents/reflection_agent.py` | Evaluates hypothesis fitness using multi-stage peer review scoring |
| **Evolution** | `agents/evolution_agent.py` | Applies genetic operators (crossover + mutation) to evolve hypotheses |
| **Supervisor** | `agents/supervisor_agent.py` | Orchestrates the full GA workflow with elitism and selection |
| **Tournament** | `agents/tournament_agent.py` | Cross-hypothesis Elo ranking for final selection |

### GA Workflow

1. **Generation**: Create initial population of T2D target hypotheses using masked multi-omics data
2. **Reflection**: Score each hypothesis on correctness, novelty, and quality
3. **Selection**: Select top-performing parents based on fitness scores
4. **Evolution**: Apply crossover (combine parent hypotheses) and mutation (refine/pivot)
5. **Elitism**: Preserve top performers across generations
6. **Repeat**: Iterate for N generations, return best hypothesis

---

## Gene Masking

The `GeneMapper` class (`pipeline/utils/gene_masking.py`) is central to preventing LLM bias:

```python
# Create mapper from datasets
mapper = GeneMapper(datasets={'GSE123': adata1, 'GSE456': adata2}, seed=42)

# Mask a gene name
masked_id = mapper.mask("PPARG")  # Returns "G00042"

# Unmask after prediction
real_gene = mapper.unmask("G00042")  # Returns "PPARG"

# Mask an entire dataframe
masked_df = mapper.mask_dataframe(de_results, gene_column='gene_symbol')
```

All multi-omics data (DE results, WGCNA modules, TF activity, pathway enrichment) is masked before being presented to the LLM, ensuring the model cannot rely on prior knowledge of gene functions.

---

## Validation

### OpenTargets Validation (`pipeline/utils/evaluation_metrics.py`)

After the GA evolves hypotheses, predicted gene targets are validated against OpenTargets disease association scores:

```python
from pipeline.utils.opentargets_validator import OpenTargetsValidator

validator = OpenTargetsValidator("pipeline/data/OT-MONDO_0005148-associated-targets-2026_2_10-v25_12.tsv")

# Validate a gene
result = validator.validate("SLC2A2")
# Returns: {'gene': 'SLC2A2', 'score': 0.7405, 'is_validated': True}

# Batch validation
results = validator.validate_genes(["SLC2A2", "PPARG", "GAD1"])
```

### T2D Result Evaluation

```bash
python pipeline/evaluate_t2d_results.py
```

---

## Configuration Reference

### GA Parameters (`pipeline/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `POPULATION_SIZE` | 6 | Initial population size |
| `NUM_GENERATIONS` | 3 | Evolution generations |
| `SELECTION_RATIO` | 0.5 | Parent selection fraction |
| `ELITISM_COUNT` | 2 | Preserved top performers |
| `TOURNAMENT_SIZE` | 4 | Tournament selection size |
| `FITNESS_CORRECTNESS_WEIGHT` | 1.0 | Correctness score weight |
| `FITNESS_NOVELTY_WEIGHT` | 1.3 | Novelty score weight |
| `FITNESS_QUALITY_WEIGHT` | 1.5 | Quality score weight |

### T2D Analysis Parameters (`pipeline/t2d_config.py`)

```python
T2D_ANALYSIS_PARAMS = {
    'min_genes': 200,           # Minimum genes per dataset
    'min_cells': 50,            # Minimum cells per condition
    'pval_threshold': 0.05,     # DE significance threshold
    'logfc_threshold': 0.5,     # Log2 fold change threshold
    'wgcna_min_module_size': 30,
    'top_n_candidates': 50,     # Candidates for LLM
}
```

### Mutation Strategies

| Strategy | Description |
|----------|-------------|
| `alternative_target` | Keep mechanism, select different target gene |
| `mechanism_refinement` | Keep target, propose detailed/alternative mechanism |
| `therapeutic_pivot` | Keep target, opposite modulation or combination |
| `tissue_focus` | Keep target, focus on specific tissue |

---

## Troubleshooting

### Network/API Issues

- **Retry configuration**: Set `max_retries` in `pipeline/external_tools/llm_client.py`
- **Cost analysis**: `python pipeline/scripts/calculate_actual_cost.py`

### Common Issues

- **State errors**: Use JSON-serializable data structures; SafeJSONEncoder handles frozensets
- **Memory issues**: Reduce `NUM_WORKERS` in `config.py`
- **Path context**: Run scripts from project root

### Debugging

```bash
# Check pipeline logs
tail -f pipeline/output/t2d_target/run_*/pipeline_*.log

# Extract predicted targets
grep "target_gene" pipeline/output/t2d_target/run_*/*.json
```
