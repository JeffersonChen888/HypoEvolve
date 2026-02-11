# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains two multi-agent scientific reasoning pipelines implementing the "AI Co-scientist" approach for automated scientific hypothesis generation and evaluation:

- **Pipeline2**: Complex multi-agent system with 7 specialized agents (drug repurposing, general Q&A)
- **Pipeline3**: Simplified genetic algorithm with 5 agents (drug repurposing, general Q&A, lethal genes modes)

## Core Architecture

### Pipeline2 (Complex Multi-Agent System)
- **Location**: `pipeline2/` directory
- **Entry Point**: `pipeline2/main.py`
- **Architecture**: Multi-agent system with asynchronous task coordination
- **Core Agents**: Generation, Reflection, Ranking, Evolution, Proximity, MetaReview, Supervisor

### Pipeline3 (Simplified Genetic Algorithm)
- **Location**: `pipeline3/` directory
- **Entry Point**: `pipeline3/main.py`
- **Architecture**: Sequential genetic algorithm with simplified agents
- **Core Agents**: Generation, Reflection, Ranking, Evolution, Supervisor, Tournament (for lethal genes mode)

### Agent Responsibilities

**Pipeline2 Agents**:
- **Generation**: Hypotheses via literature exploration, scientific debates, assumptions identification
- **Reflection**: 6-stage peer review (initial, full, deep verification, observation, simulation, tournament)
- **Ranking**: ELO tournament system with scientific debates
- **Evolution**: 7 improvement strategies for hypothesis refinement
- **Proximity**: Semantic similarity analysis and graph visualization
- **MetaReview**: System-level analysis and research overview
- **Supervisor**: Priority-based task scheduling and orchestration

**Pipeline3 Agents**:
- **Generation**: Literature exploration and hypothesis generation
- **Reflection**: 5-stage full review with fitness scoring (Related Articles, Known Aspects, Novel Components, Assumptions, Scrutiny)
- **Ranking**: ELO tournament selection (deprecated in favor of fitness-based GA selection)
- **Evolution**: 4 genetic operators (crossover: combination/inspiration; mutation: simplification/out-of-box)
- **Supervisor**: Genetic algorithm orchestration with elitism
- **Tournament**: Cross-hypothesis ELO ranking for lethal genes mode

## Development Commands

### Running Pipeline3 (Primary Development Pipeline)
```bash
# Activate virtual environment first
source venv/bin/activate

# Drug repurposing mode (default)
python pipeline3/main.py "research goal" --mode drug-repurposing

# General mode for MCQ answering
python pipeline3/main.py "research goal" --mode general

# Lethal genes mode (single pair)
python pipeline3/main.py "KLF5-ARID1A synthetic lethality" --mode lethal_genes

# Lethal genes 2 mode (with prompt file)
python pipeline3/main.py "synthetic lethality" --mode lethal_genes_2 \
    --prompt-file data/lethal_genes/individual_prompts/prompt_01_KLF5_ARID1A.txt

# Tournament mode (cross-pair ranking)
python pipeline3/main.py "synthetic lethality" --mode lethal_genes_tournament \
    --generation-prompt data/tournament/generation_prompt.txt \
    --gene-pairs data/tournament/gene_pairs.json \
    --top-k-per-pair 3

# Custom GA parameters
python pipeline3/main.py "research goal" --population-size 10 --generations 7 --selection-ratio 0.6

# Batch mode for multiple gene pairs
python pipeline3/main.py "synthetic lethality" --mode lethal_genes --batch-mode

# Model selection
python pipeline3/main.py "research goal" --model gpt-5  # or gpt-4o, o3-mini, gemini-2.5-pro, qwen2.5:32b
```

### Running Pipeline2 (Complex Multi-Agent)
```bash
cd pipeline2
python main.py "research goal" --mode drug-repurposing
python main.py "research goal" --mode general --verbose --log-file custom.log
```

### Pipeline3 Modes
- **`drug-repurposing`** (default): Drug repurposing hypothesis generation
- **`general`**: MCQ answering with FINAL_ANSWER format
- **`lethal_genes`**: Single gene pair synthetic lethality analysis
- **`lethal_genes_2`**: Gene pair analysis with custom prompt files
- **`lethal_genes_tournament`**: Two-phase system: GA per pair → cross-pair ELO tournament

### Batch Processing (Lethal Genes)
```bash
# Run Phase 1: GA evolution for each gene pair
python pipeline3/run_phase1_batched.py --start-index 0 --count 10

# Run Phase 2: Cross-pair tournament
python pipeline3/run_phase2_tournament.py --phase1-dir output/lethal_genes_2/run_xxx/
```

### Real-time UI Dashboard
```bash
# Backend (Terminal 1)
cd pipeline2/ui/backend && pip install -r requirements.txt && python monitor.py

# Frontend (Terminal 2)
cd pipeline2/ui/dashboard && npm install && npm start
```
Access at http://localhost:3000

### Requirements Installation
```bash
pip install -r pipeline2/requirements.txt
# or for Pipeline3 specific:
pip install -r pipeline3/requirements.txt
```

## Configuration

### Environment Setup
Create `.env` file in pipeline2/ or pipeline3/ with API keys:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
LLM_PROVIDER=openai  # openai/anthropic/google
LLM_MODEL=gpt-5      # gpt-5, gpt-4o, o3-mini, gemini-2.5-pro, gemini-2.0-flash
REASONING_EFFORT=medium  # GPT-5 specific: minimal/low/medium/high
TAVILY_API_KEY=...   # For web search
```

### Pipeline3 GA Parameters (config.py)
- `POPULATION_SIZE`: Initial population (default: 6)
- `NUM_GENERATIONS`: Evolution generations (default: 3)
- `SELECTION_RATIO`: Parent selection fraction (default: 0.5)
- `ELITISM_COUNT`: Top performers preserved (default: 2)
- Fitness weights: `FITNESS_CORRECTNESS_WEIGHT`, `FITNESS_NOVELTY_WEIGHT`, `FITNESS_QUALITY_WEIGHT`

### Key Directories
- `pipeline2/agents/`, `pipeline3/agents/`: Agent implementations
- `pipeline2/prompts.py`, `pipeline3/prompts.py`: All LLM prompts
- `pipeline3/output/`: Results organized by mode (drug_repurposing/, general/, lethal_genes_2/)
- `pipeline3/data/tournament/`: Gene pairs and prompt templates for tournament mode
- `scripts/`: Analysis and evaluation scripts

## Project Specific Notes

### Replication Goal
Replicating the method described by ai_coscientist.txt with 100% fidelity:
- Exact prompt usages and expected output formats
- Identical architecture, workflows, and methodology

### TCGA Cancer Types (Drug Repurposing)
- 33 TCGA cancer types defined in `pipeline2/tcga_cancer_types.py`
- Output format: `FINAL DRUG: [name]` and `CANCER TYPE: [TCGA name]`
- Agents use exact TCGA terminology (e.g., "Acute Myeloid Leukemia" not "AML")

### Lethal Genes Mode Architecture
Two-phase tournament system for synthetic lethality hypothesis ranking:
1. **Phase 1**: Run GA independently for each gene pair (population evolution)
2. **Phase 2**: Pool top-k hypotheses from all pairs → ELO tournament for cross-pair ranking

Gene pair data in `pipeline3/gene_pairs_config.py` or JSON files in `data/tournament/`

## Evaluation Systems

### BixBench Evaluation
```bash
python scripts/bixbench_evaluator.py --test-run      # Quick test (11 questions)
python scripts/bixbench_evaluator.py                 # Full evaluation (296 questions)
python scripts/bixbench_recovery.py                  # Resume interrupted evaluations
python scripts/bixbench_analyzer.py                  # Generate accuracy reports
```
- 296 MCQ questions across 53 research capsules
- Uses `--mode general` with FINAL_ANSWER format
- Smart CSV truncation for data context (99.8% token reduction)

### GPQA Evaluation
```bash
python scripts/gpqa_evaluator.py --batch-size 30 --batch-start 1 --batch-end 7  # Full dataset
python scripts/pipeline3_gpqa_evaluation.py          # Pipeline3 evaluation
python scripts/check_gpqa_accuracy.py                # Check results
python scripts/calculate_actual_costs.py             # API cost analysis
```
- 198 graduate-level physics/chemistry questions
- Output: `GPQA_trajectories/gpqa-XXX_result.json`

### DepMap Validation (Drug Repurposing)
```bash
python scripts/batch_depmap_validation.py depmap_trajectories/extracted_results/ --min-score 4
```
- Validates drug-cancer pairs against CRISPR dependency data
- Data location: `depmap_data/` (Git LFS managed)

### Lethal Genes Analysis Scripts
```bash
python scripts/evaluate_lethal_genes.py              # Evaluate lethal genes results
python scripts/analyze_top3_hypotheses.py            # Analyze top hypotheses
python scripts/validate_batch1_comprehensive.py      # Batch validation
python scripts/comprehensive_hypothesis_analysis.py  # Full analysis
```

## Troubleshooting

### Network/API Issues
- **Retry configuration**: Set `max_retries` in `pipeline3/external_tools/gpt4o.py` (default: 10,000)
- **Staged reviews**: Pipeline3 breaks reviews into 5 stages (~1,500 chars each) for reliability
- **Cost analysis**: `python scripts/calculate_actual_costs.py` for accurate token counts

### Common Issues
- **State errors**: Use JSON-serializable data structures; SafeJSONEncoder handles frozensets
- **Memory issues**: Reduce `NUM_WORKERS` in config.py
- **Path context**: Run scripts from project root: `python scripts/script_name.py`

### Debugging
```bash
# Check logs
tail -f pipeline3/output/*/pipeline3_*.log

# Extract answers from logs
grep "Predicted Answer:" GPQA_logs/*.log
grep "FINAL_ANSWER:" output/*/*.log
```

## Development Patterns

### Adding Agent Functionality
1. Implement methods in `agents/*_agent.py`
2. Update prompts in `prompts.py`
3. Add orchestration in `supervisor_agent.py`
4. Configure parameters in `config.py`

### Output Structure
Pipeline3 organizes outputs by mode and run:
```
pipeline3/output/
├── drug_repurposing/run_YYYYMMDD_HHMMSS/
├── general/run_YYYYMMDD_HHMMSS/
├── lethal_genes_2/run_YYYYMMDD_HHMMSS/
│   ├── KLF5_ARID1A/
│   ├── TP53_MDM2/
│   └── ...
└── lethal_genes_tournament/run_YYYYMMDD_HHMMSS/
    ├── phase1_results/
    ├── tournament_results_*.json
    └── final_rankings_*.csv
```