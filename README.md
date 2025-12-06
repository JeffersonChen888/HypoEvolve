# Genetic Algorithm Evaluation for Scientific Question Answering

This repository contains an evaluation framework for comparing genetic algorithm (GA) approaches with baseline methods on scientific multiple-choice questions from the GPQA dataset. The evaluation compares three approaches: a baseline LLM method and two GA-enhanced methods with different generation counts.

## Overview

The evaluation script (`scripts/comprehensive_ga_evaluation.py`) tests three approaches on 30 GPQA questions (10 Biology, 10 Chemistry, 10 Physics):

1. **Baseline**: Direct GPT-4o-mini API call (no genetic algorithm)
2. **GA-3gen**: Pipeline3 genetic algorithm with GPT-4o-mini using 3 generations
3. **GA-5gen**: Pipeline3 genetic algorithm with GPT-4o-mini using 5 generations

For each approach, the script collects:
- Accuracy (correct/incorrect answers)
- Execution time
- Token usage (input/output)
- Cost (USD)
- API call counts

## Repository Structure

```
.
├── scripts/
│   └── comprehensive_ga_evaluation.py  # Main evaluation script
├── pipeline3/                           # Genetic algorithm framework
│   ├── agents/                         # GA agents (generation, reflection, ranking, evolution)
│   ├── config.py                       # Configuration settings
│   ├── external_tools/                 # LLM API wrappers and web search
│   ├── prompts.py                      # Prompt templates
│   └── utils/                          # Utility functions (Elo scoring)
└── trajectories/                       # Output directory (created automatically)
    └── GPQA/
        └── GPT4o_Mini_New/             # Results and logs
```

## Prerequisites

- Python 3.8 or higher
- OpenAI API key with access to GPT-4o-mini
- (Optional) Web search API keys (Tavily, Serper, or SerpAPI) for literature exploration

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd DSC180A-Capstone
```

### 2. Install Dependencies

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required packages:

```bash
pip install -r pipeline3/requirements.txt
```

Or install manually:

```bash
pip install openai>=1.3.0 python-dotenv>=1.0.0
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# Optional: For literature exploration in GA
TAVILY_API_KEY=your_tavily_key  # Optional
SERPER_API_KEY=your_serper_key  # Optional
SERPAPI_API_KEY=your_serpapi_key  # Optional
```

**Important**: Never commit your `.env` file to version control. It should be in `.gitignore`.

## Data Setup

### GPQA Questions File

The evaluation script requires a file containing 30 selected GPQA questions at:

```
trajectories/GPQA/GPT4o_Mini_New/selected_30_questions.json
```

**Note**: A reference copy of this file is available in `GPT4o_Mini_New_Baseline/selected_30_questions.json` in the repository root. The script expects the file at the `trajectories/GPQA/GPT4o_Mini_New/` location, so if you're using the repository copy, you may need to copy it to the expected location or ensure the directory structure exists.

The file should have the following structure:

```json
{
  "questions": [
    {
      "record_id": "unique_id",
      "domain": "Biology|Chemistry|Physics",
      "subdomain": "optional_subdomain",
      "question": "Question text here?",
      "correct_answer": "Correct answer text",
      "incorrect_answers": ["Wrong answer 1", "Wrong answer 2", "Wrong answer 3"]
    },
    ...
  ]
}
```

**Note**: The script expects exactly 30 questions (10 from each domain). If you don't have this file, you'll need to:
1. Obtain the GPQA dataset
2. Select 30 questions (10 Biology, 10 Chemistry, 10 Physics)
3. Format them according to the structure above
4. Save as `trajectories/GPQA/GPT4o_Mini_New/selected_30_questions.json`

The script will create the directory structure automatically if it doesn't exist.

## Running the Evaluation

### Basic Usage

Run the complete evaluation:

```bash
python scripts/comprehensive_ga_evaluation.py
```

### How It Works

1. **Resume Capability**: The script automatically detects existing results and resumes from where it left off. If a question is already completed for all three approaches, it will be skipped.

2. **Progress Tracking**: Results are saved after each question's evaluation, so you can safely interrupt and resume the script.

3. **Logging**: Detailed logs are saved to:
   - `trajectories/GPQA/GPT4o_Mini_New/logs/baseline/` - Baseline execution logs
   - `trajectories/GPQA/GPT4o_Mini_New/logs/ga3gen/` - GA-3gen execution logs
   - `trajectories/GPQA/GPT4o_Mini_New/logs/ga5gen/` - GA-5gen execution logs

4. **Results File**: Main results are saved to:
   - `trajectories/GPQA/GPT4o_Mini_New/evaluation_results_30q_TIMESTAMP.json`

### Expected Runtime

- **Baseline**: ~1-2 seconds per question
- **GA-3gen**: ~2-5 minutes per question (depends on API response times)
- **GA-5gen**: ~3-8 minutes per question

For 30 questions, expect:
- Baseline: ~1-2 minutes total
- GA-3gen: ~1-2.5 hours total
- GA-5gen: ~1.5-4 hours total

**Total estimated time**: 2.5-6.5 hours for complete evaluation

### Cost Estimation

Using GPT-4o-mini pricing (as of January 2025):
- Input: $0.15 per 1M tokens
- Output: $0.60 per 1M tokens

Estimated costs for 30 questions:
- Baseline: ~$0.01-0.02
- GA-3gen: ~$0.50-1.50 (varies with token usage)
- GA-5gen: ~$0.75-2.50

**Total estimated cost**: $1.25-4.00

## Output Format

### Results JSON Structure

The main results file (`evaluation_results_30q_TIMESTAMP.json`) contains:

```json
{
  "metadata": {
    "timestamp": "20250108_140903",
    "total_questions": 30,
    "base_model": "gpt-4o-mini",
    "pricing": {
      "input_per_million": 0.15,
      "output_per_million": 0.60
    }
  },
  "questions": [
    {
      "record_id": "question_id",
      "domain": "Biology",
      "correct_answer": "A",
      "baseline": {
        "predicted": "A",
        "correct": true,
        "time_seconds": 1.23,
        "input_tokens": 500,
        "output_tokens": 200,
        "cost_usd": 0.000195,
        "api_calls": 1
      },
      "ga_3gen": { ... },
      "ga_5gen": { ... }
    },
    ...
  ],
  "summary": {
    "baseline": {
      "accuracy": 0.70,
      "total_time": 45.2,
      "total_cost": 0.015,
      "avg_time_per_q": 1.5
    },
    "ga_3gen": { ... },
    "ga_5gen": { ... }
  }
}
```

### Console Output

The script provides real-time progress updates:

```
================================================================================
Question 1/30: question_id (Biology)
================================================================================
  Running Baseline...
  Baseline: A (✓) - 1.2s - $0.000195
  Running GA-3gen...
  GA-3gen: A (✓) - 245.3s - $0.0234
  Running GA-5gen...
  GA-5gen: A (✓) - 387.6s - $0.0412
```

## Troubleshooting

### Common Issues

1. **Missing Questions File**
   ```
   Error: Questions file not found: trajectories/GPQA/GPT4o_Mini_New/selected_30_questions.json
   ```
   **Solution**: Copy the questions file from `GPT4o_Mini_New_Baseline/selected_30_questions.json` to `trajectories/GPQA/GPT4o_Mini_New/selected_30_questions.json`, or create the questions file as described in the Data Setup section.

2. **API Key Not Found**
   ```
   ValueError: OPENAI_API_KEY not found in environment variables
   ```
   **Solution**: Ensure your `.env` file exists and contains `OPENAI_API_KEY=your_key`.

3. **Import Errors**
   ```
   ImportError: OpenAI package not installed
   ```
   **Solution**: Run `pip install openai python-dotenv`.

4. **Rate Limiting**
   If you encounter rate limit errors, the script will log them. Wait a few minutes and resume the script (it will skip completed questions).

### Resuming Interrupted Evaluations

The script automatically resumes from where it left off. Simply run it again:

```bash
python scripts/comprehensive_ga_evaluation.py
```

It will:
- Detect existing results file
- Skip completed questions
- Continue with incomplete questions

## Code Structure

### Main Evaluation Script

`scripts/comprehensive_ga_evaluation.py`:
- `run_baseline()`: Runs baseline GPT-4o-mini evaluation
- `run_pipeline3_ga()`: Runs GA evaluation with specified generations
- `extract_answer_from_text()`: Extracts answer choice (A/B/C/D) from LLM response
- `parse_token_usage_from_log()`: Parses token usage from GA execution logs
- `run_comprehensive_evaluation()`: Main orchestration function

### Pipeline3 Framework

The `pipeline3/` directory contains the genetic algorithm framework:

- **agents/**: Core GA agents
  - `supervisor_agent.py`: Orchestrates the GA workflow
  - `generation_agent.py`: Generates initial hypothesis population
  - `reflection_agent.py`: Evaluates hypothesis fitness
  - `ranking_agent.py`: Ranks hypotheses using Elo tournament
  - `evolution_agent.py`: Performs crossover and mutation operations

- **external_tools/**: API wrappers
  - `gpt4o.py`: OpenAI API wrapper with token tracking
  - `web_search.py`: Literature search functionality

- **config.py**: Configuration parameters (population size, generations, etc.)
