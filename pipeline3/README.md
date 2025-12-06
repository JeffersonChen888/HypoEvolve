# Pipeline3: Simplified Genetic Algorithm for Scientific Hypothesis Evolution

Pipeline3 is a streamlined implementation of a genetic algorithm for scientific hypothesis development, focused on core evolutionary principles without the complexity of Pipeline2's multi-agent coordination.

## Overview

This pipeline implements a classical genetic algorithm with the following components:

- **Population Initialization**: Literature-grounded hypothesis generation
- **Fitness Evaluation**: Simple 4-dimension scoring (correctness, novelty, quality, testability)
- **Selection**: Tournament selection with Elo rating system
- **Crossover**: Combination and inspiration-based hypothesis merging
- **Mutation**: Simplification and out-of-box thinking variations
- **Replacement**: Elitism-based population replacement

## Architecture

### Simplified Agents

1. **GenerationAgent** (Population Initialization)
   - Literature exploration via web search
   - Basic hypothesis generation
   - Removed: Debates, iterative assumptions, research expansion

2. **ReflectionAgent** (Fitness Evaluation)
   - Initial review with 4-dimension scoring
   - Simple fitness function calculation
   - Removed: Multi-stage reviews, deep verification, simulation

3. **RankingAgent** (Selection)
   - Elo tournament with pairwise comparisons
   - Basic scoring comparison for winners
   - Removed: Complex scientific debates, multi-round tournaments

4. **EvolutionAgent** (Crossover + Mutation)
   - **Crossover**: Combination, Inspiration
   - **Mutation**: Simplification, Out-of-box thinking
   - Removed: Grounding enhancement, coherence improvements

5. **SupervisorAgent** (Workflow Orchestration)
   - Simple genetic algorithm workflow
   - Generation-by-generation execution
   - Removed: Complex task management, meta-review coordination

### Removed Components

- **Meta-review Agent**: Not essential for basic GA
- **Proximity Agent**: Not essential for basic GA
- **Complex task management**: Simplified to sequential execution
- **Multi-stage reviews**: Replaced with single fitness evaluation

## Installation

Pipeline3 uses the same dependencies as Pipeline2:

```bash
cd pipeline3
pip install -r ../pipeline2/requirements.txt
```

## Configuration

Copy environment variables from Pipeline2 or set up `.env` file:

```bash
cp ../pipeline2/.env .env
```

Key environment variables:
- `OPENAI_API_KEY`: OpenAI API key for LLM calls
- `TAVILY_API_KEY`: Tavily API key for web search (optional)
- `SERPER_API_KEY`: Serper API key for web search (optional)

## Usage

### Basic Usage

```bash
# Drug repurposing mode (default)
python main.py "Develop novel cancer therapies for pancreatic cancer"

# General scientific mode  
python main.py "Understanding protein folding mechanisms" --mode general
```

### Advanced Usage

```bash
# Custom genetic algorithm parameters
python main.py "Drug repurposing for BRCA" \
    --population-size 10 \
    --generations 7 \
    --selection-ratio 0.6 \
    --save-json \
    --verbose

# Custom output location
python main.py "Novel cancer treatments" \
    --output-dir custom_results \
    --log-file custom.log
```

## Genetic Algorithm Parameters

- `--population-size`: Initial population size (default: 8)
- `--generations`: Number of evolutionary generations (default: 5)
- `--selection-ratio`: Fraction of population selected as parents (default: 0.5)
- `--elitism-count`: Number of top performers preserved (default: 2)
- `--tournament-size`: Tournament selection size (default: 4)

## Output Files

Pipeline3 generates several output files:

1. **Log file**: `pipeline3_TIMESTAMP.log` - Detailed execution log
2. **JSON results**: `pipeline3_results_TIMESTAMP.json` - Complete results (if `--save-json`)
3. **Summary report**: `pipeline3_summary_TIMESTAMP.txt` - Human-readable summary

## Genetic Algorithm Workflow

```
1. Initialize Population (GenerationAgent)
   ├── Literature exploration via web search
   ├── Generate diverse hypotheses
   └── Set initial fitness scores

2. For each generation:
   ├── Evaluate Fitness (ReflectionAgent)
   │   ├── Score correctness, novelty, quality, testability
   │   └── Calculate combined fitness score
   │
   ├── Select Parents (RankingAgent)
   │   ├── Tournament selection
   │   └── Update Elo scores
   │
   ├── Create Offspring (EvolutionAgent)
   │   ├── Crossover: combination, inspiration
   │   ├── Mutation: simplification, out-of-box
   │   └── Evaluate offspring fitness
   │
   └── Replace Population (SupervisorAgent)
       ├── Preserve elite performers
       └── Form next generation

3. Output Results
   ├── Best hypothesis from final generation
   ├── Population evolution statistics
   └── Algorithm performance metrics
```

## Example Results

```
GENETIC ALGORITHM RESULTS:
Generations Completed: 5
Final Population Size: 8
Initial Mean Fitness: 65.4
Final Mean Fitness: 78.2
Fitness Improvement: 19.6%
Best Final Fitness: 89.3

BEST HYPOTHESIS:
Title: Metformin-Autophagy Pathway for Pancreatic Cancer Treatment
Fitness Score: 89.3
Evolution Strategy: combination
Final Drug: Metformin
Cancer Type: Pancreatic adenocarcinoma
```

## Performance Characteristics

- **Execution time**: ~5-15 minutes for default parameters
- **Memory usage**: Low (single-threaded, simple data structures)
- **API calls**: ~100-200 LLM calls for default run
- **Scalability**: Linear with population size and generations

## Comparison with Pipeline2

| Feature | Pipeline2 | Pipeline3 |
|---------|-----------|-----------|
| Agents | 7 complex agents | 4 simplified agents |
| Workflow | Multi-stage async | Sequential genetic algorithm |
| Task Management | Complex queue system | Simple generation loop |
| Reviews | 6-stage peer review | Single fitness evaluation |
| Evolution | 7 strategies | 4 core strategies |
| Execution | Parallel processing | Sequential processing |
| Complexity | High | Low |
| Maintainability | Complex | Simple |

## Customization

### Adding New Evolution Strategies

Modify `EvolutionAgent` to add new crossover or mutation operators:

```python
# In evolution_agent.py
self.mutation_strategies["new_strategy"] = "Description of new strategy"

def _new_strategy_mutation(self, parent, research_goal, offspring_id):
    # Implement new mutation strategy
    pass
```

### Adjusting Fitness Function

Modify weights in `ReflectionAgent._calculate_fitness_score()`:

```python
weights = {
    "correctness": 0.4,    # Increase correctness importance
    "quality": 0.3,        # Increase quality importance  
    "testability": 0.2,    # Reduce testability importance
    "novelty": 0.1         # Reduce novelty importance
}
```

### Custom Selection Methods

Extend `RankingAgent` with alternative selection methods:

```python
def roulette_wheel_selection(self, population, fitness_evaluations):
    # Implement roulette wheel selection
    pass
```

## Troubleshooting

### Common Issues

1. **API Key Missing**: Ensure `OPENAI_API_KEY` is set in environment
2. **Literature Search Fails**: Check web search API keys (Tavily, Serper)
3. **Memory Issues**: Reduce population size or generations
4. **Slow Execution**: Enable debug logging to identify bottlenecks

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
python main.py "research goal" --verbose
```

## Contributing

Pipeline3 is designed for simplicity and educational purposes. When contributing:

1. Maintain the genetic algorithm paradigm
2. Keep agents focused on single responsibilities
3. Avoid complex dependencies or coordination
4. Document any parameter changes
5. Test with both drug-repurposing and general modes

## License

Same license as the parent DspyPipelines project.