# HypoEvolve: Benchmarking LLM-Generated Biological Hypotheses for Scientific Discovery

Jefferson Chen, Samuel Lee, Zhiting Hu, Zhen Wang

[jec068@ucsd.edu](mailto:jec068@ucsd.edu)
[hsl023@ucsd.edu](mailto:hsl023@ucsd.edu)
[zhh019@ucsd.edu](mailto:zhh019@ucsd.edu)
[zhw085@ucsd.edu](mailto:zhw085@ucsd.edu)

University of California, San Diego

## Why This Matters

Scientific discovery depends on generating and refining hypotheses.
Traditionally, researchers:

1. Propose ideas
2. Critique them
3. Combine promising mechanisms
4. Improve them over time

However, most AI systems today generate **one answer at a time**. They don’t revise, compare, or evolve ideas the way scientists do.

We asked:

> What if AI could *evolve* hypotheses the way nature evolves organisms?

We proposed **HypoEvolve**, a system that combines large language models (LLMs) with evolutionary algorithms to iteratively refine scientific ideas.

## What We Built

HypoEvolve is a multi-agent AI system that:

* Generates scientific hypotheses
* Evaluates them systematically
* Selects the strongest ideas
* Combines and mutates them
* Repeats the process over generations

![framework](/assets/fig_framework.png)

## How It Works (High-Level)

The system has four AI agents:

1. **Generation Agent**
   Produces diverse hypotheses grounded in literature.

2. **Reflection Agent**
   Scores ideas for correctness, novelty, and quality.

3. **Evolution Agent**
   Combines strong ideas (crossover) and introduces variations (mutation).

4. **Supervisor Agent**
   Manages selection and preserves the best hypotheses.

The process repeats across generations, steadily improving hypothesis quality.

<details>
<summary>🔎 Click to View Technical Details</summary>

### Optimization Objective

We search for:

$$h* = argmax f(h)$$

Where hypothesis quality is:

$$f(h) = w_c s_c + w_n s_n + w_q s_q$$

* $s_c$: correctness
* $s_n$: novelty
* $s_q$: quality
* weights emphasize novelty and overall quality

### Evolution Loop

For each generation:

1. Structured peer review
2. Tournament selection
3. Crossover of parent hypotheses
4. Mutation (simplification or out-of-box reasoning)
5. Elitism-based replacement

### Evaluation

External validation uses:

* DepMap CRISPR gene dependency scores
* Threshold for “Excellent” ≥ 0.9

DepMap data is never used during evolution.

</details>

## Application: Drug Repurposing for Cancer

To test our system, we applied it to a real biomedical task:

> Given a cancer type, can AI propose an FDA-approved drug that might work and explain why?

The system:

* Proposes a drug
* Explains the biological mechanism
* Predicts relevance to disease biology

We evaluated results using **DepMap CRISPR knockout data**, an independent biological dataset that was *never shown* to the AI during training or evolution.

## Results

| Method          | Excellent Rate | Avg Score |
| --------------- | -------------- | ----------|
| Single-pass LLM | 42%            | 56.3%     |
| **HypoEvolve**  | **84%**        | **93.6%** |

Key findings:

* Excellent predictions doubled
* Validation scores increased dramatically
* Improvements were statistically significant (p < 0.00001)
* Gains generalized across 31 cancer types

![results](/assets/fig_summary.png)

In addition, the figure below indicates that internal LLM-evaluated fitness improves consistently across generations. The strongest gains occur in early generations, with elitism preserving high-quality hypotheses while mutation maintains diversity.

![learning_curve](/assets/fig_learning_curve.png)

## Discussion

This work suggests:

* Iterative refinement improves AI scientific reasoning
* Evolutionary search helps avoid one-shot hallucinations
* Population-based optimization can meaningfully guide hypothesis discovery

More broadly, AI systems may benefit from evolutionary pressure rather than relying solely on single prompts.

This framework could extend to:

* Biomedical discovery
* Materials science
* Climate modeling
* Mechanistic reasoning tasks

## Future Directions

* Integrate knowledge graphs into evolution
* Improve fitness evaluation beyond LLM scoring
* Apply to broader disease domains
* Incorporate experimental feedback loops
