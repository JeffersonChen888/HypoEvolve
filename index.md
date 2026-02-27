---
layout: default
---

# HypoEvolve: Benchmarking LLM-Generated Biological Hypotheses for Scientific Discovery

Jefferson Chen, Samuel Lee, Jieyuan Liu, Zhiting Hu, Zhen Wang  
University of California, San Diego  

jec068@ucsd.edu  
hsl023@ucsd.edu  
jil029@ucsd.edu  
zhh019@ucsd.edu  
zhw085@ucsd.edu  

---

## Elevator Pitch

HypoEvolve improves the quality of AI-generated biological hypotheses by introducing evolutionary refinement into large language model reasoning. Instead of generating a single answer per prompt, our system maintains a population of candidate hypotheses that are evaluated, selected, recombined, and mutated across generations. On a real-world drug repurposing task spanning 31 cancer types, this evolutionary process doubled the rate of externally validated “excellent” predictions compared to a standard single-pass LLM.

---

# Why This Matters

Scientific discovery is inherently iterative. Researchers propose ideas, critique weaknesses, merge complementary mechanisms, and refine explanations over time. Most current AI systems do not replicate this process. They produce a single response and stop.

This one-shot behavior limits reliability. Even when outputs are plausible, they may lack biological grounding. Without comparison, revision, or selection, there is no mechanism for improvement.

HypoEvolve asks a simple question: what if AI systems could evolve hypotheses the way biological systems evolve organisms?

---

# What We Built

We designed a multi-agent evolutionary framework for hypothesis refinement. The system generates a population of literature-grounded hypotheses, evaluates their quality, and applies evolutionary operators to improve them over successive generations.

![Framework overview of the multi-agent evolutionary hypothesis refinement process.](/assets/fig_framework.png)

The architecture contains four interacting agents. A Generation Agent proposes diverse candidate hypotheses. A Reflection Agent evaluates each hypothesis for biological correctness, novelty, and explanatory quality. An Evolution Agent performs crossover and mutation, combining strong ideas and introducing structured variation. Finally, a Supervisor Agent manages tournament selection and elitism, ensuring that high-performing hypotheses persist across generations.

This process repeats iteratively, allowing the population to improve over time rather than relying on a single attempt.

---

# Scope and Boundaries

This project focuses on benchmarking evolutionary refinement for AI-generated biomedical hypotheses. We implement crossover and mutation operators, structured multi-agent evaluation, and external validation using an independent biological dataset. We evaluate performance on a drug repurposing task across 31 cancer types.

We do not conduct wet-lab validation, fine-tune foundation models, or claim clinical deployment readiness. Our goal is methodological: to test whether evolutionary pressure improves AI hypothesis quality.

---

# Optimization Objective

We formalize hypothesis search as:

$$
h^* = \arg\max_h f(h)
$$

where fitness is defined as:

$$
f(h) = w_c s_c + w_n s_n + w_q s_q
$$

Here, \( s_c \) measures biological correctness, \( s_n \) measures novelty, and \( s_q \) measures explanation quality. The weighting emphasizes novelty and robustness while maintaining biological plausibility.

Across generations, hypotheses with higher fitness are more likely to survive and recombine.

---

# Application: Drug Repurposing for Cancer

To evaluate the framework, we apply it to a practical biomedical task: given a cancer type, propose an FDA-approved drug and provide a mechanistic explanation for its potential effectiveness.

Each hypothesis consists of a drug, a target pathway or gene, and a structured biological rationale. The system does not receive outcome labels during evolution.

---

# External Validation Using DepMap

We evaluate hypotheses using CRISPR gene dependency data from the Broad Institute’s DepMap project. For each proposed mechanism, we measure whether the implicated gene demonstrates strong dependency in the relevant cancer context.

A hypothesis is classified as “Excellent” if its validation score exceeds 0.9. Importantly, DepMap data is never used during hypothesis generation or evolutionary scoring. It is strictly reserved for external evaluation, preventing data leakage.

This ensures that improvements reflect genuine generalization rather than overfitting.

---

# Results

| Method          | Excellent Rate | Avg Score |
|----------------|---------------|-----------|
| Single-pass LLM | 42%           | 56.3%     |
| **HypoEvolve**  | **84%**       | **93.6%** |

![Comparison between single-pass LLM and HypoEvolve across 31 cancer types.](/assets/fig_summary.png)

HypoEvolve doubles the rate of externally validated excellent predictions and dramatically increases average validation scores. The improvement is statistically significant (p < 0.00001) and consistent across 31 cancer types.

The learning curve below shows that internal fitness improves rapidly during early generations and stabilizes as high-quality hypotheses are preserved through elitism.

![Learning curve showing hypothesis fitness improvement across generations.](/assets/fig_learning_curve.png)

---

# Interpreting the Results

The increase in Excellent Rate indicates stronger biological alignment with independent gene dependency data. Early generations eliminate weak or biologically inconsistent mechanisms, while later generations refine and stabilize high-performing hypotheses.

Because evaluation data is never used during evolution, performance gains reflect improved hypothesis structure rather than memorization.

---

# What Changes Across Generations?

Evolution primarily improves three properties. First, biologically inconsistent explanations are rapidly removed. Second, crossover introduces cross-pathway reasoning that does not appear in single-pass outputs. Third, mutation simplifies overly complex mechanisms, often increasing plausibility.

Together, these processes reduce hallucination risk and improve consistency.

---

# Limitations

Fitness evaluation partially relies on LLM-based scoring, which may introduce bias. DepMap measures gene dependency rather than complete mechanistic correctness, so validation is necessarily indirect. Evolution depth is constrained by computational resources, and biological novelty remains difficult to quantify objectively. Finally, experiments are limited to oncology; broader domains remain to be tested.

Future improvements include integrating structured biomedical knowledge graphs and incorporating experimental feedback loops.

---

# Contributions

This project introduces a population-based evolutionary framework for refining AI-generated scientific hypotheses. We formalize hypothesis fitness optimization, demonstrate statistically significant improvements under independent biological validation, and provide a reproducible benchmarking pipeline for hypothesis evolution.

More broadly, we show that evolutionary pressure can meaningfully improve AI scientific reasoning beyond single-prompt generation.

---

# Broader Implications

HypoEvolve suggests a shift from prompt engineering toward population-based reasoning systems. Instead of searching for the perfect prompt, systems may benefit from iterative selection and recombination.

This paradigm extends beyond oncology to other domains requiring mechanistic reasoning, including materials science, climate modeling, and automated scientific discovery.

---

# Code and Reproducibility

The full implementation, evaluation scripts, and experimental configurations are available here:

GitHub Repository: [ADD LINK]

All datasets used are publicly available. Random seeds are fixed to ensure reproducibility.

---

# Acknowledgments

We thank collaborators and mentors for their guidance and feedback throughout this project.

---