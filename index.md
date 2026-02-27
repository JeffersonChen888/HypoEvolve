---
layout: default
---

<script>
  window.MathJax = {
    tex: {
      inlineMath: [['\\(','\\)'], ['$', '$']],
      displayMath: [['$$','$$']]
    }
  };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# HypoEvolve: Benchmarking LLM-Generated Biological Hypotheses for Scientific Discovery

Jefferson Chen, Samuel Lee, Jieyuan Liu, Zhiting Hu, Zhen Wang  
University of California, San Diego  

[jec068@ucsd.edu](mailto:jec068@ucsd.edu)  
[hsl023@ucsd.edu](mailto:hsl023@ucsd.edu)  
[jil029@ucsd.edu](mailto:jil029@ucsd.edu)  
[zhh019@ucsd.edu](mailto:zhh019@ucsd.edu)  
[zhw085@ucsd.edu](mailto:zhw085@ucsd.edu)  

## Table of Contents

- [Introduction](#introduction)
  - [Why This Matters](#why-this-matters)
  - [What We Built](#what-we-built)
  - [Scope and Boundaries](#scope-and-boundaries)
- [Methods](#methods)
  - [Optimization Objective](#optimization-objective)
  - [Application: Drug Repurposing for Cancer](#application-drug-repurposing-for-cancer)
  - [External Validation Using DepMap](#external-validation-using-depmap)
- [Results](#results)
  - [Interpreting the Results](#interpreting-the-results)
  - [What Changes Across Generations?](#what-changes-across-generations)
- [Discussion](#discussion)
  - [Limitations](#limitations)
  - [Contributions](#contributions)
  - [Broader Implications](#broader-implications)
- [Code and Reproducibility](#code-and-reproducibility)
- [Acknowledgments](#acknowledgments)

## Introduction

### Why This Matters

Biological discovery relies on forming hypotheses. Scientists read literature, piece together scattered information, and suggest testable mechanisms. However, the amount of scientific publications has exceeded what any person can handle. Large language models (LLMs) have recently shown impressive abilities in scientific reasoning and understanding literature. However, most current AI systems do not follow this approach. They provide a single response and stop. This one-time output limits reliability. Even if the results seem valid, they may lack biological grounding. Without comparison, revision, or selection, there is no way to improve.

Therefore, we ask a simple question: what if AI could evolve hypotheses like biological systems evolve organisms? To address this question, we propose **HypoEvolve**, a simplified evolutionary sys-
tem inspired by classical genetic algorithms (GAs).

### What We Built

We created a multi-agent evolutionary framework for refining hypotheses. The system generates a group of literature-based hypotheses, assesses their quality, and applies evolutionary processes to improve them across generations.

![framework](/assets/fig_framework.png)
*Figure 1. Framework overview of the multi-agent evolutionary hypothesis refinement process.*

The framework includes four interacting agents. A Generation Agent suggests diverse candidate hypotheses. A Reflection Agent checks each hypothesis for biological accuracy, novelty, and explanatory quality. An Evolution Agent mixes and mutates ideas, combining strong concepts and adding variety. Finally, a Supervisor Agent oversees tournament selection and elitism, making sure that the best hypotheses carry on through generations.

This process works iteratively, allowing the population to improve over time instead of relying on one attempt.

### Scope and Boundaries

This project focuses on testing evolutionary refinement for AI-generated biomedical hypotheses. We use crossover and mutation processes, structured multi-agent evaluation, and external validation with an independent biological dataset. We assess performance on a drug repurposing task across 31 cancer types. We do not perform wet-lab validation, fine-tune base models, or claim readiness for clinical use. Our goal is to investigate whether evolutionary pressure enhances AI hypothesis quality.

## Methods

### Optimization Objective

We define the hypothesis search as:

$h^* = \arg\max_h f(h)$

where fitness is defined as:

$f(h) = w_c s_c + w_n s_n + w_q s_q$

Here, $s_c$ measures biological accuracy, $s_n$ measures novelty, and $s_q$ measures explanatory quality. The weighting prioritizes novelty and robustness while keeping biological plausibility.

Over generations, hypotheses with better fitness are more likely to survive and be combined.

### Application: Drug Repurposing for Cancer

To test the framework, we apply it to a real biomedical task: given a cancer type, suggest an FDA-approved drug and provide a mechanistic explanation for its potential effectiveness.

Each hypothesis includes a drug, a target pathway or gene, and a structured biological rationale. The system does not receive outcome labels during evolution.

### External Validation Using DepMap

We assess hypotheses using CRISPR gene dependency data from the Broad Institute’s DepMap project. For each proposed mechanism, we check if the implicated gene shows strong dependency in the relevant cancer context.

A hypothesis is classified as “Excellent” if its validation score exceeds 0.9. Importantly, we do not use DepMap data during hypothesis generation or scoring. It is only for external evaluation, preventing data leakage. This ensures that improvements reflect true generalization instead of overfitting.

## Results

![summary](/assets/fig_summary.png)

*Figure 2. Comparison between single-pass LLM and HypoEvolve across 31 cancer types.*

HypoEvolve doubles the rate of externally validated excellent predictions and significantly increases average validation scores. The improvement is statistically meaningful (p < 0.00001) and consistent across all 31 cancer types.

The learning curve below shows that internal fitness increases quickly during early generations and stabilizes as high-quality hypotheses are preserved through elitism.

![learning_curve](/assets/fig_learning_curve.png)

*Figure 3. Learning curve showing hypothesis fitness improvement across generations.*

### Interpreting the Results

The increase in the Excellent Rate suggests better biological alignment with independent gene dependency data. Early generations remove weak or biologically inconsistent mechanisms, while later generations refine and stabilize high-performing hypotheses.

Since evaluation data is not used during evolution, performance gains reflect improved hypothesis structure rather than memorization.

### What Changes Across Generations?

Evolution mainly enhances three properties. First, biologically inconsistent explanations are quickly eliminated. Second, crossover introduces reasoning between pathways that single-pass outputs miss. Third, mutation simplifies overly complex mechanisms, often improving plausibility. Together, these processes lower the risk of errors and improve consistency.

## Discussion

### Contributions

This project presents a population-based evolutionary framework for refining AI-generated scientific hypotheses. We define hypothesis fitness optimization, demonstrate statistically significant improvements with independent biological validation, and provide a reproducible benchmarking process for hypothesis evolution.

More broadly, we show that evolutionary pressure can significantly enhance AI scientific reasoning beyond single-prompt generation.

### Broader Implications

HypoEvolve indicates a shift from prompt engineering to population-based reasoning systems. Rather than searching for the ideal prompt, systems might benefit from iterative selection and recombination.

This approach applies beyond oncology to other fields needing mechanistic reasoning, including materials science, climate modeling, and automated scientific discovery.

### Limitations

Fitness evaluation relies partly on LLM-based scoring, which may introduce bias. DepMap measures gene dependency rather than full mechanistic correctness, so validation is inherently indirect. The depth of evolution is limited by computational resources, and biological novelty is hard to measure objectively. Additionally, our experiments focus solely on oncology; other fields still need testing.

Future improvements may include using structured biomedical knowledge graphs and incorporating experimental feedback loops.

## Artifacts

The complete implementation, evaluation scripts, and experimental configurations are available here:

GitHub Repository: <https://github.com/JeffersonChen888/HypoEvolve>

Report: <TODO>

Poster: <TODO>

All datasets used are publicly accessible. Random seeds are fixed to ensure reproducibility.

## Acknowledgments

We thank our mentors for their guidance and feedback throughout this project.
