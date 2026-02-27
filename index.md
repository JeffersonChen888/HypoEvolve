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

## Why This Matters

Scientific discovery is not a straight path. Researchers propose ideas, critique weaknesses, combine complementary mechanisms, and refine explanations over time. Most current AI systems do not follow this approach. They provide a single response and stop.

This one-time output limits reliability. Even if the results seem valid, they may lack biological grounding. Without comparison, revision, or selection, there is no way to improve.

HypoEvolve asks a simple question: what if AI could evolve hypotheses like biological systems evolve organisms?

## What We Built

We created a multi-agent evolutionary framework for refining hypotheses. The system generates a group of literature-based hypotheses, assesses their quality, and applies evolutionary processes to improve them across generations.

![Framework overview of the multi-agent evolutionary hypothesis refinement process.](/assets/fig_framework.png)

The framework includes four interacting agents. A Generation Agent suggests diverse candidate hypotheses. A Reflection Agent checks each hypothesis for biological accuracy, novelty, and explanatory quality. An Evolution Agent mixes and mutates ideas, combining strong concepts and adding variety. Finally, a Supervisor Agent oversees tournament selection and elitism, making sure that the best hypotheses carry on through generations.

This process works iteratively, allowing the population to improve over time instead of relying on one attempt.

## Scope and Boundaries

This project focuses on testing evolutionary refinement for AI-generated biomedical hypotheses. We use crossover and mutation processes, structured multi-agent evaluation, and external validation with an independent biological dataset. We assess performance on a drug repurposing task across 31 cancer types.

We do not perform wet-lab validation, fine-tune base models, or claim readiness for clinical use. Our goal is methodological: to investigate whether evolutionary pressure enhances AI hypothesis quality.

## Optimization Objective

We define the hypothesis search as:

$$
h^* = \arg\max_h f(h)
$$

where fitness is defined as:

$$
f(h) = w_c s_c + w_n s_n + w_q s_q
$$

Here, \( s_c \) measures biological accuracy, \( s_n \) measures novelty, and \( s_q \) measures explanatory quality. The weighting prioritizes novelty and robustness while keeping biological plausibility.

Over generations, hypotheses with better fitness are more likely to survive and be combined.

## Application: Drug Repurposing for Cancer

To test the framework, we apply it to a real biomedical task: given a cancer type, suggest an FDA-approved drug and provide a mechanistic explanation for its potential effectiveness.

Each hypothesis includes a drug, a target pathway or gene, and a structured biological rationale. The system does not receive outcome labels during evolution.

## External Validation Using DepMap

We assess hypotheses using CRISPR gene dependency data from the Broad Institute’s DepMap project. For each proposed mechanism, we check if the implicated gene shows strong dependency in the relevant cancer context.

A hypothesis is classified as “Excellent” if its validation score exceeds 0.9. Importantly, we do not use DepMap data during hypothesis generation or scoring. It is only for external evaluation, preventing data leakage.

This ensures that improvements reflect true generalization instead of overfitting.

## Results

| Method          | Excellent Rate | Avg Score |
|----------------|---------------|-----------|
| Single-pass LLM | 42%           | 56.3%     |
| **HypoEvolve**  | **84%**       | **93.6%** |

![Comparison between single-pass LLM and HypoEvolve across 31 cancer types.](/assets/fig_summary.png)

HypoEvolve doubles the rate of externally validated excellent predictions and significantly increases average validation scores. The improvement is statistically meaningful (p < 0.00001) and consistent across all 31 cancer types.

The learning curve below shows that internal fitness increases quickly during early generations and stabilizes as high-quality hypotheses are preserved through elitism.

![Learning curve showing hypothesis fitness improvement across generations.](/assets/fig_learning_curve.png)

## Interpreting the Results

The increase in the Excellent Rate suggests better biological alignment with independent gene dependency data. Early generations remove weak or biologically inconsistent mechanisms, while later generations refine and stabilize high-performing hypotheses.

Since evaluation data is not used during evolution, performance gains reflect improved hypothesis structure rather than memorization.

## What Changes Across Generations?

Evolution mainly enhances three properties. First, biologically inconsistent explanations are quickly eliminated. Second, crossover introduces reasoning between pathways that single-pass outputs miss. Third, mutation simplifies overly complex mechanisms, often improving plausibility. Together, these processes lower the risk of errors and improve consistency.

## Limitations

Fitness evaluation relies partly on LLM-based scoring, which may introduce bias. DepMap measures gene dependency rather than full mechanistic correctness, so validation is inherently indirect. The depth of evolution is limited by computational resources, and biological novelty is hard to measure objectively. Additionally, our experiments focus solely on oncology; other fields still need testing.

Future improvements may include using structured biomedical knowledge graphs and incorporating experimental feedback loops.

## Contributions

This project presents a population-based evolutionary framework for refining AI-generated scientific hypotheses. We define hypothesis fitness optimization, demonstrate statistically significant improvements with independent biological validation, and provide a reproducible benchmarking process for hypothesis evolution.

More broadly, we show that evolutionary pressure can significantly enhance AI scientific reasoning beyond single-prompt generation.

## Broader Implications

HypoEvolve indicates a shift from prompt engineering to population-based reasoning systems. Rather than searching for the ideal prompt, systems might benefit from iterative selection and recombination.

This approach applies beyond oncology to other fields needing mechanistic reasoning, including materials science, climate modeling, and automated scientific discovery.

## Code and Reproducibility

The complete implementation, evaluation scripts, and experimental configurations are available here:

GitHub Repository: <https://github.com/JeffersonChen888/HypoEvolve>
Report: <TODO>

All datasets used are publicly accessible. Random seeds are fixed to ensure reproducibility.

## Acknowledgments

We thank our mentors, Dr. Zhen Wang and Jieyuan Liu, for their guidance and feedback throughout this project.
