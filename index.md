---
layout: default
title: HypoEvolve
---

# HypoEvolve

Large language models generate one-shot answers. Scientific discovery requires iteration. HypoEvolve introduces evolutionary refinement into AI hypothesis generation.

---

## <span id="problem">The Problem</span>

AI systems lack iterative refinement. Without comparison or selection, hypotheses remain brittle and prone to hallucination.

---

## <span id="method">Method</span>

HypoEvolve maintains a population of hypotheses that evolve over generations.

<div class="section-highlight">
The system applies structured evaluation, tournament selection, crossover, and mutation.
</div>

![Framework](/assets/fig_framework.png)

---

## <span id="results">Results</span>

| Method | Excellent Rate | Avg Score |
|--------|---------------|-----------|
| Single-pass LLM | 42% | 56.3% |
| HypoEvolve | 84% | 93.6% |

![Results](/assets/fig_summary.png)

---

## <span id="limitations">Limitations</span>

Fitness partially relies on LLM evaluation. External validation measures gene dependency rather than full mechanistic correctness. Broader domains remain to be tested.

---

## <span id="code">Code & Reproducibility</span>

GitHub: [Add Link]

All datasets are publicly available.