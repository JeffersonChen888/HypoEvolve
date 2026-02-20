# DSC180A-Methodology-4

Name: Samuel Lee
Email: <hsl023@ucsd.edu>

Section: A08
Mentors: Zhiting Hu, Zhen Wang, Kun Zhou

**What is the most interesting topic covered in your domain this quarter?**

The most interesting topic this quarter was Weak-to-Strong Generalization. This involves training or guiding stronger models using feedback or examples from weaker models. I found it interesting because it shows how small, imperfect signals can still guide large models toward improved reasoning and decision-making.

**Describe a potential investigation you would like to pursue for your Quarter 2 Project.**

For Quarter 2, I would like to explore AI for AI research automation. Our mentors gave us several papers to read in the first few weeks, and I am most interested in this topic. In the paper FIRE-Bench, their benchmark reframes evaluation by asking agents to verify the rediscovery of well-known scientific findings from recent, important ML research. I would also like to conduct a similar project in Quarter 2.

**What is a potential change you’d make to the approach taken in your current Quarter 1 Project?**

In Quarter 1, I mainly focused on comparing the performance of weak and strong models by fine-tuning parameters and using the Performance Gain Ratio (PGR) metric. For the next stage, I want to go beyond just evaluating static metrics. I plan to include multi-turn supervision, where the strong model improves its answers based on weak feedback. This change will make the setup more interactive and better reflect real-world dynamics.

**What other techniques would you be interested in using in your project?**

I want to include in-context learning, reward-model distillation, and reinforcement learning from human feedback (RLHF) techniques. RLHF would naturally expand W2S by allowing weak or proxy feedback models to act as substitutes for human preference models, which would convert qualitative feedback into quantitative reward signals. Combining this with in-context learning and representation analysis, like probing or CCA, would help uncover how alignment behaviors and reasoning skills develop under weak or indirect supervision.