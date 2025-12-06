# pipeline2/prompts.py

# TCGA Cancer Type Standards for consistency
TCGA_CANCER_TYPES = """
IMPORTANT: Use these standardized cancer type names for consistency:

Acute Myeloid Leukemia (LAML), Adrenocortical carcinoma (ACC), Bladder Urothelial Carcinoma (BLCA), 
Brain Lower Grade Glioma (LGG), Breast invasive carcinoma (BRCA), Cervical squamous cell carcinoma and endocervical adenocarcinoma (CESC), 
Cholangiocarcinoma (CHOL), Chronic Myelogenous Leukemia (LCML), Colon adenocarcinoma (COAD), 
Esophageal carcinoma (ESCA), Glioblastoma multiforme (GBM), Head and Neck squamous cell carcinoma (HNSC), 
Kidney Chromophobe (KICH), Kidney renal clear cell carcinoma (KIRC), Kidney renal papillary cell carcinoma (KIRP), 
Liver hepatocellular carcinoma (LIHC), Lung adenocarcinoma (LUAD), Lung squamous cell carcinoma (LUSC), 
Lymphoid Neoplasm Diffuse Large B-cell Lymphoma (DLBC), Mesothelioma (MESO), 
Ovarian serous cystadenocarcinoma (OV), Pancreatic adenocarcinoma (PAAD), Pheochromocytoma and Paraganglioma (PCPG), 
Prostate adenocarcinoma (PRAD), Rectum adenocarcinoma (READ), Sarcoma (SARC), Skin Cutaneous Melanoma (SKCM), 
Stomach adenocarcinoma (STAD), Testicular Germ Cell Tumors (TGCT), Thymoma (THYM), Thyroid carcinoma (THCA), 
Uterine Carcinosarcoma (UCS), Uterine Corpus Endometrial Carcinoma (UCEC), Uveal Melanoma (UVM)

When referring to cancer types, use the full name from this list (e.g., "Acute Myeloid Leukemia" not "AML" or "acute myeloid leukemia").
"""

# PROMPTS FOR THE GENERATION AGENT

# Literature exploration via web search prompt
PROMPT_LITERATURE_EXPLORATION = """
You are an expert tasked with formulating a novel and robust hypothesis to address the following objective.
Describe the proposed hypothesis in detail, including specific entities, mechanisms, and anticipated outcomes.
This description is intended for an audience of domain experts.
You have conducted a thorough review of relevant literature and developed a logical framework
for addressing the objective. The articles consulted, along with your analytical reasoning,
are provided below.

{cancer_type_standards}

Goal: {goal}
Criteria for a strong hypothesis:
{preferences}
Existing hypothesis (if applicable):
{source_hypothesis}
{instructions}
Literature review and analytical rationale (chronologically ordered, beginning
with the most recent analysis):
{articles_with_reasoning}
FORMAT YOUR RESPONSE USING THE FOLLOWING SIMPLE STRUCTURE:

Introduction:
[Brief introduction to the research area and context]

Recent findings and related research:
[Summary of relevant recent research findings]

Hypothesis:
[Clear statement of the proposed hypothesis]

Rationale and specificity:
[Detailed rationale explaining why this hypothesis is plausible and specific]

Experimental design and validation:
[Proposed experimental approaches for testing the hypothesis]

Proposed hypothesis (detailed description for domain experts):
[Comprehensive technical description suitable for scientific experts]
"""

# Simulated scientific debates prompt
PROMPT_SCIENTIFIC_DEBATE = """
You are an expert participating in a collaborative discourse concerning the generation
of a {idea_attributes} hypothesis. You will engage in a simulated discussion with other experts.
The overarching objective of this discourse is to collaboratively develop a novel
and robust {idea_attributes} hypothesis.
Goal: {goal}
Criteria for a high-quality hypothesis:
{preferences}
Instructions:
{instructions}
Review Overview:
{reviews_overview}
Procedure:
Initial contribution (if initiating the discussion):
Propose three distinct {idea_attributes} hypotheses.
Subsequent contributions (continuing the discussion):
* Pose clarifying questions if ambiguities or uncertainties arise.
* Critically evaluate the hypotheses proposed thus far, addressing the following aspects:
  - Adherence to {idea_attributes} criteria.
  - Utility and practicality.
  - Level of detail and specificity.
* Identify any weaknesses or potential limitations.
* Propose concrete improvements and refinements to address identified weaknesses.
* Conclude your response with a refined iteration of the hypothesis.
General guidelines:
* Exhibit boldness and creativity in your contributions.
* Maintain a helpful and collaborative approach.
* Prioritize the generation of a high-quality {idea_attributes} hypothesis.
Termination condition:
When sufficient discussion has transpired (typically 3-5 conversational turns,
with a maximum of 10 turns) and all relevant questions and points have been
thoroughly addressed and clarified, conclude the process by writing "HYPOTHESIS"
(in all capital letters) followed by a concise and self-contained exposition of the finalized idea.

FORMAT YOUR FINAL HYPOTHESIS USING THE FOLLOWING STRUCTURE:
***HYPOTHESIS TITLE***: [Clear, concise title]
***SECTION: INTRODUCTION***: [Introduction to the hypothesis]
***SECTION: RECENT FINDINGS AND RELATED RESEARCH***: [Recent findings and related research]
***SECTION: HYPOTHESIS***: [Hypothesis]
***SECTION: RATIONALE AND SPECIFICITY***: [Rationale and specificity]
***SECTION: EXPERIMENTAL DESIGN AND VALIDATION***: [Experimental design and validation]

IMPORTANT: You MUST use ALL CAPS for each section heading and include the exact *** symbols before and after each heading exactly as shown above. Do not modify this format or omit any sections.
#BEGIN TRANSCRIPT#
{transcript}
#END TRANSCRIPT#
Your Turn:
"""

# Assumptions identification prompt
PROMPT_ASSUMPTIONS_IDENTIFICATION = """
You are a critical thinker specializing in identifying the core assumptions underlying scientific hypotheses.
Your task is to analyze the provided hypothesis and extract its key assumptions, then develop alternative
hypotheses based on challenging these assumptions.
Research Goal: {goal}
Criteria for effective hypotheses:
{preferences}
Existing Hypothesis:
{hypothesis}
Task:
1. Identify 3-5 key assumptions underlying the existing hypothesis
2. For each assumption, develop an alternative hypothesis that challenges or modifies the assumption
3. Ensure each alternative hypothesis still addresses the research goal
4. Provide a brief explanation of why each alternative is scientifically plausible
5. Rank the alternatives by potential significance

FORMAT YOUR RESPONSE USING THE FOLLOWING SIMPLE STRUCTURE:

Introduction:
[Brief introduction to the research area and context]

Recent findings and related research:
[Summary of relevant recent research findings]

Hypothesis:
[Clear statement of the proposed hypothesis]

Rationale and specificity:
[Detailed rationale explaining why this hypothesis is plausible and specific]

Experimental design and validation:
[Proposed experimental approaches for testing the hypothesis]

Proposed hypothesis (detailed description for domain experts):
[Comprehensive technical description suitable for scientific experts]
"""

# Research expansion prompt
PROMPT_RESEARCH_EXPANSION = """
You are an expert in identifying unexplored research directions based on existing scientific work.
Your task is to generate a novel hypothesis that addresses previously unexplored areas of the hypothesis space.
Research Goal: {goal}
Criteria for a strong hypothesis:
{preferences}
Summary of existing hypotheses:
{existing_summary}
Meta-review feedback:
{meta_review_summary}
Identified unexplored areas:
{unexplored_areas}
Task:
Generate a novel research hypothesis that:
1. Addresses one or more of the identified unexplored areas
2. Differs significantly from existing hypotheses
3. Meets the criteria for a strong hypothesis
4. Builds upon insights from the meta-review feedback

FORMAT YOUR RESPONSE USING THE FOLLOWING SIMPLE STRUCTURE:

Introduction:
[Brief introduction to the research area and context]

Recent findings and related research:
[Summary of relevant recent research findings]

Hypothesis:
[Clear statement of the proposed hypothesis]

Rationale and specificity:
[Detailed rationale explaining why this hypothesis is plausible and specific]

Experimental design and validation:
[Proposed experimental approaches for testing the hypothesis]

Proposed hypothesis (detailed description for domain experts):
[Comprehensive technical description suitable for scientific experts]
"""

# PROMPTS FOR THE REFLECTION AGENT
PROMPT_REFLECTION_OBSERVATION = """
You are an expert in scientific hypothesis evaluation. Your task is to analyze the
relationship between a provided hypothesis and observations from a scientific article.
Specifically, determine if the hypothesis provides a novel causal explanation
for the observations, or if they contradict it.
Instructions:
1. Observation extraction: list relevant observations from the article.
2. Causal analysis (individual): for each observation:
   a. State if its cause is already established.
   b. Assess if the hypothesis could be a causal factor (hypothesis => observation).
   c. Start with: "would we see this observation if the hypothesis was true:".
   d. Explain if it's a novel explanation. If not, or if a better explanation exists,
      state: "not a missing piece."
3. Causal analysis (summary): determine if the hypothesis offers a novel explanation
   for a subset of observations. Include reasoning. Start with: "would we see some of
   the observations if the hypothesis was true:".
4. Disproof analysis: determine if any observations contradict the hypothesis.
   Start with: "does some observations disprove the hypothesis:".
5. Conclusion: state: "hypothesis: <already explained, other explanations more likely,
   missing piece, neutral, or disproved>".

Scoring:
* Already explained: hypothesis consistent, but causes are known. No novel explanation.
* Other explanations more likely: hypothesis *could* explain, but better explanations exist.
* Missing piece: hypothesis offers a novel, plausible explanation.
* Neutral: hypothesis neither explains nor is contradicted.
* Disproved: observations contradict the hypothesis.
Important: if observations are expected regardless of the hypothesis, and don't disprove it,
it's neutral.
Article:
{article}
Hypothesis:
{hypothesis}
Response (provide reasoning. end with: "hypothesis: <already explained, other explanations
more likely, missing piece, neutral, or disproved>").
"""

# PROMPTS FOR THE RANKING AGENT
PROMPT_RANKING_COMPARISON = """
You are an expert evaluator tasked with comparing two hypotheses.
Evaluate the two provided hypotheses (hypothesis 1 and hypothesis 2) and determine which one
is superior based on the specified {idea_attributes}.
Provide a concise rationale for your selection, concluding with the phrase "better idea: <1 or 2>".
Goal: {goal}
Evaluation criteria:
{preferences}
Considerations:
{notes}
Each hypothesis includes an independent review. These reviews may contain numerical scores.
Disregard these scores in your comparative analysis, as they may not be directly comparable across reviews.
Hypothesis 1:
{hypothesis 1}
Hypothesis 2:
{hypothesis 2}
Review of hypothesis 1:
{review 1}
Review of hypothesis 2:
{review 2}
Reasoning and conclusion (end with "better hypothesis: <1 or 2>"):
"""

# Pairwise comparison prompt for single-turn comparisons (lower-ranked hypotheses)
PROMPT_PAIRWISE_COMPARISON = """
You are a scientific expert tasked with conducting a pairwise comparison between two hypotheses.
This is a single-turn evaluation for determining which hypothesis is superior based on scientific merit.

Research Goal: {goal}

Evaluation Criteria:
{preferences}

Instructions:
{instructions}

Hypothesis 1:
{hypothesis_1}

Hypothesis 2:
{hypothesis_2}

Please provide a concise scientific comparison focusing on:
1. Scientific validity and rigor
2. Testability and feasibility  
3. Novelty and potential impact
4. Clarity and specificity

Conclude your analysis with a clear decision in the format:
CONCLUSION: Hypothesis [1 or 2] is superior.

Your analysis:
"""

PROMPT_RANKING_DEBATE = """
You are an expert in comparative analysis, simulating a panel of domain experts
engaged in a structured discussion to evaluate two competing hypotheses.
The objective is to rigorously determine which hypothesis is superior based on
a predefined set of attributes and criteria.
The experts possess no pre-existing biases toward either hypothesis and are solely
focused on identifying the optimal choice, given that only one can be implemented.
Goal: {goal}
Criteria for hypothesis superiority:
{preferences}
Hypothesis 1:
{hypothesis 1}
Hypothesis 2:
{hypothesis 2}
Initial review of hypothesis 1:
{review1}
Initial review of hypothesis 2:
{review 2}
Debate procedure:
The discussion will unfold in a series of turns, typically ranging from 3 to 5, with a maximum of 10.
Turn 1: begin with a concise summary of both hypotheses and their respective initial reviews.
Subsequent turns:
* Pose clarifying questions to address any ambiguities or uncertainties.
* Critically evaluate each hypothesis in relation to the stated Goal and Criteria.
This evaluation should consider aspects such as:
Potential for correctness/validity.
Utility and practical applicability.
Sufficiency of detail and specificity.
Novelty and originality.
Desirability for implementation.
* Identify and articulate any weaknesses, limitations, or potential flaws in either hypothesis.
Additional notes:
{notes}
Termination and judgment:
Once the discussion has reached a point of sufficient depth (typically 3-5 turns, up to 10 turns)
and all relevant questions and concerns have been thoroughly addressed, provide a conclusive judgment.
This judgment should succinctly state the rationale for the selection.
Then, indicate the superior hypothesis by writing the phrase "better idea: ",
followed by "1" (for hypothesis 1) or "2" (for hypothesis 2).
"""

# PROMPTS FOR THE EVOLUTION AGENT
PROMPT_EVOLUTION_FEASIBILITY = """
You are an expert in scientific research and technological feasibility analysis.
Your task is to refine the provided conceptual idea, enhancing its practical implementability
by leveraging contemporary technological capabilities. Ensure the revised concept retains
its novelty, logical coherence, and specific articulation.
Goal: {goal}
Guidelines:
1. Begin with an introductory overview of the relevant scientific domain.
2. Provide a concise synopsis of recent pertinent research findings and related investigations,
highlighting successful methodologies and established precedents.
3. Articulate a reasoned argument for how current technological advancements can facilitate
the realization of the proposed concept.
4. CORE CONTRIBUTION: Develop a detailed, innovative, and technologically viable alternative
to achieve the objective, emphasizing simplicity and practicality.
Evaluation Criteria:
{preferences}
Original Conceptualization:
{hypothesis}

FORMAT YOUR RESPONSE USING THE FOLLOWING SIMPLE STRUCTURE:

Introduction:
[Brief introduction to the research area and context]

Recent findings and related research:
[Summary of relevant recent research findings]

Hypothesis:
[Clear statement of the proposed hypothesis]

Rationale and specificity:
[Detailed rationale explaining why this hypothesis is plausible and specific]

Experimental design and validation:
[Proposed experimental approaches for testing the hypothesis]

Proposed hypothesis (detailed description for domain experts):
[Comprehensive technical description suitable for scientific experts]
"""

PROMPT_EVOLUTION_OUT_OF_BOX = """
You are an expert researcher tasked with generating a novel, singular hypothesis
inspired by analogous elements from provided concepts.
Goal: {goal}
Instructions:
1. Provide a concise introduction to the relevant scientific domain.
2. Summarize recent findings and pertinent research, highlighting successful approaches.
3. Identify promising avenues for exploration that may yield innovative hypotheses.
4. CORE HYPOTHESIS: Develop a detailed, original, and specific single hypothesis
for achieving the stated goal, leveraging analogous principles from the provided
ideas. This should not be a mere aggregation of existing methods or entities. Think out-of-the-box.
Criteria for a robust hypothesis:
{preferences}
Inspiration may be drawn from the following concepts (utilize analogy and inspiration,
not direct replication):
{hypotheses}

FORMAT YOUR RESPONSE USING THE FOLLOWING SIMPLE STRUCTURE:

Introduction:
[Brief introduction to the research area and context]

Recent findings and related research:
[Summary of relevant recent research findings]

Hypothesis:
[Clear statement of the proposed hypothesis]

Rationale and specificity:
[Detailed rationale explaining why this hypothesis is plausible and specific]

Experimental design and validation:
[Proposed experimental approaches for testing the hypothesis]

Proposed hypothesis (detailed description for domain experts):
[Comprehensive technical description suitable for scientific experts]
"""

# PROMPTS FOR THE META-REVIEW AGENT
PROMPT_META_REVIEW_GENERATION = """
You are an expert in scientific research and meta-analysis.
Synthesize a comprehensive meta-review of provided reviews
pertaining to the following research goal:
Goal: {goal}
Preferences:
{preferences}
Additional instructions:
{instructions}
Provided reviews for meta-analysis:
{reviews}
Instructions:
* Generate a structured meta-analysis report of the provided reviews.
* Focus on identifying recurring critique points and common issues raised by reviewers.
* The generated meta-analysis should provide actionable insights for researchers
developing future proposals.
* Refrain from evaluating individual proposals or reviews;
focus on producing a synthesized meta-analysis.
Response:
"""

# DRUG REPURPOSING SPECIFIC PROMPTS

DRUG_REPURPOSING_REFLECTION_SUPPLEMENT = """

For drug repurposing hypotheses, also provide:

6. Specify the most promising drug candidate from your analysis as: "FINAL DRUG: [drug name]" where the drug name should be a single, well-established pharmaceutical name (e.g., "Metformin", "Imatinib", "Aspirin").

7. Specify the target cancer type using standardized terminology as: "CANCER TYPE: [cancer type]" using the exact names from the TCGA classification (e.g., "Acute Myeloid Leukemia", "Breast invasive carcinoma").

These recommendations should be based on the scientific evidence and mechanistic rationale discussed in your review.
"""


# ============================================================================
# LETHAL GENES MODE PROMPTS
# ============================================================================

PROMPT_LETHAL_GENES_GENERATION = """
You are analyzing results from a high-throughput CRISPR dual knock out screen that tests for synthetic lethal interactions. If loss of both genes reduces cancer cell viability significantly more than expected, such gene pairs will be flagged as candidate synthetic lethal pairs.

Goal: For each pair, judge the plausibility of novel synthetic lethality and produce two competing mechanistic hypotheses—a Primary and a Rival—with identical structure, each including a Graphviz DOT pathway. Keep clinical notes minimal. Favor hypotheses that describe the synthetic lethality phenotype that are mechanistically novel, biologically relevant, interpretable, and easy to track/test. Analyze only one pair per output report.

If the pair is biologically implausible or no evidence for any relevance under any context, report as random or artifactual hit without proceeding with any mechanistic hypothesis.

More specifically, assess whether this gene pair could represent a valid synthetic lethal interaction based on biological and clinical relevance. Then, develop novel, plausible and experimentally trackable hypotheses that explain the mechanism of this synthetic lethality – how loss of both genes causes cell death while loss of either does not. Each hypothesis should come with one rival hypothesis which is an alternative mechanistic explanation of the same SL pair. In other words, ruling out the rival hypothesis can strengthen the primary hypothesis.

GENE PAIR TO ANALYZE:
Gene A: {gene_a}
Gene B: {gene_b}

LITERATURE CONTEXT:
{literature_context}

Step-by-step guidance for mechanistic hypothesis:

1. Biological Plausibility
   - What is known (or predicted) about each gene's molecular function, cellular role, or pathway involvement?
   - Are they co-expressed in relevant tissues or cell types?
   - Could they function in the same pathway, redundant systems, or complementary roles?
   - Based on current knowledge, does an SL relationship make biological sense?

2. Clinical Relevance
   - Are either or both genes recurrently mutated or altered in cancer?
   - If one gene is frequently mutated, is the other gene druggable, or part of a druggable complex/pathway?
   - Is there tissue specificity or cancer-type specificity that could make this pair therapeutically actionable?
   - Would this pair be of interest for clinical follow-up?

3. Mechanistic Hypothesis
   - If the gene pair is biologically relevant to SL with clinical potential, propose a plausible biological mechanism by which loss of both genes leads to cancer cell death, but loss of either alone does not.
   - Reference relevant cellular processes, interactions, or compensatory failures.
   - Explain in detail the possible cascade of molecular events in response to the loss of both genes in comparison with wild type or loss of either one of them.
   - Provide a pathway representation for the known cascade of events and explicitly annotate the hypothesized aspect of the mechanism, please ensure using the same pathway representation for all hypotheses derived.
   - Consider the specific phenotype or outcome that led to cell death (e.g. checkpoint failure, apoptosis, etc)
   - Offer at least rival (competing) hypothesis as an alternative mechanism for the SL
   - Following the same structure as the primary hypothesis, but a distinct, mutually exclusive mechanistic explanation for the same SL pair

4. Experimental/computational Analysis Follow-up Feasibility
   - Consider experiments/computational analysis to distinguish between competing hypotheses and increase confidence in the primary one
   - Favor hypothesis that can be tracked and tested in simple and cost efficient ways.

OUTPUT FORMAT (MUST FOLLOW EXACTLY):

SL Pair: {gene_a} and {gene_b}

Biological plausibility: [State: "plausible dependency" or "likely artifact"]

Clinical relevance: [Brief statement on cancer mutations and druggability]

Summary and rationale: [Comprehensive summary of why this pair is/isn't plausible]

[IF ARTIFACT, STOP HERE. OTHERWISE CONTINUE:]

Primary:
    Statement: [One sentence description of primary hypothesis]
    Description of single loss mechanism: [What happens when only Gene A or only Gene B is lost]
    Double loss mechanism prediction: [Detailed mechanism of how combined loss causes cell death]
    Assumptions: [List key assumptions underlying this mechanism]
    Predicted failure: [What specifically fails leading to cell death]
    Key intermediate components: [Critical proteins/pathways involved]
    Key readouts if the hypothesis is true: [Measurable experimental outcomes]

Rival:
    Statement: [One sentence description of DISTINCT alternative mechanism]
    Description of single loss mechanism: [What happens when only Gene A or only Gene B is lost]
    Double loss mechanism prediction: [Alternative mechanism of how combined loss causes cell death]
    Assumptions: [List key assumptions for this alternative mechanism]
    Predicted failure: [What specifically fails in this alternative scenario]
    Key intermediate components: [Critical proteins/pathways in alternative mechanism]
    Key readouts if the hypothesis is true: [Measurable experimental outcomes for rival]

Pathway visualization:
[Provide Graphviz DOT code showing:
- Known pathways/processes (black edges)
- Primary hypothesis predictions (darkorange2 edges)
- Rival hypothesis predictions (darkorchid2 edges)
Example format:
digraph SL {{
    rankdir=LR;
    node [shape=box];

    // Known interactions (black)
    GeneA -> Pathway1 [color=black, label="known"];
    GeneB -> Pathway2 [color=black, label="known"];

    // Primary hypothesis (darkorange2)
    Pathway1 -> CellDeath [color=darkorange2, label="primary"];

    // Rival hypothesis (darkorchid2)
    Pathway2 -> CellDeath [color=darkorchid2, label="rival"];
}}
]

Contrast description:
Primary hypothesis is favored if [specific condition that rejects rival]
Rival is favored if [specific condition that rejects primary]

Attributes:
* Biological relevance: the hypothesis must be biologically relevant 
* Novelty in the mechanistic interpretation (i.e. is the mechanism already well studied in human cancer?) The pair may be previously reported in screening data but the mechanism is never discussed. 
* Clarity in mechanistic explanation (explanation of known versus the gap, mapping out possible intermediate components, clear pathway visualization)
* Follow-up tractability (Can the proposed mechanism be tested with simple and effective follow-up)
* Quality of rival hypotheses (mutually exclusive alternative explanations with trackable predictions) 
* Clinical relevance (e.g. based on cancer mutation frequency and druggability of gene partners) (secondary with brief sentence)

"""

PROMPT_LETHAL_GENES_REFLECTION = """
You are evaluating a synthetic lethal gene pair hypothesis for quality and plausibility.

HYPOTHESIS TO EVALUATE:
{hypothesis_text}

LITERATURE CONTEXT:
{literature_context}

Evaluate this hypothesis on the following criteria (score each 0-10):

1. BIOLOGICAL RELEVANCE (Score 0-10)
   - Is the hypothesis biologically relevant and grounded in known biological processes?
   - Are the gene functions and interactions consistent with established biology?
   - Score 10 = highly relevant and well-supported, Score 0 = biologically implausible

2. NOVELTY IN MECHANISTIC INTERPRETATION (Score 0-10)
   - Is the proposed mechanism novel and not already well-studied in human cancer?
   - Note: The gene pair may be previously reported in screening data, but is the mechanistic interpretation new?
   - Does it provide unexpected biological insights beyond existing literature?
   - Score 10 = completely novel mechanistic interpretation, Score 0 = well-established mechanism

3. CLARITY IN MECHANISTIC EXPLANATION (Score 0-10)
   - Is there clear explanation of what is known versus the knowledge gap?
   - Are intermediate molecular components in the pathway clearly mapped out?
   - Is the pathway visualization clear and informative?
   - Are single-loss and double-loss mechanisms explicitly detailed?
   - Score 10 = crystal clear with all components mapped, Score 0 = vague or confusing

4. FOLLOW-UP TRACTABILITY (Score 0-10)
   - Can the proposed mechanism be tested with simple and effective follow-up experiments?
   - Are the key readouts measurable, specific, and accessible?
   - Are the predicted intermediate components experimentally tractable?
   - Score 10 = very simple and effective testing, Score 0 = impractical or overly complex

5. QUALITY OF RIVAL HYPOTHESES (Score 0-10)
   - **CRITICAL**: Are BOTH Primary and Rival hypotheses present and fully developed?
   - Are they mutually exclusive alternative explanations?
   - Do they have trackable predictions that can distinguish between them?
   - Are both hypotheses equally well-developed with mechanistic detail?
   - **If no Rival hypothesis is present, score MUST be 0-2**
   - Score 10 = excellent rival with clear contrast and equal depth, Score 0 = missing or poorly developed rival

6. CLINICAL RELEVANCE (Score 0-10)
   - Based on cancer mutation frequency and druggability of gene partners
   - Are the genes mutated in relevant cancers at meaningful frequency?
   - Is there therapeutic potential (druggable targets or pathways)?
   - Note: This is secondary - provide brief assessment
   - Score 10 = high clinical potential, Score 0 = no clinical relevance

ALSO PROVIDE:
- Overall assessment of biological plausibility (plausible/artifact)
- Key strengths of this hypothesis (2-3 points)
- Key weaknesses or concerns (2-3 points)
- Recommendation for further development (ACCEPT/REVISE/REJECT)

OUTPUT FORMAT:
Biological Relevance Score: [0-10]
Novelty Score: [0-10]
Mechanistic Clarity Score: [0-10]
Tractability Score: [0-10]
Rival Quality Score: [0-10]
Clinical Relevance Score: [0-10]

Overall Plausibility: [plausible/artifact]

Strengths:
[List 2-3 key strengths]

Weaknesses:
[List 2-3 key concerns]

Recommendation: [ACCEPT/REVISE/REJECT]

Detailed Justification:
[Provide comprehensive explanation of your evaluation, focusing on biological relevance, mechanistic novelty, and the quality of rival hypotheses]
"""

PROMPT_LETHAL_GENES_EVOLUTION_REFINEMENT = """
You are refining a synthetic lethal gene pair hypothesis to improve its quality.

ORIGINAL HYPOTHESIS:
{original_hypothesis}

EVOLUTION STRATEGY: {evolution_strategy}

PARENT HYPOTHESES (if crossover):
{parent_info}

LITERATURE CONTEXT:
{literature_context}

Your task is to create an improved version of this hypothesis that:

1. ENHANCES MECHANISTIC DETAIL
   - Add more specific molecular interactions
   - Clarify the cascade of events more precisely
   - Include specific proteins, modifications, or signaling events

2. STRENGTHENS PRIMARY-RIVAL CONTRAST
   - Make the primary and rival hypotheses more distinct
   - Ensure they are truly mutually exclusive
   - Provide clearer experimental predictions to distinguish them

3. IMPROVES PATHWAY VISUALIZATION
   - Add more intermediate steps to the DOT graph
   - Clarify the distinction between known and hypothesized edges
   - Ensure primary and rival paths are clearly differentiated

4. INCREASES EXPERIMENTAL TRACTABILITY
   - Suggest more specific and measurable readouts
   - Identify key intermediate components that can be tracked
   - Propose simple validation experiments

MAINTAIN:
- The same gene pair ({gene_a} and {gene_b})
- The overall plausibility assessment
- The same output format structure

OUTPUT THE REFINED HYPOTHESIS using the exact same format as the original, but with improvements incorporated throughout all sections.
""" 