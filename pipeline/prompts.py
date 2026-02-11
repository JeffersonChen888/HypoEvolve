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

# Approved drugs with validated DepMap targets for drug repurposing
# Source: Open Targets Platform (phase 4 approved drugs with target genes in DepMap CRISPR data)
APPROVED_DRUGS_DEPMAP = [
    "SIMVASTATIN", "ATORVASTATIN", "LOVASTATIN", "FLUVASTATIN", "PRAVASTATIN", "ROSUVASTATIN",
    "METFORMIN", "PIOGLITAZONE", "ROSIGLITAZONE",
    "HYDROXYCHLOROQUINE",
    "PROPRANOLOL", "CARVEDILOL", "LOSARTAN", "VERAPAMIL", "DIGOXIN",
    "SERTRALINE", "FLUOXETINE", "THIORIDAZINE", "PIMOZIDE", "CHLORPROMAZINE",
    "OMEPRAZOLE", "ESOMEPRAZOLE", "PANTOPRAZOLE", "LANSOPRAZOLE",
    "ASPIRIN", "CELECOXIB", "SULINDAC", "INDOMETHACIN", "DICLOFENAC",
    "DOXYCYCLINE",
    "DISULFIRAM", "THALIDOMIDE", "LENALIDOMIDE", "SIROLIMUS", "EVEROLIMUS",
    "RITONAVIR", "LEFLUNOMIDE", "SULFASALAZINE", "PENTOXIFYLLINE", "VORINOSTAT", "VALPROIC ACID",
    "IMATINIB", "DASATINIB", "NILOTINIB", "SORAFENIB", "SUNITINIB",
    "ERLOTINIB", "GEFITINIB", "LAPATINIB", "VEMURAFENIB", "TRAMETINIB",
    "OLAPARIB", "RUCAPARIB", "VENETOCLAX", "IBRUTINIB",
    "PALBOCICLIB", "RIBOCICLIB", "ABEMACICLIB", "ALPELISIB", "RUXOLITINIB", "TOFACITINIB"
]

APPROVED_DRUGS_CONSTRAINT = """
⚠️ CRITICAL DRUG CONSTRAINT - READ CAREFULLY ⚠️

You MUST select your drug repurposing candidate ONLY from this approved list:

{drug_list}

These 62 drugs have been verified to have:
1. FDA approval (Phase 4)
2. Known target genes in Open Targets Platform
3. Target genes present in DepMap CRISPR dependency data

CORRECT EXAMPLES (drugs FROM the approved list):
✓ FINAL DRUG: METFORMIN (in the list)
✓ FINAL DRUG: SULFASALAZINE (in the list)
✓ FINAL DRUG: DOXYCYCLINE (in the list)
✓ FINAL DRUG: IMATINIB (in the list)

INCORRECT EXAMPLES (drugs NOT in the approved list - DO NOT USE):
✗ FINAL DRUG: TIGECYCLINE (not in the list - DO NOT USE)
✗ FINAL DRUG: AURANOFIN (not in the list - DO NOT USE)
✗ FINAL DRUG: NICLOSAMIDE (not in the list - DO NOT USE)
✗ FINAL DRUG: CISPLATIN (not in the list - DO NOT USE)
✗ FINAL DRUG: PACLITAXEL (not in the list - DO NOT USE)

VERIFICATION STEP:
Before finalizing your hypothesis, verify that your chosen drug appears in the approved list above.
If your drug is not in the list, you MUST choose a different drug from the list.

Your FINAL DRUG must be one of the 62 exact drug names shown above.
"""

# PROMPTS FOR THE GENERATION AGENT

# Literature exploration via web search prompt
PROMPT_LITERATURE_EXPLORATION = """
You are an expert tasked with formulating a novel and robust hypothesis to address the following objective.
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

REQUIRED OUTPUT FORMAT:
You MUST format your response with the following four sections, using these exact headers:

TITLE:
[A concise, descriptive title in one line]

SUMMARY:
[A single-sentence summary capturing the core idea of your hypothesis]

HYPOTHESIS:
[A clear statement of your proposed hypothesis in 2-3 sentences, including the key mechanism, entities involved, and expected outcomes]

RATIONALE:
[A detailed explanation (1-3 paragraphs) of why this hypothesis is scientifically plausible and specific. Include:
- Key mechanisms and molecular/biological entities
- Evidence from literature supporting plausibility
- Why this addresses the research goal
- What makes this hypothesis testable and falsifiable]

{mode_specific_output}

EXAMPLE FORMAT:

TITLE:
Targeting mitochondrial metabolism in KRAS-mutant cancers via complex I inhibition

SUMMARY:
KRAS-mutant cancers rely on mitochondrial oxidative phosphorylation for survival, making them vulnerable to complex I inhibitors like metformin.

HYPOTHESIS:
KRAS-mutant cancer cells exhibit heightened dependency on mitochondrial complex I activity to maintain ATP production and redox balance. Pharmacological inhibition of complex I with metformin will selectively induce energetic stress and apoptosis in KRAS-mutant tumor cells while sparing normal cells with lower metabolic demands.

RATIONALE:
Recent studies demonstrate that KRAS mutations rewire cellular metabolism, increasing reliance on oxidative phosphorylation rather than glycolysis (contrary to the Warburg effect). Complex I is the primary entry point for electrons into the mitochondrial electron transport chain, and KRAS-mutant cells show elevated complex I activity to sustain high ATP flux. Metformin, a well-characterized complex I inhibitor, has shown selective toxicity in KRAS-mutant cell lines in vitro by disrupting the NAD+/NADH ratio and triggering AMPK-mediated stress responses. This hypothesis is testable through dose-response viability assays in isogenic cell line pairs (KRAS-WT vs KRAS-mutant), Seahorse metabolic profiling to measure oxygen consumption rates, and xenograft studies measuring tumor burden under metformin treatment. The specificity can be validated by genetic rescue experiments restoring complex I activity or by supplementing alternative energy substrates.

FINAL DRUG: METFORMIN
CANCER TYPE: Lung adenocarcinoma
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

Your task is to assess whether a given gene pair could represent a valid synthetic lethal interaction, and if so, propose a plausible mechanistic hypothesis explaining how loss of both genes causes cell death while loss of either alone does not.

GENE PAIR TO ANALYZE:
Gene A: {gene_a}
Gene B: {gene_b}

LITERATURE CONTEXT:
{literature_context}

ANALYSIS GUIDANCE:

When developing your hypothesis, consider the following aspects (include all in your RATIONALE):

1. Biological Plausibility
   - What is known about each gene's molecular function, cellular role, or pathway involvement?
   - Are they co-expressed in relevant tissues or cell types?
   - Could they function in the same pathway, redundant systems, or complementary roles?
   - Based on current knowledge, does a synthetic lethal relationship make biological sense?

2. Clinical Relevance
   - Are either or both genes recurrently mutated or altered in cancer?
   - If one gene is frequently mutated, is the other gene druggable?
   - Is there tissue or cancer-type specificity that could make this pair therapeutically actionable?

3. Mechanistic Explanation
   - Propose a plausible biological mechanism by which loss of both genes leads to cell death
   - Reference relevant cellular processes, interactions, or compensatory failures
   - Explain the cascade of molecular events when both genes are lost vs single loss
   - Consider the specific phenotype or outcome (e.g., checkpoint failure, apoptosis)

4. Experimental Follow-up
   - How could this hypothesis be tested experimentally?
   - Favor approaches that are simple and cost-efficient

OUTPUT FORMAT (MUST FOLLOW EXACTLY):

TITLE:
[A concise, descriptive title for your hypothesis about this gene pair]

SUMMARY:
[A single-sentence summary of your assessment of this gene pair]

HYPOTHESIS:
[A clear statement of your proposed mechanism in 2-3 sentences, explaining how loss of both genes causes synthetic lethality]

RATIONALE:
[A detailed explanation (2-4 paragraphs) that addresses ALL of the following:
- Biological Plausibility: gene functions, pathway involvement, co-expression, does SL make sense
- Clinical Relevance: cancer mutations, druggability, tissue specificity, clinical interest
- Mechanistic Explanation: how loss of both genes causes death, cellular processes, cascade of events, phenotype
- Experimental Follow-up: how to test the hypothesis, simple/cost-efficient approaches]

FINAL_PREDICTION: [well-based OR random]

Apply rigorous criteria:
- "well-based": Requires DIRECT functional relationship - shared pathway, physical interaction, redundant function, or established synthetic lethal pair in published literature
- "random": No direct functional link found, speculative connection only, or genes operate in unrelated pathways

Be skeptical. A plausible-sounding hypothesis is NOT sufficient - require concrete evidence.

CRITICAL: You MUST end your response with the FINAL_PREDICTION line. Your response is invalid without it.
"""

PROMPT_LETHAL_GENES_REFLECTION = """
You are evaluating a synthetic lethal gene pair hypothesis for CLASSIFICATION ACCURACY.

Your goal is to assess whether the FINAL_PREDICTION (well-based vs random) is justified by the evidence.

HYPOTHESIS TO EVALUATE:
{hypothesis_text}

LITERATURE CONTEXT:
{literature_context}

Evaluate this hypothesis on the following criteria (score each 0-10):

1. DIRECT EVIDENCE STRENGTH (Score 0-10)
   - Is there DIRECT evidence for a functional relationship between these genes?
   - Physical interaction, shared essential pathway, redundant function, or published synthetic lethal pair?
   - Score 10 = direct experimental evidence exists (e.g., published SL screen, protein interaction data)
   - Score 5 = indirect pathway connection only (e.g., both in DNA repair, but no direct link)
   - Score 0 = no functional link found, purely speculative connection

2. LITERATURE SUPPORT (Score 0-10)
   - Are there published studies supporting this specific interaction?
   - Has this gene pair been studied in synthetic lethality or functional genomics context?
   - Score 10 = multiple publications directly support this pair
   - Score 5 = tangential literature (genes studied separately, not together)
   - Score 0 = no relevant literature for this pair

3. BIOLOGICAL PLAUSIBILITY (Score 0-10)
   - Is the proposed mechanism biologically sensible?
   - Are the gene functions and proposed interactions consistent with known biology?
   - Score 10 = highly plausible, well-supported mechanism
   - Score 5 = plausible but speculative
   - Score 0 = biologically implausible

4. EVIDENCE-PREDICTION CALIBRATION (Score 0-10)
   - Does the FINAL_PREDICTION match the evidence strength?
   - For "well-based": Is there sufficient DIRECT evidence (criteria 1-2 should be high)?
   - For "random": Is the prediction appropriately cautious given weak evidence?
   - Score 10 = prediction well-calibrated to evidence strength
   - Score 0 = overconfident prediction not supported by evidence

   CALIBRATION GUIDELINES:
   - "well-based" + weak evidence (criteria 1-2 avg < 5) → Score 2-4 (overconfident)
   - "well-based" + strong evidence (criteria 1-2 avg > 7) → Score 8-10 (justified)
   - "random" + weak evidence → Score 8-10 (appropriately cautious)
   - "random" + strong evidence → Score 2-4 (missed opportunity)

OUTPUT FORMAT:
Direct Evidence Score: [0-10]
Literature Support Score: [0-10]
Biological Plausibility Score: [0-10]
Evidence-Prediction Calibration Score: [0-10]

Overall Assessment: [well-calibrated/overconfident/too-cautious]

Strengths:
[List 2-3 key strengths]

Weaknesses:
[List 2-3 key concerns]

Recommendation: [ACCEPT/REVISE/REJECT]

Justification:
[Brief explanation of whether the FINAL_PREDICTION is justified by the evidence]
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

OUTPUT FORMAT (MUST FOLLOW EXACTLY):

SL Pair: {gene_a} and {gene_b}

Biological plausibility: [State: "plausible dependency" or "likely artifact"]

Clinical relevance: [Brief statement on cancer mutations and druggability]

Summary and rationale: [Comprehensive summary of why this pair is/isn't plausible]

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
[Provide Graphviz DOT code]

Contrast description:
Primary hypothesis is favored if [specific condition that rejects rival]
Rival is favored if [specific condition that rejects primary]

CRITICAL - YOU MUST END YOUR RESPONSE WITH THIS EXACT LINE:
FINAL_PREDICTION: [well-based OR random]

Example valid endings:
FINAL_PREDICTION: well-based
FINAL_PREDICTION: random

DO NOT OMIT THIS LINE. YOUR RESPONSE IS INVALID WITHOUT IT.
"""


# =============================================================================
# TOURNAMENT COMPARISON PROMPT
# =============================================================================

PROMPT_TOURNAMENT_COMPARISON = """
You are judging two synthetic lethality hypotheses from a CRISPR screen.
Compare them based on scientific rigor and reasoning strength.

HYPOTHESIS A (Gene Pair: {gene_pair_a}):
{hypothesis_a}

HYPOTHESIS B (Gene Pair: {gene_pair_b}):
{hypothesis_b}

Judge which hypothesis is STRONGER using these criteria:

1. **Biological Relevance**: Is the hypothesis biologically grounded?

2. **Novelty**: Is the mechanistic interpretation novel, not already well-studied?
   (The pair may be previously reported but mechanism never discussed)

3. **Mechanistic Clarity**: Clear explanation of known vs gaps?
   Intermediate components mapped out? Clear pathway visualization?

4. **Follow-up Tractability**: Can the mechanism be tested with simple,
   effective experiments?

5. **Rival Quality**: Are alternative explanations mutually exclusive
   with trackable predictions?

6. **Clinical Relevance** (secondary): Cancer mutation frequency?
   Druggability of gene partners?

IMPORTANT: Judge based on reasoning quality, NOT biological similarity.
Different gene pairs can be fairly compared on scientific rigor.

Provide brief analysis for each criterion, then conclude:
WINNER: [A or B]
REASONING: [One sentence explaining why]
""" 

# =============================================================================
# T2D DRUG TARGET IDENTIFICATION PROMPTS
# =============================================================================
PROMPT_T2D_GENERATION = """You are a computational biologist identifying novel drug targets for Type 2 Diabetes.

=== GENE EXPRESSION ANALYSIS (MASKED) ===
{analysis_context}

=== T2D LITERATURE CONTEXT ===
{literature_context}

=== INSTRUCTIONS ===
Based on the masked gene analysis above AND the T2D literature context, identify the TOP 3-5
most promising drug targets for Type 2 Diabetes treatment and RANK them by priority.

IMPORTANT:
- Select targets ONLY from the candidate genes listed (G00001, G00002, etc.)
- Use the DATA EVIDENCE to justify your selections and ranking
- Consider how the literature context informs therapeutic potential
- Do NOT try to guess real gene names - work only with masked identifiers
- RANK your selections from most promising (#1) to least promising (#5)

CRITICAL CONSTRAINTS:
- You MUST select gene IDs from the priority table (e.g., G00042, G00015)
- Your reasoning must be based ONLY on the data shown
- Do NOT guess real gene names or use prior knowledge about specific genes
- You MAY use knowledge about pathways (KEGG, GO terms) and TFs

=== OUTPUT FORMAT ===
TITLE: [Descriptive title for your hypothesis set]

RANKED_TARGETS:
1. [Gene ID] | Score: [X/17] | [One-sentence rationale based on data]
2. [Gene ID] | Score: [X/17] | [One-sentence rationale based on data]
3. [Gene ID] | Score: [X/17] | [One-sentence rationale based on data]
4. [Gene ID] | Score: [X/17] | [One-sentence rationale based on data] (optional)
5. [Gene ID] | Score: [X/17] | [One-sentence rationale based on data] (optional)

PRIMARY_TARGET: [Gene ID of your #1 ranked target]

SUMMARY: [2-3 sentence summary of your top target and mechanism]

DATA_EVIDENCE:
- Priority Score: [X/17 for primary target]
- Cross-dataset consistency: [Yes/No, N datasets]
- Key pathways involved: [List relevant pathways from analysis]
- WGCNA module role: [If applicable]
- Regulating TFs: [If applicable]

MECHANISM_HYPOTHESIS: [Detailed explanation of proposed mechanism for PRIMARY target based on pathway and network context]

TISSUE_RATIONALE: [Which tissues are most relevant and why, based on the data]

THERAPEUTIC_APPROACH: [Activation or inhibition? Small molecule or biologic?]

PREDICTED_OUTCOME: [Expected therapeutic benefit if primary target is modulated]

RANKING_RATIONALE: [Brief explanation of why you ranked the targets in this order - what differentiates #1 from #2, etc.]

CONFIDENCE: [HIGH/MEDIUM/LOW based on strength of data support]
"""

PROMPT_T2D_REFLECTION = """You are evaluating a RANKED drug target hypothesis set for Type 2 Diabetes.

=== HYPOTHESIS TO EVALUATE ===
{hypothesis}

=== EVALUATION CONTEXT ===
{evaluation_context}

=== SCORING CRITERIA ===
Evaluate on these dimensions (1-10 each):

1. DATA SUPPORT (weight: 25%)
   - Are the ranked targets in the top priority tier?
   - Is there cross-dataset consistency for top-ranked targets?
   - How many evidence types support the primary target?

2. RANKING QUALITY (weight: 25%)
   - Is the ranking justified by the data?
   - Are higher-ranked targets clearly stronger than lower-ranked ones?
   - Does the ranking rationale make sense?

3. MECHANISTIC COHERENCE (weight: 20%)
   - Does the proposed mechanism align with the pathway context?
   - Is the tissue rationale supported by the data?
   - Are the TF relationships considered?

4. THERAPEUTIC POTENTIAL (weight: 20%)
   - Is the therapeutic approach (activation/inhibition) appropriate?
   - Are the targets likely druggable based on network position?
   - Are there safety considerations?

5. NOVELTY (weight: 10%)
   - Does the hypothesis integrate data in a creative way?
   - Does it propose non-obvious connections?

=== OUTPUT FORMAT ===
DATA_SUPPORT_SCORE: [1-10]
DATA_SUPPORT_RATIONALE: [Explanation]

RANKING_QUALITY_SCORE: [1-10]
RANKING_QUALITY_RATIONALE: [Explanation of whether ranking is justified]

MECHANISTIC_SCORE: [1-10]
MECHANISTIC_RATIONALE: [Explanation]

THERAPEUTIC_SCORE: [1-10]
THERAPEUTIC_RATIONALE: [Explanation]

NOVELTY_SCORE: [1-10]
NOVELTY_RATIONALE: [Explanation]

OVERALL_FITNESS: [Calculated: data*0.25 + ranking*0.25 + mechanism*0.20 + therapeutic*0.20 + novelty*0.10, scaled to 0-100]

RANKED_TARGETS_ASSESSMENT:
- Target #1 quality: [STRONG/MODERATE/WEAK]
- Target #2 quality: [STRONG/MODERATE/WEAK]
- Target #3 quality: [STRONG/MODERATE/WEAK]

IMPROVEMENT_SUGGESTIONS: [Specific, actionable suggestions]
"""

PROMPT_T2D_CROSSOVER = """You are combining two T2D drug target hypotheses to create a stronger child hypothesis.

=== PARENT HYPOTHESIS 1 ===
Target: {parent1_target}
Score: {parent1_score}
{parent1_content}

=== PARENT HYPOTHESIS 2 ===
Target: {parent2_target}
Score: {parent2_score}
{parent2_content}

=== ANALYSIS DATA ===
{analysis_context}

=== TASK ===
Create a NEW hypothesis that:
1. Combines the strongest mechanistic insights from both parents
2. Selects the TOP 3-5 targets with best data support and RANK them
3. Addresses weaknesses identified in either parent
4. Maintains scientific rigor

CONSTRAINTS:
- The child MUST select targets from the priority table (G00001, G00042, etc.)
- The mechanism should be MORE comprehensive than either parent
- Base reasoning ONLY on the data provided
- RANK your targets from most promising (#1) to least promising (#5)

=== OUTPUT FORMAT ===
TITLE: [Descriptive title for your hypothesis set]

RANKED_TARGETS:
1. [Gene ID] | Score: [X/17] | [One-sentence rationale based on data]
2. [Gene ID] | Score: [X/17] | [One-sentence rationale based on data]
3. [Gene ID] | Score: [X/17] | [One-sentence rationale based on data]
4. [Gene ID] | Score: [X/17] | [One-sentence rationale based on data] (optional)
5. [Gene ID] | Score: [X/17] | [One-sentence rationale based on data] (optional)

PRIMARY_TARGET: [Gene ID of your #1 ranked target]

SUMMARY: [2-3 sentence summary combining insights from both parents]

DATA_EVIDENCE:
- Priority Score: [X/17 for primary target]
- Cross-dataset consistency: [Yes/No, N datasets]
- Key pathways involved: [List relevant pathways from analysis]
- WGCNA module role: [If applicable]
- Regulating TFs: [If applicable]

MECHANISM_HYPOTHESIS: [Detailed explanation combining mechanistic insights from both parents]

TISSUE_RATIONALE: [Which tissues are most relevant and why]

THERAPEUTIC_APPROACH: [Activation or inhibition? Small molecule or biologic?]

PREDICTED_OUTCOME: [Expected therapeutic benefit if primary target is modulated]

RANKING_RATIONALE: [Why you ranked the targets in this order]

CONFIDENCE: [HIGH/MEDIUM/LOW based on strength of data support]
"""

PROMPT_T2D_MUTATION = """You are refining a T2D drug target hypothesis through creative exploration.

=== ORIGINAL HYPOTHESIS ===
{hypothesis}

=== ANALYSIS DATA ===
{analysis_context}

=== MUTATION STRATEGY: {strategy} ===
{strategy_instructions}

=== TASK ===
Generate a MUTATED version following the strategy above.

CONSTRAINTS:
- The mutated hypothesis MUST select targets from the priority table (G00001, G00042, etc.)
- Maintain scientific rigor
- Base reasoning ONLY on data patterns
- RANK your targets from most promising (#1) to least promising (#5)

=== OUTPUT FORMAT ===
TITLE: [Descriptive title for your mutated hypothesis set]

RANKED_TARGETS:
1. [Gene ID] | Score: [X/17] | [One-sentence rationale based on data]
2. [Gene ID] | Score: [X/17] | [One-sentence rationale based on data]
3. [Gene ID] | Score: [X/17] | [One-sentence rationale based on data]
4. [Gene ID] | Score: [X/17] | [One-sentence rationale based on data] (optional)
5. [Gene ID] | Score: [X/17] | [One-sentence rationale based on data] (optional)

PRIMARY_TARGET: [Gene ID of your #1 ranked target]

SUMMARY: [2-3 sentence summary of the mutated hypothesis]

DATA_EVIDENCE:
- Priority Score: [X/17 for primary target]
- Cross-dataset consistency: [Yes/No, N datasets]
- Key pathways involved: [List relevant pathways from analysis]
- WGCNA module role: [If applicable]
- Regulating TFs: [If applicable]

MECHANISM_HYPOTHESIS: [Detailed explanation of proposed mechanism following the mutation strategy]

TISSUE_RATIONALE: [Which tissues are most relevant and why]

THERAPEUTIC_APPROACH: [Activation or inhibition? Small molecule or biologic?]

PREDICTED_OUTCOME: [Expected therapeutic benefit if primary target is modulated]

RANKING_RATIONALE: [Why you ranked the targets in this order]

CONFIDENCE: [HIGH/MEDIUM/LOW based on strength of data support]
"""

T2D_MUTATION_STRATEGIES = {
    "alternative_target": """
STRATEGY: Alternative Target Selection
- Keep the general mechanism theme
- Select a DIFFERENT target gene from the top 50 priority genes
- Explain why this alternative might be superior
""",

    "mechanism_refinement": """
STRATEGY: Mechanism Refinement
- Keep the same target gene
- Propose a MORE DETAILED or ALTERNATIVE mechanism
- Incorporate additional pathway or TF evidence not used in original
""",

    "therapeutic_pivot": """
STRATEGY: Therapeutic Approach Pivot
- Keep the same target
- Consider opposite modulation (if original was inhibition, consider activation)
- Or propose combination approach with another target
""",

    "tissue_focus": """
STRATEGY: Tissue-Specific Refinement
- Keep the same target
- Focus on a SPECIFIC tissue (islets, muscle, or adipose)
- Deepen the tissue-specific mechanistic rationale
"""
}

# =============================================================================
# T2D DRUGGABILITY-ENHANCED PROMPTS (Option A)
# =============================================================================

PROMPT_T2D_GENERATION_WITH_DRUGGABILITY = """You are a computational biologist identifying novel drug targets for Type 2 Diabetes.

=== GENE EXPRESSION ANALYSIS (MASKED) ===
{analysis_context}

=== IDG PROTEIN FAMILIES (SAFE, NON-IDENTIFYING) ===
{druggability_context}

=== T2D PATHOPHYSIOLOGY CONTEXT ===
{literature_context}

=== INSTRUCTIONS ===
Based on the masked gene analysis, IDG protein families, AND the T2D pathophysiology context,
identify the TOP 3-5 most promising drug targets for Type 2 Diabetes treatment and RANK them.

IMPORTANT:
- Select targets ONLY from the candidate genes listed (G00001, G00002, etc.)
- Use BOTH expression data AND IDG family to inform your selections
- Consider location (Membrane/Cytoplasm/Nucleus) for therapeutic accessibility
- Do NOT try to guess real gene names - work only with masked identifiers
- RANK your selections from most promising (#1) to least promising (#5)

IDG FAMILY CONTEXT (for reference only):
Proteins are classified into standard IDG families (shared by 100s of genes):
- GPCR: G protein-coupled receptor (~800 human proteins)
- Kinase: Protein kinase (~500 human proteins)
- IC: Ion channel (~300 human proteins)
- NR: Nuclear receptor (~48 human proteins)
- TF: Transcription factor
- Enzyme: All enzymes
- Transporter: All membrane transporters
- Epigenetic: Chromatin/histone modifier

NOTE: Family membership alone does NOT indicate disease relevance.
Prioritize based on DATA EVIDENCE (expression, pathways, consistency), not family.

CRITICAL CONSTRAINTS:
- You MUST select gene IDs from the priority table (e.g., G00042, G00015)
- Your reasoning must be based ONLY on the data shown
- Do NOT guess real gene names or use prior knowledge about specific genes
- You MAY use knowledge about pathways (KEGG, GO terms) and TFs
- You MAY use the druggability features to prioritize targets

=== OUTPUT FORMAT ===
TITLE: [Descriptive title for your hypothesis set]

RANKED_TARGETS:
1. [Gene ID] | Score: [X/17] | IDG Family: [family] | [One-sentence rationale based on DATA]
2. [Gene ID] | Score: [X/17] | IDG Family: [family] | [One-sentence rationale based on DATA]
3. [Gene ID] | Score: [X/17] | IDG Family: [family] | [One-sentence rationale based on DATA]
4. [Gene ID] | Score: [X/17] | IDG Family: [family] | [One-sentence rationale based on DATA] (optional)
5. [Gene ID] | Score: [X/17] | IDG Family: [family] | [One-sentence rationale based on DATA] (optional)

Where [family] must be one of: GPCR, Kinase, IC, NR, TF, Enzyme, Transporter, Epigenetic, Unknown

PRIMARY_TARGET: [Gene ID of your #1 ranked target]

SUMMARY: [2-3 sentence summary of your top target and mechanism]

DATA_EVIDENCE:
- Priority Score: [X/17 for primary target]
- Cross-dataset consistency: [Yes/No, N datasets]
- Key pathways involved: [List relevant pathways from analysis]
- WGCNA module role: [If applicable]
- Regulating TFs: [If applicable]

TARGET_CONTEXT:
- IDG Family: [One of: GPCR, Kinase, IC, NR, TF, Enzyme, Transporter, Epigenetic, Unknown]
- Location: [Membrane/Cytoplasm/Nucleus/Secreted/Unknown]
- Why this is a good candidate: [Reasoning based on DATA EVIDENCE, not family]

MECHANISM_HYPOTHESIS: [Detailed explanation of proposed mechanism for PRIMARY target based on pathway and network context]

TISSUE_RATIONALE: [Which tissues are most relevant and why, based on the data]

THERAPEUTIC_APPROACH: [Activation or inhibition? Small molecule or biologic? Why?]

PREDICTED_OUTCOME: [Expected therapeutic benefit if primary target is modulated]

RANKING_RATIONALE: [Why you ranked the targets in this order - prioritize DATA EVIDENCE over IDG family]

CONFIDENCE: [HIGH/MEDIUM/LOW based on strength of data support]
"""

T2D_ANTI_GAMING_WARNING = """
=== ANTI-GAMING WARNING ===
The gene IDs are randomly shuffled and provide NO information about gene identity.
Do NOT attempt to:
- Guess gene names from patterns in the masked IDs
- Use external knowledge about "typical" T2D genes
- Assume any masked ID corresponds to any specific gene

Your hypothesis will be evaluated based on:
1. Consistency with the DATA shown above
2. Logical reasoning from expression patterns and pathway context
3. Appropriate use of druggability features (NOT gene identity)

Gaming attempts (e.g., assuming G00001 is "insulin" or G00042 is "PDX1")
will result in hypotheses that fail validation.
"""

PROMPT_T2D_REFLECTION_WITH_DRUGGABILITY = """You are evaluating a RANKED drug target hypothesis set for Type 2 Diabetes.

=== HYPOTHESIS TO EVALUATE ===
{hypothesis}

=== EVALUATION CONTEXT ===
{evaluation_context}

=== IDG PROTEIN FAMILIES FOR RANKED TARGETS ===
{druggability_for_target}

=== SCORING CRITERIA ===
Evaluate on these dimensions (1-10 each):

1. DATA SUPPORT (weight: 35%)
   - Are the ranked targets in the top priority tier?
   - Is there cross-dataset consistency for top-ranked targets?
   - How many evidence types support the primary target?
   - Is the ranking based on DATA EVIDENCE, not just protein family?

2. RANKING QUALITY (weight: 25%)
   - Is the ranking justified primarily by DATA EVIDENCE?
   - Are higher-ranked targets clearly stronger in the data?
   - Does the ranking rationale cite specific data points?

3. MECHANISTIC COHERENCE (weight: 20%)
   - Does the proposed mechanism align with the pathway context?
   - Is the tissue rationale supported by the data?
   - Are the TF relationships considered?

4. IDG FAMILY CONTEXT (weight: 10%)
   - Is the IDG family correctly identified?
   - Is the therapeutic approach appropriate for the family?
   - NOTE: Family alone should NOT drive ranking decisions

5. NOVELTY (weight: 5%)
   - Does the hypothesis integrate data in a creative way?
   - Does it propose non-obvious connections?

6. THERAPEUTIC FEASIBILITY (weight: 5%)
   - Are the proposed modalities realistic?
   - Are there obvious safety concerns?

=== OUTPUT FORMAT ===
DATA_SUPPORT_SCORE: [1-10]
DATA_SUPPORT_RATIONALE: [Explanation - is ranking based on DATA?]

RANKING_QUALITY_SCORE: [1-10]
RANKING_QUALITY_RATIONALE: [Explanation - does ranking follow data evidence?]

MECHANISTIC_SCORE: [1-10]
MECHANISTIC_RATIONALE: [Explanation]

IDG_FAMILY_SCORE: [1-10]
IDG_FAMILY_RATIONALE: [Explanation - is family correctly used as context, not as ranking criterion?]

NOVELTY_SCORE: [1-10]
NOVELTY_RATIONALE: [Explanation]

FEASIBILITY_SCORE: [1-10]
FEASIBILITY_RATIONALE: [Explanation]

OVERALL_FITNESS: [Calculated: data*0.35 + ranking*0.25 + mechanism*0.20 + idg_family*0.10 + novelty*0.05 + feasibility*0.05, scaled to 0-100]

RANKED_TARGETS_ASSESSMENT:
- Target #1: [STRONG/MODERATE/WEAK] based on DATA (IDG Family: [family])
- Target #2: [STRONG/MODERATE/WEAK] based on DATA (IDG Family: [family])
- Target #3: [STRONG/MODERATE/WEAK] based on DATA (IDG Family: [family])

IMPROVEMENT_SUGGESTIONS: [Specific, actionable suggestions to improve data-based ranking]
"""