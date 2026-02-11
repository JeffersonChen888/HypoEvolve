"""
Simplified Evolution Agent for Pipeline3 - Genetic Algorithm Implementation

This agent handles crossover and mutation in the genetic algorithm by applying
4 core evolution strategies from Pipeline2.

Removed from Pipeline2:
- Enhancement through grounding  
- Coherence improvements
- Feasibility improvements

Kept from Pipeline2:
- Combination (crossover)
- Inspiration from existing hypotheses (crossover) 
- Simplification (mutation)
- Out-of-box thinking (mutation)
"""

import logging
import random
import uuid
import time
from typing import Dict, List, Any, Optional

from external_tools.llm_client import llm_generate
from prompts import APPROVED_DRUGS_DEPMAP, APPROVED_DRUGS_CONSTRAINT


class EvolutionAgent:
    """
    Evolution Agent applies 4 core strategies from Pipeline2 for genetic operations.
    """

    # Maximum retries for drug compliance
    MAX_DRUG_RETRIES = 3

    def __init__(self, feasibility_prompt=None, out_of_box_prompt=None, mode="drug-repurposing"):
        """
        Initialize the Evolution Agent with prompt templates.

        Args:
            feasibility_prompt: Not used (kept for compatibility)
            out_of_box_prompt: Prompt template for out-of-box thinking
            mode: Pipeline mode ("drug-repurposing" or "general")
        """
        self.mode = mode
        logging.info("Evolution Agent initialized")
        self.out_of_box_prompt = out_of_box_prompt
        self.t2d_runner = None  # ADD THIS LINE

        # Keep only 4 strategies from Pipeline2
        self.evolution_strategies = {
            "inspiration": "Inspiration from existing hypotheses - creates new hypotheses inspired by single or multiple top-ranked hypotheses.",
            "combination": "Combination - directly combines the best aspects of several top-ranking hypotheses to create new hypotheses.",
            "simplification": "Simplification - simplifies hypotheses for easier verification and testing.",
            "out_of_box_thinking": "Out-of-box thinking - explores out-of-the-box ideas by moving away from a subset of hypotheses and generating divergent ones."
        }

    def _validate_drug_compliance(self, hypothesis: Dict[str, Any]) -> tuple:
        """
        Validate that the hypothesis uses an approved drug.

        Returns:
            tuple: (is_valid, issue_message)
        """
        if self.mode != "drug-repurposing":
            return True, None

        drug = hypothesis.get('final_drug', '')
        if not drug:
            return False, "No drug specified"

        if drug.upper() in [d.upper() for d in APPROVED_DRUGS_DEPMAP]:
            return True, None

        return False, f"Drug '{drug}' is not in the approved list"

    def _get_retry_prompt(self, original_response: str, issue: str) -> str:
        """Generate a retry prompt to fix compliance issues."""
        drug_list = ", ".join(APPROVED_DRUGS_DEPMAP)
        return f"""Your previous response had an issue: {issue}

The drug you proposed is NOT in the approved list. You MUST choose a drug from this list ONLY:
{drug_list}

Please regenerate your hypothesis using ONLY a drug from the approved list above.
Keep the same scientific reasoning but change the drug to one from the list.

Your previous response was:
{original_response[:2000]}...

Now provide a corrected response with a drug FROM THE APPROVED LIST:

TITLE:
[Keep similar title but with approved drug]

SUMMARY:
[Keep similar summary but with approved drug]

HYPOTHESIS:
[Keep similar hypothesis but with approved drug]

RATIONALE:
[Keep similar rationale but with approved drug]

FINAL DRUG: [MUST be from the approved list above]
CANCER TYPE: [Same cancer type]
"""

    def _extract_gene_names(self, research_goal: str) -> List[str]:
        """Extract gene names from research goal text.
        
        Only extracts genes for lethal_genes modes where gene validation matters.
        For other modes (drug-repurposing, general), returns empty list.
        """
        import re

        # Only extract genes for lethal_genes modes
        # For general mode (GPQA), we don't need gene validation
        # For drug-repurposing, genes are not part of the research goal
        if self.mode not in ["lethal_genes", "lethal_genes_2"]:
            return []

        # In drug-repurposing mode, don't extract genes from cancer type names
        # Genes come from drug targets, not research goal text
        # This prevents "Head and Neck" being parsed as genes ['Head', 'Neck']
        if "drug repurposing" in research_goal.lower() or "repurposing" in research_goal.lower():
            return []

        # Match gene pairs like "GENE1 and GENE2" (for lethal_genes mode)
        match = re.search(r'([A-Z0-9\-_]+)\s+and\s+([A-Z0-9\-_]+)', research_goal, re.IGNORECASE)
        if match:
            return [match.group(1), match.group(2)]

        # Fallback: extract all capitalized gene-like patterns
        genes = re.findall(r'\b[A-Z][A-Z0-9\-_]{2,}\b', research_goal)
        return genes[:2] if len(genes) >= 2 else genes

    def _validate_gene_mentions(self, hypothesis_text: str, expected_genes: List[str]) -> bool:
        """Check if hypothesis mentions the expected genes."""
        if not expected_genes:
            return True

        hypothesis_upper = hypothesis_text.upper()

        for gene in expected_genes:
            gene_upper = gene.upper()
            mentions = hypothesis_upper.count(gene_upper)
            if mentions == 0:
                logging.warning(f"Gene '{gene}' not found in evolved hypothesis")
                return False

        return True

    def _validate_final_prediction(self, hypothesis: Dict[str, Any]) -> tuple:
        """
        Validate that hypothesis has a valid FINAL_PREDICTION for lethal_genes mode.

        Args:
            hypothesis: The hypothesis to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.mode not in ["lethal_genes", "lethal_genes_2"]:
            return True, None

        prediction = hypothesis.get("final_prediction", "")
        if not prediction:
            return False, "FINAL_PREDICTION is missing"

        prediction = prediction.lower().strip()
        if prediction not in ["well-based", "random"]:
            return False, f"FINAL_PREDICTION must be 'well-based' or 'random', got '{prediction}'"

        return True, None

    def _get_mode_requirements(self) -> str:
        """Get mode-specific output requirements for evolution prompts."""
        if self.mode == "drug-repurposing":
            import sys
            from pathlib import Path
            # Add project root to path for scripts imports
            project_root = Path(__file__).parent.parent.parent
            if str(project_root / "scripts") not in sys.path:
                sys.path.insert(0, str(project_root / "scripts"))
            from tcga_cancer_types import TCGA_CANCER_TYPES
            # Format cancer types WITHOUT abbreviations - use only full names
            cancer_list = ", ".join([v for k, v in TCGA_CANCER_TYPES.items() if k not in ["CNTL", "MISC", "FPPP"]])
            drug_list = ", ".join(APPROVED_DRUGS_DEPMAP)
            drug_constraint = APPROVED_DRUGS_CONSTRAINT.format(drug_list=drug_list)
            return f"""

        VALID CANCER TYPES (use the FULL NAME exactly as shown, NO abbreviations):
        {cancer_list}

        CORRECT EXAMPLES:
        ✓ CANCER TYPE: Acute Myeloid Leukemia
        ✓ CANCER TYPE: Lung adenocarcinoma
        ✓ CANCER TYPE: Breast invasive carcinoma

        INCORRECT EXAMPLES (DO NOT USE):
        ✗ CANCER TYPE: LAML (abbreviation not allowed)
        ✗ CANCER TYPE: Acute Myeloid Leukemia (LAML) (do not add abbreviation)
        ✗ CANCER TYPE: AML (abbreviation not allowed)

        {drug_constraint}

        **DRUG REPURPOSING MODE - REQUIRED OUTPUT:**
        At the END of your response, you MUST include these two lines exactly:
        FINAL DRUG: [MUST be one drug from the approved list - verify it is in the list]
        CANCER TYPE: [MUST be exact full name from cancer types list - NO abbreviations]
        """
        elif self.mode == "t2d-target":
            return """T2D TARGET REQUIREMENTS:
            - You MUST select a target gene from the integrated priority table (e.g., G00042)
            - Your reasoning must be based ONLY on the data patterns provided
            - Do NOT guess real gene identities - use only masked IDs
            - Consider genes with multiple evidence types (DE, pathways, WGCNA, TF)
            - Cross-dataset consistency is a strong positive signal"""
        return ""

    def _get_mode_specific_output_format(self) -> str:
        """Get mode-specific output format section for prompts."""
        if self.mode == "drug-repurposing":
            return """FINAL DRUG: [MUST be one drug name from the approved list above - check the list carefully]
CANCER TYPE: [MUST be exact full name from cancer types list above - NO abbreviations]

EXAMPLE (following the format):
FINAL DRUG: METFORMIN
CANCER TYPE: Lung adenocarcinoma"""
        elif self.mode == "general":
            return """**CRITICAL - REQUIRED OUTPUT:**
At the VERY END of your response, you MUST include this line EXACTLY:
FINAL_ANSWER: [A, B, C, or D]

Where [A, B, C, or D] is replaced with the single letter of your answer choice.

EXAMPLES of correct format:
FINAL_ANSWER: A
FINAL_ANSWER: B
FINAL_ANSWER: C
FINAL_ANSWER: D

⚠️ YOUR RESPONSE IS INVALID WITHOUT THE FINAL_ANSWER LINE. ⚠️
Do NOT omit this line. Do NOT use any other format."""
        elif self.mode in ["lethal_genes", "lethal_genes_2"]:
            return """FINAL_PREDICTION: [well-based OR random]

Apply rigorous criteria:
- "well-based": Requires DIRECT functional relationship - shared pathway, physical interaction, redundant function, or established SL in literature
- "random": No direct functional link, speculative connection only, or unrelated pathways

Be skeptical. A plausible-sounding hypothesis is NOT sufficient."""
        elif self.mode == "t2d-target":
            return """
                TARGET_GENE: [Masked gene ID from priority table, e.g., G00042]
                SUMMARY: [2-3 sentence summary]
                MECHANISM_HYPOTHESIS: [Data-driven mechanism]
                TISSUE_RATIONALE: [Tissue-specific considerations]
                THERAPEUTIC_APPROACH: [Activation/Inhibition approach]
                CONFIDENCE: [HIGH/MEDIUM/LOW]"""
        return ""

    def evolve_hypothesis(self, hypothesis: dict, reflection: str, 
                          research_goal: str = "", preferences: str = "") -> dict:
        """
        Main function to evolve a hypothesis using a randomly selected strategy.
        
        Args:
            hypothesis: The hypothesis to evolve
            reflection: Reflection data to guide evolution
            research_goal: The scientific research goal
            preferences: Criteria for hypothesis evaluation
            
        Returns:
            Dictionary containing evolved hypothesis and metadata
        """
        # Randomly select one of the 4 strategies
        strategy = random.choice(list(self.evolution_strategies.keys()))
        logging.info(f"Evolving hypothesis using strategy: {strategy}")
        
        if strategy == "combination":
            return self.combine_hypotheses([hypothesis], reflection, research_goal, preferences)
        elif strategy == "inspiration":  
            return self.inspire_from_hypotheses([hypothesis], reflection, research_goal, preferences)
        elif strategy == "simplification":
            return self.simplify_hypothesis(hypothesis, reflection, research_goal, preferences)
        elif strategy == "out_of_box_thinking":
            return self.out_of_box_thinking(hypothesis, reflection, research_goal, preferences)

    def combine_hypotheses(self, top_hypotheses: List[dict], reflection: str,
                          research_goal: str, preferences: str) -> dict:
        """
        Directly combines the best aspects of several top-ranking hypotheses.
        This implements the 'Combination' strategy from the AI co-scientist paper.

        Args:
            top_hypotheses: List of top-ranked hypotheses to combine
            reflection: Reflection review to guide combination
            research_goal: The scientific research goal
            preferences: Criteria for hypothesis evaluation

        Returns:
            Dictionary with combined hypothesis
        """
        logging.info(f"Combining {len(top_hypotheses)} top-ranked hypotheses")

        # Format the hypotheses for the prompt
        hypotheses_text = ""
        for i, hyp in enumerate(top_hypotheses, 1):
            hypotheses_text += f"""
        Hypothesis {i}:
        Title: {hyp.get('title', '')}
        Description: {hyp.get('description', '')}

        """

        # Add mode-specific requirements to prompt
        drug_repurposing_requirements = self._get_mode_requirements()
        lethal_genes_requirements = ""  # Primary/Rival structure removed - using standard TITLE/SUMMARY format

        # Extract genes for constraint
        genes = self._extract_gene_names(research_goal)
        gene_constraint = f"""
        **CRITICAL CONSTRAINT - GENE IDENTITY:**
        You MUST discuss the exact genes: {' and '.join(genes)}
        DO NOT substitute, replace, or add other genes (e.g., ATR, CHK1, PARP1, RAD51, XRCC1).
        Even if other genes seem more relevant or well-studied, stick to: {' and '.join(genes)}
        """ if genes else ""

        # Build mode-specific output format
        mode_specific_output = self._get_mode_specific_output_format()

        prompt = f"""
        You are an expert in scientific synthesis and hypothesis integration.
        Your task is to DIRECTLY COMBINE the best aspects of several top-ranking hypotheses
        to create a unified, more powerful hypothesis.

        Research Goal: {research_goal}
        {gene_constraint}

        Evaluation Criteria:
        {preferences}

        Top-Ranked Hypotheses to Combine:
        {hypotheses_text}

        Reflection on Source Hypotheses:
        {reflection}

        Please create a COMBINED hypothesis that:
        1. Integrates the strongest aspects from each source hypothesis
        2. Synthesizes their mechanisms into a coherent unified theory
        3. Preserves the best features while addressing individual weaknesses
        4. Creates a more comprehensive and powerful explanation
        5. Maintains clear testability and experimental approaches

        REQUIRED OUTPUT FORMAT:
        You MUST format your response with the following sections, using these exact headers:

        TITLE:
        [A concise, descriptive title reflecting the synthesis of parent hypotheses]

        SUMMARY:
        [A single-sentence summary of the combined hypothesis]

        HYPOTHESIS:
        [A clear statement of the combined hypothesis in 2-3 sentences]

        RATIONALE:
        [A detailed explanation (1-3 paragraphs) including:
        - Why this hypothesis is scientifically plausible and specific
        - How parent hypotheses are integrated and synthesized
        - Synergistic benefits of the combination over individual parents
        - Why this combination creates a more comprehensive and powerful explanation
        - What makes this hypothesis testable and falsifiable]

        {mode_specific_output}
        {lethal_genes_requirements}
        """
        
        # Use the first hypothesis as the base for parsing
        base_hypothesis = top_hypotheses[0] if top_hypotheses else {}

        # Retry up to 3 times if gene substitution or drug compliance issue detected
        max_retries = 3
        evolved_hypothesis = None
        raw_response = None
        for attempt in range(max_retries):
            logging.info("===HYPOTHESIS_EVOLUTION_START===")
            raw_response = llm_generate(prompt)
            logging.info("===HYPOTHESIS_EVOLUTION_END===")
            evolved_hypothesis = self._parse_evolved_response(raw_response, base_hypothesis, "combination")

            # Validate gene mentions
            if genes and not self._validate_gene_mentions(evolved_hypothesis.get('description', ''), genes):
                logging.warning(f"Gene substitution detected in combination (attempt {attempt + 1}/{max_retries}). Expected genes: {genes}")
                if attempt < max_retries - 1:
                    logging.info("Retrying combination operator...")
                    continue
                else:
                    logging.error(f"Gene substitution persists after {max_retries} attempts. Aborting combination operator.")
                    return None

            # Validate drug compliance
            is_valid, issue = self._validate_drug_compliance(evolved_hypothesis)
            if not is_valid:
                logging.warning(f"Drug compliance failed in combination (attempt {attempt + 1}/{max_retries}): {issue}")
                if attempt < max_retries - 1:
                    # Use retry prompt for next attempt
                    prompt = self._get_retry_prompt(raw_response, issue)
                    logging.info("Retrying with drug compliance correction...")
                    continue
                else:
                    logging.error(f"Drug compliance failed after {max_retries} attempts: {issue}")

            # Validate FINAL_PREDICTION for lethal_genes mode
            is_valid, issue = self._validate_final_prediction(evolved_hypothesis)
            if not is_valid:
                logging.warning(f"FINAL_PREDICTION validation failed in combination (attempt {attempt + 1}/{max_retries}): {issue}")
                if attempt < max_retries - 1:
                    logging.info("Retrying combination operator for FINAL_PREDICTION...")
                    continue
                else:
                    logging.error(f"FINAL_PREDICTION validation failed after {max_retries} attempts: {issue}")
                    return None

            break

        evolved_hypothesis["drug_compliance_retries"] = attempt if attempt > 0 else 0

        # Add combination metadata
        evolved_hypothesis["evolution_strategy"] = "combination"
        # Deduplicate source IDs while preserving order
        evolved_hypothesis["source_hypothesis_ids"] = list(dict.fromkeys(h.get("id") for h in top_hypotheses))
        evolved_hypothesis["evolution_justification"] = f"Combined {len(top_hypotheses)} top-ranked hypotheses"
        
        return {
            "evolved_hypothesis": evolved_hypothesis,
            "evolution_justification": f"Successfully combined {len(top_hypotheses)} top-ranked hypotheses into unified theory"
        }

    def inspire_from_hypotheses(self, top_hypotheses: List[dict], reflection: str,
                              research_goal: str, preferences: str) -> dict:
        """
        Creates new hypotheses inspired by multiple top-ranked hypotheses.
        This implements the 'Inspiration from existing hypotheses' strategy from the AI co-scientist paper.
        Uses Pipeline2's multiple hypotheses logic exactly.

        Args:
            top_hypotheses: List of top-ranked hypotheses for inspiration
            reflection: Reflection review to guide inspiration
            research_goal: The scientific research goal
            preferences: Criteria for hypothesis evaluation

        Returns:
            Dictionary with inspired hypothesis
        """
        # Use Pipeline2's multiple hypotheses logic exactly
        hypotheses_list = top_hypotheses
        mode = "multiple"

        logging.info(f"Creating hypothesis inspired by {len(hypotheses_list)} {mode} hypothesis(es)")

        # Format hypotheses using Pipeline2's multiple mode formatting
        hypotheses_text = ""
        for i, hyp in enumerate(hypotheses_list, 1):
            hypotheses_text += f"\nHypothesis {i}:\nTitle: {hyp.get('title', '')}\nDescription: {hyp.get('description', '')}\n"
        inspiration_focus = "the collective insights from the multiple source hypotheses"

        # Add mode-specific requirements to prompt
        drug_repurposing_requirements = self._get_mode_requirements()
        lethal_genes_requirements = ""  # Primary/Rival structure removed - using standard TITLE/SUMMARY format

        # Use Pipeline2's exact prompt for multiple hypotheses
        # Extract genes for constraint
        genes = self._extract_gene_names(research_goal)
        gene_constraint = f"""
        **CRITICAL CONSTRAINT - GENE IDENTITY:**
        You MUST discuss the exact genes: {' and '.join(genes)}
        DO NOT substitute, replace, or add other genes (e.g., ATR, CHK1, PARP1, RAD51, XRCC1).
        Even if other genes seem more relevant or well-studied, stick to: {' and '.join(genes)}
        """ if genes else ""

        # Build mode-specific output format
        mode_specific_output = self._get_mode_specific_output_format()

        prompt = f"""
        You are an expert in scientific creativity and hypothesis generation.
        Your task is to create a NEW hypothesis that is INSPIRED by the given high-quality hypothesis(es),
        but explores different aspects or mechanisms while maintaining scientific rigor.

        Research Goal: {research_goal}
        {gene_constraint}

        Evaluation Criteria:
        {preferences}

        Source Hypothesis(es) for inspiration:
        {hypotheses_text}

        Reflection on Source Hypothesis(es):
        {reflection}

        Please create a NEW hypothesis that:
        1. Is inspired by {inspiration_focus}
        2. Explores different mechanisms, pathways, or approaches
        3. Maintains the same research goal alignment
        4. Is novel and distinct from the source hypothesis(es)
        5. Has clear testability and experimental validation approaches

        REQUIRED OUTPUT FORMAT:
        You MUST format your response with the following sections, using these exact headers:

        TITLE:
        [A concise, descriptive title for your new inspired hypothesis]

        SUMMARY:
        [A single-sentence summary of the new hypothesis]

        HYPOTHESIS:
        [A clear statement of the new hypothesis in 2-3 sentences]

        RATIONALE:
        [A detailed explanation (1-3 paragraphs) including:
        - Why this hypothesis is scientifically plausible and specific
        - How it's inspired by the source hypothesis(es)
        - What mechanisms, pathways, or approaches differ from the source
        - What makes this novel and distinct
        - What makes this hypothesis testable and falsifiable]

        {mode_specific_output}
        {lethal_genes_requirements}
        """
        
        # Use first hypothesis as base for parsing (Pipeline2 approach)
        base_hypothesis = hypotheses_list[0]

        # Retry up to 3 times if gene substitution or drug compliance issue detected
        max_retries = 3
        evolved_hypothesis = None
        raw_response = None
        for attempt in range(max_retries):
            logging.info("===HYPOTHESIS_EVOLUTION_START===")
            raw_response = llm_generate(prompt)
            logging.info("===HYPOTHESIS_EVOLUTION_END===")
            evolved_hypothesis = self._parse_evolved_response(raw_response, base_hypothesis, "inspiration")

            # Validate gene mentions
            if genes and not self._validate_gene_mentions(evolved_hypothesis.get('description', ''), genes):
                logging.warning(f"Gene substitution detected in inspiration (attempt {attempt + 1}/{max_retries}). Expected genes: {genes}")
                if attempt < max_retries - 1:
                    logging.info("Retrying inspiration operator...")
                    continue
                else:
                    logging.error(f"Gene substitution persists after {max_retries} attempts. Aborting inspiration operator.")
                    return None

            # Validate drug compliance
            is_valid, issue = self._validate_drug_compliance(evolved_hypothesis)
            if not is_valid:
                logging.warning(f"Drug compliance failed in inspiration (attempt {attempt + 1}/{max_retries}): {issue}")
                if attempt < max_retries - 1:
                    prompt = self._get_retry_prompt(raw_response, issue)
                    logging.info("Retrying with drug compliance correction...")
                    continue
                else:
                    logging.error(f"Drug compliance failed after {max_retries} attempts: {issue}")

            # Validate FINAL_PREDICTION for lethal_genes mode
            is_valid, issue = self._validate_final_prediction(evolved_hypothesis)
            if not is_valid:
                logging.warning(f"FINAL_PREDICTION validation failed in inspiration (attempt {attempt + 1}/{max_retries}): {issue}")
                if attempt < max_retries - 1:
                    logging.info("Retrying inspiration operator for FINAL_PREDICTION...")
                    continue
                else:
                    logging.error(f"FINAL_PREDICTION validation failed after {max_retries} attempts: {issue}")
                    return None

            break

        evolved_hypothesis["drug_compliance_retries"] = attempt if attempt > 0 else 0

        # Add inspiration metadata using Pipeline2's multiple mode logic
        evolved_hypothesis["evolution_strategy"] = "inspiration"
        # Deduplicate source IDs while preserving order
        evolved_hypothesis["source_hypothesis_ids"] = list(dict.fromkeys(h.get("id") for h in hypotheses_list))
        evolved_hypothesis["evolution_justification"] = f"Inspired by {len(hypotheses_list)} top-ranked hypotheses"
        justification_msg = f"Created new hypothesis inspired by {len(hypotheses_list)} top-ranked hypotheses"
        
        return {
            "evolved_hypothesis": evolved_hypothesis,
            "evolution_justification": justification_msg
        }

    def simplify_hypothesis(self, hypothesis: dict, reflection: str,
                         research_goal: str, preferences: str) -> dict:
        """
        Simplifies a hypothesis for easier verification and testing.

        Args:
            hypothesis: The hypothesis to simplify
            reflection: Reflection review to guide simplification
            research_goal: The original research goal
            preferences: Criteria for hypothesis evaluation

        Returns:
            Dictionary with simplified hypothesis and justification
        """
        hypothesis_id = hypothesis.get("id", "unknown")
        logging.info(f"Simplifying hypothesis: {hypothesis_id}")

        # Add mode-specific requirements to prompt
        drug_repurposing_requirements = self._get_mode_requirements()
        lethal_genes_requirements = ""  # Primary/Rival structure removed - using standard TITLE/SUMMARY format

        # Extract genes for constraint
        genes = self._extract_gene_names(research_goal)
        gene_constraint = f"""
        **CRITICAL CONSTRAINT - GENE IDENTITY:**
        You MUST discuss the exact genes: {' and '.join(genes)}
        DO NOT substitute, replace, or add other genes (e.g., ATR, CHK1, PARP1, RAD51, XRCC1).
        Even if other genes seem more relevant or well-studied, stick to: {' and '.join(genes)}
        """ if genes else ""

        # Build mode-specific output format
        mode_specific_output = self._get_mode_specific_output_format()

        prompt = f"""
        You are an expert in scientific communication and hypothesis refinement.
        Your task is to simplify the provided hypothesis while preserving its core scientific value.

        Research Goal: {research_goal}
        {gene_constraint}

        Original Hypothesis:
        Title: {hypothesis.get('title', 'Untitled')}
        Summary: {hypothesis.get('summary', 'No summary')}
        Hypothesis: {hypothesis.get('hypothesis_statement', 'No hypothesis statement')}
        Rationale: {hypothesis.get('rationale', 'No rationale')}

        Reviewer Feedback:
        {reflection}

        Please simplify this hypothesis by:
        1. Improving clarity and readability of the mechanistic explanation
        2. Making the causal relationships clearer and more explicit
        3. Reducing unnecessary verbosity while maintaining scientific depth
        4. Ensuring the simplified version remains testable and falsifiable
        5. Preserving the core scientific insight and novelty

        **CRITICAL REQUIREMENTS:**
        - Simplification means clarifying and improving organization, NOT removing essential scientific content
        - The output should be MORE clear but EQUALLY comprehensive
        - Scientific depth and rigor must be maintained

        REQUIRED OUTPUT FORMAT:
        You MUST format your response with the following sections, using these exact headers:

        TITLE:
        [A clearer, more concise title]

        SUMMARY:
        [A single-sentence summary with improved clarity]

        HYPOTHESIS:
        [A clearer statement of the hypothesis in 2-3 sentences]

        RATIONALE:
        [A simplified, more readable explanation (1-3 paragraphs) that:
        - Maintains scientific plausibility and specificity
        - Uses clearer causal relationships
        - Reduces verbosity while preserving scientific depth
        - Preserves core insights and novelty
        - Maintains testability and falsifiability]

        {mode_specific_output}
        {lethal_genes_requirements}
        """
        
        # Generate the simplified hypothesis with retry logic
        max_retries = 3
        evolved_hypothesis = None
        raw_response = None
        for attempt in range(max_retries):
            logging.info("===HYPOTHESIS_EVOLUTION_START===")
            raw_response = llm_generate(prompt)
            logging.info("===HYPOTHESIS_EVOLUTION_END===")

            # Parse the response
            evolved_hypothesis = self._parse_evolved_response(raw_response, hypothesis, "simplification")

            # Validate gene mentions
            if genes and not self._validate_gene_mentions(evolved_hypothesis.get('description', ''), genes):
                logging.warning(f"Gene substitution detected in simplification (attempt {attempt + 1}/{max_retries}). Expected genes: {genes}")
                if attempt < max_retries - 1:
                    logging.info("Retrying simplification operator...")
                    continue
                else:
                    logging.error(f"Gene substitution persists after {max_retries} attempts. Aborting simplification operator.")
                    return None

            # Validate drug compliance
            is_valid, issue = self._validate_drug_compliance(evolved_hypothesis)
            if not is_valid:
                logging.warning(f"Drug compliance failed in simplification (attempt {attempt + 1}/{max_retries}): {issue}")
                if attempt < max_retries - 1:
                    prompt = self._get_retry_prompt(raw_response, issue)
                    logging.info("Retrying with drug compliance correction...")
                    continue
                else:
                    logging.error(f"Drug compliance failed after {max_retries} attempts: {issue}")

            # Validate FINAL_PREDICTION for lethal_genes mode
            is_valid, issue = self._validate_final_prediction(evolved_hypothesis)
            if not is_valid:
                logging.warning(f"FINAL_PREDICTION validation failed in simplification (attempt {attempt + 1}/{max_retries}): {issue}")
                if attempt < max_retries - 1:
                    logging.info("Retrying simplification operator for FINAL_PREDICTION...")
                    continue
                else:
                    logging.error(f"FINAL_PREDICTION validation failed after {max_retries} attempts: {issue}")
                    return None

            break

        evolved_hypothesis["drug_compliance_retries"] = attempt if attempt > 0 else 0

        # Add evolution metadata (using Pipeline2's exact logic)
        evolved_hypothesis["evolution_strategy"] = "simplification"
        evolved_hypothesis["source_hypothesis_id"] = hypothesis.get("id")
        evolved_hypothesis["evolution_justification"] = "Simplification - simplifies hypotheses for easier verification and testing."
        
        return {
            "evolved_hypothesis": evolved_hypothesis,
            "evolution_justification": "Simplified the hypothesis for clarity while preserving core insights."
        }

    def out_of_box_thinking(self, hypothesis: dict, reflection: str,
                          research_goal: str, preferences: str) -> dict:
        """
        Explores out-of-the-box ideas by generating divergent hypotheses using Pipeline2's exact logic.

        Args:
            hypothesis: The base hypothesis to diverge from
            reflection: Reflection review to guide out-of-box thinking
            research_goal: The scientific research goal
            preferences: Criteria for hypothesis evaluation

        Returns:
            Dictionary with out-of-box hypothesis
        """
        hypothesis_id = hypothesis.get("id", "unknown")
        logging.info(f"Evolving hypothesis {hypothesis_id} using strategy: out_of_box_thinking")

        # Extract key information from hypothesis and reflection (Pipeline2 logic)
        title = hypothesis.get("title", "Untitled")
        description = hypothesis.get("description", "No description")
        weaknesses = self._extract_weaknesses_from_reflection(reflection)
        strengths = self._extract_strengths_from_reflection(reflection)

        # Add mode-specific requirements to prompt
        drug_repurposing_requirements = self._get_mode_requirements()
        lethal_genes_requirements = ""  # Primary/Rival structure removed - using standard TITLE/SUMMARY format

        # Pipeline2's out-of-box prompt logic
        # Note: Pipeline2 supports custom out_of_box_prompt, but Pipeline3 uses default
        # Extract genes for constraint
        genes = self._extract_gene_names(research_goal)
        gene_constraint = f"""
        **CRITICAL CONSTRAINT - GENE IDENTITY:**
        You MUST discuss the exact genes: {' and '.join(genes)}
        DO NOT substitute, replace, or add other genes (e.g., ATR, CHK1, PARP1, RAD51, XRCC1).
        Even if other genes seem more relevant or well-studied, stick to: {' and '.join(genes)}
        """ if genes else ""

        # Build mode-specific output format
        mode_specific_output = self._get_mode_specific_output_format()

        prompt = f"""
        You are an expert in creative scientific thinking.
        Your task is to explore out-of-the-box ideas by moving away from conventional approaches
        and generating divergent hypotheses.

        Research Goal: {research_goal}
        {gene_constraint}

        Original Hypothesis:
        Title: {title}
        Summary: {hypothesis.get('summary', 'No summary')}
        Hypothesis: {hypothesis.get('hypothesis_statement', 'No hypothesis statement')}
        Rationale: {hypothesis.get('rationale', 'No rationale')}

        Key Strengths to Preserve:
        {' '.join(f'- {s}' for s in strengths)}

        Key Weaknesses to Address:
        {' '.join(f'- {w}' for w in weaknesses)}

        Please create a novel hypothesis that:
        1. Explores unconventional mechanisms or approaches
        2. Challenges existing paradigms while remaining scientifically grounded
        3. Addresses the research goal from a completely different angle
        4. Maintains testability and feasibility

        REQUIRED OUTPUT FORMAT:
        You MUST format your response with the following sections, using these exact headers:

        TITLE:
        [A creative, unconventional title reflecting the out-of-box approach]

        SUMMARY:
        [A single-sentence summary of the novel hypothesis]

        HYPOTHESIS:
        [A clear statement of the out-of-box hypothesis in 2-3 sentences]

        RATIONALE:
        [A detailed explanation (1-3 paragraphs) including:
        - Why this unconventional hypothesis is scientifically plausible
        - How it challenges existing paradigms or conventional thinking
        - What makes this approach novel and addresses weaknesses of the original
        - Why addressing the research goal from this angle is valuable
        - What makes this hypothesis testable and feasible despite being unconventional]

        {mode_specific_output}
        {lethal_genes_requirements}
        """

        # Generate the evolved hypothesis with retry logic
        max_retries = 3
        evolved_hypothesis = None
        raw_response = None
        for attempt in range(max_retries):
            logging.info("===HYPOTHESIS_EVOLUTION_START===")
            raw_response = llm_generate(prompt)
            logging.info("===HYPOTHESIS_EVOLUTION_END===")

            # Parse the response
            evolved_hypothesis = self._parse_evolved_response(raw_response, hypothesis, "out_of_box_thinking")

            # Validate gene mentions
            if genes and not self._validate_gene_mentions(evolved_hypothesis.get('description', ''), genes):
                logging.warning(f"Gene substitution detected in out_of_box_thinking (attempt {attempt + 1}/{max_retries}). Expected genes: {genes}")
                if attempt < max_retries - 1:
                    logging.info("Retrying out_of_box_thinking operator...")
                    continue
                else:
                    logging.error(f"Gene substitution persists after {max_retries} attempts. Aborting out_of_box_thinking operator.")
                    return None

            # Validate drug compliance
            is_valid, issue = self._validate_drug_compliance(evolved_hypothesis)
            if not is_valid:
                logging.warning(f"Drug compliance failed in out_of_box_thinking (attempt {attempt + 1}/{max_retries}): {issue}")
                if attempt < max_retries - 1:
                    prompt = self._get_retry_prompt(raw_response, issue)
                    logging.info("Retrying with drug compliance correction...")
                    continue
                else:
                    logging.error(f"Drug compliance failed after {max_retries} attempts: {issue}")

            # Validate FINAL_PREDICTION for lethal_genes mode
            is_valid, issue = self._validate_final_prediction(evolved_hypothesis)
            if not is_valid:
                logging.warning(f"FINAL_PREDICTION validation failed in out_of_box_thinking (attempt {attempt + 1}/{max_retries}): {issue}")
                if attempt < max_retries - 1:
                    logging.info("Retrying out_of_box_thinking operator for FINAL_PREDICTION...")
                    continue
                else:
                    logging.error(f"FINAL_PREDICTION validation failed after {max_retries} attempts: {issue}")
                    return None

            break

        evolved_hypothesis["drug_compliance_retries"] = attempt if attempt > 0 else 0

        # Add evolution metadata (Pipeline2 style)
        evolved_hypothesis["evolution_strategy"] = "out_of_box_thinking"
        evolved_hypothesis["source_hypothesis_id"] = hypothesis.get("id")
        evolved_hypothesis["evolution_justification"] = "Out-of-box thinking - explores out-of-the-box ideas by moving away from a subset of hypotheses and generating divergent ones."

        return {
            "evolved_hypothesis": evolved_hypothesis,
            "evolution_justification": f"Applied out_of_box_thinking strategy to evolve the hypothesis."
        }

    def _extract_weaknesses_from_reflection(self, reflection: str) -> List[str]:
        """Extract weaknesses from reflection text."""
        # If reflection is a dictionary with weaknesses field
        if isinstance(reflection, dict) and "weaknesses" in reflection:
            weaknesses = reflection["weaknesses"]
            if isinstance(weaknesses, list):
                return weaknesses
            elif isinstance(weaknesses, str):
                return [w.strip() for w in weaknesses.split('\n') if w.strip()]

        # Otherwise, try to extract from text
        weaknesses = []
        if isinstance(reflection, str):
            # Try multiple patterns to extract weaknesses section
            patterns = [
                "weaknesses:", "weaknesses", "limitations:", "limitations",
                "areas for improvement:", "areas for improvement"
            ]

            for pattern in patterns:
                if pattern in reflection.lower():
                    sections = reflection.lower().split(pattern)
                    if len(sections) > 1:
                        # Try to find the end of the weakness section
                        weakness_section = sections[1]
                        for end_pattern in ["strengths:", "\n\n", "conclusion:"]:
                            if end_pattern in weakness_section.lower():
                                weakness_section = weakness_section.split(end_pattern)[0]

                        # Process the weakness section
                        weakness_lines = [line.strip() for line in weakness_section.split("\n") if line.strip()]
                        processed_weaknesses = []
                        for line in weakness_lines:
                            if line.startswith("- ") or line.startswith("* "):
                                processed_weaknesses.append(line[2:].strip())
                            elif line.startswith("1. ") or line.startswith("2. ") or line.startswith("3. "):
                                processed_weaknesses.append(line[3:].strip())
                            elif len(line) < 100:  # Only include reasonably short lines
                                processed_weaknesses.append(line)

                        if processed_weaknesses:
                            weaknesses = processed_weaknesses
                            break

        # Fallback weaknesses if none found
        if not weaknesses:
            weaknesses = [
                "Lack of specificity in the proposed mechanism",
                "Limited connection to existing literature",
                "Unclear experimental validation approach",
                "May not be fully testable with current technology"
            ]

        return weaknesses[:3]  # Return up to 3 weaknesses

    def _extract_strengths_from_reflection(self, reflection: str) -> List[str]:
        """Extract strengths from reflection text."""
        # If reflection is a dictionary with strengths field
        if isinstance(reflection, dict) and "strengths" in reflection:
            strengths = reflection["strengths"]
            if isinstance(strengths, list):
                return strengths
            elif isinstance(strengths, str):
                return [s.strip() for s in strengths.split('\n') if s.strip()]

        # Otherwise, try to extract from text
        strengths = []
        if isinstance(reflection, str):
            # Try multiple patterns to extract strengths section
            patterns = [
                "strengths:", "strengths", "positive aspects:", "positive aspects",
                "strong points:", "strong points"
            ]

            for pattern in patterns:
                if pattern in reflection.lower():
                    sections = reflection.lower().split(pattern)
                    if len(sections) > 1:
                        # Try to find the end of the strengths section
                        strength_section = sections[1]
                        for end_pattern in ["weaknesses:", "\n\n", "limitations:"]:
                            if end_pattern in strength_section.lower():
                                strength_section = strength_section.split(end_pattern)[0]

                        # Process the strength section
                        strength_lines = [line.strip() for line in strength_section.split("\n") if line.strip()]
                        processed_strengths = []
                        for line in strength_lines:
                            if line.startswith("- ") or line.startswith("* "):
                                processed_strengths.append(line[2:].strip())
                            elif line.startswith("1. ") or line.startswith("2. ") or line.startswith("3. "):
                                processed_strengths.append(line[3:].strip())
                            elif len(line) < 100:  # Only include reasonably short lines
                                processed_strengths.append(line)

                        if processed_strengths:
                            strengths = processed_strengths
                            break

        # Fallback strengths if none found
        if not strengths:
            strengths = [
                "Novel approach to the research question",
                "Addresses an important gap in current understanding",
                "Potentially high impact if validated",
                "Builds on established scientific principles"
            ]

        return strengths[:3]  # Return up to 3 strengths

    def _parse_lethal_genes_sections(self, text: str) -> Dict[str, Any]:
        """
        Parse lethal_genes_2 hypothesis sections including PRIMARY and RIVAL structures.
        Copied from generation_agent.py to ensure consistent parsing.

        Args:
            text: Raw hypothesis text

        Returns:
            Dictionary with parsed sections
        """
        import re

        sections = {
            "biological_plausibility": "",
            "clinical_relevance": "",
            "summary_and_rationale": "",
            "primary_hypothesis": {},
            "rival_hypothesis": {},
            "pathway_dot": "",
            "contrast_description": "",
            "final_prediction": ""
        }

        # Extract biological plausibility
        plaus_match = re.search(
            r'Biological plausibility:\s*(.+?)(?=\n\n|Clinical relevance:|\Z)',
            text, re.DOTALL | re.IGNORECASE
        )
        if plaus_match:
            sections["biological_plausibility"] = plaus_match.group(1).strip()

        # Extract clinical relevance
        clin_match = re.search(
            r'Clinical relevance:\s*(.+?)(?=\n\n|Summary and rationale:|\Z)',
            text, re.DOTALL | re.IGNORECASE
        )
        if clin_match:
            sections["clinical_relevance"] = clin_match.group(1).strip()

        # Extract summary and rationale
        summ_match = re.search(
            r'Summary and rationale:\s*(.+?)(?=\n\nPrimary:|\Z)',
            text, re.DOTALL | re.IGNORECASE
        )
        if summ_match:
            sections["summary_and_rationale"] = summ_match.group(1).strip()

        # Extract Primary hypothesis - handle multiple formats:
        # "Primary:", "PRIMARY HYPOTHESIS", "### PRIMARY", "**Primary**"
        primary_match = re.search(
            r'(?:###?\s*)?(?:\*\*)?(?:PRIMARY(?:\s+HYPOTHESIS)?|Primary):?(?:\*\*)?\s*\n?(.+?)(?=\n\s*(?:###?\s*)?(?:\*\*)?(?:RIVAL|Rival)|\Z)',
            text, re.DOTALL | re.IGNORECASE
        )
        if primary_match:
            sections["primary_hypothesis"] = self._parse_hypothesis_subsections(
                primary_match.group(1)
            )
        else:
            logging.warning(f"Failed to parse PRIMARY hypothesis section in evolution. Text preview: {text[:300]}...")

        # Extract Rival hypothesis - handle multiple formats
        rival_match = re.search(
            r'(?:###?\s*)?(?:\*\*)?(?:RIVAL(?:\s+HYPOTHESIS)?|Rival):?(?:\*\*)?\s*\n?(.+?)(?=\n\s*(?:###?\s*)?(?:Pathway|PATHWAY|CLINICAL|Clinical|Contrast|CONTRAST)|\Z)',
            text, re.DOTALL | re.IGNORECASE
        )
        if rival_match:
            sections["rival_hypothesis"] = self._parse_hypothesis_subsections(
                rival_match.group(1)
            )
        else:
            logging.warning(f"Failed to parse RIVAL hypothesis section in evolution. Text preview: {text[:300]}...")

        # Extract DOT graph
        dot_match = re.search(
            r'digraph\s+\w*\s*\{(.+?)\}',
            text, re.DOTALL
        )
        if dot_match:
            sections["pathway_dot"] = f"digraph SL {{{dot_match.group(1)}}}"

        # Extract contrast description
        contrast_match = re.search(
            r'Contrast description:\s*(.+?)(?=\n\nAttributes|\n\nFINAL_PREDICTION|\Z)',
            text, re.DOTALL | re.IGNORECASE
        )
        if contrast_match:
            sections["contrast_description"] = contrast_match.group(1).strip()

        # Extract FINAL_PREDICTION (required field, no fallback)
        # Normalize unicode hyphens to ASCII before matching
        normalized_text = text.replace('‑', '-').replace('–', '-').replace('—', '-')
        prediction_match = re.search(
            r'FINAL_PREDICTION:\s*(well-based|random)',
            normalized_text, re.IGNORECASE
        )
        if prediction_match:
            sections["final_prediction"] = prediction_match.group(1).lower()
        else:
            logging.warning("FINAL_PREDICTION not found in LLM response")
            sections["final_prediction"] = ""

        return sections

    def _parse_hypothesis_subsections(self, text: str) -> Dict[str, str]:
        """
        Parse primary or rival hypothesis subsections.
        Copied from generation_agent.py to ensure consistent parsing.

        Args:
            text: Hypothesis section text

        Returns:
            Dictionary with subsection fields
        """
        import re

        subsections = {
            "statement": "",
            "single_loss_mechanism": "",
            "double_loss_mechanism": "",
            "assumptions": "",
            "predicted_failure": "",
            "key_intermediate_components": "",
            "key_readouts": ""
        }

        # Extract each subsection
        patterns = {
            "statement": r'Statement:\s*(.+?)(?=\n\s*Description of single loss|\Z)',
            "single_loss_mechanism": r'Description of single loss mechanism:\s*(.+?)(?=\n\s*Double loss mechanism|\Z)',
            "double_loss_mechanism": r'Double loss mechanism prediction:\s*(.+?)(?=\n\s*Assumptions:|\Z)',
            "assumptions": r'Assumptions:\s*(.+?)(?=\n\s*Predicted failure|\Z)',
            "predicted_failure": r'Predicted failure:\s*(.+?)(?=\n\s*Key intermediate components|\Z)',
            "key_intermediate_components": r'Key intermediate components:\s*(.+?)(?=\n\s*Key readouts|\Z)',
            "key_readouts": r'Key readouts if the hypothesis is true:\s*(.+?)(?=\n\n|\Z)'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                subsections[key] = match.group(1).strip()

        return subsections

    def _parse_evolved_response(self, raw_response: str, original_hypothesis: Dict[str, Any],
                              strategy: str) -> Dict[str, Any]:
        """
        Parse the evolved hypothesis from the raw response using the new unified format.

        Extracts: TITLE, SUMMARY, HYPOTHESIS, RATIONALE, and mode-specific outputs
        (FINAL DRUG/CANCER TYPE for drug-repurposing, FINAL_ANSWER for general)

        Args:
            raw_response: The raw response from the LLM
            original_hypothesis: The original hypothesis being evolved
            strategy: The evolution strategy used

        Returns:
            Dictionary with the parsed evolved hypothesis
        """
        import re

        # Helper function to extract section content
        def extract_section(pattern_name, text, next_pattern=None):
            """Extract content between pattern_name and next_pattern"""
            pattern = rf'{pattern_name}:\s*\n?(.*?)(?=\n(?:{next_pattern}:|FINAL|Primary:|Rival:|$))'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
            return None

        # For drug-repurposing, general, and lethal_genes modes, extract unified TITLE/SUMMARY fields
        title = extract_section("TITLE", raw_response, "SUMMARY") or f"Evolved hypothesis ({strategy})"
        summary = extract_section("SUMMARY", raw_response, "HYPOTHESIS") or "No summary provided"
        hypothesis_statement = extract_section("HYPOTHESIS", raw_response, "RATIONALE") or "No hypothesis statement provided"
        rationale = extract_section("RATIONALE", raw_response, "FINAL") or "No rationale provided"

        # Extract mode-specific output fields
        final_drug = None
        cancer_type = None
        final_answer = None
        final_prediction = None

        if self.mode == "drug-repurposing":
            drug_match = re.search(r'FINAL DRUG:\s*([^\n]+)', raw_response, re.IGNORECASE)
            if drug_match:
                final_drug = drug_match.group(1).strip()

            cancer_match = re.search(r'CANCER TYPE:\s*([^\n]+)', raw_response, re.IGNORECASE)
            if cancer_match:
                cancer_type = cancer_match.group(1).strip()

        elif self.mode == "general":
            # Match FINAL_ANSWER: A, FINAL_ANSWER: [A], or FINAL_ANSWER: (A)
            answer_match = re.search(r'FINAL_ANSWER:\s*[\[\(]?([A-D])[\]\)]?', raw_response, re.IGNORECASE)
            if answer_match:
                final_answer = answer_match.group(1).strip().upper()
            else:
                # Fallback: search for any FINAL_ANSWER pattern in the response
                all_matches = re.findall(r'FINAL_ANSWER:\s*[\[\(]?([A-D])[\]\)]?', raw_response, re.IGNORECASE)
                if all_matches:
                    final_answer = all_matches[-1].upper()  # Take the last one

        elif self.mode in ["lethal_genes", "lethal_genes_2"]:
            # Normalize unicode hyphens to ASCII before matching
            normalized_response = raw_response.replace('‑', '-').replace('–', '-').replace('—', '-')
            prediction_match = re.search(r'FINAL_PREDICTION:\s*(well-based|random)', normalized_response, re.IGNORECASE)
            if prediction_match:
                final_prediction = prediction_match.group(1).strip().lower()

        # Generate a unique ID
        base_id = original_hypothesis.get("id", "hyp")
        suffix = strategy[:3]
        evolved_id = f"{base_id}-{suffix}-{str(uuid.uuid4())[:6]}"

        # Calculate new ELO score
        base_elo = original_hypothesis.get("elo_score", 1000)
        elo_boost = random.randint(20, 50)

        # Create evolved hypothesis with new unified format
        evolved_hyp = {
            "id": evolved_id,
            "title": title,
            "summary": summary,
            "hypothesis_statement": hypothesis_statement,
            "rationale": rationale,
            "elo_score": base_elo + elo_boost,
            "origin": "evolution",
            "parent_id": original_hypothesis.get("id"),
            "evolution_strategy": strategy,
            "evolution_timestamp": time.time(),
            "reviews": [],
            "cluster_id": original_hypothesis.get("cluster_id"),
            "gene_a": original_hypothesis.get("gene_a"),
            "gene_b": original_hypothesis.get("gene_b"),
            "gene_pair_name": original_hypothesis.get("gene_pair_name")
        }

        # Add mode-specific fields
        if self.mode == "drug-repurposing":
            evolved_hyp["final_drug"] = final_drug
            evolved_hyp["cancer_type"] = cancer_type
        elif self.mode == "general":
            evolved_hyp["final_answer"] = final_answer
        elif self.mode in ["lethal_genes", "lethal_genes_2"]:
            evolved_hyp["final_prediction"] = final_prediction
            evolved_hyp["description"] = raw_response  # Keep raw text for gene validation

        return evolved_hyp

    # Genetic Algorithm Methods for Pipeline3
    def perform_crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any],
                          research_goal: str = "", preferences: str = "") -> Dict[str, Any]:
        """
        Perform crossover between two parent hypotheses.
        
        Args:
            parent1: First parent hypothesis
            parent2: Second parent hypothesis
            research_goal: The scientific research goal
            preferences: Criteria for hypothesis evaluation
            
        Returns:
            Offspring hypothesis from crossover
        """
        if self.mode == "t2d-target":
            return self._t2d_crossover(parent1, parent2, research_goal)

        # Get reflection data from first parent (Pipeline2 approach for multiple-parent functions)
        reflection = ""
        if parent1.get("reviews"):
            reflection = parent1["reviews"][-1]  # Get latest review from first parent

        # Use combination or inspiration strategy
        strategy = random.choice(["combination", "inspiration"])

        if strategy == "combination":
            result = self.combine_hypotheses([parent1, parent2], reflection, research_goal, preferences)
        else:
            result = self.inspire_from_hypotheses([parent1, parent2], reflection, research_goal, preferences)
            
        return result["evolved_hypothesis"]

    def perform_mutation(self, hypothesis: Dict[str, Any],
                        research_goal: str = "", preferences: str = "") -> Dict[str, Any]:
        """
        Perform mutation on a hypothesis.
        
        Args:
            hypothesis: Hypothesis to mutate
            research_goal: The scientific research goal
            preferences: Criteria for hypothesis evaluation
            
        Returns:
            Mutated hypothesis
        """
        if self.mode == "t2d-target":
            return self._t2d_mutate(hypothesis, research_goal)
        # Get reflection data from hypothesis (Pipeline2 approach)
        reflection = ""
        if hypothesis.get("reviews"):
            reflection = hypothesis["reviews"][-1]  # Get latest review

        # Use simplification or out-of-box strategy
        strategy = random.choice(["simplification", "out_of_box_thinking"])

        if strategy == "simplification":
            result = self.simplify_hypothesis(hypothesis, reflection, research_goal, preferences)
        else:
            result = self.out_of_box_thinking(hypothesis, reflection, research_goal, preferences)
            
        return result["evolved_hypothesis"]

    # =========================================================================
    # T2D MODE SPECIFIC METHODS
    # =========================================================================
    
    def _t2d_crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any],
                       research_goal: str) -> Dict[str, Any]:
        """
        Crossover operation for T2D hypotheses using analysis context.
        
        Args:
            parent1: First parent hypothesis
            parent2: Second parent hypothesis
            research_goal: Research goal string
        
        Returns:
            Child hypothesis
        """
        from prompts import PROMPT_T2D_CROSSOVER
        
        if self.t2d_runner is None:
            raise ValueError("T2D runner not attached. Set evolution_agent.t2d_runner first.")
        
        # Get analysis context
        analysis_context = self.t2d_runner.format_for_evolution_agent(parent1, parent2)
        
        prompt = PROMPT_T2D_CROSSOVER.format(
            parent1_target=parent1.get('target_gene_masked', 'Unknown'),
            parent1_score=parent1.get('fitness_score', 0),
            parent1_content=parent1.get('description', ''),
            parent2_target=parent2.get('target_gene_masked', 'Unknown'),
            parent2_score=parent2.get('fitness_score', 0),
            parent2_content=parent2.get('description', ''),
            analysis_context=analysis_context
        )
        
        logging.info("===T2D_CROSSOVER_START===")
        raw_response = llm_generate(prompt)
        logging.info("===T2D_CROSSOVER_END===")
        
        # Parse child hypothesis
        child = self._parse_t2d_evolved_hypothesis(raw_response, parent1, parent2)
        child["evolution_strategy"] = "t2d_crossover"
        child["parent_ids"] = [parent1.get('id'), parent2.get('id')]
        
        # Validate target is in valid list
        valid_targets = self.t2d_runner.get_valid_target_genes()
        if child.get('target_gene_masked') not in valid_targets:
            logging.warning(f"Crossover produced invalid target: {child.get('target_gene_masked')}")
        
        return child
    
    def _t2d_mutate(self, hypothesis: Dict[str, Any], research_goal: str) -> Dict[str, Any]:
        """
        Mutation operation for T2D hypotheses.
        
        Args:
            hypothesis: Hypothesis to mutate
            research_goal: Research goal string
        
        Returns:
            Mutated hypothesis
        """
        from prompts import PROMPT_T2D_MUTATION, T2D_MUTATION_STRATEGIES
        
        if self.t2d_runner is None:
            raise ValueError("T2D runner not attached. Set evolution_agent.t2d_runner first.")
        
        # Select random mutation strategy
        strategy = random.choice(list(T2D_MUTATION_STRATEGIES.keys()))
        strategy_instructions = T2D_MUTATION_STRATEGIES[strategy]
        
        logging.info(f"T2D mutation using strategy: {strategy}")
        
        # Get analysis context
        analysis_context = self.t2d_runner.format_for_evolution_agent()
        
        prompt = PROMPT_T2D_MUTATION.format(
            hypothesis=hypothesis.get('description', ''),
            analysis_context=analysis_context,
            strategy=strategy,
            strategy_instructions=strategy_instructions
        )
        
        logging.info("===T2D_MUTATION_START===")
        raw_response = llm_generate(prompt)
        logging.info("===T2D_MUTATION_END===")
        
        # Parse mutated hypothesis
        mutated = self._parse_t2d_evolved_hypothesis(raw_response, hypothesis, None)
        mutated["evolution_strategy"] = f"t2d_mutation_{strategy}"
        mutated["parent_ids"] = [hypothesis.get('id')]
        
        # Validate target is in valid list
        valid_targets = self.t2d_runner.get_valid_target_genes()
        if mutated.get('target_gene_masked') not in valid_targets:
            logging.warning(f"Mutation produced invalid target: {mutated.get('target_gene_masked')}")
        
        return mutated
    
    def _parse_t2d_evolved_hypothesis(self, raw_response: str, 
                                       parent1: Dict[str, Any],
                                       parent2: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse evolved T2D hypothesis from LLM response.
        
        Args:
            raw_response: Raw LLM response text
            parent1: First parent hypothesis
            parent2: Second parent hypothesis (None for mutation)
        
        Returns:
            Parsed hypothesis dictionary
        """
        import re
        
        def extract(pattern, text, default=""):
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else default
        
        # Extract fields
        title = extract(r'TITLE:\s*(.+?)(?=\n|PRIMARY_TARGET:|RANKED_TARGETS:)', raw_response)
        
        # Try PRIMARY_TARGET first, then TARGET_GENE for backward compatibility
        target = extract(r'PRIMARY_TARGET:\s*(G\d{5})', raw_response)
        if not target:
            target = extract(r'TARGET_GENE:\s*(G\d{5})', raw_response)
        
        # Fallback: extract first gene from RANKED_TARGETS section (format: "1. G17025 | ...")
        if not target:
            ranked_match = re.search(r'RANKED_TARGETS:\s*\n\s*1\.\s*(G\d{5})', raw_response, re.DOTALL)
            if ranked_match:
                target = ranked_match.group(1)
                logging.info(f"Extracted target from RANKED_TARGETS: {target}")
        
        # Log warning if no target could be extracted
        if not target:
            logging.warning("Could not extract target_gene_masked from evolved hypothesis - evaluation metrics may be affected")
        
        summary = extract(r'SUMMARY:\s*(.+?)(?=\n\n|DATA_EVIDENCE:|MECHANISM)', raw_response)
        mechanism = extract(r'MECHANISM_HYPOTHESIS:\s*(.+?)(?=\n\n|TISSUE_RATIONALE:)', raw_response)
        tissue = extract(r'TISSUE_RATIONALE:\s*(.+?)(?=\n\n|THERAPEUTIC)', raw_response)
        therapeutic = extract(r'THERAPEUTIC_APPROACH:\s*(.+?)(?=\n\n|PREDICTED)', raw_response)
        outcome = extract(r'PREDICTED_OUTCOME:\s*(.+?)(?=\n\n|CONFIDENCE)', raw_response)
        confidence = extract(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', raw_response, "MEDIUM")
        
        # Determine generation number
        gen1 = parent1.get('generation', 0) if parent1 else 0
        gen2 = parent2.get('generation', 0) if parent2 else 0
        new_generation = max(gen1, gen2) + 1
        
        # Calculate ELO
        base_elo = parent1.get('elo_score', 1200) if parent1 else 1200
        elo_boost = random.randint(20, 50)
        
        hypothesis = {
            "id": f"t2d-{new_generation:02d}-{str(uuid.uuid4())[:8]}",
            "title": title or "T2D Evolved Hypothesis",
            "target_gene_masked": target,
            "summary": summary,
            "mechanism_hypothesis": mechanism,
            "tissue_rationale": tissue,
            "therapeutic_approach": therapeutic,
            "predicted_outcome": outcome,
            "confidence_level": confidence.upper(),
            "description": raw_response,
            "elo_score": base_elo + elo_boost,
            "fitness_score": None,
            "origin": "t2d_evolution",
            "generation": new_generation,
            "evolution_timestamp": time.time()
        }
        
        return hypothesis

    def create_offspring(self, parents: List[Dict[str, Any]], research_goal: str,
                        offspring_size: int, preferences: str = "") -> List[Dict[str, Any]]:
        """
        Create offspring population using genetic operations (Population Management).
        
        Args:
            parents: Selected parent hypotheses
            research_goal: The scientific research goal
            offspring_size: Number of offspring to create
            preferences: Evaluation preferences
            
        Returns:
            List of offspring hypotheses
        """
        logging.info(f"Creating {offspring_size} offspring from {len(parents)} parents")
        
        # GA Parameters (can be made configurable)
        crossover_rate = 0.6  # 60% of offspring from crossover
        
        # Calculate operation counts
        crossover_count = int(offspring_size * crossover_rate)
        mutation_count = offspring_size - crossover_count
        
        logging.info(f"Genetic operations: {crossover_count} crossover + {mutation_count} mutation")
        
        offspring = []
        
        # Crossover operations
        for i in range(crossover_count):
            # Select 2 parents (with replacement if needed)
            if len(parents) >= 2:
                parent1, parent2 = random.sample(parents, 2)
            else:
                # Fallback: duplicate single parent
                parent1 = parent2 = parents[0]
            
            child = self.perform_crossover(parent1, parent2, research_goal, preferences)
            offspring.append(child)
        
        # Mutation operations
        for i in range(mutation_count):
            # Select random parent for mutation
            parent = random.choice(parents)
            mutated_child = self.perform_mutation(parent, research_goal, preferences)
            offspring.append(mutated_child)
        
        logging.info(f"Successfully created {len(offspring)} offspring")
        
        return offspring