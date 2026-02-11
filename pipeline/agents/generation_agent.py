"""
Simplified Generation Agent for Pipeline3 - Genetic Algorithm Implementation

This agent handles population initialization in the genetic algorithm by generating
initial hypotheses through literature exploration only.

Removed from Pipeline2:
- Simulated debates
- Iterative assumptions
- Research expansion
- Multiple generation strategies

Kept:
- Literature exploration via web search
- Basic hypothesis generation
"""

import logging
import os
import random
import time
import uuid
from typing import Dict, List, Any

from external_tools.llm_client import llm_generate
from external_tools.web_search import perform_web_search
from external_tools.web_search import search_literature
from prompts import PROMPT_LITERATURE_EXPLORATION, APPROVED_DRUGS_DEPMAP, APPROVED_DRUGS_CONSTRAINT
# from tcga_cancer_types import get_tcga_cancer_types_prompt  # Function doesn't exist


class GenerationAgent:
    """
    Simplified Generation Agent focused solely on population initialization
    through literature-grounded hypothesis generation.
    """

    # Maximum retries for drug compliance
    MAX_DRUG_RETRIES = 3

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

    def __init__(self, mode: str = "drug-repurposing", run_folder: str = None):
        """
        Initialize the simplified generation agent.

        Args:
            mode: Pipeline mode ("drug-repurposing" or "general")
            run_folder: Folder path for this specific run (for per-pair logs in batch mode)
        """
        self.mode = mode
        self.t2d_runner = None  # T2D pipeline runner (set by supervisor)
        self.use_druggability = False  # Option A flag (set by supervisor)

        self.run_folder = run_folder
        logging.info(f"Initialized simplified Generation Agent in {mode} mode")
    
    def generate_initial_population(self, research_goal: str,
                                   population_size: int = 5,
                                   num_papers: int = 3,
                                   preferences: str = "",
                                   source_hypothesis: str = "",
                                   instructions: str = "") -> Dict[str, Any]:
        """
        Generate initial population of hypotheses for the genetic algorithm.

        Args:
            research_goal: The scientific research goal
            population_size: Number of hypotheses to generate
            num_papers: Number of papers to use for literature exploration
            preferences: Criteria for hypothesis evaluation
            source_hypothesis: Optional existing hypothesis to build upon
            instructions: Optional additional instructions for hypothesis generation

        Returns:
            Dictionary containing the generated hypothesis population
        """
        logging.info(f"Generating initial population of {population_size} hypotheses")

        # Handle lethal_genes mode differently
        if self.mode == "lethal_genes":
            # Research goal should be in format "Gene_A:Gene_B" or just "Gene_A Gene_B"
            return self._generate_lethal_genes_population(research_goal, population_size, num_papers)
        # After the lethal_genes_2 check, add:
        if self.mode == "t2d-target":
            return self._generate_t2d_population(research_goal, population_size)
        # Step 1: Generate search queries in one call and perform literature exploration
        search_queries = self._generate_search_queries_batch(research_goal)
        all_papers, literature_context = self._search_with_queries(search_queries, num_papers)
        
        # Step 3: Generate hypotheses using literature exploration prompt (Pipeline2 approach)
        hypotheses = []
        for i in range(population_size):
            # Include cancer type standards and mode-specific output requirements
            cancer_type_standards = ""
            mode_specific_output = ""

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
                # Format approved drug list
                drug_list = ", ".join(APPROVED_DRUGS_DEPMAP)
                drug_constraint = APPROVED_DRUGS_CONSTRAINT.format(drug_list=drug_list)
                cancer_type_standards = f"""DRUG REPURPOSING TASK REQUIREMENTS:
You must identify a specific drug repurposing candidate for the target cancer type.

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

{drug_constraint}"""

                mode_specific_output = """FINAL DRUG: [MUST be one drug name from the approved list above - check the list carefully]
CANCER TYPE: [MUST be exact full name from cancer types list above - NO abbreviations]

EXAMPLE (following the format):
FINAL DRUG: METFORMIN
CANCER TYPE: Lung adenocarcinoma"""

            elif self.mode == "general":
                mode_specific_output = """**CRITICAL - REQUIRED OUTPUT:**
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

            prompt = PROMPT_LITERATURE_EXPLORATION.format(
                goal=research_goal,
                preferences=preferences,
                source_hypothesis=source_hypothesis,
                instructions=instructions,
                articles_with_reasoning=literature_context,
                cancer_type_standards=cancer_type_standards,
                mode_specific_output=mode_specific_output
            )
            
            # Log information about what literature context is being passed to the LLM
            if literature_context and "No relevant scientific literature was found" not in literature_context:
                logging.info(f"Passing literature analysis to LLM for hypothesis {i+1}: {len(literature_context)} characters of analysis")
                logging.info(f"Literature analysis includes content from {len(all_papers)} papers total")
            else:
                logging.info(f"No literature analysis available for hypothesis {i+1} - LLM will use general knowledge")
            
            # Generate hypothesis with retry logic for drug compliance
            hypothesis_text = llm_generate(prompt)
            hypothesis = self._parse_hypothesis_response(hypothesis_text, i+1)

            # Validate and retry if drug not in approved list
            is_valid, issue = self._validate_drug_compliance(hypothesis)
            retry_count = 0

            while not is_valid and retry_count < self.MAX_DRUG_RETRIES:
                retry_count += 1
                logging.warning(f"Hypothesis {i+1} drug compliance failed: {issue}. Retry {retry_count}/{self.MAX_DRUG_RETRIES}")

                retry_prompt = self._get_retry_prompt(hypothesis_text, issue)
                hypothesis_text = llm_generate(retry_prompt)
                hypothesis = self._parse_hypothesis_response(hypothesis_text, i+1)
                is_valid, issue = self._validate_drug_compliance(hypothesis)

            if not is_valid:
                logging.error(f"Hypothesis {i+1} failed drug compliance after {self.MAX_DRUG_RETRIES} retries: {issue}")
            else:
                if retry_count > 0:
                    logging.info(f"Hypothesis {i+1} drug compliance fixed after {retry_count} retry(s)")

            hypothesis["generation_method"] = "literature_exploration"
            hypothesis["search_iterations"] = len(search_queries)
            hypothesis["search_queries"] = search_queries
            hypothesis["num_papers_found"] = len(all_papers)
            hypothesis["drug_compliance_retries"] = retry_count

            hypotheses.append(hypothesis)
        
        logging.info(f"Generated {len(hypotheses)} hypotheses for initial population")
        
        return {
            "hypotheses": hypotheses,
            "literature_context": literature_context,
            "papers_found": len(all_papers),
            "search_queries": search_queries
        }

    def _generate_t2d_population(self, research_goal: str,
                              population_size: int) -> Dict[str, Any]:
        """
        Generate initial population for T2D drug target identification.

        Uses the pre-computed analysis context from T2DPipelineRunner
        AND literature search for T2D background and pathway context.
        Optionally includes druggability features (Option A mode).

        Args:
            research_goal: Research goal string
            population_size: Number of hypotheses to generate

        Returns:
            Dict with hypotheses and metadata
        """
        from prompts import PROMPT_T2D_GENERATION

        # Get analysis context from supervisor
        if not hasattr(self, 't2d_runner') or self.t2d_runner is None:
            raise ValueError("T2D runner not attached. Set generation_agent.t2d_runner first.")

        # Check if we should use druggability features (Option A)
        use_druggability = getattr(self, 'use_druggability', False)

        # Get appropriate context based on mode
        if use_druggability and hasattr(self.t2d_runner, 'druggability_features'):
            from prompts import PROMPT_T2D_GENERATION_WITH_DRUGGABILITY, T2D_ANTI_GAMING_WARNING
            analysis_context = self.t2d_runner.format_for_generation_agent_with_druggability()
            druggability_context = self.t2d_runner.format_druggability_for_llm()
            prompt_template = PROMPT_T2D_GENERATION_WITH_DRUGGABILITY
            logging.info("Using Option A: Druggability-enhanced generation")
        else:
            analysis_context = self.t2d_runner.format_for_generation_agent()
            druggability_context = "Druggability features not available in this mode."
            prompt_template = PROMPT_T2D_GENERATION
            logging.info("Using Option C: Expression-only generation")

        valid_targets = self.t2d_runner.get_valid_target_genes()

        logging.info(f"Generating {population_size} T2D hypotheses...")
        logging.info(f"Valid targets: {len(valid_targets)} genes")

        # === Literature Search for T2D Context ===
        literature_context = self._search_t2d_literature()
        logging.info(f"Retrieved T2D literature context: {len(literature_context)} characters")

        hypotheses = []
        selected_genes = []  # Track already-selected genes for diversity

        for i in range(population_size):
            logging.info(f"Generating hypothesis {i+1}/{population_size}")

            # Build diversity constraint if genes already selected
            diversity_note = ""
            if selected_genes:
                diversity_note = f"""

=== DIVERSITY REQUIREMENT ===
The following genes have ALREADY been selected for other hypotheses in this population:
{', '.join(selected_genes)}

You MUST select a DIFFERENT gene from the priority table. Do NOT select any of the genes listed above.
Choose a gene with strong evidence that has NOT yet been selected.
"""

            # Format prompt based on mode
            if use_druggability and hasattr(self.t2d_runner, 'druggability_features'):
                prompt = prompt_template.format(
                    analysis_context=analysis_context,
                    druggability_context=druggability_context,
                    literature_context=literature_context
                ) + diversity_note
            else:
                prompt = prompt_template.format(
                    analysis_context=analysis_context,
                    literature_context=literature_context
                ) + diversity_note

            # Generate hypothesis
            response = llm_generate(prompt)

            # Parse response
            hypothesis = self._parse_t2d_hypothesis(response, i + 1)

            # Track selected gene for diversity
            target_gene = hypothesis.get('target_gene_masked')
            if target_gene:
                selected_genes.append(target_gene)

            # Validate target is in valid list
            if target_gene not in valid_targets:
                logging.warning(f"Hypothesis {i+1} selected invalid target: {target_gene}")
            elif target_gene in selected_genes[:-1]:  # Check if duplicate (excluding current)
                logging.warning(f"Hypothesis {i+1} selected duplicate target: {target_gene}")

            hypotheses.append(hypothesis)

        return {
            "hypotheses": hypotheses,
            "analysis_context": analysis_context,
            "druggability_context": druggability_context if use_druggability else None,
            "literature_context": literature_context,  # Store for reflection
            "valid_targets": valid_targets,
            "papers_found": literature_context.count("Title:"),
            "search_queries": ["T2D pathophysiology", "insulin resistance mechanisms", "pathway-based"],
            "mode": "Option A (druggability)" if use_druggability else "Option C (expression-only)"
        }

    def _search_t2d_literature(self) -> str:
        """
        Search for T2D PATHOPHYSIOLOGY literature without revealing drug targets.

        Searches for:
        1. Disease mechanisms and pathophysiology (NOT drug targets)
        2. Pathway-based literature (from GSEA results)
        3. Cellular dysfunction mechanisms

        IMPORTANT: Queries must NOT include "drug target", "therapeutic target",
        or similar terms that could leak known target information to the LLM.

        Returns:
            Combined literature context string
        """
        # Base T2D queries - PATHOPHYSIOLOGY ONLY (no drug targets)
        # This prevents information leakage about known therapeutic targets
        base_queries = [
            "type 2 diabetes pathophysiology molecular mechanism",
            "insulin resistance cellular dysfunction mechanism",
            "beta cell failure diabetes pathogenesis",
        ]
        
        # Get pathway-specific queries from GSEA results (if available)
        pathway_queries = self._get_pathway_search_queries()
        
        # Combine queries (limit to 5 total to avoid overwhelming)
        all_queries = base_queries + pathway_queries[:2]
        
        logging.info(f"Searching T2D literature with {len(all_queries)} queries")
        
        # Perform searches
        all_papers = []
        combined_analysis = ""
        
        for i, query in enumerate(all_queries):
            logging.info(f"T2D literature search {i+1}/{len(all_queries)}: {query}")
            
            papers = self._gather_literature(query, num_papers=3)
            
            # Filter duplicates
            new_papers = [p for p in papers if not any(
                existing.get('title', '').lower() == p.get('title', '').lower() 
                for existing in all_papers
            )]
            
            if new_papers:
                all_papers.extend(new_papers)
                iteration_analysis = self._analyze_literature(new_papers, query)
                combined_analysis += f"\n\n--- {query.upper()} ---\n\n{iteration_analysis}"
        
        if all_papers:
            combined_analysis += "\n\n--- T2D LITERATURE SYNTHESIS ---\n\n"
            combined_analysis += self._synthesize_t2d_findings(all_papers)
        else:
            combined_analysis = "No specific T2D literature found. Proceeding with data-driven analysis only."
        
        logging.info(f"T2D literature search complete: {len(all_papers)} papers found")
        return combined_analysis

    def _get_pathway_search_queries(self) -> List[str]:
        """
        Generate search queries based on enriched pathways from GSEA.

        Returns:
            List of pathway-based search queries
        """
        pathway_queries = []

        if self.t2d_runner is None:
            return pathway_queries

        try:
            # Get top enriched pathways from GSEA results
            enriched_pathways = self.t2d_runner.get_top_enriched_pathways(n=3)

            for pathway in enriched_pathways:
                # Clean pathway name for better search queries
                # Remove database prefix (e.g., "GO_Biological_Process_2023__")
                # Remove GO IDs (e.g., "(GO:0042254)")
                clean_pathway = pathway

                # Remove database prefix (everything before __)
                if "__" in clean_pathway:
                    clean_pathway = clean_pathway.split("__", 1)[1]

                # Remove GO/KEGG IDs in parentheses
                import re
                clean_pathway = re.sub(r'\s*\([A-Z]+:\d+\)\s*', '', clean_pathway)
                clean_pathway = re.sub(r'\s*\(hsa\d+\)\s*', '', clean_pathway)  # KEGG IDs

                # Clean up extra whitespace
                clean_pathway = ' '.join(clean_pathway.split())

                if clean_pathway:
                    # Create diabetes-specific query for each pathway
                    query = f"{clean_pathway} diabetes type 2"
                    pathway_queries.append(query)
                    logging.info(f"Added pathway query: {query} (from: {pathway[:50]}...)")

        except Exception as e:
            logging.warning(f"Could not get pathway queries: {e}")

        return pathway_queries

    def _synthesize_t2d_findings(self, papers: List[Dict[str, Any]]) -> str:
        """
        Synthesize T2D PATHOPHYSIOLOGY literature - disease mechanisms only.

        IMPORTANT: This synthesis must NOT mention drug targets, treatments,
        or therapeutic interventions to prevent information leakage.

        Args:
            papers: List of papers found

        Returns:
            Synthesis string focusing on disease biology only
        """
        paper_titles = [p.get('title', 'Untitled') for p in papers[:10]]

        prompt = f"""
        Synthesize the following T2D PATHOPHYSIOLOGY papers:

        Papers: {', '.join(paper_titles)}

        Focus ONLY on DISEASE BIOLOGY:
        1. What cellular and molecular processes are DYSREGULATED in T2D?
        2. What tissues and cell types are primarily affected?
        3. What are the UPSTREAM CAUSES of insulin resistance and beta-cell failure?
        4. What biological pathways show aberrant activity in T2D patients?

        FORBIDDEN (do NOT mention):
        - Drug names or treatments
        - Clinical trial results
        - "Drug targets" or "therapeutic targets"
        - Any specific genes known to be targeted by approved drugs
        - Therapeutic interventions or treatment approaches

        Provide a concise synthesis (3-4 paragraphs) describing T2D DISEASE BIOLOGY ONLY.
        This should help understand what goes wrong in T2D, not how to treat it.
        """

        return llm_generate(prompt)

    def _parse_t2d_hypothesis(self, response: str, number: int) -> Dict[str, Any]:
        """Parse T2D hypothesis response into structured format.

        Supports both single-target (legacy) and multi-gene ranked format.
        Multi-gene format includes RANKED_TARGETS and PRIMARY_TARGET fields.
        """
        import re
        import uuid
        import time

        # Create clean response without markdown bolding for easier regex matching
        # This fixes issues where LLM outputs **PRIMARY_TARGET:** which breaks strict regex
        clean_response = response.replace("**", "")

        # Extract fields using regex
        def extract(pattern, text, default=""):
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else default

        title = extract(r'TITLE:\s*(.+?)(?=\n|RANKED_TARGETS:|TARGET_GENE:|TARGET:)', clean_response)

        # Try new multi-gene ranked format first
        ranked_targets = self._parse_ranked_targets(clean_response)
        primary_target = extract(r'PRIMARY_TARGET:\s*(G\d{5})', clean_response)
        ranking_rationale = extract(r'RANKING_RATIONALE:\s*(.+?)(?=\n\n|CONFIDENCE:|$)', clean_response)

        # Fall back to legacy single-target format if no ranked targets found
        if not ranked_targets:
            # Try TARGET_GENE: first (original format)
            single_target = extract(r'TARGET_GENE:\s*(G\d{5})', clean_response)
            # Also try TARGET: (crossover/mutation format fallback)
            if not single_target:
                single_target = extract(r'^TARGET:\s*(G\d{5})', clean_response)
            if single_target:
                ranked_targets = [{"rank": 1, "gene_id": single_target, "rationale": "Primary target"}]
                primary_target = single_target

        # Use primary target from ranked list if not explicitly specified
        if not primary_target and ranked_targets:
            primary_target = ranked_targets[0]["gene_id"]

        # Log warning if no valid target found
        if not primary_target:
            logging.warning(f"Could not extract primary target from hypothesis response")
            # Log the first few lines of response for debugging
            logging.debug(f"First 500 chars of response: {clean_response[:500]}")

        summary = extract(r'SUMMARY:\s*(.+?)(?=\n\n|DATA_EVIDENCE:)', clean_response)
        mechanism = extract(r'MECHANISM_HYPOTHESIS:\s*(.+?)(?=\n\n|TISSUE_RATIONALE:)', clean_response)
        tissue = extract(r'TISSUE_RATIONALE:\s*(.+?)(?=\n\n|THERAPEUTIC_APPROACH:)', clean_response)
        therapeutic = extract(r'THERAPEUTIC_APPROACH:\s*(.+?)(?=\n\n|PREDICTED_OUTCOME:)', clean_response)
        outcome = extract(r'PREDICTED_OUTCOME:\s*(.+?)(?=\n\n|RANKING_RATIONALE:|CONFIDENCE:)', clean_response)
        confidence = extract(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', clean_response, "MEDIUM")

        hypothesis = {
            "id": f"t2d-{number:03d}-{str(uuid.uuid4())[:8]}",
            "title": title or f"T2D Target Hypothesis {number}",
            "target_gene_masked": primary_target,  # Primary target for backward compatibility
            "ranked_targets": ranked_targets,  # Full ranked list
            "n_ranked_targets": len(ranked_targets),
            "ranking_rationale": ranking_rationale,
            "summary": summary,
            "mechanism_hypothesis": mechanism,
            "tissue_rationale": tissue,
            "therapeutic_approach": therapeutic,
            "predicted_outcome": outcome,
            "confidence_level": confidence.upper(),
            "description": response,  # Store full response including original formatting
            "elo_score": 1200,
            "fitness_score": None,
            "origin": "t2d_generation",
            "generation": 0,
            "generation_timestamp": time.time()
        }

        return hypothesis

    def _parse_ranked_targets(self, response: str) -> List[Dict[str, Any]]:
        """Parse RANKED_TARGETS section from T2D hypothesis response.

        Expected format:
        RANKED_TARGETS:
        1. G00042 | Score: 15/17 | [rationale]
        2. G00015 | Score: 14/17 | [rationale]

        Returns:
            List of dicts with rank, gene_id, score, rationale
        """
        import re

        ranked_targets = []

        # Find RANKED_TARGETS section - improved regex for robustness (relaxed lookahead)
        section_match = re.search(
            r'RANKED_TARGETS:\s*\n(.*?)(?=\n\s*PRIMARY_TARGET:|$)',
            response, re.DOTALL | re.IGNORECASE
        )

        if not section_match:
            return []

        section_text = section_match.group(1)

        # Parse each ranked line
        # Format: "1. G00042 | Score: 15/17 | Druggability: Kinase | rationale" or
        #         "1. G00042 | Score: 15/17 | rationale"
        pattern = r'(\d+)\.\s*(G\d{5})\s*\|\s*Score:\s*(\d+)/17\s*\|(?:\s*Druggability:\s*([^|]+)\s*\|)?\s*(.+?)(?=\n\d+\.|$)'
        matches = re.findall(pattern, section_text, re.DOTALL)

        for match in matches:
            rank = int(match[0])
            gene_id = match[1]
            score = int(match[2])
            druggability = match[3].strip() if match[3] else None
            rationale = match[4].strip()

            target_entry = {
                "rank": rank,
                "gene_id": gene_id,
                "priority_score": score,
                "rationale": rationale
            }

            if druggability:
                target_entry["druggability"] = druggability

            ranked_targets.append(target_entry)

        # Sort by rank to ensure correct order
        ranked_targets.sort(key=lambda x: x["rank"])

        return ranked_targets
    
    def _generate_lethal_genes_population(self, research_goal: str,
                                         population_size: int,
                                         num_papers: int) -> Dict[str, Any]:
        """
        Generate initial population for lethal genes mode.

        Args:
            research_goal: Gene pair in format "GeneA:GeneB" or "GeneA GeneB"
            population_size: Number of hypotheses to generate
            num_papers: Number of papers for literature search

        Returns:
            Dictionary containing hypotheses and metadata
        """
        # Parse gene pair from research goal
        gene_pair = self._parse_gene_pair(research_goal)
        if not gene_pair:
            raise ValueError(f"Could not parse gene pair from: {research_goal}")

        gene_a, gene_b = gene_pair
        logging.info(f"Generating lethal genes hypotheses for pair: {gene_a} - {gene_b}")

        # Perform literature search for the gene pair
        search_queries = [
            f"{gene_a} AND {gene_b} AND synthetic lethality",
            f"{gene_a} AND {gene_b} AND cancer",
            f"{gene_a} gene function",
            f"{gene_b} gene function"
        ]

        all_papers, literature_context = self._search_with_queries(search_queries, num_papers)

        # Generate multiple hypotheses for the same gene pair
        hypotheses = []
        for i in range(population_size):
            hypothesis_text = llm_generate(
                self._get_lethal_genes_prompt(gene_a, gene_b, literature_context)
            )

            # Process into structured format using unified parser
            hypothesis = self._parse_hypothesis_response(hypothesis_text, i + 1)

            # Add lethal genes specific fields
            hypothesis["gene_a"] = gene_a
            hypothesis["gene_b"] = gene_b
            hypothesis["generation_method"] = "lethal_genes_literature"
            hypothesis["search_queries"] = search_queries
            hypothesis["num_papers_found"] = len(all_papers)
            hypothesis["literature_context"] = literature_context  # Store for later use in reflection
            hypothesis["description"] = hypothesis_text  # Keep raw text for gene validation

            hypotheses.append(hypothesis)

        logging.info(f"Generated {len(hypotheses)} lethal genes hypotheses")

        return {
            "hypotheses": hypotheses,
            "literature_context": literature_context,
            "papers_found": len(all_papers),
            "search_queries": search_queries
        }

    def generate_batch_lethal_genes(self, num_papers: int = 10) -> Dict[str, Any]:
        """
        Generate hypotheses for multiple gene pairs from config (batch mode).

        In batch mode:
        - Generate ONE hypothesis per gene pair from ACTIVE_GENE_PAIRS
        - All hypotheses compete in the genetic algorithm together
        - Population size = number of active gene pairs

        Args:
            num_papers: Maximum papers to retrieve per gene pair

        Returns:
            Dictionary containing all hypotheses and metadata
        """
        from gene_pairs_config import ACTIVE_GENE_PAIRS

        logging.info(f"Starting batch lethal genes generation for {len(ACTIVE_GENE_PAIRS)} gene pairs")

        all_hypotheses = []

        for idx, gene_pair in enumerate(ACTIVE_GENE_PAIRS, 1):
            gene_a, gene_b = gene_pair
            logging.info(f"Processing gene pair {idx}/{len(ACTIVE_GENE_PAIRS)}: {gene_a} - {gene_b}")

            # Create per-pair log file if run_folder is specified
            pair_handler = None
            if self.run_folder:
                pair_log_file = os.path.join(self.run_folder, f"pair{idx}_{gene_a}-{gene_b}.log")
                pair_handler = logging.FileHandler(pair_log_file, mode='w')
                pair_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
                logging.getLogger().addHandler(pair_handler)
                logging.info(f"Created per-pair log: {pair_log_file}")

            # Perform literature search for this gene pair
            search_queries = [
                f"{gene_a} AND {gene_b} AND synthetic lethality",
                f"{gene_a} AND {gene_b} AND cancer",
                f"{gene_a} gene function",
                f"{gene_b} gene function"
            ]

            all_papers, literature_context = self._search_with_queries(search_queries, num_papers)

            # Generate ONE hypothesis for this gene pair
            hypothesis_text = llm_generate(
                self._get_lethal_genes_prompt(gene_a, gene_b, literature_context)
            )

            # Process into structured format
            hypothesis = self._process_lethal_genes_hypothesis(
                hypothesis_text, gene_a, gene_b, f"pair-{idx}"
            )
            hypothesis["generation_method"] = "lethal_genes_batch"
            hypothesis["search_queries"] = search_queries
            hypothesis["num_papers_found"] = len(all_papers)
            hypothesis["literature_context"] = literature_context
            hypothesis["gene_pair_index"] = idx  # Track which pair this is

            all_hypotheses.append(hypothesis)
            logging.info(f"Generated hypothesis for {gene_a} - {gene_b} (pair {idx}/{len(ACTIVE_GENE_PAIRS)})")

            # Remove per-pair log handler to prevent cross-contamination
            if pair_handler:
                pair_handler.close()
                logging.getLogger().removeHandler(pair_handler)

        logging.info(f"Batch generation complete: {len(all_hypotheses)} hypotheses for {len(ACTIVE_GENE_PAIRS)} gene pairs")

        return {
            "hypotheses": all_hypotheses,
            "num_gene_pairs": len(ACTIVE_GENE_PAIRS),
            "batch_mode": True
        }

    def generate_from_complete_prompt(self, complete_prompt: str,
                                      population_size: int = 5,
                                      gene_pair_name: str = "unknown") -> Dict[str, Any]:
        """
        Generate multiple hypotheses from a complete prompt file (lethal_genes_2 mode).

        This method is used when a complete, pre-written prompt is provided
        (e.g., from data/lethal_genes/individual_prompts/).

        Args:
            complete_prompt: Complete prompt text loaded from file
            population_size: Number of hypotheses to generate (default: 5)
            gene_pair_name: Gene pair identifier for metadata (e.g., "KLF5_ARID1A")

        Returns:
            Dictionary containing:
                - hypotheses: List of hypothesis dicts
                - gene_pair_name: Identifier for this gene pair
                - prompt_length: Length of complete prompt in characters
        """
        logging.info(f"Generating {population_size} hypotheses from complete prompt for {gene_pair_name}")
        logging.info(f"Complete prompt length: {len(complete_prompt)} characters")

        hypotheses = []
        for i in range(population_size):
            logging.info(f"Generating hypothesis {i+1}/{population_size} for {gene_pair_name}")

            # Call LLM with complete prompt
            logging.info("===HYPOTHESIS_GENERATION_START===")
            hypothesis_text = llm_generate(complete_prompt)
            logging.info("===HYPOTHESIS_GENERATION_END===")

            # Process the response into structured format
            hypothesis = self._process_lethal_genes_2_hypothesis(
                hypothesis_text, gene_pair_name, f"lg2-{i+1}"
            )
            hypothesis["generation_method"] = "lethal_genes_2_complete_prompt"
            hypothesis["gene_pair_name"] = gene_pair_name
            hypothesis["prompt_length"] = len(complete_prompt)

            hypotheses.append(hypothesis)

        logging.info(f"Generated {len(hypotheses)} hypotheses for {gene_pair_name}")

        return {
            "hypotheses": hypotheses,
            "gene_pair_name": gene_pair_name,
            "prompt_length": len(complete_prompt)
        }

    def generate_from_tournament_prompt(self, gene_pair: tuple,
                                        prompt_template: str,
                                        population_size: int = 3) -> List[Dict[str, Any]]:
        """
        Generate hypotheses from a prompt template for lethal_genes_tournament mode.

        This method takes a prompt template with [GENE_A] and [GENE_B] placeholders
        and generates hypotheses for a specific gene pair.

        Args:
            gene_pair: Tuple of (gene_a, gene_b)
            prompt_template: Prompt template with [GENE_A] and [GENE_B] placeholders
            population_size: Number of hypotheses to generate (default: 3)

        Returns:
            List of hypothesis dictionaries with structured format
        """
        gene_a, gene_b = gene_pair
        gene_pair_name = f"{gene_a}_{gene_b}"

        logging.info(f"Generating {population_size} hypotheses for tournament mode: {gene_a} - {gene_b}")

        # Substitute gene pair placeholders in template
        prompt = prompt_template.replace("[GENE_A]", gene_a).replace("[GENE_B]", gene_b)
        # Also handle lowercase variants
        prompt = prompt.replace("[gene_a]", gene_a).replace("[gene_b]", gene_b)
        # Handle with curly braces too
        prompt = prompt.replace("{GENE_A}", gene_a).replace("{GENE_B}", gene_b)

        hypotheses = []
        for i in range(population_size):
            logging.info(f"Generating hypothesis {i+1}/{population_size} for {gene_pair_name}")

            # Call LLM with substituted prompt
            logging.info("===HYPOTHESIS_GENERATION_START===")
            hypothesis_text = llm_generate(prompt)
            logging.info("===HYPOTHESIS_GENERATION_END===")

            # Process the response into structured format (same as lethal_genes_2)
            hypothesis = self._process_lethal_genes_2_hypothesis(
                hypothesis_text, gene_pair_name, f"tour-{i+1}"
            )

            # Add tournament-specific metadata
            hypothesis["generation_method"] = "tournament_prompt"
            hypothesis["gene_pair_name"] = gene_pair_name
            hypothesis["gene_a"] = gene_a
            hypothesis["gene_b"] = gene_b
            hypothesis["prompt_length"] = len(prompt)
            hypothesis["generation"] = 0
            hypothesis["parent_ids"] = []
            hypothesis["evolution_strategy"] = "initial_generation"
            hypothesis["elo_score"] = 1200

            hypotheses.append(hypothesis)

        logging.info(f"Generated {len(hypotheses)} hypotheses for {gene_pair_name}")
        return hypotheses

    def _parse_gene_pair(self, research_goal: str) -> tuple:
        """
        Parse gene pair from research goal string.

        Args:
            research_goal: String containing gene pair

        Returns:
            Tuple of (gene_a, gene_b) or None if parsing fails
        """
        import re

        # Try various formats
        # Format 1: "Gene_A:Gene_B"
        if ':' in research_goal:
            parts = research_goal.split(':')
            if len(parts) == 2:
                return (parts[0].strip(), parts[1].strip())

        # Format 2: "Gene_A and Gene_B"
        if ' and ' in research_goal.lower():
            parts = re.split(r'\s+and\s+', research_goal, flags=re.IGNORECASE)
            if len(parts) == 2:
                return (parts[0].strip(), parts[1].strip())

        # Format 3: "Gene_A Gene_B" (space-separated)
        parts = research_goal.split()
        if len(parts) == 2:
            return (parts[0].strip(), parts[1].strip())

        return None

    def _get_lethal_genes_prompt(self, gene_a: str, gene_b: str,
                                 literature_context: str) -> str:
        """
        Get the prompt for lethal genes hypothesis generation.

        Args:
            gene_a: First gene
            gene_b: Second gene
            literature_context: Literature search results

        Returns:
            Formatted prompt string
        """
        from prompts import PROMPT_LETHAL_GENES_GENERATION

        return PROMPT_LETHAL_GENES_GENERATION.format(
            gene_a=gene_a,
            gene_b=gene_b,
            literature_context=literature_context if literature_context else "No literature found."
        )

    def _process_lethal_genes_hypothesis(self, hypothesis_text: str,
                                        gene_a: str, gene_b: str,
                                        id_suffix: str) -> Dict[str, Any]:
        """
        Process raw lethal genes hypothesis text into structured format.

        Args:
            hypothesis_text: Raw text from LLM
            gene_a: First gene
            gene_b: Second gene
            id_suffix: Suffix for hypothesis ID

        Returns:
            Structured hypothesis dictionary
        """
        import re

        # Initialize structure
        hypothesis = {
            "id": f"hyp_{id_suffix}_{uuid.uuid4().hex[:8]}",
            "gene_a": gene_a,
            "gene_b": gene_b,
            "title": f"Synthetic Lethality Analysis: {gene_a} - {gene_b}",
            "description": hypothesis_text,
            "testability_notes": "",
            "elo_score": 1200,
            "origin": f"generation_{id_suffix}",
            "reviews": [],
            "cluster_id": None,
            "generation_timestamp": time.time()
        }

        # Parse sections from hypothesis text
        sections = self._parse_lethal_genes_sections(hypothesis_text)

        # Add parsed sections to hypothesis
        hypothesis.update(sections)

        return hypothesis

    def _process_lethal_genes_2_hypothesis(self, hypothesis_text: str,
                                          gene_pair_name: str,
                                          id_suffix: str) -> Dict[str, Any]:
        """
        Process raw lethal genes 2 hypothesis text into structured format.

        Used for lethal_genes_2 mode where complete prompts are provided
        with gene pairs in various formats (e.g., "KLF5_ARID1A",
        "WRN_microsatellite_instability_MSI").

        Args:
            hypothesis_text: Raw text from LLM
            gene_pair_name: Gene pair identifier from filename (e.g., "KLF5_ARID1A")
            id_suffix: Suffix for hypothesis ID

        Returns:
            Structured hypothesis dictionary
        """
        import re

        # Initialize structure
        hypothesis = {
            "id": f"hyp_{id_suffix}_{uuid.uuid4().hex[:8]}",
            "gene_pair_name": gene_pair_name,
            "title": f"Synthetic Lethality Analysis: {gene_pair_name.replace('_', ' ')}",
            "description": hypothesis_text,
            "testability_notes": "",
            "elo_score": 1200,
            "origin": f"generation_{id_suffix}",
            "reviews": [],
            "cluster_id": None,
            "generation_timestamp": time.time()
        }

        # Parse sections from hypothesis text using existing parser
        sections = self._parse_lethal_genes_sections(hypothesis_text)

        # Add parsed sections to hypothesis
        hypothesis.update(sections)

        return hypothesis

    def _parse_lethal_genes_sections(self, text: str) -> Dict[str, Any]:
        """
        Parse lethal genes hypothesis output into structured sections.

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
            logging.warning(f"Failed to parse PRIMARY hypothesis section. Text preview: {text[:300]}...")

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
            logging.warning(f"Failed to parse RIVAL hypothesis section. Text preview: {text[:300]}...")

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

    def _explore_literature(self, research_goal: str, num_papers: int, num_iterations: int = 3, 
                          source_hypothesis: str = "", preferences: str = "", 
                          instructions: str = "") -> tuple:
        """
        Explore literature using Pipeline2's iterative approach.
        
        Args:
            research_goal: The research goal to search for
            num_papers: Maximum number of papers to retrieve per iteration
            num_iterations: Number of search iterations to perform
            source_hypothesis: Optional source hypothesis for context
            preferences: Optional user preferences
            instructions: Optional special instructions
            
        Returns:
            Tuple of (all_papers, combined_analysis, used_search_queries)
        """
        logging.info(f"Starting literature exploration with {num_iterations} iterations")
        
        combined_analysis = ""
        # Track all papers found across iterations to avoid duplicates
        all_papers = []
        # Track search queries to avoid duplicates
        used_search_queries = []
        
        # Perform multiple iterations of search and analysis
        for iteration in range(num_iterations):
            logging.info(f"Literature exploration iteration {iteration+1}/{num_iterations}")
            
            # Generate a search query for this iteration
            current_query = self._generate_search_query(
                research_goal=research_goal,
                iteration=iteration,
                source_hypothesis=source_hypothesis if iteration == 0 else "",
                papers=all_papers,
                current_analysis=combined_analysis,
                used_queries=used_search_queries
            )
            
            # Skip this iteration if duplicate query detected
            if current_query is None:
                logging.info(f"Skipping iteration {iteration+1} due to duplicate query")
                continue
            
            # Add the query to our tracking list
            used_search_queries.append(current_query)
            logging.info(f"Using search query for iteration {iteration+1}: {current_query}")
            
            # Step 1: Gather relevant literature through search
            papers = self._gather_literature(current_query, num_papers)
            
            # Filter out duplicate papers (those we've already seen)
            new_papers = [p for p in papers if not any(
                existing['title'].lower() == p['title'].lower() for existing in all_papers
            )]

            # Log paper discovery results
            if not new_papers:
                logging.warning(f"No new papers found in iteration {iteration+1}. Continuing with existing results.")
                # If it's the first iteration and no papers were found at all, add a note
                if iteration == 0 and not all_papers:
                    combined_analysis = "Note: No relevant literature was found for this research goal. The hypothesis generation will proceed without literature context.\n\n"

                # If we have no papers at all, break out of the iterations
                if not all_papers:
                    break
                continue
            else:
                logging.info(f"Found {len(new_papers)} new papers in iteration {iteration+1} (total: {len(all_papers) + len(new_papers)})")

            # Add new papers to our collection
            all_papers.extend(new_papers)

            # Step 2: Analyze new literature and extract insights
            iteration_analysis = self._analyze_literature(new_papers, research_goal)

            # Add this iteration's analysis to the combined analysis
            combined_analysis += f"\n\n--- ITERATION {iteration+1} LITERATURE ANALYSIS ---\n\n"
            combined_analysis += f"Search Query: {current_query}\n\n"
            combined_analysis += iteration_analysis
        
        # Add a final synthesis across all iterations
        if all_papers:
            combined_analysis += "\n\n--- FINAL SYNTHESIS ACROSS ALL ITERATIONS ---\n\n"
            combined_analysis += self._synthesize_findings(all_papers, research_goal)
        
        logging.info(f"Literature exploration completed. Found {len(all_papers)} papers across {len(used_search_queries)} queries")
        return all_papers, combined_analysis, used_search_queries
    
    
    def _generate_search_query(self, research_goal: str, iteration: int,
                              source_hypothesis: str = "", papers: List[Dict[str, Any]] = None,
                              current_analysis: str = "", used_queries: List[str] = None) -> str:
        """
        Generate effective search queries using LLM with strategic approach from Pipeline2.
        Uses a strategic approach with predefined query patterns for different iterations.
        
        Args:
            research_goal: The original research goal
            iteration: The current iteration number (0-based)
            source_hypothesis: Optional existing hypothesis to build upon (for first iteration)
            papers: Not used - kept for API compatibility
            current_analysis: Not used - kept for API compatibility
            used_queries: List of previously used search queries to avoid duplicates
            
        Returns:
            String with the next search query to use
        """
        # Initialize parameters
        used_queries = used_queries or []
        
        # Generate query like a researcher looking for papers to cite
        if iteration == 0:
            # First iteration: Foundational/theoretical papers
            query_prompt = f"""
A researcher working on this problem needs to find foundational papers to cite:

{research_goal}

What GENERAL search query (maximum 6 words) would find the most important theoretical papers in this field?

REQUIREMENTS:
- Use ONLY general field terms (e.g., "bioinformatics", "sequence analysis", "pattern recognition")
- NO specific details from the problem (no exact sequences, numbers, or algorithm specifics)
- Maximum 6 words total
- Think: What field is this? What foundational concepts would experts cite?

Examples of correct general queries:
- "Pauli matrices quantum mechanics" (3 words, for spin problems)
- "eigenvalue problems linear algebra" (3 words, for matrix problems)
- "quantum operator theory" (3 words, for quantum mechanics)

Generate one GENERAL search query for foundational papers (max 6 words):
QUERY:"""
        
        elif iteration == 1:
            # Second iteration: Methods/computational papers
            query_prompt = f"""
A researcher working on this problem needs to find papers about methods and techniques:

{research_goal}

Previous query used: {used_queries[0] if used_queries else 'none'}

What GENERAL search query (maximum 6 words) would find papers about methods and techniques for this field?

REQUIREMENTS:
- Use ONLY general method terms (e.g., "algorithms", "computational methods", "analysis techniques")
- NO specific details from the problem (no exact sequences, numbers, or algorithm specifics)
- Maximum 6 words total
- Focus on computational/analytical methods for this field

Examples of correct general method queries:
- "spin eigenvalue algorithms" (3 words)
- "quantum matrix diagonalization" (3 words)
- "computational quantum mechanics" (3 words)

Generate one GENERAL search query for methods papers (max 6 words):
QUERY:"""

        elif iteration == 2:
            # Third iteration: Applications/experimental papers
            query_prompt = f"""
A researcher working on this problem needs to find application or experimental papers:

{research_goal}

Previous queries used: {', '.join(used_queries)}

What GENERAL search query (maximum 6 words) would find papers showing applications and experiments in this field?

REQUIREMENTS:
- Use ONLY general application terms (e.g., "applications", "experiments", "implementations")
- NO specific details from the problem (no exact sequences, numbers, or algorithm specifics)
- Maximum 6 words total
- Focus on practical applications and experiments in this field

Examples of correct general application queries:
- "spin measurement experiments" (3 words)
- "quantum systems applications" (3 words)
- "Pauli matrices solid state physics" (5 words)

Generate one GENERAL search query for application papers (max 6 words):
QUERY:"""
            
        else:
            # Additional iterations: Related fields or review papers
            query_prompt = f"""
A researcher working on this problem needs to find related work or review papers:

{research_goal}

Previous queries used: {', '.join(used_queries)}

What GENERAL search query (maximum 6 words) would find review papers and broader context for this field?

REQUIREMENTS:
- Use ONLY general field terms (e.g., "review", "survey", broad field names)
- NO specific details from the problem (no exact sequences, numbers, or algorithm specifics)
- Maximum 6 words total
- Focus on reviews and broader field context

Examples of correct general review queries:
- "quantum mechanics review" (3 words)
- "spin systems physics" (3 words)
- "linear algebra quantum applications" (4 words)

Generate one GENERAL search query for review papers (max 6 words):
QUERY:"""
        
        # Generate the query using researcher mindset
        response = llm_generate(query_prompt)
        
        # Extract query from response
        if "QUERY:" in response:
            query = response.split("QUERY:")[1].strip()
        else:
            query = response.strip()
        
        # Clean up quotes and extra formatting
        query = query.strip('"').strip("'").strip()
        
        # Validate that we got a real query from the LLM  
        if len(query.split()) < 2:
            raise ValueError(f"LLM failed to generate valid query. Got: '{query}'. Need real API access, not mock client.")
        
        # Clean up the query
        query = query.strip().replace('\n', ' ').replace('  ', ' ')
        
        # Ensure we don't exceed reasonable length (academic search works best with 2-6 terms)
        words = query.split()
        if len(words) > 6:
            query = ' '.join(words[:6])
        
        # Check if this query is a duplicate of a previously used query
        if query.lower() in [q.lower() for q in used_queries]:
            logging.info(f"Skipping duplicate query: '{query}'")
            return None  # Skip this iteration
            
        return query

    def _generate_search_queries_batch(self, research_goal: str) -> List[str]:
        """
        Generate 3 search queries in a single API call.

        Args:
            research_goal: The original research goal

        Returns:
            List of 3 search query strings
        """
        query_prompt = f"""
Generate 3 GENERAL search queries (max 6 words each) for this research problem:

{research_goal}

REQUIREMENTS FOR EACH QUERY:
- Use ONLY general field terms (e.g., "bioinformatics", "quantum mechanics", "pattern recognition")
- NO specific details from the problem (no exact sequences, numbers, or algorithm specifics)
- Maximum 6 words per query
- Focus on: 1) theoretical concepts, 2) methods/algorithms, 3) applications/experiments

Examples of correct general queries:
- "quantum mechanics eigenvalue problems" (4 words)
- "pattern recognition algorithms" (3 words)
- "DNA sequence computational analysis" (4 words)

Format your response as:
1. [query 1]
2. [query 2]
3. [query 3]
"""

        response = llm_generate(query_prompt)

        # Parse queries from numbered list
        queries = []
        for line in response.split('\n'):
            line = line.strip()
            if line and any(line.startswith(f"{i}.") for i in range(1, 4)):
                # Extract query after number
                query = line.split('.', 1)[1].strip()
                query = query.strip('"').strip("'").strip()
                if len(query.split()) <= 6:
                    queries.append(query)

        # Ensure we have exactly 3 queries, fallback if parsing failed
        if len(queries) < 3:
            queries = ["theoretical concepts", "computational methods", "experimental applications"][:3]

        return queries[:3]

    def _search_with_queries(self, search_queries: List[str], num_papers: int) -> tuple:
        """
        Perform literature search using the generated queries.

        Args:
            search_queries: List of search queries
            num_papers: Maximum papers per query

        Returns:
            Tuple of (all_papers, combined_analysis)
        """
        logging.info(f"Starting literature search with {len(search_queries)} queries")

        all_papers = []
        combined_analysis = ""

        for i, query in enumerate(search_queries):
            logging.info(f"Literature search iteration {i+1}/{len(search_queries)}")
            logging.info(f"Using search query: {query}")

            # Gather papers for this query
            papers = self._gather_literature(query, num_papers)

            # Filter out duplicates
            new_papers = [p for p in papers if not any(
                existing['title'].lower() == p['title'].lower() for existing in all_papers
            )]

            if new_papers:
                logging.info(f"Found {len(new_papers)} new papers in iteration {i+1} (total: {len(all_papers) + len(new_papers)})")
                all_papers.extend(new_papers)

                # Analyze these papers
                iteration_analysis = self._analyze_literature(new_papers, f"Query: {query}")
                combined_analysis += f"\n\n--- ITERATION {i+1} LITERATURE ANALYSIS ---\n\n"
                combined_analysis += f"Search Query: {query}\n\n"
                combined_analysis += iteration_analysis
            else:
                logging.warning(f"No new papers found in iteration {i+1}. Continuing with existing results.")

        # Add final synthesis if we have papers
        if all_papers:
            combined_analysis += "\n\n--- FINAL SYNTHESIS ACROSS ALL QUERIES ---\n\n"
            combined_analysis += self._synthesize_findings(all_papers, "Combined literature search")

        logging.info(f"Literature search completed. Found {len(all_papers)} papers across {len(search_queries)} queries")
        return all_papers, combined_analysis

    def _gather_literature(self, research_goal: str, num_papers: int) -> List[Dict[str, Any]]:
        """
        Gather relevant scientific literature for a research goal.

        Returns:
            List of paper information dictionaries
        """
        papers = search_literature(research_goal, max_results=num_papers)
        if not papers:
            logging.warning(f"No literature found for research goal: {research_goal}. Continuing with empty results.")
        else:
            logging.info(f"Found {len(papers)} papers for search query: {research_goal}")
        return papers
    
    def _synthesize_findings(self, papers: List[Dict[str, Any]], research_goal: str) -> str:
        """
        Create a final synthesis of findings across all papers.
        
        Args:
            papers: List of all papers found across iterations
            research_goal: The scientific research goal
            
        Returns:
            String with synthesis of findings
        """
        synthesis = "Comprehensive Synthesis of Findings Across All Literature:\n\n"
        
        # Group papers by topic or relevance (simplified approach)
        # In a more advanced implementation, this could use clustering or topic modeling
        paper_titles = [p.get('title', 'Untitled paper') for p in papers]
        
        # Create the synthesis prompt
        prompt = f"""
        Create a comprehensive synthesis of the findings from the following papers related to: {research_goal}
        
        PAPERS:
        {', '.join(paper_titles)}
        
        Please synthesize:
        1. The main themes and findings across these papers
        2. Areas of consensus among researchers
        3. Contradictory findings or debates in the field
        4. Research gaps and opportunities for novel contributions
        5. Methodological approaches commonly used
        
        FORMAT YOUR RESPONSE WITH THESE SECTIONS:
        MAIN THEMES AND FINDINGS:
        [Your synthesis]
        
        CONSENSUS AREAS:
        [Your synthesis]
        
        CONTRADICTIONS AND DEBATES:
        [Your synthesis]
        
        RESEARCH GAPS:
        [Your synthesis]
        
        METHODOLOGICAL APPROACHES:
        [Your synthesis]
        """
        
        # Generate the synthesis
        synthesis += llm_generate(prompt)
        
        return synthesis
    
    def _analyze_literature(self, papers: List[Dict[str, Any]], research_goal: str) -> str:
        """
        Analyze collected literature to create context for hypothesis generation.
        
        Args:
            papers: List of papers to analyze
            research_goal: The research goal
            
        Returns:
            Literature analysis summary
        """
        if not papers:
            logging.info("No papers provided - using general knowledge")
            return "No relevant literature found. Hypothesis generation will rely on general scientific knowledge."
        
        logging.info(f"Analyzing {len(papers)} papers for literature context")
        
        # Create literature summary
        analysis = "Literature Analysis Summary:\n\n"
        
        # Add paper summaries
        for i, paper in enumerate(papers[:5], 1):  # Limit to top 5 papers
            title = paper.get('title', f'Paper {i}')
            abstract = paper.get('abstract', paper.get('snippet', 'No abstract available'))
            
            analysis += f"{i}. {title}\n"
            if abstract and len(abstract) > 50:
                # Truncate long abstracts
                abstract_truncated = abstract[:300] + "..." if len(abstract) > 300 else abstract
                analysis += f"   Summary: {abstract_truncated}\n\n"
        
        # Add synthesis
        analysis += "Key Research Insights:\n"
        analysis += f"- {len(papers)} relevant papers identified for {research_goal}\n"
        analysis += "- Literature provides foundation for evidence-based hypothesis generation\n"
        analysis += "- Multiple research perspectives available for novel hypothesis development\n"
        
        logging.info(f"Generated literature analysis with {len(analysis)} characters")
        return analysis
    
    def _generate_single_hypothesis(self, research_goal: str, 
                                   literature_context: str, 
                                   hypothesis_number: int) -> Dict[str, Any]:
        """
        Generate a single hypothesis based on research goal and literature.
        
        Args:
            research_goal: The research goal
            literature_context: Literature analysis context
            hypothesis_number: Sequential number for this hypothesis
            
        Returns:
            Dictionary containing the generated hypothesis
        """
        # Include cancer type standards and drug constraint for drug repurposing
        cancer_type_context = ""
        drug_constraint_text = ""
        if self.mode == "drug-repurposing":
            drug_list = ", ".join(APPROVED_DRUGS_DEPMAP)
            drug_constraint_text = APPROVED_DRUGS_CONSTRAINT.format(drug_list=drug_list)

        prompt = f"""
        You are a scientific researcher tasked with generating a novel hypothesis based on the provided research goal and literature context.

        Research Goal: {research_goal}

        {cancer_type_context}

        {drug_constraint_text if self.mode == "drug-repurposing" else ""}

        Literature Context:
        {literature_context}

        Generate a novel, testable scientific hypothesis that:
        1. Directly addresses the research goal
        2. Is grounded in the provided literature context
        3. Proposes a specific mechanism or relationship
        4. Can be experimentally validated
        5. Offers potential for significant scientific impact

        {"For drug repurposing tasks, ensure your hypothesis identifies specific drugs from the approved list above and cancer types using TCGA terminology." if self.mode == "drug-repurposing" else ""}

        Format your response as:

        HYPOTHESIS TITLE: [Concise, descriptive title]

        HYPOTHESIS DESCRIPTION: [Detailed description of the hypothesis, including proposed mechanisms, expected relationships, and scientific rationale. Should be 3-4 sentences.]

        TESTABILITY: [Brief description of how this hypothesis could be experimentally tested]

        {"FINAL DRUG: [drug name from approved list]" if self.mode == "drug-repurposing" else ""}
        {"CANCER TYPE: [TCGA cancer type]" if self.mode == "drug-repurposing" else ""}
        """
        
        # Generate hypothesis
        raw_response = llm_generate(prompt)
        
        # Parse response
        hypothesis = self._parse_hypothesis_response(raw_response, hypothesis_number)
        
        logging.info(f"Generated hypothesis {hypothesis_number}: {hypothesis.get('title', 'Untitled')}")
        
        return hypothesis
    
    def _parse_hypothesis_response(self, raw_response: str, hypothesis_number: int) -> Dict[str, Any]:
        """
        Parse the LLM response into a structured hypothesis using the new unified format.

        Extracts: TITLE, SUMMARY, HYPOTHESIS, RATIONALE, and mode-specific outputs
        (FINAL DRUG/CANCER TYPE for drug-repurposing, FINAL_ANSWER for general)

        Args:
            raw_response: Raw response from LLM
            hypothesis_number: Sequential number for this hypothesis

        Returns:
            Structured hypothesis dictionary
        """
        import re

        # Helper function to extract section content
        def extract_section(pattern_name, text, next_pattern=None):
            """Extract content between pattern_name and next_pattern"""
            pattern = rf'{pattern_name}:\s*\n?(.*?)(?=\n(?:{next_pattern}:|FINAL|$))'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
            return None

        # Extract core fields
        title = extract_section("TITLE", raw_response, "SUMMARY") or "Untitled Hypothesis"
        summary = extract_section("SUMMARY", raw_response, "HYPOTHESIS") or "No summary provided"
        hypothesis_statement = extract_section("HYPOTHESIS", raw_response, "RATIONALE") or "No hypothesis statement provided"
        rationale = extract_section("RATIONALE", raw_response, "FINAL") or "No rationale provided"

        # Extract mode-specific output fields
        final_drug = None
        cancer_type = None
        final_answer = None

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

        # Extract final_prediction for lethal_genes modes
        final_prediction = None
        if self.mode in ["lethal_genes", "lethal_genes_2"]:
            # Normalize unicode hyphens to ASCII before matching
            normalized_response = raw_response.replace('‑', '-').replace('–', '-').replace('—', '-')
            prediction_match = re.search(r'FINAL_PREDICTION:\s*(well-based|random)', normalized_response, re.IGNORECASE)
            if prediction_match:
                final_prediction = prediction_match.group(1).strip().lower()

        # Create hypothesis structure with new unified format
        hypothesis = {
            "id": f"gen-{hypothesis_number:03d}-{str(uuid.uuid4())[:8]}",
            "title": title,
            "summary": summary,
            "hypothesis_statement": hypothesis_statement,
            "rationale": rationale,
            "origin": "generation",
            "generation_method": "literature_exploration",
            "elo_score": 1200,  # Default ELO score
            "fitness_score": None,  # Will be set by reflection agent
            "generation_depth": 0,
            "created_from": "initial_population"
        }

        # Add mode-specific fields
        if self.mode == "drug-repurposing":
            hypothesis["final_drug"] = final_drug
            hypothesis["cancer_type"] = cancer_type
        elif self.mode == "general":
            hypothesis["final_answer"] = final_answer
        elif self.mode in ["lethal_genes", "lethal_genes_2"]:
            hypothesis["final_prediction"] = final_prediction

        return hypothesis
    
    def _process_hypothesis(self, hypothesis_text: str, id_suffix: str) -> Dict[str, Any]:
        """Process raw hypothesis text into a structured format (from Pipeline2).
        
        Parses hypothesis text that follows the format:
        ***HYPOTHESIS TITLE***: [Clear, concise title]
        ***SECTION: INTRODUCTION***: [Introduction to the hypothesis]
        ***SECTION: RECENT FINDINGS AND RELATED RESEARCH***: [Recent findings and related research]
        ***SECTION: HYPOTHESIS***: [Hypothesis]
        ***SECTION: RATIONALE AND SPECIFICITY***: [Rationale and specificity]
        ***SECTION: EXPERIMENTAL DESIGN AND VALIDATION***: [Experimental design and validation]
        
        Note: The entire structured output from GPT (with all sections above) is considered 
        a complete scientific hypothesis, even though there's a "HYPOTHESIS" section within it.
        The "hypothesis_core" field extracted from that section serves as an abbreviation or 
        summary of the full, structured hypothesis.
        """
        # Initialize structured sections
        title = "Untitled Hypothesis"
        introduction = ""
        recent_findings = ""
        hypothesis = ""
        rationale = ""
        experimental_design = ""
        
        # Extract sections based on the standardized format with *** markers
        full_text = hypothesis_text.strip()
        
        # Define the expected section patterns - updated for simple format without *** markers
        section_patterns = {
            "INTRODUCTION": r"Introduction:\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)",
            "RECENT FINDINGS": r"Recent findings and related research:\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)",
            "HYPOTHESIS": r"Hypothesis:\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)",
            "RATIONALE": r"Rationale and specificity:\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)",
            "EXPERIMENTAL DESIGN": r"Experimental design and validation:\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)",
            "PROPOSED HYPOTHESIS": r"Proposed hypothesis \(detailed description for domain experts\):\s*(.*?)(?=\n\n|\Z)"
        }
        
        # First try to extract sections using the standardized format
        import re
        
        # Join all lines to handle potential line breaks within the text
        single_line_text = re.sub(r'\n+', ' ', full_text)
        
        # Extract each section based on the patterns
        extracted_sections = {}
        for section_name, pattern in section_patterns.items():
            matches = re.findall(pattern, single_line_text, re.DOTALL | re.IGNORECASE)
            if matches:
                extracted_sections[section_name] = matches[0].strip()
        
        # If we found sections using the new simple format, use them
        if extracted_sections:
            # Use proposed hypothesis as title if available, otherwise extract from hypothesis section
            title = extracted_sections.get("PROPOSED HYPOTHESIS", extracted_sections.get("HYPOTHESIS", title))
            introduction = extracted_sections.get("INTRODUCTION", introduction)
            recent_findings = extracted_sections.get("RECENT FINDINGS", recent_findings)
            hypothesis = extracted_sections.get("HYPOTHESIS", hypothesis)
            rationale = extracted_sections.get("RATIONALE", rationale)
            experimental_design = extracted_sections.get("EXPERIMENTAL DESIGN", experimental_design)
        else:
            # If we couldn't extract using the standardized format, try alternative extraction
            
            # Try simple section extraction based on common formatting patterns
            sections = {}
            current_section = None
            current_content = []
            
            for line in full_text.split('\n'):
                line_stripped = line.strip()
                
                # Check for section headers in various formats
                section_match = re.match(r'^(?:\*{1,3})?(?:SECTION: )?([A-Z][A-Z\s]+)(?:\*{1,3})?:(.*)$', line_stripped)
                if section_match:
                    section_name = section_match.group(1).strip()
                    content_start = section_match.group(2).strip()
                    
                    # If we found a new section, save the previous one
                    if current_section and current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                        current_content = []
                    
                    current_section = section_name
                    if content_start:
                        current_content.append(content_start)
                elif current_section:
                    current_content.append(line)
            
            # Save the last section
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            
            # Map extracted sections to our standardized fields
            for section_name, content in sections.items():
                if "TITLE" in section_name:
                    title = content
                elif "INTRO" in section_name:
                    introduction = content
                elif "FINDINGS" in section_name or "RESEARCH" in section_name:
                    recent_findings = content
                elif "HYPOTHESIS" in section_name:
                    hypothesis = content
                elif "RATIONALE" in section_name or "SPECIFIC" in section_name:
                    rationale = content
                elif "EXPERIMENT" in section_name or "DESIGN" in section_name or "TEST" in section_name or "VALIDATION" in section_name:
                    experimental_design = content
        
        # Ensure we have a hypothesis - if not, use the entire text as fallback
        if not hypothesis:
            hypothesis = full_text
            
        # Combine sections for the description field
        description = ""
        if introduction:
            description += "Introduction:\n" + introduction + "\n\n"
        if recent_findings:
            description += "Recent findings and related research:\n" + recent_findings + "\n\n"
        if hypothesis:
            description += "Hypothesis:\n" + hypothesis + "\n\n"
        if rationale:
            description += "Rationale and specificity:\n" + rationale + "\n\n"
        if experimental_design:
            description += "Experimental design and validation:\n" + experimental_design
        
        # If we couldn't construct a structured description, use the full text
        if not description.strip():
            description = full_text
        
        # Use experimental design as testability if no explicit testability is provided
        testability = experimental_design if experimental_design else "No specific testability notes provided."

        # Extract FINAL DRUG and CANCER TYPE for drug-repurposing mode
        final_drug = None
        cancer_type = None
        if self.mode == "drug-repurposing":
            import re
            drug_match = re.search(r'FINAL DRUG:\s*([^\n]+)', full_text, re.IGNORECASE)
            if drug_match:
                final_drug = drug_match.group(1).strip()
            cancer_match = re.search(r'CANCER TYPE:\s*([^\n]+)', full_text, re.IGNORECASE)
            if cancer_match:
                cancer_type = cancer_match.group(1).strip()

        return {
            "id": self._generate_id(id_suffix),
            "title": title,
            "description": description,
            "testability_notes": testability,
            "introduction": introduction,
            "recent_findings": recent_findings,
            "hypothesis_core": hypothesis,
            "rationale": rationale,
            "experimental_design": experimental_design,
            "final_drug": final_drug,
            "cancer_type": cancer_type,
            "elo_score": 1200,  # Initial ELO score
            "origin": f"generation_{id_suffix}",
            "reviews": [],
            "cluster_id": None,
            "generation_timestamp": time.time()
        }
    
    def _generate_id(self, suffix: str) -> str:
        """Generate a unique ID for a hypothesis."""
        return f"hyp_{suffix}_{uuid.uuid4().hex[:8]}"