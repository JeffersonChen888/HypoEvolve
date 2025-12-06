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

from external_tools.gpt4o import gpt4o_generate
from external_tools.web_search import perform_web_search
from external_tools.web_search import search_literature
from prompts import PROMPT_LITERATURE_EXPLORATION
# from tcga_cancer_types import get_tcga_cancer_types_prompt  # Function doesn't exist


class GenerationAgent:
    """
    Simplified Generation Agent focused solely on population initialization
    through literature-grounded hypothesis generation.
    """
    
    def __init__(self, mode: str = "drug-repurposing", run_folder: str = None):
        """
        Initialize the simplified generation agent.

        Args:
            mode: Pipeline mode ("drug-repurposing" or "general")
            run_folder: Folder path for this specific run (for per-pair logs in batch mode)
        """
        self.mode = mode
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

        # Step 1: Generate search queries in one call and perform literature exploration
        search_queries = self._generate_search_queries_batch(research_goal)
        all_papers, literature_context = self._search_with_queries(search_queries, num_papers)
        
        # Step 3: Generate hypotheses using literature exploration prompt (Pipeline2 approach)
        hypotheses = []
        for i in range(population_size):
            # Include cancer type standards for drug repurposing tasks
            cancer_type_standards = ""
            if self.mode == "drug-repurposing":
                from tcga_cancer_types import TCGA_CANCER_TYPES
                cancer_type_standards = TCGA_CANCER_TYPES
            
            prompt = PROMPT_LITERATURE_EXPLORATION.format(
                goal=research_goal,
                preferences=preferences,
                source_hypothesis=source_hypothesis,
                instructions=instructions,
                articles_with_reasoning=literature_context,
                cancer_type_standards=cancer_type_standards
            )
            
            # Log information about what literature context is being passed to the LLM
            if literature_context and "No relevant scientific literature was found" not in literature_context:
                logging.info(f"Passing literature analysis to LLM for hypothesis {i+1}: {len(literature_context)} characters of analysis")
                logging.info(f"Literature analysis includes content from {len(all_papers)} papers total")
            else:
                logging.info(f"No literature analysis available for hypothesis {i+1} - LLM will use general knowledge")
            
            # Generate hypothesis
            hypothesis_text = gpt4o_generate(prompt)
            
            # Process into structured format
            hypothesis = self._process_hypothesis(hypothesis_text, f"lit-{i+1}")
            hypothesis["generation_method"] = "literature_exploration"
            hypothesis["search_iterations"] = len(search_queries)
            hypothesis["search_queries"] = search_queries
            hypothesis["num_papers_found"] = len(all_papers)
            
            hypotheses.append(hypothesis)
        
        logging.info(f"Generated {len(hypotheses)} hypotheses for initial population")
        
        return {
            "hypotheses": hypotheses,
            "literature_context": literature_context,
            "papers_found": len(all_papers),
            "search_queries": search_queries
        }

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
            hypothesis_text = gpt4o_generate(
                self._get_lethal_genes_prompt(gene_a, gene_b, literature_context)
            )

            # Process into structured format
            hypothesis = self._process_lethal_genes_hypothesis(
                hypothesis_text, gene_a, gene_b, f"sl-{i+1}"
            )
            hypothesis["generation_method"] = "lethal_genes_literature"
            hypothesis["search_queries"] = search_queries
            hypothesis["num_papers_found"] = len(all_papers)
            hypothesis["literature_context"] = literature_context  # Store for later use in reflection

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
            hypothesis_text = gpt4o_generate(
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
            hypothesis_text = gpt4o_generate(complete_prompt)
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
            "contrast_description": ""
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

        # Extract Primary hypothesis
        primary_match = re.search(
            r'Primary:\s*(.+?)(?=\n\nRival:|\Z)',
            text, re.DOTALL | re.IGNORECASE
        )
        if primary_match:
            sections["primary_hypothesis"] = self._parse_hypothesis_subsections(
                primary_match.group(1)
            )

        # Extract Rival hypothesis
        rival_match = re.search(
            r'Rival:\s*(.+?)(?=\n\nPathway visualization:|\Z)',
            text, re.DOTALL | re.IGNORECASE
        )
        if rival_match:
            sections["rival_hypothesis"] = self._parse_hypothesis_subsections(
                rival_match.group(1)
            )

        # Extract DOT graph
        dot_match = re.search(
            r'digraph\s+\w*\s*\{(.+?)\}',
            text, re.DOTALL
        )
        if dot_match:
            sections["pathway_dot"] = f"digraph SL {{{dot_match.group(1)}}}"

        # Extract contrast description
        contrast_match = re.search(
            r'Contrast description:\s*(.+?)(?=\n\nAttributes|\Z)',
            text, re.DOTALL | re.IGNORECASE
        )
        if contrast_match:
            sections["contrast_description"] = contrast_match.group(1).strip()

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
        response = gpt4o_generate(query_prompt)
        
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

        response = gpt4o_generate(query_prompt)

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
        synthesis += gpt4o_generate(prompt)
        
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
        # Include cancer type standards for drug repurposing
        cancer_type_context = ""
        if self.mode == "drug-repurposing":
            cancer_type_context = ""  # Simplified for GPQA evaluation
        
        prompt = f"""
        You are a scientific researcher tasked with generating a novel hypothesis based on the provided research goal and literature context.

        Research Goal: {research_goal}

        {cancer_type_context}

        Literature Context:
        {literature_context}

        Generate a novel, testable scientific hypothesis that:
        1. Directly addresses the research goal
        2. Is grounded in the provided literature context
        3. Proposes a specific mechanism or relationship
        4. Can be experimentally validated
        5. Offers potential for significant scientific impact

        {"For drug repurposing tasks, ensure your hypothesis identifies specific drugs and cancer types using TCGA terminology." if self.mode == "drug-repurposing" else ""}

        Format your response as:

        HYPOTHESIS TITLE: [Concise, descriptive title]

        HYPOTHESIS DESCRIPTION: [Detailed description of the hypothesis, including proposed mechanisms, expected relationships, and scientific rationale. Should be 3-4 sentences.]

        TESTABILITY: [Brief description of how this hypothesis could be experimentally tested]

        {"FINAL DRUG: [drug name]" if self.mode == "drug-repurposing" else ""}
        {"CANCER TYPE: [TCGA cancer type]" if self.mode == "drug-repurposing" else ""}
        """
        
        # Generate hypothesis
        raw_response = gpt4o_generate(prompt)
        
        # Parse response
        hypothesis = self._parse_hypothesis_response(raw_response, hypothesis_number)
        
        logging.info(f"Generated hypothesis {hypothesis_number}: {hypothesis.get('title', 'Untitled')}")
        
        return hypothesis
    
    def _parse_hypothesis_response(self, raw_response: str, hypothesis_number: int) -> Dict[str, Any]:
        """
        Parse the LLM response into a structured hypothesis.
        
        Args:
            raw_response: Raw response from LLM
            hypothesis_number: Sequential number for this hypothesis
            
        Returns:
            Structured hypothesis dictionary
        """
        # Extract title
        title = "Untitled Hypothesis"
        if "HYPOTHESIS TITLE:" in raw_response:
            title_line = raw_response.split("HYPOTHESIS TITLE:")[1].split("\n")[0].strip()
            if title_line:
                title = title_line
        
        # Extract description
        description = "No description provided"
        if "HYPOTHESIS DESCRIPTION:" in raw_response:
            desc_start = raw_response.find("HYPOTHESIS DESCRIPTION:") + len("HYPOTHESIS DESCRIPTION:")
            desc_end = raw_response.find("TESTABILITY:", desc_start)
            if desc_end == -1:
                desc_end = len(raw_response)
            description = raw_response[desc_start:desc_end].strip()
        
        # Extract testability
        testability = "Testability not specified"
        if "TESTABILITY:" in raw_response:
            test_start = raw_response.find("TESTABILITY:") + len("TESTABILITY:")
            test_end = raw_response.find("FINAL DRUG:", test_start)
            if test_end == -1:
                test_end = raw_response.find("CANCER TYPE:", test_start)
            if test_end == -1:
                test_end = len(raw_response)
            testability = raw_response[test_start:test_end].strip()
        
        # Extract drug and cancer type for drug repurposing mode
        final_drug = None
        cancer_type = None
        if self.mode == "drug-repurposing":
            if "FINAL DRUG:" in raw_response:
                drug_line = raw_response.split("FINAL DRUG:")[1].split("\n")[0].strip()
                if drug_line:
                    final_drug = drug_line
            
            if "CANCER TYPE:" in raw_response:
                cancer_line = raw_response.split("CANCER TYPE:")[1].split("\n")[0].strip()
                if cancer_line:
                    cancer_type = cancer_line
        
        # Create hypothesis structure
        hypothesis = {
            "id": f"gen-{hypothesis_number:03d}-{str(uuid.uuid4())[:8]}",
            "title": title,
            "description": description,
            "testability_notes": testability,
            "origin": "generation",
            "generation_method": "literature_exploration",
            "elo_score": 1200,  # Default ELO score
            "fitness_score": None,  # Will be set by reflection agent
            "generation_depth": 0,
            "created_from": "initial_population"
        }
        
        # Add drug repurposing specific fields
        if self.mode == "drug-repurposing":
            hypothesis["final_drug"] = final_drug
            hypothesis["cancer_type"] = cancer_type
        
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
            "elo_score": 1200,  # Initial ELO score
            "origin": f"generation_{id_suffix}",
            "reviews": [],
            "cluster_id": None,
            "generation_timestamp": time.time()
        }
    
    def _generate_id(self, suffix: str) -> str:
        """Generate a unique ID for a hypothesis."""
        return f"hyp_{suffix}_{uuid.uuid4().hex[:8]}"