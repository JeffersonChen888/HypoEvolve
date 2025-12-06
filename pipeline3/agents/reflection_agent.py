"""
Simplified Reflection Agent for Pipeline3 - Genetic Algorithm Implementation

This agent handles fitness evaluation in the genetic algorithm by performing
simplified initial reviews to score hypothesis quality.

Removed from Pipeline2:
- Deep verification review
- Observation review  
- Simulation review
- Tournament review
- Full review with web search
- Multi-stage review process

Kept:
- Initial review (correctness, novelty, quality scoring)
- Simple fitness function calculation
"""

import logging
import re
from typing import Dict, List, Any

from external_tools.gpt4o import gpt4o_generate
from external_tools.web_search import perform_web_search


class ReflectionAgent:
    """
    Simplified Reflection Agent focused solely on fitness evaluation
    through basic hypothesis quality assessment.
    """
    
    def __init__(self, mode: str = "drug-repurposing"):
        """
        Initialize the simplified reflection agent.
        
        Args:
            mode: Pipeline mode ("drug-repurposing" or "general")
        """
        self.mode = mode
        logging.info(f"Initialized simplified Reflection Agent in {mode} mode")
    
    def evaluate_fitness(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the fitness of a hypothesis for genetic algorithm selection.

        Args:
            hypothesis: Hypothesis to evaluate

        Returns:
            Dictionary containing fitness scores and evaluation details
        """
        hypothesis_id = hypothesis.get("id", "unknown")
        logging.info(f"Evaluating fitness for hypothesis: {hypothesis_id}")

        # Handle lethal_genes and lethal_genes_2 modes with specialized evaluation
        if self.mode in ["lethal_genes", "lethal_genes_2"]:
            return self._evaluate_lethal_genes_fitness(hypothesis)

        # Standard evaluation for drug-repurposing and general modes
        # Perform initial review first
        initial_review_result = self._perform_initial_review(
            hypothesis.get("title", ""),
            hypothesis.get("description", "")
        )

        # Then perform full review with web search (Pipeline2 approach)
        full_review_text = self._perform_full_review_with_search(
            hypothesis.get("title", ""),
            hypothesis.get("description", ""),
            initial_review_result
        )

        # Create combined review result for fitness calculation
        review_result = initial_review_result.copy()
        review_result["full_review"] = full_review_text

        # Store the full review in the hypothesis (Pipeline2 approach)
        if "reviews" not in hypothesis:
            hypothesis["reviews"] = []
        hypothesis["reviews"].append(full_review_text)

        # Calculate fitness score
        fitness_score = self._calculate_fitness_score(review_result)

        # Create evaluation result (adapted for Pipeline2's format)
        scores = review_result.get("scores", {})
        result = {
            "hypothesis_id": hypothesis_id,
            "fitness_score": fitness_score,
            "correctness_score": scores.get("correctness", 0),
            "novelty_score": scores.get("novelty", 0),
            "quality_score": scores.get("quality", 0),
            "recommendation": review_result.get("recommendation", "PROCEED"),
            "review_summary": review_result.get("text", ""),
            "evaluation_method": "initial_review"
        }

        logging.info(f"Fitness evaluation complete for {hypothesis_id}: score = {fitness_score:.2f}")

        return result
    
    def evaluate_population_fitness(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate fitness for an entire population of hypotheses.
        
        Args:
            population: List of hypotheses to evaluate
            
        Returns:
            List of fitness evaluations
        """
        logging.info(f"Evaluating fitness for population of {len(population)} hypotheses")
        
        evaluations = []
        for hypothesis in population:
            evaluation = self.evaluate_fitness(hypothesis)
            evaluations.append(evaluation)
        
        # Sort by fitness score (descending)
        evaluations.sort(key=lambda x: x.get("fitness_score", 0), reverse=True)
        
        logging.info(f"Population fitness evaluation complete. Top score: {evaluations[0].get('fitness_score', 0):.2f}")
        
        return evaluations
    
    def _perform_initial_review(self, title: str, description: str) -> dict:
        """
        Perform initial fast-pass review to assess basic hypothesis plausibility.
        This review doesn't use external tools, as specified in the paper.
        
        According to the paper, this step should:
        1. Quickly assess basic quality, correctness, and novelty
        2. Filter out flawed, non-novel or unsuitable hypotheses
        
        Returns:
            Dictionary containing structured review data with clear assessment signals
        """
        prompt = f"""
        Perform an initial fast-pass review of the following scientific hypothesis:
        
        Title: {title}
        Description: {description}
        
        In your initial review, carefully assess the following criteria:
        1. Quality: Is the hypothesis well-formulated and of high scientific quality?
        2. Correctness: Is the hypothesis aligned with established scientific principles and logically sound?
        3. Novelty: Does the hypothesis present a genuinely novel idea or approach not already established in the field?
        4. Red flags: Are there any fundamental logical flaws or inconsistencies?
        
        Your review must explicitly conclude with an overall recommendation in this format:
        "RECOMMENDATION: [PROCEED or REJECT]"
        
        Additionally, include a numeric score from 1-5 (where 1 is lowest and 5 is highest) for each criterion in this format:
        "SCORES: Quality=[1-5], Correctness=[1-5], Novelty=[1-5]"
        
        Provide a concise assessment (3-5 sentences) followed by the recommendation and scores.
        """
        
        # Generate the initial review
        review_text = gpt4o_generate(prompt)
        
        # Parse the review to extract structured data
        review_data = {
            "text": review_text,
            "recommendation": "PROCEED",  # Default value
            "scores": {
                "quality": 0,
                "correctness": 0,
                "novelty": 0
            }
        }
        
        # Extract the recommendation (PROCEED or REJECT)
        recommendation_match = re.search(r"RECOMMENDATION:\s*(PROCEED|REJECT)", review_text, re.IGNORECASE)
        if recommendation_match:
            review_data["recommendation"] = recommendation_match.group(1).upper()
        
        # Extract scores for each criterion
        scores_match = re.search(r"SCORES:\s*Quality=(\d+),\s*Correctness=(\d+),\s*Novelty=(\d+)", review_text, re.IGNORECASE)
        if scores_match:
            try:
                review_data["scores"]["quality"] = int(scores_match.group(1))
                review_data["scores"]["correctness"] = int(scores_match.group(2))
                review_data["scores"]["novelty"] = int(scores_match.group(3))
            except (IndexError, ValueError) as e:
                logging.warning(f"Error parsing scores from initial review: {e}")
        
        # Log the outcome
        logging.info(f"Initial review complete. Recommendation: {review_data['recommendation']}")

        return review_data

    def _perform_full_review_with_search(self, title: str, description: str, initial_review: dict) -> str:
        """
        Perform a full review to assess the hypothesis in detail, with web search integration
        as described in the paper.

        Args:
            title: Hypothesis title
            description: Hypothesis description
            initial_review: Dictionary containing the structured initial review data

        Returns:
            String containing the full review text
        """
        # Step 1: Generate search queries based on the hypothesis
        search_query_prompt = f"""
        Generate 3 GENERAL web search queries (maximum 6 words each) to find relevant academic articles about this hypothesis:

        Title: {title}
        Description: {description}

        REQUIREMENTS FOR EACH QUERY:
        - Use ONLY general field terms (e.g., "bioinformatics", "sequence analysis", "pattern recognition")
        - NO specific details from the hypothesis (no exact sequences, numbers, or algorithm specifics)
        - Maximum 6 words per query
        - Focus on the general scientific field and concepts

        Examples of good general queries:
        - "bioinformatics sequence analysis"
        - "pattern recognition algorithms"
        - "computational biology methods"

        Format your response as a numbered list of 3 GENERAL search queries (max 6 words each):
        """

        search_queries_text = gpt4o_generate(search_query_prompt)
        search_queries = [line.strip().replace(f"{i+1}. ", "").replace(f"{i+1}.", "")
                          for i, line in enumerate(search_queries_text.split('\n'))
                          if line.strip() and any(char.isdigit() for char in line[:3])]

        # Step 2: Perform web searches to gather relevant literature
        search_results = []
        total_papers_found = 0
        for query in search_queries:  # Limit to first 3 queries
            try:
                # Use the perform_web_search function which now connects to actual research databases
                result = perform_web_search(query)
                search_results.append({"query": query, "results": result})

                # Count papers found for this query
                if result and "No relevant papers found" not in result:
                    paper_count = len([line for line in result.split('\n')
                                     if line.strip() and not line.startswith('   ')])
                    total_papers_found += paper_count
                    logging.info(f"Web search completed for query '{query}': {paper_count} papers found")
                else:
                    logging.info(f"Web search completed for query '{query}': 0 papers found")
            except Exception as e:
                logging.error(f"Web search failed for query '{query}': {e}")
                search_results.append({"query": query, "results": "Search failed or not available"})

        # Log total search results summary
        logging.info(f"Literature search complete: {total_papers_found} papers found across {len(search_queries)} queries")

        # Step 3: Generate the full review with search results
        articles_context = "\n\n".join([
            f"Search results for '{result['query']}':\n{result['results']}"
            for result in search_results
        ])

        # Log what literature context is being passed to the reflection agent
        if articles_context.strip() and "No relevant papers found" not in articles_context:
            logging.info(f"Reflection agent received literature context: {len(articles_context)} characters")
            # Log a sample of the papers found
            for result in search_results:
                if result['results'] and "No relevant papers found" not in result['results']:
                    lines = result['results'].split('\n')
                    paper_lines = [line for line in lines if line.strip() and not line.startswith('   ')]
                    for paper_line in paper_lines[:3]:  # Log first 3 papers
                        if paper_line.strip():
                            logging.info(f"  Literature for reflection: {paper_line.strip()}")
        else:
            logging.info("No literature context available for reflection agent")

        # If no results were found or all searches failed, add a note
        if not articles_context.strip():
            articles_context = "No relevant literature was found for this hypothesis. The review will proceed without literature context."

        # Format the initial review for inclusion in the prompt
        initial_review_text = initial_review["text"]
        initial_scores = f"Initial scores: Quality={initial_review['scores']['quality']}, " \
                         f"Correctness={initial_review['scores']['correctness']}, " \
                         f"Novelty={initial_review['scores']['novelty']}"

        # Break down the full review into 5 separate API calls for better reliability
        logging.info("Performing full review in 5 separate stages")

        # Stage 1: Related Articles
        stage1_prompt = f"""
        Based on the following hypothesis and search results, identify 5-8 relevant articles:

        Title: {title}
        Description: {description}

        Search Results: {articles_context}

        Provide only the "Related Articles:" section. For each article, provide:
        - Article title and citation
        - Brief explanation of why it's relevant to the hypothesis

        Format exactly as:
        Related Articles:
        [1] Article title - Source: Brief explanation...
        [2] Article title - Source: Brief explanation...
        """

        related_articles = gpt4o_generate(stage1_prompt)
        logging.info("Stage 1 complete: Related Articles")

        # Stage 2: Known Aspects
        stage2_prompt = f"""
        Based on this hypothesis and search results, identify what's already established:

        Title: {title}
        Description: {description}

        Related Articles:
        {related_articles}

        Provide only the "Known Aspects:" section - summarize components already established in literature.

        Format exactly as:
        Known Aspects:
        • Aspect 1: Description with citations...
        • Aspect 2: Description with citations...
        """

        known_aspects = gpt4o_generate(stage2_prompt)
        logging.info("Stage 2 complete: Known Aspects")

        # Stage 3: Novel Components
        stage3_prompt = f"""
        Based on this hypothesis and literature review, identify genuinely new contributions:

        Title: {title}
        Description: {description}

        Known Aspects:
        {known_aspects}

        Provide only the "Novel Components:" section - identify what represents new contributions.

        Format exactly as:
        Novel Components:
        • Novel aspect 1: Description...
        • Novel aspect 2: Description...
        """

        novel_components = gpt4o_generate(stage3_prompt)
        logging.info("Stage 3 complete: Novel Components")

        # Stage 4: Assumptions
        stage4_prompt = f"""
        Based on this hypothesis, identify 6-8 key underlying assumptions:

        Title: {title}
        Description: {description}

        Provide only the "Assumptions of the Idea:" section with 6-8 bullet points.

        Format exactly as:
        Assumptions of the Idea:
        • Assumption 1: Description...
        • Assumption 2: Description...
        """

        assumptions = gpt4o_generate(stage4_prompt)
        logging.info("Stage 4 complete: Assumptions")

        # Stage 5: Scrutiny
        stage5_prompt = f"""
        Critically analyze each assumption for plausibility and weaknesses:

        Assumptions:
        {assumptions}

        Related Literature:
        {related_articles}

        Provide only the "Scrutiny of Assumptions and Reasoning:" section with critical analysis.

        Format exactly as:
        Scrutiny of Assumptions and Reasoning:
        • [First assumption]: Analysis with strengths/weaknesses...
        • [Second assumption]: Analysis with strengths/weaknesses...
        """

        scrutiny = gpt4o_generate(stage5_prompt)
        logging.info("Stage 5 complete: Scrutiny")

        # Combine all stages into final review
        full_review = f"""
{related_articles}

{known_aspects}

{novel_components}

{assumptions}

{scrutiny}
        """

        logging.info("Full review complete: All 5 stages combined")
        return full_review

    
    def _calculate_fitness_score(self, review_result: Dict[str, Any]) -> float:
        """
        Calculate overall fitness score using Pipeline2's weighted approach.
        
        Args:
            review_result: Dictionary containing review format with scores dict
            
        Returns:
            Overall fitness score (0-100)
        """
        import config
        
        # Extract individual scores (3 dimensions matching Pipeline2)
        scores = review_result.get("scores", {})
        correctness = scores.get("correctness", 3)
        novelty = scores.get("novelty", 3)
        quality = scores.get("quality", 3)
        
        # Calculate weighted average using Pipeline2's weights
        weights = {
            'correctness': config.FITNESS_CORRECTNESS_WEIGHT,  # 1.0
            'novelty': config.FITNESS_NOVELTY_WEIGHT,         # 1.3  
            'quality': config.FITNESS_QUALITY_WEIGHT          # 1.5
        }
        
        total_weight = sum(weights.values())
        weighted_sum = (correctness * weights['correctness'] + 
                       novelty * weights['novelty'] + 
                       quality * weights['quality'])
        
        if total_weight > 0:
            weighted_avg = weighted_sum / total_weight
            # Scale from 1-5 range to 0-100 range
            fitness_score = ((weighted_avg - 1) / 4) * 100
        else:
            fitness_score = 60.0  # Default middle score
        
        return round(fitness_score, 2)

    def _evaluate_lethal_genes_fitness(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate fitness for lethal genes hypotheses using specialized criteria.

        Args:
            hypothesis: Lethal genes hypothesis to evaluate

        Returns:
            Dictionary containing fitness scores and evaluation details
        """
        from prompts import PROMPT_LETHAL_GENES_REFLECTION

        hypothesis_id = hypothesis.get("id", "unknown")
        gene_a = hypothesis.get("gene_a", "Unknown")
        gene_b = hypothesis.get("gene_b", "Unknown")

        logging.info(f"Performing lethal genes fitness evaluation for {gene_a} x {gene_b}")

        # Format the hypothesis content for evaluation
        hypothesis_text = self._format_lethal_genes_hypothesis(hypothesis)

        # Get literature context from hypothesis (stored during generation)
        literature_context = hypothesis.get("literature_context", "No literature context available.")

        # Generate evaluation using lethal genes reflection prompt
        prompt = PROMPT_LETHAL_GENES_REFLECTION.format(
            gene_a=gene_a,
            gene_b=gene_b,
            hypothesis_text=hypothesis_text,
            literature_context=literature_context
        )

        review_text = gpt4o_generate(prompt)

        # Store review in hypothesis
        if "reviews" not in hypothesis:
            hypothesis["reviews"] = []
        hypothesis["reviews"].append(review_text)

        # Parse scores from the review
        scores = self._parse_lethal_genes_scores(review_text)

        # Calculate fitness score
        fitness_score = self._calculate_lethal_genes_fitness_score(scores)

        # Extract assessment and recommendation
        plausibility = self._extract_section(review_text, "Overall Plausibility Assessment")
        recommendation = self._extract_section(review_text, "Recommendation")

        result = {
            "hypothesis_id": hypothesis_id,
            "fitness_score": fitness_score,
            "novelty_score": scores.get("novelty", 5),
            "biological_relevance_score": scores.get("biological_relevance", 5),
            "mechanistic_clarity_score": scores.get("mechanistic_clarity", 5),
            "rival_quality_score": scores.get("rival_quality", 5),
            "tractability_score": scores.get("tractability", 5),
            "clinical_relevance_score": scores.get("clinical_relevance", 5),
            "plausibility_assessment": plausibility,
            "recommendation": recommendation,
            "review_summary": review_text,
            "evaluation_method": "lethal_genes_reflection"
        }

        logging.info(f"Lethal genes fitness evaluation complete for {hypothesis_id}: score = {fitness_score:.2f}")

        return result

    def _format_lethal_genes_hypothesis(self, hypothesis: Dict[str, Any]) -> str:
        """
        Format lethal genes hypothesis for evaluation.

        Args:
            hypothesis: Hypothesis dictionary

        Returns:
            Formatted hypothesis text
        """
        sections = []

        # Add basic info
        sections.append(f"Title: {hypothesis.get('title', 'No title')}")
        sections.append(f"Description: {hypothesis.get('description', 'No description')}")

        # Add biological plausibility
        if hypothesis.get("biological_plausibility"):
            sections.append(f"\nBiological Plausibility:\n{hypothesis['biological_plausibility']}")

        # Add clinical relevance
        if hypothesis.get("clinical_relevance"):
            sections.append(f"\nClinical Relevance:\n{hypothesis['clinical_relevance']}")

        # Add primary hypothesis
        primary = hypothesis.get("primary_hypothesis", {})
        if isinstance(primary, dict) and primary.get("statement"):
            sections.append(f"\nPrimary Hypothesis:\n{primary.get('statement', '')}")

        # Add rival hypothesis
        rival = hypothesis.get("rival_hypothesis", {})
        if isinstance(rival, dict) and rival.get("statement"):
            sections.append(f"\nRival Hypothesis:\n{rival.get('statement', '')}")

        # Add contrast description
        if hypothesis.get("contrast_description"):
            sections.append(f"\nContrast Description:\n{hypothesis['contrast_description']}")

        return "\n\n".join(sections)

    def _parse_lethal_genes_scores(self, review_text: str) -> Dict[str, int]:
        """
        Parse scores from lethal genes reflection review.

        Args:
            review_text: Review text containing scores

        Returns:
            Dictionary mapping score names to values
        """
        scores = {
            "novelty": 5,
            "biological_relevance": 5,
            "mechanistic_clarity": 5,
            "rival_quality": 5,
            "tractability": 5,
            "clinical_relevance": 5
        }

        # Try to extract scores using regex patterns
        # Pattern matches "Score: 9" or "Score 9" or "Score: 9/10"
        patterns = {
            "novelty": r"Novelty.*?Score:?\s*(\d+)",
            "biological_relevance": r"Biological [Rr]elevance.*?Score:?\s*(\d+)",
            "mechanistic_clarity": r"Mechanistic [Cc]larity.*?Score:?\s*(\d+)",
            "rival_quality": r"Rival [Qq]uality.*?Score:?\s*(\d+)",
            "tractability": r"Tractability.*?Score:?\s*(\d+)",
            "clinical_relevance": r"Clinical [Rr]elevance.*?Score:?\s*(\d+)"
        }

        for score_name, pattern in patterns.items():
            match = re.search(pattern, review_text, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    scores[score_name] = int(match.group(1))
                except (IndexError, ValueError) as e:
                    logging.warning(f"Failed to parse {score_name} score: {e}")

        return scores

    def _calculate_lethal_genes_fitness_score(self, scores: Dict[str, int]) -> float:
        """
        Calculate fitness score for lethal genes hypotheses from 6 criteria with weighted scoring.

        Args:
            scores: Dictionary of scores (each 0-10)

        Returns:
            Overall fitness score (0-100)
        """
        # Weighted scoring based on importance
        # Higher weights for: mechanistic novelty, rival quality, mechanistic clarity
        # Lower weight for: clinical relevance (secondary per specification)
        weights = {
            'biological_relevance': 1.0,
            'novelty': 1.5,              # High weight - key differentiator
            'mechanistic_clarity': 1.3,   # Essential for understanding
            'tractability': 1.0,
            'rival_quality': 1.4,         # HIGH weight - prevents oversimplification
            'clinical_relevance': 0.8     # Secondary per specification
        }

        # Calculate weighted sum
        total_weight = sum(weights.values())
        weighted_sum = sum(scores.get(key, 5) * weight for key, weight in weights.items())

        # Normalize to 0-100 scale
        # Max possible score per criterion is 10, so max weighted sum is 10 * total_weight
        max_weighted_sum = 10 * total_weight

        if max_weighted_sum > 0:
            fitness_score = (weighted_sum / max_weighted_sum) * 100
        else:
            fitness_score = 50.0  # Default middle score

        return round(fitness_score, 2)

    def _extract_section(self, text: str, section_name: str) -> str:
        """
        Extract a specific section from review text.

        Args:
            text: Full review text
            section_name: Name of section to extract

        Returns:
            Extracted section text or empty string
        """
        pattern = rf"{re.escape(section_name)}:\s*(.*?)(?=\n\n[A-Z]|\Z)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def get_fitness_statistics(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics for a set of fitness evaluations.
        
        Args:
            evaluations: List of fitness evaluation results
            
        Returns:
            Dictionary containing fitness statistics
        """
        if not evaluations:
            return {"error": "No evaluations provided"}
        
        fitness_scores = [eval_result.get("fitness_score", 0) for eval_result in evaluations]
        
        stats = {
            "population_size": len(evaluations),
            "mean_fitness": sum(fitness_scores) / len(fitness_scores),
            "max_fitness": max(fitness_scores),
            "min_fitness": min(fitness_scores),
            "fitness_range": max(fitness_scores) - min(fitness_scores),
            "top_25_percent_threshold": sorted(fitness_scores, reverse=True)[len(fitness_scores)//4] if len(fitness_scores) >= 4 else max(fitness_scores)
        }
        
        # Round values for readability
        for key in ["mean_fitness", "max_fitness", "min_fitness", "fitness_range", "top_25_percent_threshold"]:
            stats[key] = round(stats[key], 2)
        
        return stats