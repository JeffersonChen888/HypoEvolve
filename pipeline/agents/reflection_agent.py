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

from external_tools.llm_client import llm_generate
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
        self.t2d_runner = None
        
        # Novelty scoring: track baseline and previous generation predictions
        self.baseline_top_genes = []  # Set by supervisor before GA runs
        self.previous_generation_genes = []  # Updated after each generation

        logging.info(f"Initialized simplified Reflection Agent in {mode} mode")

    def _construct_description_from_hypothesis(self, hypothesis: Dict[str, Any]) -> str:
        """
        Construct a full description from the hypothesis fields.
        Supports both new unified format (summary, hypothesis_statement, rationale)
        and legacy format (description).

        Args:
            hypothesis: Hypothesis dictionary

        Returns:
            Full description string
        """
        # Try new unified format first
        if "summary" in hypothesis or "hypothesis_statement" in hypothesis or "rationale" in hypothesis:
            parts = []
            if hypothesis.get("summary"):
                parts.append(f"Summary: {hypothesis['summary']}")
            if hypothesis.get("hypothesis_statement"):
                parts.append(f"\nHypothesis: {hypothesis['hypothesis_statement']}")
            if hypothesis.get("rationale"):
                parts.append(f"\nRationale: {hypothesis['rationale']}")
            return "\n".join(parts)

        # Fall back to legacy description field
        return hypothesis.get("description", "No description available")

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

        # Add after existing mode checks:
        if self.mode == "t2d-target":
            return self._evaluate_t2d_fitness(hypothesis)

        # Standard evaluation for drug-repurposing and general modes
        # Construct full description from new unified fields or use legacy description
        full_description = self._construct_description_from_hypothesis(hypothesis)

        # Perform initial review first
        initial_review_result = self._perform_initial_review(
            hypothesis.get("title", ""),
            full_description
        )

        # Then perform full review with web search (Pipeline2 approach)
        full_review_text = self._perform_full_review_with_search(
            hypothesis.get("title", ""),
            full_description,
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

        Hypothesis Content:
        {description}

        The hypothesis above contains:
        - A summary (one-sentence core idea)
        - A hypothesis statement (2-3 sentence clear statement of the proposed mechanism/approach)
        - A rationale (detailed explanation of plausibility, supporting evidence, and testability)

        In your initial review, carefully assess the following criteria:
        1. Quality: Is the hypothesis well-formulated, scientifically rigorous, and clearly articulated?
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
        review_text = llm_generate(prompt)
        
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

        search_queries_text = llm_generate(search_query_prompt)
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

        Hypothesis Content:
        {description}

        (The hypothesis contains: summary, hypothesis statement, and detailed rationale)

        Search Results: {articles_context}

        Provide only the "Related Articles:" section. For each article, provide:
        - Article title and citation
        - Brief explanation of why it's relevant to the hypothesis

        Format exactly as:
        Related Articles:
        [1] Article title - Source: Brief explanation...
        [2] Article title - Source: Brief explanation...
        """

        related_articles = llm_generate(stage1_prompt)
        logging.info("Stage 1 complete: Related Articles")

        # Stage 2: Known Aspects
        stage2_prompt = f"""
        Based on this hypothesis and search results, identify what's already established:

        Title: {title}

        Hypothesis Content:
        {description}

        (The hypothesis contains: summary, hypothesis statement, and detailed rationale)

        Related Articles:
        {related_articles}

        Provide only the "Known Aspects:" section - summarize components already established in literature.

        Format exactly as:
        Known Aspects:
        • Aspect 1: Description with citations...
        • Aspect 2: Description with citations...
        """

        known_aspects = llm_generate(stage2_prompt)
        logging.info("Stage 2 complete: Known Aspects")

        # Stage 3: Novel Components
        stage3_prompt = f"""
        Based on this hypothesis and literature review, identify genuinely new contributions:

        Title: {title}

        Hypothesis Content:
        {description}

        (The hypothesis contains: summary, hypothesis statement, and detailed rationale)

        Known Aspects:
        {known_aspects}

        Provide only the "Novel Components:" section - identify what represents new contributions.

        Format exactly as:
        Novel Components:
        • Novel aspect 1: Description...
        • Novel aspect 2: Description...
        """

        novel_components = llm_generate(stage3_prompt)
        logging.info("Stage 3 complete: Novel Components")

        # Stage 4: Assumptions
        stage4_prompt = f"""
        Based on this hypothesis, identify 6-8 key underlying assumptions:

        Title: {title}

        Hypothesis Content:
        {description}

        (The hypothesis contains: summary, hypothesis statement, and detailed rationale)

        Provide only the "Assumptions of the Idea:" section with 6-8 bullet points.

        Format exactly as:
        Assumptions of the Idea:
        • Assumption 1: Description...
        • Assumption 2: Description...
        """

        assumptions = llm_generate(stage4_prompt)
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

        scrutiny = llm_generate(stage5_prompt)
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

        review_text = llm_generate(prompt)

        # Store review in hypothesis
        if "reviews" not in hypothesis:
            hypothesis["reviews"] = []
        hypothesis["reviews"].append(review_text)

        # Parse scores from the review
        scores = self._parse_lethal_genes_scores(review_text)

        # Calculate fitness score
        fitness_score = self._calculate_lethal_genes_fitness_score(scores)

        # Extract assessment and recommendation
        overall_assessment = self._extract_section(review_text, "Overall Assessment")
        recommendation = self._extract_section(review_text, "Recommendation")

        result = {
            "hypothesis_id": hypothesis_id,
            "fitness_score": fitness_score,
            "direct_evidence_score": scores.get("direct_evidence", 5),
            "literature_support_score": scores.get("literature_support", 5),
            "biological_plausibility_score": scores.get("biological_plausibility", 5),
            "calibration_score": scores.get("calibration", 5),
            "overall_assessment": overall_assessment,
            "recommendation": recommendation,
            "review_summary": review_text,
            "evaluation_method": "lethal_genes_reflection"
        }

        logging.info(f"Lethal genes fitness evaluation complete for {hypothesis_id}: score = {fitness_score:.2f}")

        return result

    def _evaluate_t2d_fitness(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate fitness for T2D drug target hypotheses.

        Supports both single-target and multi-gene ranked hypotheses.
        Uses druggability-aware prompts when druggability features are available.

        Args:
            hypothesis: Hypothesis dict with target_gene_masked, ranked_targets, etc.

        Returns:
            Dict with fitness score and evaluation details
        """
        from prompts import PROMPT_T2D_REFLECTION, PROMPT_T2D_REFLECTION_WITH_DRUGGABILITY

        # Get evaluation context from T2D runner
        if not hasattr(self, 't2d_runner') or self.t2d_runner is None:
            logging.warning("T2D runner not attached, using basic evaluation")
            return self._basic_fitness_evaluation(hypothesis)

        evaluation_context = self.t2d_runner.format_for_reflection_agent(hypothesis)

        # Check if druggability features are available and get them for ranked targets
        use_druggability = getattr(self, 'use_druggability', False)
        druggability_context = ""

        if use_druggability and hasattr(self.t2d_runner, 'druggability_features'):
            # Get druggability for all ranked targets
            ranked_targets = hypothesis.get('ranked_targets', [])
            if ranked_targets:
                drug_lines = ["Druggability features for ranked targets:"]
                for target in ranked_targets:
                    gene_id = target.get('gene_id')
                    drug_feat = self.t2d_runner.get_druggability_for_gene(gene_id)
                    if drug_feat:
                        drug_lines.append(
                            f"  #{target['rank']} {gene_id}: {drug_feat.get('protein_class', 'Unknown')} | "
                            f"{drug_feat.get('subcellular_location', 'Unknown')} | "
                            f"{drug_feat.get('pathway_role', 'Unknown')}"
                        )
                druggability_context = "\n".join(drug_lines)
            else:
                # Single target fallback
                primary_target = hypothesis.get('target_gene_masked')
                if primary_target:
                    drug_feat = self.t2d_runner.get_druggability_for_gene(primary_target)
                    if drug_feat:
                        druggability_context = (
                            f"Primary target {primary_target}: {drug_feat.get('protein_class', 'Unknown')} | "
                            f"{drug_feat.get('subcellular_location', 'Unknown')} | "
                            f"{drug_feat.get('pathway_role', 'Unknown')}"
                        )

        # Choose prompt based on druggability availability
        if use_druggability and druggability_context:
            prompt = PROMPT_T2D_REFLECTION_WITH_DRUGGABILITY.format(
                hypothesis=hypothesis.get('description', ''),
                evaluation_context=evaluation_context,
                druggability_for_target=druggability_context
            )
        else:
            prompt = PROMPT_T2D_REFLECTION.format(
                hypothesis=hypothesis.get('description', ''),
                evaluation_context=evaluation_context
            )

        # Get LLM evaluation
        response = llm_generate(prompt)

        # Parse scores
        import re

        def extract_score(text, field):
            pattern = rf'{field}:\s*(\d+(?:\.\d+)?)'
            match = re.search(pattern, text)
            if match:
                return min(10.0, max(1.0, float(match.group(1))))
            return 5.0

        data_score = extract_score(response, 'DATA_SUPPORT_SCORE')
        ranking_score = extract_score(response, 'RANKING_QUALITY_SCORE')
        mechanism_score = extract_score(response, 'MECHANISTIC_SCORE')
        therapeutic_score = extract_score(response, 'THERAPEUTIC_SCORE')
        novelty_score = extract_score(response, 'NOVELTY_SCORE')

        # Calculate weighted fitness based on mode
        if use_druggability and druggability_context:
            # Druggability-aware weights
            druggability_score = extract_score(response, 'DRUGGABILITY_SCORE')
            feasibility_score = extract_score(response, 'FEASIBILITY_SCORE')
            fitness = (
                data_score * 0.20 +
                ranking_score * 0.20 +
                mechanism_score * 0.15 +
                druggability_score * 0.25 +
                novelty_score * 0.10 +
                feasibility_score * 0.10
            ) * 10
        else:
            # Standard weights (without druggability)
            fitness = (
                data_score * 0.25 +
                ranking_score * 0.25 +
                mechanism_score * 0.20 +
                therapeutic_score * 0.20 +
                novelty_score * 0.10
            ) * 10

        # Extract suggestions
        suggestions_match = re.search(r'IMPROVEMENT_SUGGESTIONS:\s*(.+?)(?=\n\n|\Z)', response, re.DOTALL)
        suggestions = suggestions_match.group(1).strip() if suggestions_match else ""

        # Extract ranked targets assessment
        targets_assessment = self._parse_targets_assessment(response)
        
        # === NOVELTY BONUS ===
        # Reward predictions that explore beyond baseline/previous generations
        # Quality-weighted: only applies if data support is adequate
        novelty_bonus = self._calculate_novelty_bonus(
            predicted_genes=hypothesis.get('ranked_targets', []),
            primary_target=hypothesis.get('target_gene_masked'),
            data_support_score=data_score
        )
        
        # Apply novelty bonus (up to 20% boost)
        fitness_with_bonus = fitness * (1 + novelty_bonus)
        
        if novelty_bonus > 0:
            logging.info(f"Novelty bonus applied: {novelty_bonus*100:.1f}% -> fitness {fitness:.2f} -> {fitness_with_bonus:.2f}")

        result = {
            "hypothesis_id": hypothesis.get("id"),
            "fitness_score": round(fitness_with_bonus, 2),
            "base_fitness_score": round(fitness, 2),
            "novelty_bonus": round(novelty_bonus, 4),
            "data_support_score": data_score,
            "ranking_quality_score": ranking_score,
            "mechanism_score": mechanism_score,
            "therapeutic_score": therapeutic_score,
            "novelty_score": novelty_score,
            "improvement_suggestions": suggestions,
            "targets_assessment": targets_assessment,
            "evaluation_method": "t2d_ranked_multi_dimensional",
            "raw_response": response
        }

        # Add druggability-specific scores if available
        if use_druggability and druggability_context:
            result["druggability_score"] = extract_score(response, 'DRUGGABILITY_SCORE')
            result["feasibility_score"] = extract_score(response, 'FEASIBILITY_SCORE')

        return result

    def _parse_targets_assessment(self, response: str) -> List[Dict[str, str]]:
        """Parse RANKED_TARGETS_ASSESSMENT section from reflection response.

        Args:
            response: Raw LLM response

        Returns:
            List of target assessments with quality ratings
        """
        import re

        assessments = []

        # Find assessment lines
        # Format: "- Target #1: STRONG (Druggability: Kinase)" or "- Target #1 quality: STRONG"
        pattern = r'Target\s*#(\d+)(?:\s*quality)?:\s*(STRONG|MODERATE|WEAK)(?:\s*\(Druggability:\s*([^)]+)\))?'
        matches = re.findall(pattern, response, re.IGNORECASE)

        for match in matches:
            rank = int(match[0])
            quality = match[1].upper()
            druggability = match[2].strip() if match[2] else None

            assessment = {
                "rank": rank,
                "quality": quality
            }
            if druggability:
                assessment["druggability"] = druggability

            assessments.append(assessment)

        return assessments

    def _calculate_novelty_bonus(self, predicted_genes: List[Dict], primary_target: str = None, 
                                   data_support_score: float = 5.0) -> float:
        """
        Calculate QUALITY-WEIGHTED novelty bonus for predictions.
        
        Key design: Novelty bonus is SCALED by data support score to prevent
        rewarding random/weak predictions just because they're "different".
        
        Rewards hypotheses that:
        1. Predict genes NOT in baseline top 20 (up to 15% bonus)
        2. Predict genes NOT in previous generation's top 10 (up to 5% bonus)
        BUT only if data support is adequate (score >= 5).
        
        Args:
            predicted_genes: List of ranked target dicts with 'gene_id' keys
            primary_target: Single primary target gene ID (fallback)
            data_support_score: How well the prediction is supported by data (1-10)
            
        Returns:
            Float bonus (0.0 to 0.20 = up to 20% bonus for novel predictions)
        """
        # No bonus for poorly-supported predictions
        # This prevents rewarding random/weak genes just because they're different
        if data_support_score < 5.0:
            logging.debug(f"No novelty bonus: data support {data_support_score:.1f} < 5.0")
            return 0.0
        
        # Get predicted gene IDs
        if predicted_genes:
            predicted_set = set(g.get('gene_id') for g in predicted_genes[:5] if g.get('gene_id'))
        elif primary_target:
            predicted_set = {primary_target}
        else:
            return 0.0
        
        if not predicted_set:
            return 0.0
        
        # Quality multiplier: scales bonus based on data support
        # Score 5 -> 0.17, Score 7 -> 0.50, Score 10 -> 1.0
        quality_multiplier = min((data_support_score - 4) / 6, 1.0)
        
        raw_bonus = 0.0
        
        # Bonus 1: Novel from baseline (up to 15%)
        if self.baseline_top_genes:
            baseline_set = set(self.baseline_top_genes[:20])
            novel_from_baseline = len(predicted_set - baseline_set)
            # 3% per novel gene, max 15%
            baseline_bonus = min(novel_from_baseline * 0.03, 0.15)
            raw_bonus += baseline_bonus
            
            if baseline_bonus > 0:
                logging.debug(f"Baseline novelty: {novel_from_baseline} novel genes -> {baseline_bonus*100:.1f}% raw bonus")
        
        # Bonus 2: Novel from previous generation (up to 5%)
        if self.previous_generation_genes:
            prev_gen_set = set(self.previous_generation_genes[:10])
            novel_from_prev = len(predicted_set - prev_gen_set)
            # 1% per novel gene, max 5%
            prev_gen_bonus = min(novel_from_prev * 0.01, 0.05)
            raw_bonus += prev_gen_bonus
            
            if prev_gen_bonus > 0:
                logging.debug(f"Generation novelty: {novel_from_prev} novel genes -> {prev_gen_bonus*100:.1f}% raw bonus")
        
        # Apply quality scaling
        total_bonus = min(raw_bonus, 0.20) * quality_multiplier
        
        if total_bonus > 0 and raw_bonus > 0:
            logging.debug(f"Quality-weighted novelty: {raw_bonus*100:.1f}% raw * {quality_multiplier:.2f} quality = {total_bonus*100:.1f}% bonus")
        
        return total_bonus

    def _basic_fitness_evaluation(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Basic fitness evaluation when T2D runner is not available."""
        return {
            "hypothesis_id": hypothesis.get("id"),
            "fitness_score": 50.0,
            "data_support_score": 5.0,
            "ranking_quality_score": 5.0,
            "mechanism_score": 5.0,
            "therapeutic_score": 5.0,
            "novelty_score": 5.0,
            "improvement_suggestions": "T2D runner not available for detailed evaluation",
            "evaluation_method": "basic_fallback"
        }

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

        Uses 4 classification-focused criteria:
        1. Direct Evidence Strength
        2. Literature Support
        3. Biological Plausibility
        4. Evidence-Prediction Calibration

        Args:
            review_text: Review text containing scores

        Returns:
            Dictionary mapping score names to values
        """
        scores = {
            "direct_evidence": 5,
            "literature_support": 5,
            "biological_plausibility": 5,
            "calibration": 5
        }

        # Try to extract scores using regex patterns
        # Pattern matches "Score: 9" or "Score 9" or "**Score:** 9" (markdown formatting)
        patterns = {
            "direct_evidence": r"\*?\*?Direct\s+Evidence\s+Score:\*?\*?\s*(\d+)",
            "literature_support": r"\*?\*?Literature\s+Support\s+Score:\*?\*?\s*(\d+)",
            "biological_plausibility": r"\*?\*?Biological\s+Plausibility\s+Score:\*?\*?\s*(\d+)",
            "calibration": r"\*?\*?Evidence-Prediction\s+Calibration\s+Score:\*?\*?\s*(\d+)"
        }

        for score_name, pattern in patterns.items():
            match = re.search(pattern, review_text, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    scores[score_name] = int(match.group(1))
                except (IndexError, ValueError) as e:
                    logging.warning(f"Failed to parse {score_name} score: {e}")

        # Warn if all scores remain at default (likely parsing failure)
        if all(score == 5 for score in scores.values()):
            logging.warning("Score parsing may have failed - all scores are default value 5")
            logging.debug(f"Review text sample: {review_text[:500]}")

        return scores

    def _calculate_lethal_genes_fitness_score(self, scores: Dict[str, int]) -> float:
        """
        Calculate fitness score for lethal genes hypotheses using classification-focused criteria.

        Weights prioritize evidence strength and prediction calibration to align
        fitness with classification accuracy rather than hypothesis quality.

        Args:
            scores: Dictionary of scores (each 0-10)

        Returns:
            Overall fitness score (0-100)
        """
        # Classification-focused weights
        # High weights for: direct evidence and calibration (most important for accuracy)
        weights = {
            'direct_evidence': 2.0,          # HIGHEST - most important for classification
            'literature_support': 1.5,
            'biological_plausibility': 1.0,
            'calibration': 2.0               # HIGHEST - penalizes overconfidence
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