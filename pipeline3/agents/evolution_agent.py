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

from external_tools.gpt4o import gpt4o_generate


class EvolutionAgent:
    """
    Evolution Agent applies 4 core strategies from Pipeline2 for genetic operations.
    """
    
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
        # Keep only 4 strategies from Pipeline2
        self.evolution_strategies = {
            "inspiration": "Inspiration from existing hypotheses - creates new hypotheses inspired by single or multiple top-ranked hypotheses.",
            "combination": "Combination - directly combines the best aspects of several top-ranking hypotheses to create new hypotheses.",
            "simplification": "Simplification - simplifies hypotheses for easier verification and testing.",
            "out_of_box_thinking": "Out-of-box thinking - explores out-of-the-box ideas by moving away from a subset of hypotheses and generating divergent ones."
        }

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
        
        prompt = f"""
        You are an expert in scientific synthesis and hypothesis integration.
        Your task is to DIRECTLY COMBINE the best aspects of several top-ranking hypotheses
        to create a unified, more powerful hypothesis.
        
        Research Goal: {research_goal}
        
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
        
        The combined hypothesis should:
        - Have a clear, descriptive title reflecting the synthesis
        - Provide detailed description showing how elements are integrated
        - Explain the synergistic benefits of the combination
        - Maintain scientific rigor and testability
        
        Your response:
        """
        
        # Use the first hypothesis as the base for parsing
        base_hypothesis = top_hypotheses[0] if top_hypotheses else {}
        logging.info("===HYPOTHESIS_EVOLUTION_START===")
        raw_response = gpt4o_generate(prompt)
        logging.info("===HYPOTHESIS_EVOLUTION_END===")
        evolved_hypothesis = self._parse_evolved_response(raw_response, base_hypothesis, "combination")
        
        # Add combination metadata
        evolved_hypothesis["evolution_strategy"] = "combination"
        evolved_hypothesis["source_hypothesis_ids"] = [h.get("id") for h in top_hypotheses]
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
        
        # Use Pipeline2's exact prompt for multiple hypotheses
        prompt = f"""
        You are an expert in scientific creativity and hypothesis generation.
        Your task is to create a NEW hypothesis that is INSPIRED by the given high-quality hypothesis(es),
        but explores different aspects or mechanisms while maintaining scientific rigor.
        
        Research Goal: {research_goal}
        
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
        
        Provide your inspired hypothesis with:
        - A clear, descriptive title
        - Detailed description of the hypothesis
        - Explanation of how it's inspired by (but different from) the source(s)
        
        Your response:
        """
        
        # Use first hypothesis as base for parsing (Pipeline2 approach)
        base_hypothesis = hypotheses_list[0]
        logging.info("===HYPOTHESIS_EVOLUTION_START===")
        raw_response = gpt4o_generate(prompt)
        logging.info("===HYPOTHESIS_EVOLUTION_END===")
        evolved_hypothesis = self._parse_evolved_response(raw_response, base_hypothesis, "inspiration")
        
        # Add inspiration metadata using Pipeline2's multiple mode logic
        evolved_hypothesis["evolution_strategy"] = "inspiration"
        evolved_hypothesis["source_hypothesis_ids"] = [h.get("id") for h in hypotheses_list]
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
        
        prompt = f"""
        You are an expert in scientific communication and hypothesis refinement.
        Your task is to simplify the provided hypothesis while preserving its core scientific value.

        Research Goal: {research_goal}

        Original Hypothesis:
        Title: {hypothesis.get('title', 'Untitled')}
        Description: {hypothesis.get('description', 'No description')}

        Reviewer Feedback:
        {reflection}

        Please simplify this hypothesis by:
        1. Improving clarity and readability of the mechanistic explanation
        2. Making the causal relationships clearer and more explicit
        3. Reducing unnecessary verbosity while maintaining scientific depth
        4. Ensuring the simplified version remains testable and falsifiable
        5. Preserving the core scientific insight and novelty

        **CRITICAL REQUIREMENTS - DO NOT VIOLATE:**
        - If the hypothesis contains Primary and Rival sub-hypotheses, BOTH must be fully retained
        - All key mechanistic details must be preserved (single-loss mechanism, double-loss mechanism, intermediate components)
        - Scientific depth and rigor must be maintained - do not convert to layman's language
        - All major sections (Biological Plausibility, Clinical Relevance, mechanisms, assumptions, pathway visualization) must remain intact
        - Simplification means clarifying and improving organization, NOT removing essential scientific content
        - The output should be MORE clear but EQUALLY comprehensive

        The simplified hypothesis should improve readability while preserving all scientific content and structure.
        """
        
        # Generate the simplified hypothesis
        logging.info("===HYPOTHESIS_EVOLUTION_START===")
        raw_response = gpt4o_generate(prompt)
        logging.info("===HYPOTHESIS_EVOLUTION_END===")

        # Parse the response
        evolved_hypothesis = self._parse_evolved_response(raw_response, hypothesis, "simplification")
        
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

        # Pipeline2's out-of-box prompt logic
        # Note: Pipeline2 supports custom out_of_box_prompt, but Pipeline3 uses default
        prompt = f"""
        You are an expert in creative scientific thinking.
        Your task is to explore out-of-the-box ideas by moving away from conventional approaches
        and generating divergent hypotheses.

        Research Goal: {research_goal}

        Original Hypothesis:
        Title: {title}
        Description: {description}

        Key Strengths to Preserve:
        {' '.join(f'- {s}' for s in strengths)}

        Key Weaknesses to Address:
        {' '.join(f'- {w}' for w in weaknesses)}

        Please create a novel hypothesis that:
        1. Explores unconventional mechanisms or approaches
        2. Challenges existing paradigms while remaining scientifically grounded
        3. Addresses the research goal from a completely different angle
        4. Maintains testability and feasibility

        The out-of-box hypothesis should have a clear title and detailed description.
        """

        # Generate the evolved hypothesis
        logging.info("===HYPOTHESIS_EVOLUTION_START===")
        raw_response = gpt4o_generate(prompt)
        logging.info("===HYPOTHESIS_EVOLUTION_END===")

        # Parse the response
        evolved_hypothesis = self._parse_evolved_response(raw_response, hypothesis, "out_of_box_thinking")

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

    def _parse_evolved_response(self, raw_response: str, original_hypothesis: Dict[str, Any], 
                              strategy: str) -> Dict[str, Any]:
        """
        Parse the evolved hypothesis from the raw response with robust handling (Pipeline2 exact).
        
        Args:
            raw_response: The raw response from the LLM
            original_hypothesis: The original hypothesis being evolved
            strategy: The evolution strategy used
            
        Returns:
            Dictionary with the parsed evolved hypothesis
        """
        # Improved robust parsing logic
        lines = raw_response.strip().split('\n')
        
        # Extract title with multiple patterns
        title = None
        title_pattern_found = False
        
        # First look for explicit title markers
        for i, line in enumerate(lines[:15]):  # Look in first 15 lines
            if line.lower().startswith("title:"):
                title = line.split(":", 1)[1].strip()
                title_pattern_found = True
                break
            elif "title:" in line.lower():
                parts = line.lower().split("title:")
                if len(parts) > 1:
                    # Get the text after "title:" until the end of the line
                    title = line[line.lower().index("title:") + 6:].strip()
                    title_pattern_found = True
                    break
            elif line.startswith("# "):
                title = line[2:].strip()
                title_pattern_found = True
                break
        
        # If no explicit title pattern found, use heuristics
        if not title:
            if len(lines) > 0:
                # Use first non-empty line if it's relatively short
                first_non_empty = next((line for line in lines if line.strip()), "")
                if first_non_empty and len(first_non_empty) < 100:
                    title = first_non_empty.strip()
                else:
                    # Generate a title based on the strategy and original title
                    original_title = original_hypothesis.get("title", "Untitled Hypothesis")
                    title = f"Evolved {original_title} ({strategy.replace('_', ' ').title()})"
        
        # Clean up title if needed
        if title and len(title) > 150:
            title = title[:147] + "..."
        
        # Extract description
        description = raw_response.strip()
        
        # If we found a title pattern, remove it from the description
        if title_pattern_found and title:
            # Try different approaches to remove the title from description
            if description.lower().startswith("title:"):
                # Remove the first line
                description_lines = description.split("\n")
                description = "\n".join(description_lines[1:]).strip()
            elif "title:" in description.lower():
                # Try to remove just the title part
                title_index = description.lower().find("title:")
                title_line_end = description.find("\n", title_index)
                if title_line_end > 0:
                    description = description[:title_index] + description[title_line_end:].strip()
        
        # In case the entire response is just the title
        if description == title:
            description = f"Evolved hypothesis using {strategy} strategy. Further details needed."
        
        # Generate a unique ID for the evolved hypothesis
        base_id = original_hypothesis.get("id", "hyp")
        suffix = strategy[:3]
        evolved_id = f"{base_id}-{suffix}-{str(uuid.uuid4())[:6]}"
        
        # Calculate a new ELO score with a small improvement
        base_elo = original_hypothesis.get("elo_score", 1000)
        elo_boost = random.randint(20, 50)  # Small improvement
        
        return {
            "id": evolved_id,
            "title": title,
            "description": description,
            "testability_notes": original_hypothesis.get("testability_notes", ""),
            "elo_score": base_elo + elo_boost,
            "origin": "evolution",
            "parent_id": original_hypothesis.get("id"),
            "evolution_strategy": strategy,
            "evolution_timestamp": time.time(),
            "reviews": [],
            "cluster_id": original_hypothesis.get("cluster_id"),
            # Preserve gene pair identity for batch mode (lethal_genes)
            "gene_a": original_hypothesis.get("gene_a"),
            "gene_b": original_hypothesis.get("gene_b")
        }

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