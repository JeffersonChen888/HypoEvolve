"""
Simplified Supervisor Agent for Pipeline3 - Genetic Algorithm Implementation

This agent orchestrates the genetic algorithm workflow by coordinating
the simplified agents in a streamlined evolutionary process.

Simplified Workflow:
1. Generate initial hypothesis population
2. Evaluate fitness of each hypothesis  
3. Select best performers through tournament selection
4. Create offspring through crossover and mutation
5. Replace population and repeat

Removed from Pipeline2:
- Complex task management system
- Meta-review agent coordination
- Proximity agent coordination
- Multi-stage review orchestration
- Complex callback systems
- Parallel task processing
"""

import logging
import time
from typing import Dict, List, Any, Optional

from agents.generation_agent import GenerationAgent
from agents.reflection_agent import ReflectionAgent
from agents.evolution_agent import EvolutionAgent
from agents.tournament_agent import TournamentAgent  # Used in lethal_genes_tournament mode only
from config import DETAILED_STATE_LOGGING


class SupervisorAgent:
    """
    Simplified Supervisor Agent that orchestrates a genetic algorithm
    for scientific hypothesis evolution.
    """
    
    def __init__(self, research_goal: str, mode: str = "drug-repurposing", batch_mode: bool = False, run_folder: str = None, prompt_file: str = None):
        """
        Initialize the simplified supervisor agent.

        Args:
            research_goal: The scientific research goal
            mode: Pipeline mode ("drug-repurposing", "general", "lethal_genes", or "lethal_genes_2")
            batch_mode: If True, process multiple gene pairs from config (lethal_genes only)
            run_folder: Folder path for this specific run (for per-pair logs in batch mode)
            prompt_file: Path to complete prompt file (lethal_genes_2 only)
        """
        self.research_goal = research_goal
        self.mode = mode
        self.batch_mode = batch_mode
        self.run_folder = run_folder
        self.prompt_file = prompt_file

        # Initialize agents
        self.generation_agent = GenerationAgent(mode=mode, run_folder=run_folder)
        self.reflection_agent = ReflectionAgent(mode=mode)
        self.evolution_agent = EvolutionAgent(mode=mode)
        
        # Import config for default values
        import config
        
        # Genetic algorithm parameters (use config defaults, can be overridden later)
        self.population_size = config.POPULATION_SIZE
        self.num_generations = config.NUM_GENERATIONS
        self.selection_ratio = config.SELECTION_RATIO
        self.elitism_count = config.ELITISM_COUNT

        # Ensure elitism count doesn't exceed population size (leave room for at least 1 offspring)
        if self.elitism_count >= self.population_size:
            self.elitism_count = max(1, self.population_size - 1)
            logging.warning(f"Elitism count adjusted to {self.elitism_count} (was {config.ELITISM_COUNT}) to allow offspring generation with population size {self.population_size}")

        # State tracking
        self.current_generation = 0
        self.population = []
        self.fitness_evaluations = []
        self.generation_history = []

        # Evolutionary lineage tracking (for lethal_genes mode complete history)
        self.all_hypotheses = []  # Complete list of ALL hypotheses across generations
        self.generation_metadata = []  # Per-generation statistics

        # T2D mode support
        self.t2d_runner = None
        
        # Novelty scoring: track baseline predictions for diversity bonus
        self.baseline_predictions = []

        logging.info(f"Initialized simplified Supervisor Agent for genetic algorithm")
        logging.info(f"Research goal: {research_goal}")
        logging.info(f"Mode: {mode}")
    
    def set_baseline_predictions(self, baseline_predictions: list) -> None:
        """
        Cache baseline predictions for novelty scoring.
        
        This allows the reflection agent to give bonus scores to hypotheses
        that explore genes NOT in the baseline top predictions.
        
        Args:
            baseline_predictions: List of baseline prediction dicts with 'gene_id' keys
        """
        self.baseline_predictions = baseline_predictions
        
        # Extract gene IDs and pass to reflection agent
        # Note: baseline LLM returns 'target_gene_masked', not 'gene_id'
        baseline_gene_ids = [p.get('target_gene_masked') for p in baseline_predictions if p.get('target_gene_masked')]
        self.reflection_agent.baseline_top_genes = baseline_gene_ids
        
        logging.info(f"Set {len(baseline_gene_ids)} baseline genes for novelty scoring: {baseline_gene_ids[:5]}...")
    
    def run_genetic_algorithm(self) -> Dict[str, Any]:
        """
        Run the complete genetic algorithm process.
        
        Returns:
            Dictionary containing final results and statistics
        """
        logging.info("=" * 80)
        logging.info("STARTING GENETIC ALGORITHM FOR SCIENTIFIC HYPOTHESIS EVOLUTION")
        logging.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Step 1: Initialize population
            self._initialize_population()
            
            # Step 2: Run evolutionary generations
            for generation in range(self.num_generations):
                self.current_generation = generation + 1
                self._run_generation()
            
            # Step 3: Final evaluation and results
            final_results = self._compile_final_results()
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            logging.info("=" * 80)
            logging.info(f"GENETIC ALGORITHM COMPLETED in {execution_time:.1f} seconds")
            logging.info(f"Final population size: {len(self.population)}")
            logging.info(f"Generations completed: {self.current_generation}")
            if DETAILED_STATE_LOGGING:
                logging.info(f"Final algorithm state: {self._get_detailed_state_summary()}")
            logging.info("=" * 80)
            
            return final_results
            
        except Exception as e:
            logging.error(f"Error in genetic algorithm execution: {e}")
            return {"error": str(e), "partial_results": self._get_current_state()}

    def run_tournament_mode(self, gene_pairs: List[tuple], generation_prompt: str,
                           judging_criteria: str = None,
                           population_size: int = 3,
                           num_generations: int = 2,
                           top_k_per_pair: int = 3) -> Dict[str, Any]:
        """
        Run the two-phase lethal genes tournament pipeline.

        Phase 1: Process each gene pair sequentially through genetic algorithm evolution
        Phase 2: Pool top hypotheses from all pairs and run Swiss-style cross-pair tournament

        Args:
            gene_pairs: List of (gene_a, gene_b) tuples
            generation_prompt: Prompt template with [GENE_A] and [GENE_B] placeholders
            judging_criteria: Custom judging criteria for tournament (optional)
            population_size: Number of hypotheses per pair (default: 3)
            num_generations: Number of GA generations per pair (default: 2)
            top_k_per_pair: Number of top hypotheses to take from each pair (default: 3)

        Returns:
            Dictionary containing:
                - phase1_results: Per-pair evolution results
                - tournament_results: Swiss tournament results
                - final_rankings: Unified ranking across all gene pairs
        """
        import random

        logging.info("=" * 80)
        logging.info("STARTING LETHAL GENES TOURNAMENT MODE")
        logging.info(f"Gene pairs: {len(gene_pairs)}")
        logging.info(f"Population per pair: {population_size}")
        logging.info(f"Generations per pair: {num_generations}")
        logging.info(f"Top K per pair: {top_k_per_pair}")
        logging.info("=" * 80)

        start_time = time.time()
        all_top_hypotheses = []
        phase1_results = []

        # ==================== PHASE 1: PER-PAIR EVOLUTION ====================
        logging.info("=" * 60)
        logging.info("PHASE 1: PER-PAIR GENETIC ALGORITHM EVOLUTION")
        logging.info("=" * 60)

        for i, gene_pair in enumerate(gene_pairs):
            gene_a, gene_b = gene_pair
            pair_name = f"{gene_a}_{gene_b}"

            logging.info(f"\n{'='*40}")
            logging.info(f"Processing pair {i+1}/{len(gene_pairs)}: {gene_a} - {gene_b}")
            logging.info(f"{'='*40}")

            try:
                # Step 1: Generate initial population for this pair
                logging.info(f"Generating {population_size} initial hypotheses...")
                population = self.generation_agent.generate_from_tournament_prompt(
                    gene_pair=gene_pair,
                    prompt_template=generation_prompt,
                    population_size=population_size
                )
                logging.info(f"Generated {len(population)} hypotheses")

                # Step 2: Evaluate initial fitness
                logging.info("Evaluating initial fitness...")
                fitness_results = self.reflection_agent.evaluate_population_fitness(population)

                # Update population with fitness scores
                fitness_lookup = {r["hypothesis_id"]: r["fitness_score"] for r in fitness_results}
                for hyp in population:
                    hyp["fitness_score"] = fitness_lookup.get(hyp["id"], 0)

                # Log initial fitness
                for hyp in population:
                    logging.info(f"  Initial: {hyp.get('id', 'unknown')[:20]} - fitness: {hyp.get('fitness_score', 0)}")

                # Step 3: Evolve for N generations
                for gen in range(1, num_generations + 1):
                    logging.info(f"\n--- Generation {gen}/{num_generations} ---")

                    # Calculate offspring size (preserve at least 1 elite)
                    elitism_count = max(1, population_size // 3)
                    offspring_size = population_size - elitism_count

                    # Create offspring using tournament selection within this pair's population
                    offspring = []
                    skipped_count = 0
                    for j in range(offspring_size):
                        # 60% crossover, 40% mutation
                        if random.random() < 0.6 and len(population) >= 2:
                            # Crossover - ensure parent1 != parent2
                            parent1 = self._tournament_select(population, tournament_size=2)
                            # Select parent2 from remaining population (excluding parent1)
                            remaining = [h for h in population if h.get("id") != parent1.get("id")]
                            if not remaining:
                                logging.warning(f"Not enough unique parents for crossover, skipping offspring {j+1}")
                                skipped_count += 1
                                continue
                            parent2 = self._tournament_select(remaining, tournament_size=min(2, len(remaining)))
                            child = self.evolution_agent.perform_crossover(
                                parent1, parent2, f"synthetic lethality for {gene_a} and {gene_b}", ""
                            )
                            if child is None:
                                logging.warning(f"Crossover failed for {pair_name} (gene validation failed), skipping offspring {j+1}")
                                skipped_count += 1
                                continue
                            # Preserve gene pair metadata
                            child["gene_a"] = gene_a
                            child["gene_b"] = gene_b
                        else:
                            # Mutation
                            parent = self._tournament_select(population, tournament_size=2)
                            child = self.evolution_agent.perform_mutation(
                                parent, f"synthetic lethality for {gene_a} and {gene_b}", ""
                            )
                            if child is None:
                                logging.warning(f"Mutation failed for {pair_name} (gene validation failed), skipping offspring {j+1}")
                                skipped_count += 1
                                continue
                            # Preserve gene pair metadata
                            child["gene_a"] = gene_a
                            child["gene_b"] = gene_b

                        child["elo_score"] = 1200  # Default ELO for tournament
                        offspring.append(child)

                    if skipped_count > 0:
                        logging.warning(f"Generation {gen}: {skipped_count}/{offspring_size} offspring skipped due to evolution failures")

                    # Evaluate offspring fitness
                    if offspring:
                        logging.info(f"Evaluating {len(offspring)} offspring...")
                        offspring_fitness = self.reflection_agent.evaluate_population_fitness(offspring)
                        offspring_lookup = {r["hypothesis_id"]: r["fitness_score"] for r in offspring_fitness}
                        for child in offspring:
                            child["fitness_score"] = offspring_lookup.get(child["id"], 0)

                    # Replace population (elitism + offspring)
                    elite = sorted(population, key=lambda x: x.get("fitness_score", 0), reverse=True)[:elitism_count]
                    population = elite + offspring

                    # Log generation results
                    fitnesses = [h.get("fitness_score", 0) for h in population]
                    logging.info(f"Generation {gen} complete: mean={sum(fitnesses)/len(fitnesses):.2f}, max={max(fitnesses):.2f}")

                # Step 4: Take top K from this pair
                population_sorted = sorted(population, key=lambda x: x.get("fitness_score", 0), reverse=True)
                top_k = population_sorted[:top_k_per_pair]

                logging.info(f"\nTop {top_k_per_pair} hypotheses for {pair_name}:")
                for rank, hyp in enumerate(top_k, 1):
                    logging.info(f"  {rank}. {hyp.get('id', 'unknown')[:30]} - fitness: {hyp.get('fitness_score', 0):.2f}")

                all_top_hypotheses.extend(top_k)

                # Store phase 1 results for this pair
                phase1_results.append({
                    "gene_pair": pair_name,
                    "gene_a": gene_a,
                    "gene_b": gene_b,
                    "top_hypotheses": top_k,
                    "final_population_size": len(population),
                    "generations_completed": num_generations
                })

            except Exception as e:
                logging.error(f"Error processing pair {pair_name}: {e}")
                import traceback
                traceback.print_exc()
                phase1_results.append({
                    "gene_pair": pair_name,
                    "gene_a": gene_a,
                    "gene_b": gene_b,
                    "error": str(e),
                    "top_hypotheses": [],
                    "failed": True  # Explicit failure flag
                })

        # Calculate Phase 1 statistics including failures
        successful_pairs = [p for p in phase1_results if not p.get("failed", False)]
        failed_pairs = [p for p in phase1_results if p.get("failed", False)]

        logging.info(f"\n{'='*60}")
        logging.info(f"PHASE 1 COMPLETE:")
        logging.info(f"  - Successful pairs: {len(successful_pairs)}/{len(gene_pairs)}")
        logging.info(f"  - Failed pairs: {len(failed_pairs)}/{len(gene_pairs)}")
        logging.info(f"  - Hypotheses pooled: {len(all_top_hypotheses)}")
        if failed_pairs:
            logging.warning(f"  - FAILED PAIRS: {[p['gene_pair'] for p in failed_pairs]}")
        logging.info(f"{'='*60}")

        # ==================== PHASE 2: SWISS-STYLE TOURNAMENT ====================
        logging.info("\n" + "=" * 60)
        logging.info("PHASE 2: CROSS-PAIR SWISS-STYLE TOURNAMENT")
        logging.info("=" * 60)

        if len(all_top_hypotheses) < 2:
            logging.warning("Not enough hypotheses for tournament")
            tournament_results = {
                "rankings": all_top_hypotheses,
                "match_history": [],
                "rounds_completed": 0,
                "stability_achieved": False,  # NOT stable - tournament couldn't run
                "skipped": True,  # Flag indicating tournament was skipped
                "skip_reason": f"Only {len(all_top_hypotheses)} hypotheses available (minimum 2 required)"
            }
        else:
            tournament_agent = TournamentAgent(judging_criteria=judging_criteria)
            tournament_results = tournament_agent.run_swiss_tournament(all_top_hypotheses)

        # Compile final results
        end_time = time.time()
        execution_time = end_time - start_time

        final_results = {
            "mode": "lethal_genes_tournament",
            "execution_time_seconds": round(execution_time, 2),
            "parameters": {
                "gene_pairs_count": len(gene_pairs),
                "population_per_pair": population_size,
                "generations_per_pair": num_generations,
                "top_k_per_pair": top_k_per_pair,
                "total_hypotheses_pooled": len(all_top_hypotheses)
            },
            "phase1_summary": {
                "successful_pairs": len(successful_pairs),
                "failed_pairs": len(failed_pairs),
                "failed_pair_names": [p["gene_pair"] for p in failed_pairs],
                "total_hypotheses": len(all_top_hypotheses)
            },
            "phase1_results": phase1_results,
            "tournament_results": {
                "rounds_completed": tournament_results.get("rounds_completed", 0),
                "stability_achieved": tournament_results.get("stability_achieved", False),
                "total_matches": len(tournament_results.get("match_history", [])),
                "skipped": tournament_results.get("skipped", False),
                "skip_reason": tournament_results.get("skip_reason", None),
                "draw_count": sum(1 for m in tournament_results.get("match_history", []) if m.get("is_draw", False))
            },
            "final_rankings": tournament_results.get("rankings", []),
            "match_history": tournament_results.get("match_history", [])
        }

        logging.info("=" * 80)
        logging.info(f"LETHAL GENES TOURNAMENT COMPLETED in {execution_time:.1f} seconds")
        logging.info(f"Phase 1: {len(successful_pairs)}/{len(gene_pairs)} pairs successful")
        if failed_pairs:
            logging.warning(f"Phase 1 FAILURES: {len(failed_pairs)} pairs failed - {[p['gene_pair'] for p in failed_pairs]}")
        if tournament_results.get("skipped"):
            logging.warning(f"Tournament SKIPPED: {tournament_results.get('skip_reason')}")
        else:
            logging.info(f"Tournament rounds: {tournament_results.get('rounds_completed', 0)}")
            logging.info(f"Ranking stability achieved: {tournament_results.get('stability_achieved', False)}")
            logging.info(f"Total matches played: {len(tournament_results.get('match_history', []))}")
            draw_count = sum(1 for m in tournament_results.get("match_history", []) if m.get("is_draw", False))
            if draw_count > 0:
                logging.warning(f"Draws (comparison failures): {draw_count}")
        logging.info("=" * 80)

        # Log top 10 final rankings
        if final_results["final_rankings"]:
            logging.info("\nTOP 10 FINAL RANKINGS (Cross-Pair):")
            for rank, hyp in enumerate(final_results["final_rankings"][:10], 1):
                gene_pair = f"{hyp.get('gene_a', '?')}-{hyp.get('gene_b', '?')}"
                elo = hyp.get("final_elo", hyp.get("elo_score", 1200))
                logging.info(f"  {rank}. [{gene_pair}] ELO: {elo}")
        else:
            logging.warning("No rankings available - all gene pairs failed")

        return final_results

    def _initialize_population(self) -> None:
        """
        Initialize the starting population of hypotheses.
        """
        # Handle lethal_genes_2 mode: generate from complete prompt file
        if self.mode == "lethal_genes_2" and self.prompt_file:
            from utils.prompt_loader import get_prompt_info

            logging.info(f"Lethal_genes_2 mode: Loading complete prompt from file: {self.prompt_file}")
            gene_pair_name, complete_prompt = get_prompt_info(self.prompt_file)
            logging.info(f"Gene pair: {gene_pair_name}")
            logging.info(f"Prompt length: {len(complete_prompt)} characters")
            logging.info(f"Generating population of {self.population_size} hypotheses from complete prompt")

            generation_result = self.generation_agent.generate_from_complete_prompt(
                complete_prompt=complete_prompt,
                population_size=self.population_size,
                gene_pair_name=gene_pair_name
            )

        # Handle batch mode for lethal genes
        elif self.batch_mode and self.mode == "lethal_genes":
            from gene_pairs_config import NUM_ACTIVE_PAIRS

            # Override population size to match number of gene pairs
            self.population_size = NUM_ACTIVE_PAIRS
            # Adjust elitism if needed
            if self.elitism_count >= self.population_size:
                self.elitism_count = max(1, self.population_size - 1)
                logging.warning(f"Batch mode: Elitism adjusted to {self.elitism_count} for population size {self.population_size}")

            logging.info(f"Batch mode: Initializing population of {self.population_size} hypotheses (one per gene pair)")
            logging.info("Starting batch hypothesis generation with literature exploration")

            generation_result = self.generation_agent.generate_batch_lethal_genes(num_papers=3)
        elif self.mode == "t2d-target":
            logging.info("T2D mode: Using pre-computed analysis pipeline")

            if self.t2d_runner is None:
                raise ValueError("T2D runner must be attached before running. Set supervisor.t2d_runner = runner")

            # Attach runner to sub-agents
            self.generation_agent.t2d_runner = self.t2d_runner
            self.reflection_agent.t2d_runner = self.t2d_runner
            self.evolution_agent.t2d_runner = self.t2d_runner

            # Pass druggability mode flag (Option A vs Option C)
            use_druggability = getattr(self, 'use_druggability', False)
            self.generation_agent.use_druggability = use_druggability
            self.reflection_agent.use_druggability = use_druggability
            self.evolution_agent.use_druggability = use_druggability
            logging.info(f"T2D mode: {'Option A (druggability)' if use_druggability else 'Option C (expression-only)'}")

            # Generate initial population
            generation_result = self.generation_agent.generate_initial_population(
                research_goal=self.research_goal,
                population_size=self.population_size,
                num_papers=0  # Not used in T2D mode
            )
            # Note: Don't return early - let the code flow through to set self.population
            
        else:
            # Normal single gene pair mode
            logging.info(f"Initializing population of {self.population_size} hypotheses")
            logging.info(f"Research goal: {self.research_goal[:100]}...")
            logging.info("Starting initial hypothesis generation with literature exploration")

            generation_result = self.generation_agent.generate_initial_population(
                research_goal=self.research_goal,
                population_size=self.population_size,
                num_papers=3
            )

        self.population = generation_result.get("hypotheses", [])

        logging.info(f"Generated initial population: {len(self.population)} hypotheses")

        # Add lineage tracking fields to initial hypotheses (generation 0)
        for hyp in self.population:
            hyp["generation"] = 0
            hyp["parent_ids"] = []  # No parents for initial generation
            hyp["evolution_strategy"] = "initial_generation"
            hyp["is_elite"] = False  # Initial generation not elite yet

        # Store initial hypotheses in complete history
        self.all_hypotheses.extend(self.population)
        logging.info(f"Stored {len(self.population)} generation 0 hypotheses in evolutionary history")

        # Log details about generated hypotheses
        for i, hyp in enumerate(self.population):
            title = hyp.get('title', 'No title')[:50]
            logging.info(f"  Hypothesis {i+1}: {title}...")

        # Log literature exploration results
        search_queries = generation_result.get("search_queries", [])
        logging.info(f"Literature exploration used {len(search_queries)} search queries: {search_queries}")
        if DETAILED_STATE_LOGGING:
            logging.info(f"Initial population state: {self._get_detailed_state_summary()}")

        # Evaluate initial fitness
        self._evaluate_population_fitness()
        
        # Record initial generation
        self._record_generation_stats()
    
    def _run_generation(self) -> None:
        """
        Run a single generation of the genetic algorithm.
        """
        logging.info(f"Running generation {self.current_generation}")
        
        # GA-aligned approach: per-offspring tournament selection (sample 2, pick fitter)
        
        # Step 1 & 2 Combined: GA-aligned offspring creation with tournament selection
        offspring = self._create_offspring_ga_aligned()
        
        # Step 3: Evaluation - assess offspring fitness
        offspring_evaluations = self._evaluate_offspring_fitness(offspring)
        
        # Step 4: Replacement - form next generation (no pre-selected parents in GA approach)
        self._replace_population(None, offspring, offspring_evaluations)
        
        # Step 5: Record generation statistics
        self._record_generation_stats()
        
        logging.info(f"Generation {self.current_generation} complete")
        if DETAILED_STATE_LOGGING:
            logging.info(f"Generation {self.current_generation} final state: {self._get_detailed_state_summary()}")
    
    def _evaluate_population_fitness(self) -> None:
        """
        Evaluate fitness for the current population.
        """
        logging.info(f"Evaluating fitness for {len(self.population)} hypotheses")

        # Log each hypothesis before fitness evaluation
        for i, hyp in enumerate(self.population):
            title = hyp.get('title', 'No title')[:50]
            existing_fitness = hyp.get('fitness', 'No fitness')
            logging.info(f"  Evaluating Hypothesis {i+1}: {title}... (current fitness: {existing_fitness})")

        self.fitness_evaluations = self.reflection_agent.evaluate_population_fitness(
            self.population
        )

        # Log fitness results
        logging.info(f"Fitness evaluation completed for {len(self.fitness_evaluations)} hypotheses")
        for eval_result in self.fitness_evaluations:
            hyp_id = eval_result.get('hypothesis_id', 'Unknown')
            fitness = eval_result.get('fitness_score', 'Unknown')
            logging.info(f"  Hypothesis {hyp_id}: fitness = {fitness}")
        
        # Update hypothesis fitness scores
        fitness_lookup = {eval_result["hypothesis_id"]: eval_result["fitness_score"] 
                         for eval_result in self.fitness_evaluations}
        
        for hypothesis in self.population:
            hyp_id = hypothesis.get("id")
            if hyp_id in fitness_lookup:
                hypothesis["fitness_score"] = fitness_lookup[hyp_id]
        
        logging.info("Population fitness evaluation complete")
        if DETAILED_STATE_LOGGING:
            logging.info(f"Post-fitness evaluation state: {self._get_detailed_state_summary()}")
    
    def _select_parents(self) -> List[Dict[str, Any]]:
        """
        Select parents using integrated tournament + fitness scoring.
        
        Returns:
            List of selected parent hypotheses
        """
        num_parents = max(2, int(len(self.population) * self.selection_ratio))
        
        # GA-aligned approach: Use only reflection fitness scores for selection
        # (Tournament functions deprecated but kept for reference)
        self.population.sort(key=lambda x: x.get("fitness_score", 0), reverse=True)
        parents = self.population[:num_parents]
        
        logging.info(f"Selected {len(parents)} parents using fitness-based selection (GA-aligned)")
        
        return parents
    
    def _create_offspring_ga_aligned(self) -> List[Dict[str, Any]]:
        """
        Create offspring using GA-aligned per-offspring tournament selection.
        Each offspring creation independently selects parents from entire population.

        In batch mode (lethal_genes with multiple gene pairs):
        - Crossover is DISABLED to preserve gene pair identity
        - Only mutation is used to refine hypotheses for each pair
        - This prevents mixing mechanisms from different gene pairs
        """
        import random

        offspring_size = len(self.population) - self.elitism_count

        # Batch mode: mutation only (no crossover to preserve gene pair identity)
        if self.batch_mode:
            crossover_rate = 0.0
            logging.info(f"Batch mode: Creating {offspring_size} offspring using MUTATION ONLY (preserving gene pair identity)")
        else:
            crossover_rate = 0.6  # 60% crossover, 40% mutation
            logging.info(f"Creating {offspring_size} offspring using GA-aligned tournament selection")
            logging.info(f"Genetic operation rates: {crossover_rate*100:.0f}% crossover, {(1-crossover_rate)*100:.0f}% mutation")

        offspring = []

        for i in range(offspring_size):
            if random.random() < crossover_rate:
                # Crossover: Tournament select 2 parents independently
                parent1 = self._tournament_select(self.population, tournament_size=2)
                parent2 = self._tournament_select(self.population, tournament_size=2)

                logging.info(f"Crossover {i+1}/{offspring_size}: parents '{parent1.get('title', 'Unknown')[:30]}...' x '{parent2.get('title', 'Unknown')[:30]}...'")

                child = self.evolution_agent.perform_crossover(
                    parent1, parent2, self.research_goal, ""
                )

                # Handle failed crossover (operator returned None)
                if child is None:
                    logging.warning(f"Crossover {i+1} returned None, skipping this offspring")
                    continue

                # Add lineage tracking to offspring
                child["generation"] = self.current_generation
                child["parent_ids"] = [parent1.get("id"), parent2.get("id")]
                # evolution_strategy already set by perform_crossover

                child_title = child.get('title', 'Unknown')[:40]
                logging.info(f"  -> Created offspring: '{child_title}...'")
                logging.debug(f"Crossover offspring {i+1}: parents {parent1.get('id', 'unknown')} x {parent2.get('id', 'unknown')}")
            else:
                # Mutation: Tournament select 1 parent
                parent = self._tournament_select(self.population, tournament_size=2)

                logging.info(f"Mutation {i+1}/{offspring_size}: parent '{parent.get('title', 'Unknown')[:30]}...'")

                child = self.evolution_agent.perform_mutation(
                    parent, self.research_goal, ""
                )

                # Handle failed mutation (operator returned None)
                if child is None:
                    logging.warning(f"Mutation {i+1} returned None, skipping this offspring")
                    continue

                # Add lineage tracking to offspring
                child["generation"] = self.current_generation
                child["parent_ids"] = [parent.get("id")]
                # evolution_strategy already set by perform_mutation

                child_title = child.get('title', 'Unknown')[:40]
                logging.info(f"  -> Created mutated offspring: '{child_title}...'")
                logging.debug(f"Mutation offspring {i+1}: parent {parent.get('id', 'unknown')}")

            offspring.append(child)

            # Store offspring in complete history
            self.all_hypotheses.append(child)
        
        logging.info(f"Successfully created {len(offspring)} offspring using GA selection")
        return offspring
    
    def _tournament_select(self, population: List[Dict[str, Any]], tournament_size: int = 2) -> Dict[str, Any]:
        """
        Local tournament selection for GA-aligned parent selection.
        No API calls needed - just local fitness comparison.
        
        Args:
            population: Population to select from
            tournament_size: Size of tournament (default: 2)
            
        Returns:
            Selected individual (winner of tournament)
        """
        import random
        
        # Randomly sample tournament_size individuals
        tournament = random.sample(population, min(tournament_size, len(population)))
        
        # Return the individual with highest fitness score
        winner = max(tournament, key=lambda x: x.get('fitness_score', 0))
        
        return winner
        
    def _calculate_combined_fitness(self, tournament_results: Dict[str, Any]) -> None:
        """DEPRECATED: Combined fitness calculation no longer used in GA-aligned approach."""
        """
        Calculate combined fitness from reflection scores and tournament Elo scores.
        
        Args:
            tournament_results: Results from tournament with updated Elo scores
        """
        logging.info("Calculating combined fitness scores")
        
        # Create lookup for updated Elo scores
        elo_lookup = {}
        for ranking in tournament_results.get("rankings", []):
            hyp_id = ranking["hypothesis"].get("id")
            elo_score = ranking.get("elo_score", 1200)
            elo_lookup[hyp_id] = elo_score
        
        # Calculate combined fitness for each hypothesis
        for hypothesis in self.population:
            hyp_id = hypothesis.get("id")
            
            # Get reflection fitness score (0-100)
            reflection_score = hypothesis.get("fitness_score", 50)
            
            # Get tournament Elo score and normalize to 0-100
            elo_score = elo_lookup.get(hyp_id, 1200)
            normalized_elo = ((elo_score - 1000) / 400) * 100
            normalized_elo = max(0, min(100, normalized_elo))  # Clamp to 0-100
            
            # Combine scores: 40% reflection quality, 60% tournament performance
            combined_fitness = (reflection_score * 0.4) + (normalized_elo * 0.6)
            
            # Update hypothesis with combined fitness
            hypothesis["combined_fitness"] = combined_fitness
            hypothesis["tournament_elo"] = elo_score
            hypothesis["normalized_elo"] = normalized_elo
        
        logging.info("Combined fitness calculation complete")
    
    def _create_offspring(self, parents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create offspring from selected parents.
        
        Args:
            parents: Selected parent hypotheses
            
        Returns:
            List of offspring hypotheses
        """
        offspring_size = len(self.population) - self.elitism_count
        
        offspring = self.evolution_agent.create_offspring(
            parents=parents,
            research_goal=self.research_goal,
            offspring_size=offspring_size
        )
        
        logging.info(f"Created {len(offspring)} offspring")
        
        return offspring
    
    def _evaluate_offspring_fitness(self, offspring: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate fitness for offspring hypotheses.
        
        Args:
            offspring: List of offspring hypotheses
            
        Returns:
            List of fitness evaluations for offspring
        """
        logging.info(f"Evaluating fitness for {len(offspring)} offspring")
        
        offspring_evaluations = self.reflection_agent.evaluate_population_fitness(offspring)
        
        # Update offspring fitness scores
        fitness_lookup = {eval_result["hypothesis_id"]: eval_result["fitness_score"] 
                         for eval_result in offspring_evaluations}
        
        for child in offspring:
            child_id = child.get("id")
            if child_id in fitness_lookup:
                child["fitness_score"] = fitness_lookup[child_id]
        
        return offspring_evaluations
    
    def _replace_population(self, parents: List[Dict[str, Any]], 
                           offspring: List[Dict[str, Any]],
                           offspring_evaluations: List[Dict[str, Any]]) -> None:
        """
        Replace current population with new generation.
        
        Args:
            parents: Parent hypotheses (None in GA-aligned approach)
            offspring: Offspring hypotheses
            offspring_evaluations: Offspring fitness evaluations
        """
        # Implement elitism - preserve top performers
        elite_hypotheses = sorted(self.population,
                                key=lambda x: x.get("fitness_score", 0),
                                reverse=True)[:self.elitism_count]

        # Mark elite hypotheses for this generation
        elite_ids = {h.get("id") for h in elite_hypotheses}
        for hyp in elite_hypotheses:
            hyp["is_elite"] = True
            hyp["elite_generation"] = self.current_generation

        # Mark offspring as non-elite
        for hyp in offspring:
            hyp["is_elite"] = False

        # Combine offspring with elite hypotheses
        new_population = elite_hypotheses + offspring
        
        # If we have too many, select the best
        if len(new_population) > self.population_size:
            new_population = sorted(new_population,
                                  key=lambda x: x.get("fitness_score", 0),
                                  reverse=True)[:self.population_size]
        
        self.population = new_population
        
        # Update fitness evaluations
        all_evaluations = self.fitness_evaluations + offspring_evaluations
        self.fitness_evaluations = [eval_result for eval_result in all_evaluations
                                  if any(h.get("id") == eval_result["hypothesis_id"] 
                                        for h in self.population)]
        
        logging.info(f"Population replaced: {len(self.population)} hypotheses in new generation")
        if DETAILED_STATE_LOGGING:
            logging.info(f"New generation state: {self._get_detailed_state_summary()}")
    
    def _record_generation_stats(self) -> None:
        """
        Record statistics for the current generation.
        """
        fitness_scores = [h.get("fitness_score", 0) for h in self.population]
        
        generation_stats = {
            "generation": self.current_generation,
            "population_size": len(self.population),
            "mean_fitness": sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0,
            "max_fitness": max(fitness_scores) if fitness_scores else 0,
            "min_fitness": min(fitness_scores) if fitness_scores else 0,
            "fitness_std": self._calculate_std(fitness_scores),
            "best_hypothesis": max(self.population, key=lambda x: x.get("fitness_score", 0)) if self.population else None
        }
        
        # Round numerical values
        for key in ["mean_fitness", "max_fitness", "min_fitness", "fitness_std"]:
            generation_stats[key] = round(generation_stats[key], 2)
        
        self.generation_history.append(generation_stats)
        
        # === NOVELTY SCORING: Track top genes for next generation ===
        # Extract target genes from top hypotheses for novelty comparison
        if self.mode == "t2d-target":
            top_hypotheses = sorted(self.population, key=lambda x: x.get("fitness_score", 0), reverse=True)[:10]
            top_genes = []
            for hyp in top_hypotheses:
                # Get primary target or first ranked target
                if hyp.get('target_gene_masked'):
                    top_genes.append(hyp['target_gene_masked'])
                elif hyp.get('ranked_targets'):
                    for target in hyp['ranked_targets'][:3]:
                        if target.get('gene_id'):
                            top_genes.append(target['gene_id'])
            
            # Update reflection agent with previous generation's top genes
            self.reflection_agent.previous_generation_genes = top_genes[:10]
            logging.info(f"Updated previous_generation_genes for novelty scoring: {top_genes[:5]}...")
        
        logging.info(f"Generation {self.current_generation} stats: "
                    f"mean={generation_stats['mean_fitness']}, "
                    f"max={generation_stats['max_fitness']}, "
                    f"min={generation_stats['min_fitness']}")
    
    def _calculate_std(self, values: List[float]) -> float:
        """
        Calculate standard deviation of values.
        
        Args:
            values: List of numerical values
            
        Returns:
            Standard deviation
        """
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _compile_final_results(self) -> Dict[str, Any]:
        """
        Compile final results and statistics.
        
        Returns:
            Dictionary containing comprehensive results
        """
        # Rank final population by fitness score
        final_ranking = sorted(self.population,
                              key=lambda x: x.get("fitness_score", 0),
                              reverse=True)
        
        # Get top hypothesis
        best_hypothesis = final_ranking[0] if final_ranking else None
        
        # Calculate evolution statistics
        initial_stats = self.generation_history[0] if self.generation_history else {}
        final_stats = self.generation_history[-1] if self.generation_history else {}
        
        fitness_improvement = 0
        if initial_stats.get("mean_fitness", 0) > 0:
            fitness_improvement = (
                (final_stats.get("mean_fitness", 0) - initial_stats.get("mean_fitness", 0)) /
                initial_stats.get("mean_fitness", 0)
            ) * 100
        
        results = {
            "genetic_algorithm_results": {
                "research_goal": self.research_goal,
                "mode": self.mode,
                "generations_completed": self.current_generation,
                "final_population_size": len(self.population),
                "initial_mean_fitness": initial_stats.get("mean_fitness", 0),
                "final_mean_fitness": final_stats.get("mean_fitness", 0),
                "fitness_improvement_percent": round(fitness_improvement, 2),
                "best_final_fitness": final_stats.get("max_fitness", 0)
            },
            "best_hypothesis": best_hypothesis,
            "final_population": final_ranking,
            "generation_history": self.generation_history,
            "evolution_summary": {
                "selection_method": "tournament_selection",
                "crossover_operations": ["combination", "inspiration"],
                "mutation_operations": ["simplification", "out_of_box"],
                "elitism_preserved": self.elitism_count,
                "selection_ratio": self.selection_ratio
            }
        }

        # Include complete evolutionary lineage for lethal_genes modes
        if self.mode in ["lethal_genes", "lethal_genes_2"]:
            results["all_hypotheses"] = self.all_hypotheses
            results["total_hypotheses_generated"] = len(self.all_hypotheses)
            logging.info(f"Including complete evolutionary lineage: {len(self.all_hypotheses)} total hypotheses across all generations")

        return results
    
    def _get_current_state(self) -> Dict[str, Any]:
        """
        Get current state for error recovery.

        Returns:
            Dictionary containing current algorithm state
        """
        return {
            "current_generation": self.current_generation,
            "population": self.population,
            "fitness_evaluations": self.fitness_evaluations,
            "generation_history": self.generation_history
        }

    def _get_detailed_state_summary(self) -> str:
        """
        Get detailed state summary for logging (only when DETAILED_STATE_LOGGING is enabled).

        Returns:
            Formatted string with current algorithm state details
        """
        if not hasattr(self, 'population') or not self.population:
            return "population_size=0"

        fitnesses = [h.get('fitness', 0) for h in self.population if h.get('fitness') is not None]
        if not fitnesses:
            return f"population_size={len(self.population)}, fitness=unset"

        return (f"population_size={len(self.population)}, "
                f"fitness_range=[{min(fitnesses):.2f}-{max(fitnesses):.2f}], "
                f"avg_fitness={sum(fitnesses)/len(fitnesses):.2f}")
    
    def get_best_hypothesis(self) -> Optional[Dict[str, Any]]:
        """
        Get the best hypothesis from current population.
        
        Returns:
            Best hypothesis or None if no population exists
        """
        if not self.population:
            return None
        
        return max(self.population, key=lambda x: x.get("fitness_score", 0))
    
    def get_algorithm_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the genetic algorithm execution.
        
        Returns:
            Dictionary containing detailed statistics
        """
        if not self.generation_history:
            return {"error": "No generation history available"}
        
        fitness_evolution = [gen["mean_fitness"] for gen in self.generation_history]
        max_fitness_evolution = [gen["max_fitness"] for gen in self.generation_history]
        
        stats = {
            "algorithm_parameters": {
                "population_size": self.population_size,
                "num_generations": self.num_generations,
                "selection_ratio": self.selection_ratio,
                "elitism_count": self.elitism_count
            },
            "fitness_evolution": {
                "mean_fitness_by_generation": fitness_evolution,
                "max_fitness_by_generation": max_fitness_evolution,
                "final_improvement": fitness_evolution[-1] - fitness_evolution[0] if len(fitness_evolution) > 1 else 0,
                "convergence_rate": self._calculate_convergence_rate(fitness_evolution)
            },
            "population_diversity": {
                "final_fitness_std": self.generation_history[-1]["fitness_std"],
                "fitness_range": self.generation_history[-1]["max_fitness"] - self.generation_history[-1]["min_fitness"]
            }
        }
        
        return stats
    
    def _calculate_convergence_rate(self, fitness_values: List[float]) -> float:
        """
        Calculate convergence rate of the algorithm.
        
        Args:
            fitness_values: List of mean fitness values by generation
            
        Returns:
            Convergence rate (slope of fitness improvement)
        """
        if len(fitness_values) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(fitness_values)
        x_values = list(range(n))
        
        x_mean = sum(x_values) / n
        y_mean = sum(fitness_values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, fitness_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator