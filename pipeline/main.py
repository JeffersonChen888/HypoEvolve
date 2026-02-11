#!/usr/bin/env python3
"""
Pipeline3 Main Entry Point - Simplified Genetic Algorithm for Scientific Hypothesis Evolution

This is a streamlined genetic algorithm implementation focused on core evolutionary
principles for scientific hypothesis development.

Usage:
    python pipeline3/main.py "research goal" [--mode drug-repurposing|general] [--log-file path]

Example:
    python pipeline3/main.py "Develop novel cancer therapies" --mode drug-repurposing
    python pipeline3/main.py "Understanding protein folding mechanisms" --mode general
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add pipeline3 directory to path
current_dir = Path(__file__).parent
project_root = current_dir.parent  # Parent of pipeline3 is project root
sys.path.insert(0, str(current_dir))

# Import pipeline3 modules
import config
from agents.supervisor_agent import SupervisorAgent
from t2d_config import T2D_ANALYSIS_PARAMS


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pipeline3: Simplified Genetic Algorithm for Scientific Hypothesis Evolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline3/main.py "Develop novel cancer therapies" --mode drug-repurposing
  python pipeline3/main.py "Understanding protein folding" --mode general
  python pipeline3/main.py "Drug repurposing for BRCA" --population-size 10 --generations 7
        """
    )
    
    parser.add_argument(
        "research_goal",
        type=str,
        help="The scientific research goal to explore"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["drug-repurposing", "general", "lethal_genes", "lethal_genes_2", "lethal_genes_tournament", "t2d-target"],  # ADD "t2d-target"
        default="t2d-target",
        help="Pipeline mode (default: t2d-target)"
    )

    # T2D mode arguments
    parser.add_argument(
        "--t2d-data-dir",
        type=str,
        default=None,
        help="Directory containing T2D dataset .h5ad files (t2d-target mode)"
    )

    parser.add_argument(
        "--t2d-datasets",
        type=str,
        default=None,
        help="Comma-separated list of GEO IDs to analyze (default: all in data dir)"
    )

    parser.add_argument(
        "--skip-wgcna",
        action="store_true",
        help="Skip WGCNA analysis (faster but less comprehensive)"
    )

    parser.add_argument(
        "--skip-tf",
        action="store_true",
        help="Skip TF activity analysis"
    )

    parser.add_argument(
        "--gene-pairs-file",
        type=str,
        default=None,
        help="Path to TSV file containing gene pairs for lethal_genes mode (format: GeneA<tab>GeneB<tab>type)"
    )

    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to complete prompt file for lethal_genes_2 mode (e.g., data/lethal_genes/individual_prompts/prompt_01_KLF5_ARID1A.txt)"
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Shared run ID for batch processing (used by batch scripts to group outputs under same folder)"
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=config.POPULATION_SIZE,
        help=f"Initial population size (default: {config.POPULATION_SIZE})"
    )
    
    parser.add_argument(
        "--generations",
        type=int,
        default=config.NUM_GENERATIONS,
        help=f"Number of generations (default: {config.NUM_GENERATIONS})"
    )
    
    parser.add_argument(
        "--selection-ratio",
        type=float,
        default=config.SELECTION_RATIO,
        help=f"Selection ratio for parents (default: {config.SELECTION_RATIO})"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Custom log file path"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory (default: auto-determined by mode: output/lethal_genes/, output/drug_repurposing/, output/general/)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model to use (default: gpt-5-mini). Options: gpt-5, gpt-5-mini, gpt-5-nano, gpt-4o, o3-mini, gemini-2.5-pro, gemini-2.0-flash, qwen2.5:32b (Ollama)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--save-json",
        action="store_true",
        default=True,
        help="Save results as JSON file (default: True)"
    )

    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Batch mode: process multiple gene pairs from gene_pairs_config.py (lethal_genes mode only)"
    )

    # Tournament mode arguments
    parser.add_argument(
        "--generation-prompt",
        type=str,
        default=None,
        help="Path to generation prompt template file with [GENE_A] and [GENE_B] placeholders (lethal_genes_tournament mode)"
    )

    parser.add_argument(
        "--gene-pairs",
        type=str,
        default=None,
        help="Path to JSON file with gene pairs array: [[\"GENE_A\", \"GENE_B\"], ...] (lethal_genes_tournament mode)"
    )

    parser.add_argument(
        "--judging-criteria",
        type=str,
        default=None,
        help="Path to custom judging criteria file for tournament comparisons (optional, has default)"
    )

    parser.add_argument(
        "--top-k-per-pair",
        type=int,
        default=3,
        help="Number of top hypotheses to select from each gene pair for tournament (default: 3)"
    )

    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    if not args.research_goal.strip():
        raise ValueError("Research goal cannot be empty")

    if args.population_size < 2:
        raise ValueError("Population size must be at least 2")

    if args.generations < 1:
        raise ValueError("Number of generations must be at least 1")

    if not (0.1 <= args.selection_ratio <= 1.0):
        raise ValueError("Selection ratio must be between 0.1 and 1.0")

    # Validate tournament mode requirements
    if args.mode == "lethal_genes_tournament":
        if not args.generation_prompt:
            raise ValueError("--generation-prompt is required for lethal_genes_tournament mode")
        if not args.gene_pairs:
            raise ValueError("--gene-pairs is required for lethal_genes_tournament mode")
        if not os.path.exists(args.generation_prompt):
            raise ValueError(f"Generation prompt file not found: {args.generation_prompt}")
        if not os.path.exists(args.gene_pairs):
            raise ValueError(f"Gene pairs file not found: {args.gene_pairs}")
        if args.top_k_per_pair < 1:
            raise ValueError("--top-k-per-pair must be at least 1")

    return True


def setup_environment(args):
    """Setup environment and logging."""
    # Create timestamped run folder organized by mode
    timestamp = config.time.strftime("%Y%m%d_%H%M%S")

    # Use provided run_id or create new one
    if hasattr(args, 'run_id') and args.run_id:
        run_id = args.run_id
    else:
        run_id = f"run_{timestamp}"

    # Determine base output directory (absolute path)
    if args.output_dir:
        # User specified custom output directory
        base_dir = args.output_dir
        if not os.path.isabs(base_dir):
            # Make relative paths absolute from project root
            base_dir = os.path.join(str(project_root), base_dir)
    else:
        # Auto-determine from mode: pipeline/output/lethal_genes_2/, etc.
        base_dir = os.path.join(str(project_root), "pipeline", "output", args.mode.replace("-", "_"))

    # For lethal_genes_2 mode, create gene-pair-specific folder under shared run_id
    if args.mode == "lethal_genes_2" and hasattr(args, 'prompt_file') and args.prompt_file:
        from utils.prompt_loader import extract_gene_pair_name
        gene_pair_name = extract_gene_pair_name(args.prompt_file)
        # Structure: pipeline/output/lethal_genes_2/{run_id}/{gene_pair_name}/
        run_folder = os.path.join(base_dir, run_id, gene_pair_name)
    else:
        # Standard structure: {base_dir}/{run_id}/
        run_folder = os.path.join(base_dir, run_id)

    os.makedirs(run_folder, exist_ok=True)

    # Override model if specified
    if args.model:
        from external_tools import llm_client
        llm_client.MODEL_NAME = args.model
        logging.info(f"Model overridden to: {args.model}")

    # Setup logging
    if args.log_file:
        log_file = args.log_file
    else:
        log_file = os.path.join(run_folder, f"pipeline_{timestamp}.log")

    logger = config.setup_logging(log_file)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    return logger, log_file, run_folder


def print_welcome_message(args, run_folder):
    """Print welcome message and configuration."""
    print("=" * 80)
    print("PIPELINE3: SIMPLIFIED GENETIC ALGORITHM FOR SCIENTIFIC HYPOTHESES")
    print("=" * 80)
    print(f"Research Goal: {args.research_goal}")
    print(f"Mode: {args.mode}")
    from external_tools.llm_client import MODEL_NAME
    print(f"Model: {args.model if args.model else MODEL_NAME}")
    print(f"Population Size: {args.population_size}")
    print(f"Generations: {args.generations}")
    print(f"Selection Ratio: {args.selection_ratio}")
    print(f"Output Directory: {run_folder}")
    print("=" * 80)


def create_supervisor(args, run_folder=None):
    """Create and configure supervisor agent."""
    # For lethal_genes_2 mode, prompt_file is required
    if args.mode == "lethal_genes_2" and not args.prompt_file:
        raise ValueError("--prompt-file is required for lethal_genes_2 mode")

    supervisor = SupervisorAgent(
        research_goal=args.research_goal,
        mode=args.mode,
        batch_mode=args.batch_mode,
        run_folder=run_folder,
        prompt_file=args.prompt_file if hasattr(args, 'prompt_file') else None
    )
    
    # Override default parameters with command line arguments
    # Note: In batch mode, population_size is set automatically to NUM_ACTIVE_PAIRS
    if not args.batch_mode:
        supervisor.population_size = args.population_size

    supervisor.num_generations = args.generations
    supervisor.selection_ratio = args.selection_ratio

    # Re-validate elitism count after population_size override
    if supervisor.elitism_count >= supervisor.population_size:
        old_elitism = supervisor.elitism_count
        supervisor.elitism_count = max(1, supervisor.population_size - 1)
        logging.warning(f"Elitism count adjusted to {supervisor.elitism_count} (was {old_elitism}) to allow offspring generation with population size {supervisor.population_size}")

    return supervisor


def save_results(results, args, log_file, run_folder=None):
    """Save results to files."""
    timestamp = config.time.strftime("%Y%m%d_%H%M%S")

    # Determine output path (use run_folder if available, fallback to output_dir)
    output_path = run_folder if run_folder else args.output_dir

    # Special handling for lethal_genes batch mode
    if args.mode == "lethal_genes":
        # Batch mode: output all hypotheses sorted by score
        if args.batch_mode and "all_hypotheses" in results:
            all_hyps = results["all_hypotheses"]
            # Sort by fitness score descending
            sorted_hyps = sorted(all_hyps, key=lambda h: h.get("fitness_score", 0), reverse=True)

            # Save sorted hypotheses
            sorted_output_file = os.path.join(
                output_path,
                f"all_hypotheses_sorted_{timestamp}.json"
            )
            try:
                with open(sorted_output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "total_hypotheses": len(sorted_hyps),
                        "generations": args.generations,
                        "gene_pairs_processed": results.get("total_hypotheses_generated", 0) // (args.generations + 1) if args.generations > 0 else len(sorted_hyps),
                        "hypotheses_sorted_by_score": sorted_hyps
                    }, f, indent=2, default=str)
                print(f"All hypotheses (sorted by score) saved to: {sorted_output_file}")
                print(f"Total hypotheses: {len(sorted_hyps)}")
                print(f"Top score: {sorted_hyps[0].get('fitness_score', 0):.2f} - {sorted_hyps[0].get('gene_a', 'N/A')} x {sorted_hyps[0].get('gene_b', 'N/A')}")
            except Exception as e:
                logging.error(f"Failed to save sorted hypotheses: {e}")

            # Save comprehensive genealogy JSON
            genealogy_file = os.path.join(
                output_path,
                f"genealogy_{timestamp}.json"
            )
            try:
                # Group hypotheses by generation
                hyps_by_gen = {}
                for hyp in all_hyps:
                    gen = hyp.get("generation", 0)
                    if gen not in hyps_by_gen:
                        hyps_by_gen[gen] = []
                    hyps_by_gen[gen].append({
                        "id": hyp.get("id"),
                        "gene_a": hyp.get("gene_a"),
                        "gene_b": hyp.get("gene_b"),
                        "generation": gen,
                        "fitness_score": hyp.get("fitness_score", 0),
                        "parent_ids": hyp.get("parent_ids", []),
                        "evolution_strategy": hyp.get("evolution_strategy", "unknown"),
                        "is_elite": hyp.get("is_elite", False),
                        "elite_generation": hyp.get("elite_generation"),
                        "title": hyp.get("title", "")[:100]
                    })

                # Create genealogy structure
                genealogy_data = {
                    "metadata": {
                        "timestamp": timestamp,
                        "total_hypotheses": len(all_hyps),
                        "total_generations": args.generations + 1,  # +1 for generation 0
                        "population_size": args.population_size,
                        "selection_ratio": args.selection_ratio,
                        "elitism_count": results.get("evolution_summary", {}).get("elitism_preserved", 0)
                    },
                    "generations": []
                }

                # Add each generation
                for gen in sorted(hyps_by_gen.keys()):
                    gen_hyps = hyps_by_gen[gen]
                    elite_count = sum(1 for h in gen_hyps if h.get("is_elite"))
                    genealogy_data["generations"].append({
                        "generation": gen,
                        "hypothesis_count": len(gen_hyps),
                        "elite_count": elite_count,
                        "fitness_stats": {
                            "min": min((h["fitness_score"] for h in gen_hyps), default=0),
                            "max": max((h["fitness_score"] for h in gen_hyps), default=0),
                            "mean": sum(h["fitness_score"] for h in gen_hyps) / len(gen_hyps) if gen_hyps else 0
                        },
                        "hypotheses": gen_hyps
                    })

                with open(genealogy_file, 'w', encoding='utf-8') as f:
                    json.dump(genealogy_data, f, indent=2, default=str)
                print(f"Genealogy data saved to: {genealogy_file}")
                logging.info(f"Saved genealogy data with {len(all_hyps)} hypotheses across {len(hyps_by_gen)} generations")
            except Exception as e:
                logging.error(f"Failed to save genealogy JSON: {e}")

            return  # Skip individual gene pair output for batch mode

    # Standard JSON output (auto-save for lethal_genes, or if explicitly requested)
    if args.save_json or args.mode == "lethal_genes":
        json_file = os.path.join(output_path, f"pipeline_results_{timestamp}.json")
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to: {json_file}")
        except Exception as e:
            logging.error(f"Failed to save JSON results: {e}")

    # Save summary report
    summary_file = os.path.join(output_path, f"pipeline_summary_{timestamp}.txt")
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            write_summary_report(f, results, args, log_file)
        print(f"Summary saved to: {summary_file}")
    except Exception as e:
        logging.error(f"Failed to save summary: {e}")


def write_summary_report(file, results, args, log_file):
    """Write summary report to file."""
    file.write("PIPELINE3 GENETIC ALGORITHM RESULTS\n")
    file.write("=" * 50 + "\n\n")
    
    # Configuration
    file.write("CONFIGURATION:\n")
    file.write(f"Research Goal: {args.research_goal}\n")
    file.write(f"Mode: {args.mode}\n")
    file.write(f"Population Size: {args.population_size}\n")
    file.write(f"Generations: {args.generations}\n")
    file.write(f"Selection Ratio: {args.selection_ratio}\n")
    file.write(f"Log File: {log_file}\n\n")
    
    # Algorithm results
    if "genetic_algorithm_results" in results:
        ga_results = results["genetic_algorithm_results"]
        file.write("GENETIC ALGORITHM RESULTS:\n")
        file.write(f"Generations Completed: {ga_results.get('generations_completed', 0)}\n")
        file.write(f"Final Population Size: {ga_results.get('final_population_size', 0)}\n")
        file.write(f"Initial Mean Fitness: {ga_results.get('initial_mean_fitness', 0):.2f}\n")
        file.write(f"Final Mean Fitness: {ga_results.get('final_mean_fitness', 0):.2f}\n")
        file.write(f"Fitness Improvement: {ga_results.get('fitness_improvement_percent', 0):.2f}%\n")
        file.write(f"Best Final Fitness: {ga_results.get('best_final_fitness', 0):.2f}\n\n")
    
    # Best hypothesis
    if "best_hypothesis" in results and results["best_hypothesis"]:
        best = results["best_hypothesis"]
        file.write("BEST HYPOTHESIS:\n")
        file.write(f"Title: {best.get('title', 'Unknown')}\n")

        # Display new unified fields if available (drug-repurposing/general modes)
        if 'summary' in best or 'hypothesis_statement' in best or 'rationale' in best:
            if best.get('summary'):
                file.write(f"Summary: {best.get('summary')}\n")
            if best.get('hypothesis_statement'):
                file.write(f"Hypothesis: {best.get('hypothesis_statement')}\n")
            if best.get('rationale'):
                file.write(f"Rationale: {best.get('rationale')}\n")
        else:
            # Fall back to legacy description field
            file.write(f"Description: {best.get('description', 'No description')}\n")

        file.write(f"Fitness Score: {best.get('fitness_score', 0):.2f}\n")
        file.write(f"ELO Score: {best.get('elo_score', 0)}\n")
        file.write(f"Evolution Strategy: {best.get('evolution_strategy', 'N/A')}\n")

        if args.mode == "drug-repurposing":
            file.write(f"Final Drug: {best.get('final_drug', 'Not specified')}\n")
            file.write(f"Cancer Type: {best.get('cancer_type', 'Not specified')}\n")
        elif args.mode == "general":
            file.write(f"Final Answer: {best.get('final_answer', 'Not specified')}\n")
        elif args.mode == "lethal_genes":
            file.write(f"Gene Pair: {best.get('gene_a', 'N/A')} - {best.get('gene_b', 'N/A')}\n")
            file.write(f"FINAL_PREDICTION: {best.get('final_prediction', 'Not specified')}\n")
            file.write(f"Summary: {best.get('summary', 'Not specified')}\n")
            # Rationale includes biological plausibility, clinical relevance, mechanistic explanation
            rationale = best.get('rationale', '')
            if rationale:
                # Truncate long rationale for summary file
                rationale_preview = rationale[:500] + "..." if len(rationale) > 500 else rationale
                file.write(f"Rationale: {rationale_preview}\n")

        file.write(f"\n")
    
    # Generation history summary
    if "generation_history" in results:
        file.write("GENERATION HISTORY:\n")
        for gen in results["generation_history"]:
            file.write(f"Generation {gen['generation']}: "
                      f"mean={gen['mean_fitness']:.2f}, "
                      f"max={gen['max_fitness']:.2f}, "
                      f"std={gen['fitness_std']:.2f}\n")


def print_results_summary(results):
    """Print results summary to console."""
    print("\n" + "=" * 80)
    print("GENETIC ALGORITHM EXECUTION COMPLETED")
    print("=" * 80)
    
    if "genetic_algorithm_results" in results:
        ga_results = results["genetic_algorithm_results"]
        print(f"Generations completed: {ga_results.get('generations_completed', 0)}")
        print(f"Final population size: {ga_results.get('final_population_size', 0)}")
        print(f"Fitness improvement: {ga_results.get('fitness_improvement_percent', 0):.2f}%")
        print(f"Best final fitness: {ga_results.get('best_final_fitness', 0):.2f}")
    
    if "best_hypothesis" in results and results["best_hypothesis"]:
        best = results["best_hypothesis"]
        print(f"\nBest Hypothesis: {best.get('title', 'Unknown')}")
        print(f"Fitness Score: {best.get('fitness_score', 0):.2f}")
        print(f"Evolution Strategy: {best.get('evolution_strategy', 'N/A')}")
    
    print("=" * 80)


def load_tournament_inputs(args):
    """Load input files for tournament mode."""
    # Load generation prompt template
    with open(args.generation_prompt, 'r', encoding='utf-8') as f:
        generation_prompt = f.read()

    # Load gene pairs
    with open(args.gene_pairs, 'r', encoding='utf-8') as f:
        gene_pairs_data = json.load(f)

    # Convert to list of tuples
    gene_pairs = [tuple(pair) for pair in gene_pairs_data]

    # Load optional judging criteria
    judging_criteria = None
    if args.judging_criteria and os.path.exists(args.judging_criteria):
        with open(args.judging_criteria, 'r', encoding='utf-8') as f:
            judging_criteria = f.read()

    return generation_prompt, gene_pairs, judging_criteria


def save_tournament_results(results, args, log_file, run_folder):
    """Save results for tournament mode."""
    timestamp = config.time.strftime("%Y%m%d_%H%M%S")

    # Save comprehensive JSON results
    json_file = os.path.join(run_folder, f"tournament_results_{timestamp}.json")
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Full results saved to: {json_file}")
    except Exception as e:
        logging.error(f"Failed to save tournament results: {e}")

    # Save final rankings as CSV for easy viewing
    rankings_file = os.path.join(run_folder, f"final_rankings_{timestamp}.csv")
    try:
        with open(rankings_file, 'w', encoding='utf-8') as f:
            f.write("Rank,Gene_A,Gene_B,ELO_Score,Fitness_Score,Wins,Losses,Matches\n")
            for rank, hyp in enumerate(results.get("final_rankings", []), 1):
                gene_a = hyp.get("gene_a", "?")
                gene_b = hyp.get("gene_b", "?")
                elo = hyp.get("final_elo", hyp.get("elo_score", 1200))
                fitness = hyp.get("fitness_score", 0)
                wins = hyp.get("tournament_wins", 0)
                losses = hyp.get("tournament_losses", 0)
                matches = hyp.get("tournament_matches", 0)
                f.write(f"{rank},{gene_a},{gene_b},{elo},{fitness:.2f},{wins},{losses},{matches}\n")
        print(f"Final rankings saved to: {rankings_file}")
    except Exception as e:
        logging.error(f"Failed to save rankings CSV: {e}")

    # Save phase 1 per-pair results
    phase1_dir = os.path.join(run_folder, "phase1_results")
    os.makedirs(phase1_dir, exist_ok=True)
    for pair_result in results.get("phase1_results", []):
        pair_name = pair_result.get("gene_pair", "unknown")
        pair_file = os.path.join(phase1_dir, f"{pair_name}.json")
        try:
            with open(pair_file, 'w', encoding='utf-8') as f:
                json.dump(pair_result, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Failed to save pair results {pair_name}: {e}")

    print(f"Phase 1 results saved to: {phase1_dir}")


def print_tournament_summary(results):
    """Print tournament results summary."""
    print("\n" + "=" * 80)
    print("LETHAL GENES TOURNAMENT COMPLETED")
    print("=" * 80)

    params = results.get("parameters", {})
    print(f"Gene pairs processed: {params.get('gene_pairs_count', 0)}")
    print(f"Total hypotheses pooled: {params.get('total_hypotheses_pooled', 0)}")
    print(f"Execution time: {results.get('execution_time_seconds', 0):.1f} seconds")

    tournament = results.get("tournament_results", {})
    print(f"\nTournament rounds: {tournament.get('rounds_completed', 0)}")
    print(f"Total matches: {tournament.get('total_matches', 0)}")
    print(f"Ranking stability achieved: {tournament.get('stability_achieved', False)}")

    print("\n--- TOP 10 FINAL RANKINGS ---")
    for rank, hyp in enumerate(results.get("final_rankings", [])[:10], 1):
        gene_pair = f"{hyp.get('gene_a', '?')}-{hyp.get('gene_b', '?')}"
        elo = hyp.get("final_elo", hyp.get("elo_score", 1200))
        fitness = hyp.get("fitness_score", 0)
        print(f"  {rank}. [{gene_pair}] ELO: {elo}, Fitness: {fitness:.2f}")

    print("=" * 80)


def main():
    """Main entry point for Pipeline3."""
    try:
        # Parse and validate arguments
        args = parse_arguments()
        validate_arguments(args)

        # Setup environment
        logger, log_file, run_folder = setup_environment(args)

        # Print welcome message
        print_welcome_message(args, run_folder)

        # Handle tournament mode separately
        if args.mode == "lethal_genes_tournament":
            # Load tournament inputs
            generation_prompt, gene_pairs, judging_criteria = load_tournament_inputs(args)
            logging.info(f"Loaded {len(gene_pairs)} gene pairs for tournament")

            # Create supervisor (mode doesn't need to be tournament - agents use lethal_genes internally)
            supervisor = SupervisorAgent(
                research_goal=args.research_goal,
                mode="lethal_genes_2",  # Use lethal_genes_2 style generation
                run_folder=run_folder
            )

            # Override GA parameters for tournament mode
            supervisor.population_size = args.population_size
            supervisor.num_generations = args.generations

            # Run tournament mode
            results = supervisor.run_tournament_mode(
                gene_pairs=gene_pairs,
                generation_prompt=generation_prompt,
                judging_criteria=judging_criteria,
                population_size=args.population_size,
                num_generations=args.generations,
                top_k_per_pair=args.top_k_per_pair
            )

            # Print tournament summary
            print_tournament_summary(results)

            # Save tournament results
            save_tournament_results(results, args, log_file, run_folder)

        elif args.mode == "t2d-target":
            # T2D Drug Target Identification mode
            from utils.t2d_pipeline_runner import T2DPipelineRunner
            from t2d_config import T2D_DATA_DIR, USE_DRUGGABILITY

            data_dir = args.t2d_data_dir or T2D_DATA_DIR

            # Parse dataset list
            dataset_names = None
            if args.t2d_datasets:
                dataset_names = [d.strip() for d in args.t2d_datasets.split(",")]

            logging.info(f"T2D mode: Loading datasets from {data_dir}")

            # Run T2D pipeline
            t2d_runner = T2DPipelineRunner(data_dir, run_folder)
            t2d_runner.load_datasets(dataset_names)
            t2d_runner.run_analysis(
                run_wgcna=not args.skip_wgcna,
                run_tf=not args.skip_tf
            )

            # ================================================================
            # OPTION A vs OPTION C DECISION
            # ================================================================
            use_druggability = USE_DRUGGABILITY
            leakage_safe = True

            if use_druggability:
                logging.info("Option A: Attempting druggability-enhanced mode...")

                # Extract druggability features
                top_n = T2D_ANALYSIS_PARAMS.get("top_candidates_for_llm", 500)
                druggability_df = t2d_runner.run_druggability_extraction(top_n_genes=top_n)

                if druggability_df is not None:
                    # Run leakage test
                    leakage_results = t2d_runner.run_leakage_test()

                    if leakage_results.get('overall_safe', False):
                        logging.info("Leakage test PASSED - using Option A (druggability-enhanced)")
                        leakage_safe = True
                    else:
                        logging.warning("Leakage test FAILED - falling back to Option C (expression-only)")
                        leakage_safe = False
                        use_druggability = False
                else:
                    logging.warning("Druggability extraction failed - using Option C (expression-only)")
                    use_druggability = False

            if not use_druggability:
                logging.info("Option C: Using expression-only mode (no druggability features)")

            # ================================================================
            # LOAD GROUND TRUTH FOR EVALUATION
            # ================================================================
            logging.info("Loading ground truth from OpenTargets...")
            t2d_runner.load_ground_truth()

            # Create supervisor with T2D context
            supervisor = SupervisorAgent(
                research_goal=args.research_goal,
                mode="t2d-target",
                run_folder=run_folder
            )

            # Attach T2D runner to supervisor
            supervisor.t2d_runner = t2d_runner
            supervisor.population_size = args.population_size
            supervisor.num_generations = args.generations

            # Store druggability mode flag for generation agent
            supervisor.use_druggability = use_druggability

            # ================================================================
            # RUN BASELINE BEFORE GA (for novelty scoring)
            # ================================================================
            logging.info("Running baseline LLM predictions before GA (for novelty scoring)...")
            try:
                from utils.evaluation_metrics import T2DEvaluationMetrics
                
                # Get analysis context for baseline
                analysis_context = t2d_runner.format_for_generation_agent()  # uses config default
                
                # Run baseline prediction
                evaluator = T2DEvaluationMetrics()
                baseline_predictions = evaluator.run_baseline_llm(
                    analysis_context=analysis_context,
                    n_predictions=5
                )
                
                # Pass baseline predictions to supervisor for novelty scoring
                supervisor.set_baseline_predictions(baseline_predictions)
                logging.info(f"Baseline predictions set for novelty scoring: {[p.get('target_gene_masked') for p in baseline_predictions]}")
            except Exception as e:
                logging.warning(f"Failed to run early baseline for novelty scoring: {e}")
                # Continue without novelty scoring if baseline fails

            # Run GA
            results = supervisor.run_genetic_algorithm()

            # ================================================================
            # POST-PROCESSING: UNMASK, SAVE, EVALUATE
            # ================================================================

            # Collect all hypotheses for candidate saving
            all_hypotheses = results.get('all_hypotheses', [])
            if "best_hypothesis" in results:
                all_hypotheses = [results["best_hypothesis"]] + all_hypotheses

            # Save candidate genes for future reference
            t2d_runner.save_candidate_genes(all_hypotheses)

            # Unmask final predictions
            if "best_hypothesis" in results:
                unmasked = t2d_runner.unmask_final_predictions([results["best_hypothesis"]])
                results["best_hypothesis"] = unmasked[0]

            # ================================================================
            # COMPREHENSIVE EVALUATION (3 metrics)
            # ================================================================
            logging.info("Running comprehensive evaluation (OpenTargets correlation, baseline comparison, @K metrics)...")

            # Get analysis and literature context for baseline comparison
            analysis_context = t2d_runner.format_for_generation_agent()  # uses config default
            literature_context = results.get('population_data', {}).get('literature_context', '')

            # Run full evaluation with all three metrics
            full_eval_results = t2d_runner.run_full_evaluation(
                predictions=all_hypotheses,
                analysis_context=analysis_context,
                literature_context=literature_context,
                run_baseline=True
            )
            results["full_evaluation"] = full_eval_results

            # Also run basic ground truth evaluation for backward compatibility
            eval_results = t2d_runner.evaluate_predictions(all_hypotheses)
            results["evaluation"] = eval_results

            # Add metadata
            results["pipeline_mode"] = "Option A (druggability)" if use_druggability else "Option C (expression-only)"
            results["leakage_test_passed"] = leakage_safe if use_druggability else "N/A"

            # Save gene mapping for reproducibility
            t2d_runner.save_gene_mapping(os.path.join(run_folder, "gene_mapping.json"))

            # Save full evaluation report to file
            if 'evaluation_report' in full_eval_results:
                eval_report_path = os.path.join(run_folder, "evaluation_report.txt")
                with open(eval_report_path, 'w') as f:
                    f.write(full_eval_results['evaluation_report'])
                logging.info(f"Evaluation report saved to: {eval_report_path}")

            # Print and save results
            print_results_summary(results)
            save_results(results, args, log_file, run_folder)

            # Print comprehensive evaluation summary
            print("\n" + "=" * 70)
            print("T2D COMPREHENSIVE EVALUATION SUMMARY")
            print("=" * 70)
            print(f"Pipeline Mode: {results['pipeline_mode']}")

            # OpenTargets correlation
            corr = full_eval_results.get('opentargets_correlation', {})
            if 'error' not in corr:
                print(f"\n1. OPENTARGETS CORRELATION")
                print(f"   Spearman r = {corr.get('spearman_correlation', 'N/A')}")
                print(f"   Genes evaluated: {corr.get('n_genes_evaluated', 0)}")
            else:
                print(f"\n1. OPENTARGETS CORRELATION: {corr.get('error', 'N/A')}")

            # Baseline comparison
            baseline = full_eval_results.get('baseline_comparison', {})
            if baseline and 'summary' in baseline:
                summary = baseline['summary']
                print(f"\n2. BASELINE LLM COMPARISON")
                print(f"   Framework hits: {summary.get('framework_total_hits', 0)}")
                print(f"   Baseline hits: {summary.get('baseline_total_hits', 0)}")
                print(f"   Improvement: {summary.get('hit_improvement', 0):+d} hits")
                print(f"   Framework outperforms: {summary.get('framework_outperforms', False)}")

            # @K metrics
            at_k = full_eval_results.get('at_k_metrics', {})
            if 'error' not in at_k:
                print(f"\n3. @K METRICS")
                print(f"   MRR: {at_k.get('mrr', 0):.4f}")
                print(f"   MAP: {at_k.get('map', 0):.4f}")
                print(f"   First hit at rank: {at_k.get('first_hit_rank', 'None')}")

                # Print @K metrics
                for k_key, k_data in at_k.get('metrics_by_k', {}).items():
                    print(f"   {k_key}: P={k_data.get('precision', 0):.3f}, R={k_data.get('recall', 0):.3f}, Hits={k_data.get('hits', 0)}")

                # Tier hits
                tier_metrics = at_k.get('tier_metrics', {})
                if tier_metrics:
                    print(f"\n   Hits by Tier:")
                    for tier, tier_data in tier_metrics.items():
                        print(f"     {tier.upper()}: {tier_data.get('hits', 0)}/{tier_data.get('total_in_tier', 0)}")
                        if tier_data.get('hit_genes'):
                            print(f"       Identified: {', '.join(tier_data['hit_genes'][:5])}")

            print("=" * 70)

        else:
            # Standard GA modes
            supervisor = create_supervisor(args, run_folder)
            results = supervisor.run_genetic_algorithm()

            # Print results summary
            print_results_summary(results)

            # Save results
            save_results(results, args, log_file, run_folder)

        print(f"\nExecution completed successfully!")
        print(f"Log file: {log_file}")

        return 0

    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
        logging.info("Execution interrupted by user")
        return 1

    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())