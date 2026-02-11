#!/usr/bin/env python3
"""
Benchmark Iterations Script (Optimized)
Runs the T2D pipeline for 5 generations, capturing checkpoints at 1, 3, and 5 generations.
This avoids restarting the pipeline multiple times.
Uses configuration from t2d_config.py for population size.
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any

# Add pipeline root to path
current_dir = Path(__file__).parent
repo_root = current_dir.parent
pipeline_dir = repo_root / "pipeline"
sys.path.insert(0, str(pipeline_dir))

from main import parse_arguments
from t2d_config import T2D_DATA_DIR, T2D_OUTPUT_DIR, GENETIC_ALGORITHM_CONFIG, T2D_ANALYSIS_PARAMS
from utils.t2d_pipeline_runner import T2DPipelineRunner
import agents.supervisor_agent
from agents.supervisor_agent import SupervisorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"benchmark_iterations_optimized_{time.strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

# -----------------------------------------------------------------------------
# CUSTOM SUPERVISOR FOR CHECKPOINTING
# -----------------------------------------------------------------------------
class BenchmarkSupervisor(SupervisorAgent):
    """
    Custom Supervisor that checkpoints metrics at specific generations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark_checkpoints = {} # gen -> results
        self.checkpoint_gens = [1, 3, 5]
        
    def run_genetic_algorithm(self) -> Dict[str, Any]:
        # Dynamic checkpoints based on total generations
        # Always encompass 1, 3, 5 if within range, plus the final generation
        simulated_benchmarks = [1, 3, 5, 10, 20]
        self.checkpoint_gens = sorted(list(set(
            [g for g in simulated_benchmarks if g <= self.num_generations] + [self.num_generations]
        )))
        
        logging.info("=" * 80)
        logging.info("STARTING BENCHMARK GENETIC ALGORITHM (OPTIMIZED)")
        logging.info(f"Configuration: {self.num_generations} generations, {self.population_size} population")
        logging.info(f"Checkpoints: {self.checkpoint_gens}")
        logging.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Step 1: Initialize population
            self._initialize_population()
            
            # Step 2: Run evolutionary generations
            for generation in range(self.num_generations):
                self.current_generation = generation + 1
                self._run_generation()
                
                # CHECKPOINTING
                if self.current_generation in self.checkpoint_gens:
                    logging.info(f"--- CAPTURING CHECKPOINT GEN {self.current_generation} ---")
                    
                    # Compile interim results
                    # We create a deep copy to avoid mutation issues later
                    interim_results = self._compile_final_results()
                    
                    # Store minimal needed data
                    self.benchmark_checkpoints[self.current_generation] = {
                        'best_hypothesis': interim_results.get('best_hypothesis'),
                        'best_fitness': interim_results.get('best_fitness'),
                        'mean_fitness': interim_results.get('mean_fitness'),
                        'top_fitness': interim_results.get('top_fitness'),
                        'elapsed_time': time.time() - start_time
                    }
            
            # Step 3: Final results
            final_results = self._compile_final_results()
            final_results['benchmark_checkpoints'] = self.benchmark_checkpoints
            
            end_time = time.time()
            execution_time = end_time - start_time
            logging.info(f"GENETIC GA GENETIC ALGORITHM COMPLETED in {execution_time:.1f} seconds")
            
            return final_results
            
        except Exception as e:
            logging.error(f"Error in benchmark GA: {e}")
            return {"error": str(e)}

# Apply Monkey Patch
agents.supervisor_agent.SupervisorAgent = BenchmarkSupervisor


# -----------------------------------------------------------------------------
# BENCHMARK RUNNER
# -----------------------------------------------------------------------------
def run_benchmark(dataset_dir, output_basedir):
    results = []
    
    logging.info(f"STARTING OPTIMIZED BENCHMARK RUN")
    
    # Get parameters from config
    max_generations = GENETIC_ALGORITHM_CONFIG.get("num_generations", 5)
    pop_size = GENETIC_ALGORITHM_CONFIG.get("population_size", 20)
    
    logging.info(f"Using configuration: generations={max_generations}, pop_size={pop_size}")
    
    start_time = time.time()
    
    # Setup run directory
    run_id = f"benchmark_opt_{time.strftime('%H%M%S')}"
    run_dir = os.path.join(output_basedir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Initialize runner
    runner = T2DPipelineRunner(dataset_dir, run_dir)
    
    # Load datasets
    runner.load_datasets(["GSE221156_pseudobulk"])
    
    # Run Analysis
    # Check for PyWGCNA
    try:
        import PyWGCNA
        run_wgcna_flag = True
    except ImportError:
        logging.warning("PyWGCNA not found, skipping WGCNA analysis")
        run_wgcna_flag = False

    runner.run_analysis(run_wgcna=run_wgcna_flag, run_tf=True)
    
    # Extract features
    top_n = T2D_ANALYSIS_PARAMS.get("top_candidates_for_llm", 500)
    runner.run_druggability_extraction(top_n_genes=top_n)
    runner.run_leakage_test()
    
    # Run Evolution (will use BenchmarkSupervisor)
    metrics = runner.run_evolutionary_optimization(
        objective="Identify novel drug targets for Type 2 Diabetes",
        population_size=pop_size,
        n_generations=max_generations
    )
    
    # =========================================================================
    # SAVE RESULTS (Similar to main.py)
    # =========================================================================
    logging.info(f"Saving full results to {run_dir}...")
    
    # 1. Save Gene Mapping
    runner.save_gene_mapping(os.path.join(run_dir, "gene_mapping.json"))
    
    # 2. Unmask Best Hypothesis for saving
    if "best_hypothesis" in metrics and metrics["best_hypothesis"]:
        try:
            unmasked = runner.unmask_final_predictions([metrics["best_hypothesis"]])
            metrics["best_hypothesis"] = unmasked[0]
        except Exception as e:
            logging.error(f"Failed to unmask best hypothesis for saving: {e}")
        
    # 3. Save Pipeline Results JSON
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_file = os.path.join(run_dir, f"pipeline_results_{timestamp}.json")
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, default=str)
        logging.info(f"Pipeline results saved to: {json_file}")
    except Exception as e:
        logging.error(f"Failed to save JSON results: {e}")
    # =========================================================================
    
    # Process Checkpoints
    checkpoints = metrics.get('benchmark_checkpoints', {})
    if not checkpoints:
        logging.warning("No checkpoints found! Using final results only.")
        checkpoints[max_generations] = metrics
        
    for gen, data in checkpoints.items():
        logging.info(f"Processing Generation {gen} results...")
        
        # Get hypothesis
        best_hyp = data.get('best_hypothesis')
        if not best_hyp:
            continue
            
        # We need to unmask it to evaluate properly
        # Wrap in list
        unmasked_list = runner.unmask_final_predictions([best_hyp])
        unmasked_hyp = unmasked_list[0]
        
        # Run Full Evaluation
        # Note: We pass list of 1 hypothesis
        evaluation = runner.run_full_evaluation(
            [unmasked_hyp],
            run_baseline=True # Baseline only runs once and caches? Check implementation.
                              # T2DEvaluationMetrics.run_baseline_llm calls LLM. 
                              # T2DPipelineRunner stores 'full_evaluation_results'.
        )
        
        # Collect Metrics
        res = {
            "generations": gen,
            "runtime_sec": round(data.get('elapsed_time', 0), 2),
            "final_fitness": data.get('best_fitness', 0),
            "novelty_score": unmasked_hyp.get('novelty_score', 0),
            "mean_fitness": data.get('mean_fitness', [])[-1] if data.get('mean_fitness') else 0,
        }
        
        # Add Evaluation Metrics
        if 'baseline_comparison' in evaluation:
            res['win_rate'] = evaluation['baseline_comparison'].get('win_rate', 0)
            res['framework_wins'] = evaluation['baseline_comparison'].get('framework_wins', 0)
        
        if 'at_k_metrics' in evaluation:
            res['mrr'] = evaluation['at_k_metrics'].get('mrr', 0)
            res['map'] = evaluation['at_k_metrics'].get('map', 0)
            
        # Add Rank Stats for Tier 1
        if 'at_k_metrics' in evaluation and 'tier_metrics' in evaluation['at_k_metrics']:
            tier1 = evaluation['at_k_metrics']['tier_metrics'].get('tier1', {})
            res['tier1_hits'] = tier1.get('hits', 0)
            res['tier1_recall'] = tier1.get('recall', 0)
            
        results.append(res)
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=T2D_DATA_DIR)
    parser.add_argument("--output-dir", default=T2D_OUTPUT_DIR)
    args = parser.parse_args()
    
    df = run_benchmark(args.data_dir, args.output_dir)
    print("\nBENCHMARK RESULTS (OPTIMIZED):")
    print(df.to_string())
    
    df.to_csv("benchmark_results_optimized.csv", index=False)
