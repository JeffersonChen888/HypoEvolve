#!/usr/bin/env python3
"""
Benchmark Models Script
Runs the T2D pipeline with different LLM models/providers to compare performance.
Uses subprocesses to ensure clean configuration loading for each run.
"""

import os
import sys
import json
import time
import subprocess
import glob
import pandas as pd
from pathlib import Path

def run_benchmark_models():
    # Define models to test
    # Note: Requires valid API keys in environment or .env file
    models_to_test = [
        {"provider": "openai", "model": "gpt-4o", "reasoning": "medium"},
        # {"provider": "anthropic", "model": "claude-3-5-sonnet-20240620", "reasoning": "medium"},
        # {"provider": "google", "model": "gemini-1.5-pro", "reasoning": "medium"},
    ]
    
    results = []
    base_dir = Path(__file__).parent.parent
    main_script = base_dir / "pipeline" / "main.py"
    
    for config in models_to_test:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {config['provider']} - {config['model']}")
        print(f"{'='*60}")
        
        # Prepare environment
        env = os.environ.copy()
        env["LLM_PROVIDER"] = config["provider"]
        env["OPENAI_MODEL_NAME"] = config["model"] # Assuming config supports this override
        env["REASONING_EFFORT"] = config["reasoning"]
        # Force a specific run name or capture the latest
        
        # Run pipeline
        cmd = [
            sys.executable, str(main_script),
            "Identify novel drug targets for Type 2 Diabetes",
            "--mode", "t2d-target",
            "--skip-wgcna", # fast mode
            "--skip-tf"
        ]
        
        start_time = time.time()
        try:
            # Run and capture output to find the run directory
            result = subprocess.run(
                cmd, 
                env=env, 
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout[-500:]) # Print last 500 chars
            
            # Find the latest run directory
            output_dir = base_dir / "output" / "t2d_target"
            list_of_files = glob.glob(str(output_dir / "run_*"))
            latest_run = max(list_of_files, key=os.path.getctime)
            print(f"Run completed in: {latest_run}")
            
            # Parse results from that run
            results_file = Path(latest_run) / "final_results.json" # or pipeline_results_*.json
            
            # Since filename has timestamp, find any json in that dir
            json_files = list(Path(latest_run).glob("pipeline_results_*.json"))
            if json_files:
                with open(json_files[0], 'r') as f:
                    data = json.load(f)
                    
                # Extract metrics
                metrics = {
                    "model": config["model"],
                    "provider": config["provider"],
                    "runtime": round(time.time() - start_time, 2),
                    "novelty_score": data.get("best_hypothesis", {}).get("novelty_score", 0),
                    "fitness": data.get("best_fitness", 0),
                }
                
                # Check for evaluation report
                eval_file = Path(latest_run) / "full_evaluation_results.json" # If saved separately
                # Or extract from logs/pipeline results if embedded
                
                results.append(metrics)
            else:
                print("No results JSON found.")
                
        except subprocess.CalledProcessError as e:
            print(f"Error running model {config['model']}:")
            print(e.stderr)
            
    # Save results
    if results:
        df = pd.DataFrame(results)
        print("\nBENCHMARK RESULTS:")
        print(df.to_string())
        df.to_csv("benchmark_models_results.csv", index=False)

if __name__ == "__main__":
    run_benchmark_models()
