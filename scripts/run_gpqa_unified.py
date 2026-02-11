#!/usr/bin/env python3
"""
GPQA Unified Evaluation Script

Runs GA-5gen and captures checkpoints at generation 0 (baseline), 3 (ga3), and 5 (ga5).
This is more efficient than running 3 separate evaluations.

Usage:
    python run_gpqa_unified.py --start 0 --end 10
    python run_gpqa_unified.py --start 0 --end 198  # Full dataset
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Add pipeline to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

import pandas as pd

# Pricing for gpt-5-mini
PRICE_PER_INPUT_TOKEN = 1.10 / 1_000_000   # $1.10 per 1M input tokens
PRICE_PER_OUTPUT_TOKEN = 4.40 / 1_000_000  # $4.40 per 1M output tokens

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "output" / "gpqa_unified"


def setup_logging(log_file: Path = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=handlers
    )


def load_gpqa_dataset() -> pd.DataFrame:
    """Load the GPQA Diamond dataset."""
    data_path = PROJECT_ROOT / "data" / "gpqa" / "gpqa_diamond.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"GPQA dataset not found at {data_path}")
    return pd.read_csv(data_path)


class AnswerExtractionError(Exception):
    """Raised when answer extraction fails and we should stop processing."""
    pass


def extract_answer_from_text(text: str) -> str:
    """
    Extract answer letter from LLM response text.
    Uses comprehensive patterns from comprehensive_ga_evaluation.py.
    """
    if not text:
        return 'UNKNOWN'

    text_upper = text.upper()

    # Prioritized patterns (most specific first, search from end backwards)
    patterns = [
        # FINAL_ANSWER format (Pipeline standard)
        r'FINAL_ANSWER:\s*\[?([A-D])\]?',
        r'FINAL ANSWER:\s*\[?([A-D])\]?',

        # JSON format {"Answer": "X"}
        r'"[Aa]nswer"\s*:\s*"([A-D])"',

        # Bold markdown variants
        r'\*\*CORRECT OPTION:\s*([A-D])\*\*',
        r'\*\*OPTION\s+([A-D])\*\*:',
        r'\*\*([A-D])\)\s*[^*]*\*\*',
        r'\*\*([A-D])\*\*',

        # "Answer:" prefix patterns
        r'ANSWER:\s*([A-D])\)',
        r'ANSWER:\s*([A-D])\b',

        # "Correct" statements
        r'CORRECT\s+(?:ANSWER|CHOICE|OPTION)\s+IS\s*[:;\s]*([A-D])\)',
        r'CORRECT\s+(?:ANSWER|CHOICE|OPTION)\s+IS\s*[:;\s]*([A-D])\b',
        r'(?:THE\s+)?CORRECT\s+(?:ANSWER|CHOICE|OPTION).*?([A-D])\)',

        # Plain answer statements
        r'(?:ANSWER|CHOICE)\s+IS\s*[:;\n\s]*\s*([A-D])\)',
        r'(?:ANSWER|CHOICE)\s+IS\s*[:;\n\s]*\s*([A-D])\b',
        r'\b([A-D])\s*(?:is|would be)\s+(?:the\s+)?(?:correct|right|best)',

        # Selection statements
        r'(?:SELECT|CHOOSE)\s+([A-D])',
        r'OPTION\s+([A-D])',
    ]

    # Search all patterns from END to beginning (prefer final answer)
    for pattern in patterns:
        matches = list(re.finditer(pattern, text_upper, re.IGNORECASE | re.DOTALL))
        if matches:
            return matches[-1].group(1).upper()

    # Fallback: look for standalone letter in last lines
    lines = text.strip().split('\n')
    for line in reversed(lines[-5:]):
        line_clean = line.strip().upper()
        if line_clean in ['A', 'B', 'C', 'D']:
            return line_clean
        # Check for "A)" or "(A)" format
        match = re.search(r'^([A-D])\)|\(([A-D])\)$', line_clean)
        if match:
            return (match.group(1) or match.group(2)).upper()

    return 'UNKNOWN'


def extract_answer_from_hypothesis(hypothesis: dict, raise_on_unknown: bool = True) -> str:
    """
    Extract answer from a hypothesis object.
    Comprehensively searches all fields like comprehensive_ga_evaluation.py.

    Args:
        hypothesis: The hypothesis dict from the pipeline
        raise_on_unknown: If True, raise AnswerExtractionError instead of returning 'UNKNOWN'

    Returns:
        The extracted answer letter (A, B, C, or D)

    Raises:
        AnswerExtractionError: If raise_on_unknown=True and no answer could be extracted
    """
    if not hypothesis:
        if raise_on_unknown:
            raise AnswerExtractionError("Hypothesis is empty or None")
        return 'UNKNOWN'

    # First check: direct final_answer field (most reliable)
    final_answer = hypothesis.get('final_answer')
    if final_answer:
        answer = str(final_answer).strip().upper()
        if len(answer) == 1 and answer in 'ABCD':
            return answer
        # Try to extract letter from longer string
        match = re.search(r'\b([A-D])\b', answer)
        if match:
            return match.group(1).upper()

    # Collect ALL text fields from the hypothesis (comprehensive search)
    text_fields = []

    # Standard fields
    for key in ['title', 'description', 'summary', 'rationale',
                'hypothesis_statement', 'testability_notes', 'evolution_justification']:
        value = hypothesis.get(key, '')
        if value:
            text_fields.append(str(value))

    # Reviews (list of strings)
    reviews = hypothesis.get('reviews', [])
    if isinstance(reviews, list):
        for review in reviews:
            if isinstance(review, str):
                text_fields.append(review)
    elif isinstance(reviews, str):
        text_fields.append(reviews)

    # Combine and search
    combined_text = ' '.join(text_fields)
    answer = extract_answer_from_text(combined_text)

    if answer != 'UNKNOWN':
        return answer

    # Additional JSON pattern search
    json_pattern = r'"Answer"\s*:\s*"([A-D])"'
    matches = re.findall(json_pattern, combined_text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    # If we still can't find an answer
    if raise_on_unknown:
        preview = combined_text[:200] if combined_text else "No text found"
        raise AnswerExtractionError(
            f"Could not extract answer from hypothesis. "
            f"final_answer field: {final_answer}, "
            f"Text preview: {preview}..."
        )

    return 'UNKNOWN'


def parse_token_usage_from_log(log_file: Path) -> dict:
    """Parse token usage from log file."""
    result = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'cost_usd': 0.0,
        'api_calls': 0
    }
    
    if not log_file or not log_file.exists():
        return result
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if 'TOKEN_USAGE:' in line:
                    # Parse: TOKEN_USAGE: input=X, output=Y, total=Z, cost=$W
                    input_match = re.search(r'input=(\d+)', line)
                    output_match = re.search(r'output=(\d+)', line)
                    
                    if input_match and output_match:
                        result['input_tokens'] += int(input_match.group(1))
                        result['output_tokens'] += int(output_match.group(1))
                        result['api_calls'] += 1
        
        result['total_tokens'] = result['input_tokens'] + result['output_tokens']
        result['cost_usd'] = (
            result['input_tokens'] * PRICE_PER_INPUT_TOKEN +
            result['output_tokens'] * PRICE_PER_OUTPUT_TOKEN
        )
    except Exception as e:
        logging.warning(f"Error parsing log file: {e}")
    
    return result


def run_unified_ga(question: str, correct_answer: str, log_file: Path) -> dict:
    """
    Run GA-5gen and capture checkpoints at generations 0, 3, and 5.
    
    Returns:
        Dict with 'baseline', 'ga3', and 'ga5' results
    """
    from agents.supervisor_agent import SupervisorAgent
    import config
    
    # Override config for 5 generations
    config.NUM_GENERATIONS = 5
    config.POPULATION_SIZE = 6
    
    # Setup logging to file
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
        logging.getLogger().addHandler(file_handler)
    
    start_time = time.time()
    
    try:
        # Initialize and run GA with 5 generations
        supervisor = SupervisorAgent(
            research_goal=question,
            mode="general"
        )
        # Override config values on the instance as well
        supervisor.num_generations = 5
        supervisor.population_size = 6
        
        results = supervisor.run_genetic_algorithm()
        elapsed = time.time() - start_time
        
        # Extract results from generation_history
        generation_history = results.get('generation_history', [])
        
        # Initialize checkpoint results
        checkpoints = {
            'baseline': {'generation': 0, 'best_hypothesis': None},
            'ga3': {'generation': 3, 'best_hypothesis': None},
            'ga5': {'generation': 5, 'best_hypothesis': None}
        }
        
        # Map generation numbers to checkpoint names
        gen_to_checkpoint = {0: 'baseline', 3: 'ga3', 5: 'ga5'}
        
        for gen_stats in generation_history:
            gen_num = gen_stats.get('generation', -1)
            if gen_num in gen_to_checkpoint:
                checkpoint_name = gen_to_checkpoint[gen_num]
                checkpoints[checkpoint_name]['best_hypothesis'] = gen_stats.get('best_hypothesis')
        
        # Parse token usage from log
        token_usage = parse_token_usage_from_log(log_file)
        
        # Build results for each checkpoint
        output = {}
        extraction_errors = []

        for checkpoint_name, checkpoint_data in checkpoints.items():
            best_hyp = checkpoint_data.get('best_hypothesis', {})

            # Try to extract answer - raise error on failure
            try:
                predicted = extract_answer_from_hypothesis(best_hyp, raise_on_unknown=True)
            except AnswerExtractionError as extraction_err:
                # Log the extraction failure with details
                logging.error(
                    f"ANSWER EXTRACTION FAILED for {checkpoint_name} (gen {checkpoint_data['generation']}): "
                    f"{extraction_err}"
                )
                extraction_errors.append(f"{checkpoint_name}: {extraction_err}")

                # Fallback: try to extract from log file as last resort
                predicted = 'EXTRACTION_FAILED'
                if log_file and log_file.exists():
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            log_content = f.read()
                        # Find ALL FINAL_ANSWER occurrences
                        answers = re.findall(r'FINAL_ANSWER:\s*\[?([A-D])\]?', log_content, re.IGNORECASE)
                        if answers:
                            # Use the answer at appropriate position based on generation
                            if checkpoint_name == 'baseline' and len(answers) >= 1:
                                predicted = answers[0].upper()
                                logging.info(f"  Recovered answer from log for {checkpoint_name}: {predicted}")
                            elif checkpoint_name == 'ga3' and len(answers) >= 4:
                                # After gen 3, take answer around position 3-4
                                predicted = answers[min(3, len(answers)-1)].upper()
                                logging.info(f"  Recovered answer from log for {checkpoint_name}: {predicted}")
                            elif checkpoint_name == 'ga5' and answers:
                                # Take last answer for ga5
                                predicted = answers[-1].upper()
                                logging.info(f"  Recovered answer from log for {checkpoint_name}: {predicted}")
                    except Exception as log_err:
                        logging.warning(f"  Failed to extract from log file: {log_err}")

            is_correct = predicted == correct_answer.upper() if predicted in 'ABCD' else False

            # Estimate token usage per checkpoint (proportional to generations)
            if checkpoint_name == 'baseline':
                token_fraction = 0.15  # Initial population is ~15% of total
            elif checkpoint_name == 'ga3':
                token_fraction = 0.55  # Gen 0-3 is ~55% of total
            else:
                token_fraction = 1.0   # Full run

            output[checkpoint_name] = {
                'predicted': predicted,
                'correct': is_correct,
                'time_seconds': elapsed * token_fraction,
                'input_tokens': int(token_usage['input_tokens'] * token_fraction),
                'output_tokens': int(token_usage['output_tokens'] * token_fraction),
                'total_tokens': int(token_usage['total_tokens'] * token_fraction),
                'cost_usd': round(token_usage['cost_usd'] * token_fraction, 6),
                'api_calls': int(token_usage['api_calls'] * token_fraction),
                'generations': checkpoint_data['generation'],
                'best_hypothesis': best_hyp if best_hyp else {},
                'note': 'Captured from unified GA-5gen run'
            }

        # If any extraction failed, raise error to stop processing
        if extraction_errors and any(o['predicted'] == 'EXTRACTION_FAILED' for o in output.values()):
            raise AnswerExtractionError(
                f"Answer extraction failed for question. Errors: {'; '.join(extraction_errors)}"
            )

        return output

    except AnswerExtractionError:
        # Re-raise extraction errors to stop the process
        raise

    except Exception as e:
        logging.error(f"Error in unified GA: {e}")
        import traceback
        traceback.print_exc()
        elapsed = time.time() - start_time

        # Return error results for all checkpoints
        error_result = {
            'predicted': 'ERROR',
            'correct': False,
            'time_seconds': elapsed,
            'error': str(e),
            'best_hypothesis': {}
        }
        return {
            'baseline': error_result.copy(),
            'ga3': error_result.copy(),
            'ga5': error_result.copy()
        }
    finally:
        # Remove file handler
        if log_file:
            for handler in logging.getLogger().handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    if handler.baseFilename == str(log_file):
                        logging.getLogger().removeHandler(handler)
                        handler.close()


def save_results(data: dict, output_file: Path):
    """Save results to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)


def run_evaluation(start_idx: int, end_idx: int, model: str, output_file: Path = None):
    """Run unified evaluation on specified question range."""
    # Load dataset
    df = load_gpqa_dataset()
    
    # Clamp indices
    start_idx = max(0, start_idx)
    end_idx = min(len(df), end_idx)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Default output file
    if output_file is None:
        output_file = OUTPUT_DIR / f"gpqa_unified_{timestamp}.json"
    
    # Setup logging
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir / f"main_{timestamp}.log")
    
    logging.info(f"Starting GPQA Unified Evaluation")
    logging.info(f"Model: {model}")
    logging.info(f"Questions: {start_idx} to {end_idx-1}")
    
    # Initialize results structure
    results_data = {
        'metadata': {
            'timestamp': timestamp,
            'model': model,
            'question_range': f"{start_idx}-{end_idx-1}",
            'total_questions': end_idx - start_idx,
            'evaluation_type': 'unified_ga5gen'
        },
        'questions': [],
        'summary': {
            'baseline': {'correct': 0, 'total': 0, 'accuracy': 0, 'total_time': 0, 'total_cost': 0},
            'ga3': {'correct': 0, 'total': 0, 'accuracy': 0, 'total_time': 0, 'total_cost': 0},
            'ga5': {'correct': 0, 'total': 0, 'accuracy': 0, 'total_time': 0, 'total_cost': 0}
        }
    }
    
    # Process each question
    for idx in range(start_idx, end_idx):
        row = df.iloc[idx]
        question = row['question']
        correct_answer = row['answer'].strip().upper()
        
        logging.info(f"\n{'='*80}")
        logging.info(f"Question {idx+1-start_idx}/{end_idx-start_idx} (index {idx})")
        logging.info(f"Correct answer: {correct_answer}")
        logging.info(f"{'='*80}")
        
        # Create log file for this question
        log_file = log_dir / f"q{idx}_{timestamp}.log"

        # Run unified GA - with proper error handling
        try:
            checkpoint_results = run_unified_ga(question, correct_answer, log_file)
        except AnswerExtractionError as e:
            logging.error(f"FATAL: Answer extraction failed for question {idx}: {e}")
            logging.error("Stopping evaluation. Please check the pipeline output format.")
            # Save partial results before stopping
            save_results(results_data, output_file)
            raise  # Re-raise to stop the process

        # Build question result
        question_result = {
            'question_index': idx,
            'correct_answer': correct_answer,
            'question_preview': question[:200] + '...' if len(question) > 200 else question,
            'baseline': checkpoint_results.get('baseline', {}),
            'ga3': checkpoint_results.get('ga3', {}),
            'ga5': checkpoint_results.get('ga5', {})
        }
        
        # Log results
        for checkpoint in ['baseline', 'ga3', 'ga5']:
            result = checkpoint_results.get(checkpoint, {})
            status = "✓" if result.get('correct') else "✗"
            logging.info(f"  {checkpoint.upper()}: {status} - Predicted: {result.get('predicted', 'UNKNOWN')}")
        
        results_data['questions'].append(question_result)
        
        # Update summary
        for checkpoint in ['baseline', 'ga3', 'ga5']:
            result = checkpoint_results.get(checkpoint, {})
            results_data['summary'][checkpoint]['total'] += 1
            if result.get('correct'):
                results_data['summary'][checkpoint]['correct'] += 1
            results_data['summary'][checkpoint]['total_time'] += result.get('time_seconds', 0)
            results_data['summary'][checkpoint]['total_cost'] += result.get('cost_usd', 0)
        
        # Calculate accuracies
        for checkpoint in ['baseline', 'ga3', 'ga5']:
            total = results_data['summary'][checkpoint]['total']
            correct = results_data['summary'][checkpoint]['correct']
            results_data['summary'][checkpoint]['accuracy'] = round(correct / total * 100, 2) if total > 0 else 0
            results_data['summary'][checkpoint]['avg_time_per_q'] = round(
                results_data['summary'][checkpoint]['total_time'] / total, 2
            ) if total > 0 else 0
        
        # Save after each question
        save_results(results_data, output_file)
    
    # Final summary
    logging.info(f"\n{'='*80}")
    logging.info("EVALUATION COMPLETE")
    logging.info(f"{'='*80}")
    
    for checkpoint in ['baseline', 'ga3', 'ga5']:
        summary = results_data['summary'][checkpoint]
        logging.info(f"\n{checkpoint.upper()}:")
        logging.info(f"  Accuracy: {summary['accuracy']}% ({summary['correct']}/{summary['total']})")
        logging.info(f"  Total Time: {summary['total_time']:.1f}s")
        logging.info(f"  Total Cost: ${summary['total_cost']:.4f}")
    
    logging.info(f"\nResults saved to: {output_file}")
    
    return results_data


def main():
    parser = argparse.ArgumentParser(description='GPQA Unified Evaluation (GA-5gen with checkpoints)')
    parser.add_argument('--start', type=int, default=0, help='Start question index')
    parser.add_argument('--end', type=int, default=198, help='End question index (exclusive)')
    parser.add_argument('--model', type=str, default='gpt-5-mini', help='Model to use')
    parser.add_argument('--output-file', type=str, default=None, help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Set model in environment
    os.environ['OPENAI_MODEL'] = args.model
    
    # Also override in llm_client
    try:
        from external_tools import llm_client
        llm_client.MODEL_NAME = args.model
    except ImportError:
        pass
    
    output_file = Path(args.output_file) if args.output_file else None
    
    run_evaluation(
        start_idx=args.start,
        end_idx=args.end,
        model=args.model,
        output_file=output_file
    )


if __name__ == '__main__':
    main()
