#!/usr/bin/env python3
"""
GPQA Diamond Evaluation Script - Baseline + GA Comparison

Evaluates GPQA Diamond dataset (all questions) with:
1. Baseline: Single LLM call (no genetic algorithm)
2. GA-3gen: Pipeline + 3 generations
3. GA-5gen: Pipeline + 5 generations

Follows the same pattern as comprehensive_ga_evaluation.py.

Usage:
    # Run all evaluations on all questions
    python scripts/run_gpqa_ga_evaluation.py --all

    # Run only baseline on first 10 questions
    python scripts/run_gpqa_ga_evaluation.py --baseline --start 0 --end 10

    # Run baseline + GA-3gen on all questions
    python scripts/run_gpqa_ga_evaluation.py --baseline --ga3 --all

    # Specify model
    python scripts/run_gpqa_ga_evaluation.py --all --model gpt-5-mini
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
from typing import Dict, List, Optional

import pandas as pd

# Add pipeline to path
project_root = Path(__file__).parent.parent
pipeline_dir = project_root / "pipeline"
sys.path.insert(0, str(pipeline_dir))
sys.path.insert(0, str(project_root))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import OpenAI for baseline (direct API call like comprehensive_ga_evaluation.py)
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except ImportError:
    raise ImportError("OpenAI package not installed. Run: pip install openai")

# Default model
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5-mini")

# Pricing for gpt-5-mini (as of January 2025)
PRICE_PER_INPUT_TOKEN = 0.25 / 1000000   # $0.25 per 1M input tokens
PRICE_PER_OUTPUT_TOKEN = 2 / 1000000     # $2 per 1M output tokens

# Results directory
RESULTS_DIR = project_root / "output" / "gpqa"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(log_file: Optional[Path] = None):
    """Setup basic logging for the evaluation script."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='w'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True
    )


def load_gpqa_dataset() -> pd.DataFrame:
    """Load GPQA Diamond dataset from CSV."""
    data_path = project_root / "data" / "gpqa" / "gpqa_diamond.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"GPQA dataset not found at {data_path}. "
            f"Run 'python download_gpqa.py' to download."
        )
    
    df = pd.read_csv(data_path)
    logging.info(f"Loaded GPQA Diamond dataset: {len(df)} questions")
    return df


def format_question_prompt(question_text: str) -> str:
    """
    Format the GPQA question for LLM prompt.
    The question already contains the options (A, B, C, D).
    """
    return f"""You are an expert scientist with deep knowledge in physics, chemistry, and biology.
Analyze the following graduate-level multiple choice question and provide the best answer.

QUESTION:
{question_text}

INSTRUCTIONS:
1. Carefully read and understand the question
2. Consider each answer option systematically
3. Apply your scientific knowledge to evaluate the options
4. Select the single best answer

You MUST end your response with:
FINAL_ANSWER: [A, B, C, or D]

Your analysis:
"""


def extract_answer_from_text(text: str) -> str:
    """
    Extract answer choice (A, B, C, D) from response text.
    Tries multiple patterns to handle various LLM response formats.
    """
    # Pattern 1: FINAL_ANSWER: [X] format (with brackets)
    match = re.search(r'FINAL_ANSWER:\s*\[([A-D])\]', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 2: FINAL_ANSWER: X format (without brackets)
    match = re.search(r'FINAL_ANSWER:\s*([A-D])\b', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 2: JSON format {"Answer": "X"}
    match = re.search(r'"[Aa]nswer"\s*:\s*"([A-D])"', text)
    if match:
        return match.group(1).upper()
    
    # Pattern 3: "The answer is X" or "correct answer is X"
    match = re.search(r'(?:the\s+)?(?:correct\s+)?answer\s+is\s*:?\s*([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 4: Bold markdown **A** or **Option A**
    match = re.search(r'\*\*([A-D])\*\*', text)
    if match:
        return match.group(1).upper()
    
    # Pattern 5: Standalone letter in last lines
    lines = text.strip().split('\n')
    for line in reversed(lines[-5:]):
        line_clean = line.strip()
        if line_clean in ['A', 'B', 'C', 'D']:
            return line_clean
        # Check for "A)" or "(A)" format
        match = re.search(r'^([A-D])\)|\(([A-D])\)$', line_clean)
        if match:
            return (match.group(1) or match.group(2)).upper()
    
    return 'UNKNOWN'


def run_baseline(question: str, correct_answer: str, 
                 log_file: Optional[Path] = None) -> Dict:
    """
    Run baseline evaluation: single LLM API call.
    Matches the pattern from comprehensive_ga_evaluation.py.
    
    Args:
        question: The question text (includes options)
        correct_answer: The correct answer letter (A, B, C, or D)
        log_file: Optional log file path
    
    Returns:
        Dictionary with evaluation results
    """
    logging.info(f"  Running BASELINE ({MODEL_NAME}, single call)...")
    
    prompt = format_question_prompt(question)
    
    # Write to log file if provided
    if log_file:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"BASELINE EVALUATION\n")
            f.write(f"Model: {MODEL_NAME}\n")
            f.write("="*80 + "\n\n")
            f.write("QUESTION:\n")
            f.write(prompt + "\n\n")
            f.write("="*80 + "\n")
    
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant with expertise in scientific research."},
                {"role": "user", "content": prompt}
            ],
            # max_tokens=2000,
            # temperature=0.7
        )
        
        elapsed = time.time() - start_time
        
        # Extract response and usage
        response_text = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        
        # Calculate cost
        cost = (input_tokens * PRICE_PER_INPUT_TOKEN +
                output_tokens * PRICE_PER_OUTPUT_TOKEN)
        
        # Extract predicted answer
        predicted = extract_answer_from_text(response_text)
        is_correct = predicted == correct_answer.upper()
        
        # Log to file if provided
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\nAPI RESPONSE:\n")
                f.write(response_text + "\n\n")
                f.write("="*80 + "\n")
                f.write(f"TOKEN_USAGE: input={input_tokens}, output={output_tokens}, total={total_tokens}, cost=${cost:.6f}\n")
                f.write(f"PREDICTED_ANSWER: {predicted}\n")
                f.write(f"CORRECT_ANSWER: {correct_answer}\n")
                f.write(f"RESULT: {'CORRECT' if is_correct else 'INCORRECT'}\n")
                f.write(f"TIME_ELAPSED: {elapsed:.2f} seconds\n")
                f.write("="*80 + "\n")
        
        return {
            'predicted': predicted,
            'correct': is_correct,
            'time_seconds': elapsed,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'cost_usd': cost,
            'api_calls': 1,
            'response_text': response_text
        }
        
    except Exception as e:
        logging.error(f"  Baseline failed: {e}")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\nERROR: {str(e)}\n")
        return {
            'predicted': 'ERROR',
            'correct': False,
            'time_seconds': time.time() - start_time,
            'input_tokens': 0,
            'output_tokens': 0,
            'cost_usd': 0.0,
            'api_calls': 0,
            'error': str(e)
        }


def run_pipeline_ga(question: str, correct_answer: str, 
                    num_generations: int,
                    log_file: Optional[Path] = None) -> Dict:
    """
    Run Pipeline genetic algorithm evaluation.
    Matches the pattern from comprehensive_ga_evaluation.py.
    
    Args:
        question: The question text (includes options)
        correct_answer: The correct answer letter
        num_generations: Number of GA generations (3 or 5)
        log_file: Optional log file path
    
    Returns:
        Dictionary with evaluation results
    """
    logging.info(f"  Running GA-{num_generations}gen ({MODEL_NAME}, {num_generations} generations)...")
    
    # Import Pipeline modules
    import config as pipeline_config
    from agents.supervisor_agent import SupervisorAgent
    from external_tools import llm_client
    
    # Set model
    llm_client.MODEL_NAME = MODEL_NAME
    
    # Format the research goal (question)
    research_goal = format_question_prompt(question)
    
    # Override config for this run
    original_generations = pipeline_config.NUM_GENERATIONS
    pipeline_config.NUM_GENERATIONS = num_generations
    
    # Setup log file if provided
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        pipeline_config.setup_logging(str(log_file))
    
    start_time = time.time()
    
    try:
        # Initialize and run Pipeline
        supervisor = SupervisorAgent(
            research_goal=research_goal,
            mode="general"  # MCQ mode
        )
        
        results = supervisor.run_genetic_algorithm()
        elapsed = time.time() - start_time
        
        # Extract predicted answer from best hypothesis
        best_hypothesis = results.get('best_hypothesis', {}) if results else {}

        try:
            # Try with raise_on_unknown=True first to get proper errors
            predicted = extract_answer_from_hypothesis(best_hypothesis, raise_on_unknown=True)
        except AnswerExtractionError as extraction_err:
            logging.warning(f"    Primary extraction failed: {extraction_err}")
            predicted = 'UNKNOWN'

            # Fallback: if best_hypothesis is empty (GA crashed), try extracting from log file
            if log_file and log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_content = f.read()
                    # Find all FINAL_ANSWER occurrences and take the last one
                    answers = re.findall(r'FINAL_ANSWER:\s*\[?([A-D])\]?', log_content, re.IGNORECASE)
                    if answers:
                        predicted = answers[-1].upper()
                        logging.info(f"    Extracted answer from log file: {predicted}")
                except Exception as e:
                    logging.warning(f"    Failed to extract from log: {e}")

            # If still UNKNOWN, raise error to stop processing
            if predicted == 'UNKNOWN':
                raise AnswerExtractionError(
                    f"FATAL: Could not extract answer for GA-{num_generations}gen. "
                    f"Original error: {extraction_err}"
                )

        is_correct = predicted == correct_answer.upper()
        
        # Parse real token usage from log file
        token_usage = parse_token_usage_from_log(log_file)
        
        # If log parsing succeeded, use real values
        if token_usage['api_calls'] > 0:
            return {
                'predicted': predicted,
                'correct': is_correct,
                'time_seconds': elapsed,
                'input_tokens': token_usage['input_tokens'],
                'output_tokens': token_usage['output_tokens'],
                'total_tokens': token_usage['total_tokens'],
                'cost_usd': token_usage['cost_usd'],
                'api_calls': token_usage['api_calls'],
                'generations': num_generations,
                'best_hypothesis': best_hypothesis,
                'note': 'Token usage from real API logs'
            }
        else:
            # Fallback: estimate tokens
            estimated_input = estimate_input_tokens(num_generations)
            estimated_output = estimate_output_tokens(num_generations)
            cost = (estimated_input * PRICE_PER_INPUT_TOKEN +
                    estimated_output * PRICE_PER_OUTPUT_TOKEN)
            
            return {
                'predicted': predicted,
                'correct': is_correct,
                'time_seconds': elapsed,
                'input_tokens': estimated_input,
                'output_tokens': estimated_output,
                'cost_usd': cost,
                'api_calls': estimate_api_calls(num_generations),
                'generations': num_generations,
                'best_hypothesis': best_hypothesis,
                'note': 'Token usage estimated (no log data)'
            }
        
    except Exception as e:
        logging.error(f"  GA-{num_generations}gen failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'predicted': 'ERROR',
            'correct': False,
            'time_seconds': time.time() - start_time,
            'input_tokens': 0,
            'output_tokens': 0,
            'cost_usd': 0.0,
            'api_calls': 0,
            'generations': num_generations,
            'error': str(e)
        }
    
    finally:
        # Restore original config
        pipeline_config.NUM_GENERATIONS = original_generations


class AnswerExtractionError(Exception):
    """Raised when answer extraction fails."""
    pass


def extract_answer_from_hypothesis(hypothesis: Dict, raise_on_unknown: bool = False) -> str:
    """
    Extract answer from Pipeline's best hypothesis.
    Searches through all fields of the hypothesis, prioritizing final_answer field.

    Args:
        hypothesis: The hypothesis dict from the pipeline
        raise_on_unknown: If True, raise AnswerExtractionError instead of returning 'UNKNOWN'

    Returns:
        The extracted answer letter (A, B, C, or D)
    """
    if not hypothesis:
        if raise_on_unknown:
            raise AnswerExtractionError("Hypothesis is empty or None")
        return 'UNKNOWN'

    # FIRST: Check for direct final_answer field (most reliable - this is what pipeline sets)
    final_answer = hypothesis.get('final_answer')
    if final_answer:
        answer = str(final_answer).strip().upper()
        if len(answer) == 1 and answer in 'ABCD':
            logging.debug(f"Found answer in final_answer field: {answer}")
            return answer
        # Try to extract letter from longer string
        match = re.search(r'\b([A-D])\b', answer)
        if match:
            return match.group(1).upper()

    # Collect all text fields from the hypothesis
    text_fields = []

    # Add title, description, testability_notes
    for key in ['title', 'description', 'testability_notes', 'evolution_justification',
                'rationale', 'summary', 'hypothesis_statement']:
        value = hypothesis.get(key, '')
        if value:
            text_fields.append(str(value))

    # Add reviews
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

    # If UNKNOWN, try JSON pattern
    if answer == 'UNKNOWN':
        json_pattern = r'"Answer"\s*:\s*"([A-D])"'
        matches = re.findall(json_pattern, combined_text, re.IGNORECASE)
        if matches:
            answer = matches[-1].upper()

    if answer == 'UNKNOWN' and raise_on_unknown:
        preview = combined_text[:200] if combined_text else "No text found"
        raise AnswerExtractionError(
            f"Could not extract answer from hypothesis. "
            f"final_answer field: {final_answer}, "
            f"Text preview: {preview}..."
        )

    return answer


def estimate_input_tokens(num_generations: int) -> int:
    """Estimate input tokens for Pipeline based on generations."""
    population_size = 6  # Default
    base_tokens = population_size * 2000  # Initial generation
    per_gen_tokens = population_size * 1500 + 4 * 1200  # Reflections + Evolution
    return base_tokens + (num_generations * per_gen_tokens)


def estimate_output_tokens(num_generations: int) -> int:
    """Estimate output tokens for Pipeline based on generations."""
    population_size = 6
    base_tokens = population_size * 1500
    per_gen_tokens = population_size * 2000 + 4 * 1500
    return base_tokens + (num_generations * per_gen_tokens)


def estimate_api_calls(num_generations: int) -> int:
    """Estimate number of API calls for Pipeline."""
    population_size = 6
    initial_calls = population_size * 2
    per_gen_calls = population_size + 4
    return initial_calls + (num_generations * per_gen_calls)


def parse_token_usage_from_log(log_file: Optional[Path]) -> Dict:
    """Parse real token usage from log file."""
    if not log_file or not log_file.exists():
        return {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
            'cost_usd': 0.0,
            'api_calls': 0
        }
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern: TOKEN_USAGE: input=123, output=456, total=579, cost=$0.001234
    pattern = r'TOKEN_USAGE: input=(\d+), output=(\d+), total=(\d+), cost=\$(\d+\.\d+)'
    matches = re.findall(pattern, content)
    
    if not matches:
        return {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
            'cost_usd': 0.0,
            'api_calls': 0
        }
    
    total_input = sum(int(m[0]) for m in matches)
    total_output = sum(int(m[1]) for m in matches)
    total_tokens = sum(int(m[2]) for m in matches)
    total_cost = sum(float(m[3]) for m in matches)
    
    return {
        'input_tokens': total_input,
        'output_tokens': total_output,
        'total_tokens': total_tokens,
        'cost_usd': total_cost,
        'api_calls': len(matches)
    }


def save_results(results_data: Dict, file_path: Path):
    """Save results to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    logging.info(f"Results saved to: {file_path}")


def load_existing_results(results_file: Path) -> Optional[Dict]:
    """Load existing results for resume."""
    if results_file.exists():
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load existing results: {e}")
    return None


def is_approach_complete(approach_result: Dict) -> bool:
    """Check if a single approach completed successfully."""
    if not approach_result:
        return False
    if approach_result.get('predicted') in ['ERROR', 'UNKNOWN', None, '']:
        return False
    return True


def run_comprehensive_evaluation(start_idx: int = 0, end_idx: int = None,
                                  run_baseline_flag: bool = True,
                                  run_ga3_flag: bool = True,
                                  run_ga5_flag: bool = True,
                                  model: str = None,
                                  output_file: str = None):
    """
    Main evaluation function - runs all three approaches.
    Matches the pattern from comprehensive_ga_evaluation.py.
    """
    global MODEL_NAME
    if model:
        MODEL_NAME = model
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup logging
    log_file = RESULTS_DIR / f"gpqa_eval_{timestamp}.log"
    setup_logging(log_file)
    
    logging.info("="*80)
    logging.info("GPQA DIAMOND EVALUATION")
    logging.info(f"Model: {MODEL_NAME}")
    logging.info(f"Timestamp: {timestamp}")
    logging.info("="*80)
    
    # Load dataset
    df = load_gpqa_dataset()
    
    if end_idx is None:
        end_idx = len(df)
    
    start_idx = max(0, start_idx)
    end_idx = min(end_idx, len(df))
    
    logging.info(f"Evaluating questions {start_idx} to {end_idx-1} ({end_idx - start_idx} total)")
    logging.info(f"Baseline: {run_baseline_flag}, GA-3gen: {run_ga3_flag}, GA-5gen: {run_ga5_flag}\n")
    
    # Results file
    if output_file:
        results_file = Path(output_file)
        results_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        results_file = RESULTS_DIR / f"gpqa_evaluation_{timestamp}.json"
    
    # Initialize results structure
    results_data = {
        'metadata': {
            'timestamp': timestamp,
            'model': MODEL_NAME,
            'question_range': f"{start_idx}-{end_idx-1}",
            'total_questions': end_idx - start_idx,
        },
        'questions': [],
        'summary': {
            'baseline': {'correct': 0, 'total': 0, 'accuracy': 0, 'total_time': 0, 'total_cost': 0},
            'ga_3gen': {'correct': 0, 'total': 0, 'accuracy': 0, 'total_time': 0, 'total_cost': 0},
            'ga_5gen': {'correct': 0, 'total': 0, 'accuracy': 0, 'total_time': 0, 'total_cost': 0}
        }
    }
    
    # Process each question
    for idx in range(start_idx, end_idx):
        row = df.iloc[idx]
        question = row['question']
        correct_answer = row['answer']
        
        logging.info(f"\n{'='*80}")
        logging.info(f"Question {idx + 1}/{end_idx} (index {idx})")
        logging.info(f"Correct answer: {correct_answer}")
        logging.info(f"{'='*80}")
        
        question_result = {
            'question_index': idx,
            'correct_answer': correct_answer,
            'question_preview': question[:200] + "..." if len(question) > 200 else question,
            'baseline': {},
            'ga_3gen': {},
            'ga_5gen': {}
        }
        
        # 1. Run baseline
        if run_baseline_flag:
            log_path = RESULTS_DIR / f"logs/baseline/q{idx}_{timestamp}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            baseline_result = run_baseline(question, correct_answer, log_path)
            question_result['baseline'] = baseline_result
            
            status = "✓" if baseline_result['correct'] else "✗"
            logging.info(f"  Baseline: {status} - Predicted: {baseline_result['predicted']} - {baseline_result['time_seconds']:.1f}s")
        
        # 2. Run GA with 3 generations
        if run_ga3_flag:
            log_path = RESULTS_DIR / f"logs/ga3gen/q{idx}_{timestamp}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            ga3_result = run_pipeline_ga(question, correct_answer, 3, log_path)
            question_result['ga_3gen'] = ga3_result
            
            status = "✓" if ga3_result['correct'] else "✗"
            logging.info(f"  GA-3gen: {status} - Predicted: {ga3_result['predicted']} - {ga3_result['time_seconds']:.1f}s")
        
        # 3. Run GA with 5 generations
        if run_ga5_flag:
            log_path = RESULTS_DIR / f"logs/ga5gen/q{idx}_{timestamp}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            ga5_result = run_pipeline_ga(question, correct_answer, 5, log_path)
            question_result['ga_5gen'] = ga5_result
            
            status = "✓" if ga5_result['correct'] else "✗"
            logging.info(f"  GA-5gen: {status} - Predicted: {ga5_result['predicted']} - {ga5_result['time_seconds']:.1f}s")
        
        results_data['questions'].append(question_result)
        
        # Calculate summary statistics after each question (for live updates)
        for approach in ['baseline', 'ga_3gen', 'ga_5gen']:
            correct = sum(1 for q in results_data['questions'] if q[approach].get('correct', False))
            total = sum(1 for q in results_data['questions'] if q[approach])
            total_time = sum(q[approach].get('time_seconds', 0) for q in results_data['questions'])
            total_cost = sum(q[approach].get('cost_usd', 0) for q in results_data['questions'])
            
            results_data['summary'][approach] = {
                'correct': correct,
                'total': total,
                'accuracy': round(correct / total * 100, 2) if total > 0 else 0,
                'total_time': round(total_time, 2),
                'total_cost': round(total_cost, 6),
                'avg_time_per_q': round(total_time / total, 2) if total > 0 else 0
            }
        
        # Save after each question (for resume)
        save_results(results_data, results_file)
    
    # Calculate summary statistics
    for approach in ['baseline', 'ga_3gen', 'ga_5gen']:
        correct = sum(1 for q in results_data['questions'] if q[approach].get('correct', False))
        total = sum(1 for q in results_data['questions'] if q[approach])
        total_time = sum(q[approach].get('time_seconds', 0) for q in results_data['questions'])
        total_cost = sum(q[approach].get('cost_usd', 0) for q in results_data['questions'])
        
        results_data['summary'][approach] = {
            'correct': correct,
            'total': total,
            'accuracy': round(correct / total * 100, 2) if total > 0 else 0,
            'total_time': round(total_time, 2),
            'total_cost': round(total_cost, 6),
            'avg_time_per_q': round(total_time / total, 2) if total > 0 else 0
        }
    
    # Final save
    save_results(results_data, results_file)
    
    # Print summary
    logging.info(f"\n{'='*80}")
    logging.info("EVALUATION COMPLETE")
    logging.info(f"{'='*80}")
    
    for approach in ['baseline', 'ga_3gen', 'ga_5gen']:
        stats = results_data['summary'][approach]
        if stats['total'] > 0:
            logging.info(f"\n{approach.upper()}:")
            logging.info(f"  Accuracy: {stats['accuracy']}% ({stats['correct']}/{stats['total']})")
            logging.info(f"  Total Time: {stats['total_time']:.1f}s")
            logging.info(f"  Total Cost: ${stats['total_cost']:.4f}")
    
    logging.info(f"\nResults saved to: {results_file}")
    
    return results_data


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GPQA Diamond with baseline and GA approaches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all approaches on all questions
  python scripts/run_gpqa_ga_evaluation.py --all
  
  # Run only baseline on first 10 questions  
  python scripts/run_gpqa_ga_evaluation.py --baseline --start 0 --end 10
  
  # Run baseline + GA-3gen on all questions
  python scripts/run_gpqa_ga_evaluation.py --baseline --ga3 --all
  
  # Specify model
  python scripts/run_gpqa_ga_evaluation.py --all --model gpt-5-mini
        """
    )
    
    # Method selection
    parser.add_argument("--baseline", action="store_true", help="Run baseline (single LLM call)")
    parser.add_argument("--ga3", action="store_true", help="Run GA with 3 generations")
    parser.add_argument("--ga5", action="store_true", help="Run GA with 5 generations")
    parser.add_argument("--all-methods", action="store_true", help="Run all three methods")
    
    # Question selection
    parser.add_argument("--index", type=int, help="Evaluate single question by index")
    parser.add_argument("--start", type=int, default=0, help="Starting question index")
    parser.add_argument("--end", type=int, default=None, help="Ending question index (exclusive)")
    parser.add_argument("--all", action="store_true", help="Evaluate all questions")
    
    # Model and output
    parser.add_argument("--model", type=str, default=None, help="Model to use (default: gpt-5-mini)")
    parser.add_argument("--output-file", type=str, default=None, help="Output file path (for parallel runs)")
    
    args = parser.parse_args()
    
    # Determine which methods to run
    if args.all_methods:
        run_baseline_flag = True
        run_ga3_flag = True
        run_ga5_flag = True
    else:
        run_baseline_flag = args.baseline
        run_ga3_flag = args.ga3
        run_ga5_flag = args.ga5
    
    # If no method specified, run all
    if not (run_baseline_flag or run_ga3_flag or run_ga5_flag):
        run_baseline_flag = True
        run_ga3_flag = True
        run_ga5_flag = True
    
    # Determine question range
    if args.index is not None:
        start_idx = args.index
        end_idx = args.index + 1
    elif args.all:
        start_idx = 0
        end_idx = None
    else:
        start_idx = args.start
        end_idx = args.end if args.end else args.start + 10
    
    # Run evaluation
    results = run_comprehensive_evaluation(
        start_idx=start_idx,
        end_idx=end_idx,
        run_baseline_flag=run_baseline_flag,
        run_ga3_flag=run_ga3_flag,
        run_ga5_flag=run_ga5_flag,
        model=args.model,
        output_file=args.output_file
    )
    
    # Print final summary
    print(f"\n{'='*60}")
    print("GPQA EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {results['metadata']['model']}")
    print(f"Questions: {results['metadata']['question_range']}")
    
    for approach in ['baseline', 'ga_3gen', 'ga_5gen']:
        stats = results['summary'][approach]
        if stats['total'] > 0:
            print(f"\n{approach.upper()}:")
            print(f"  Accuracy: {stats['accuracy']}% ({stats['correct']}/{stats['total']})")
            print(f"  Avg time/question: {stats['avg_time_per_q']}s")
            print(f"  Total cost: ${stats['total_cost']:.4f}")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
