#!/usr/bin/env python3
"""
Comprehensive GA Evaluation Script

Evaluates 30 GPQA questions (10 Biology, 10 Chemistry, 10 Physics) with:
1. Baseline: gpt-4o-mini (no genetic algorithm)
2. GA-3gen: Pipeline3 + gpt-4o-mini + 3 generations
3. GA-5gen: Pipeline3 + gpt-4o-mini + 5 generations

Collects accuracy, time, tokens, and costs for all approaches.
"""

import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class AnswerExtractionError(Exception):
    """Raised when answer extraction fails and we should stop processing."""
    pass

# Add pipeline3 to path
pipeline3_dir = Path(__file__).parent.parent / "pipeline3"
sys.path.insert(0, str(pipeline3_dir))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import OpenAI for baseline
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except ImportError:
    raise ImportError("OpenAI package not installed. Run: pip install openai")

# Set model name BEFORE importing Pipeline3 (so it reads the correct model)
MODEL_NAME = "gpt-5-mini"
os.environ['OPENAI_MODEL'] = MODEL_NAME

# Import Pipeline3 modules
import pipeline3.config as config
from pipeline3.agents.supervisor_agent import SupervisorAgent
# Also directly update the gpt4o module's MODEL_NAME to ensure it's correct
from pipeline3.external_tools import gpt4o
gpt4o.MODEL_NAME = MODEL_NAME

# Pricing for gpt-5o-mini (as of January 2025)
PRICE_PER_INPUT_TOKEN_MINI = 0.25 / 1000000   # $0.25 per 1M input tokens
PRICE_PER_OUTPUT_TOKEN_MINI = 2 / 1000000  # $2 per 1M output tokens

# Results directory
RESULTS_DIR = Path("trajectories/GPQA/GPT5_Mini_New")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Find existing results file or create new timestamp
existing_files = sorted(RESULTS_DIR.glob("evaluation_results_30q_*.json"))
if existing_files:
    # Reuse existing file's timestamp for resume
    RESULTS_FILE = existing_files[-1]  # Use most recent
    # Extract timestamp from filename: evaluation_results_30q_20251008_140903.json -> 20251008_140903
    parts = RESULTS_FILE.stem.split('_')
    TIMESTAMP = '_'.join(parts[-2:])  # Get last two parts: date_time
    print(f"Found existing results file: {RESULTS_FILE.name}")
    print(f"Resuming with timestamp: {TIMESTAMP}")
else:
    # Create new timestamp
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_FILE = RESULTS_DIR / f"evaluation_results_30q_{TIMESTAMP}.json"
    print(f"Starting new evaluation with timestamp: {TIMESTAMP}")


def setup_logging():
    """Setup basic logging for the evaluation script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_30_questions(questions_file: Path) -> List[Dict]:
    """Load the 30 selected questions from JSON file."""
    with open(questions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['questions']


def format_question_as_mcq(question_data: Dict) -> Tuple[str, str]:
    """
    Format question as multiple choice with options A, B, C, D.
    Returns (formatted_question, correct_answer_letter)
    """
    question_text = question_data['question']
    correct_answer = question_data['correct_answer']
    incorrect_answers = question_data['incorrect_answers']

    # Combine all answers (correct first, then incorrect)
    all_answers = [correct_answer] + incorrect_answers

    # Format as MCQ
    formatted = f"""Answer this scientific question by selecting the correct option (A, B, C, or D):

Question: {question_text}

Options:
A) {all_answers[0]}
B) {all_answers[1]}
C) {all_answers[2]}
D) {all_answers[3]}

You must respond in JSON format with the following structure:
{{
  "Answer": "A or B or C or D",
  "Reasoning": "your reasoning here"
}}"""

    # Correct answer is always A (since we put it first)
    # YOU NEED TO CHANGE THIS PART TO RETURN THE CORRECT ANSWER FOR FULL GPQA EVALUATION
    return formatted, 'A'


def run_baseline(question_data: Dict, log_file: Optional[Path] = None) -> Dict:
    """
    Run baseline evaluation: single gpt-4o-mini API call.

    Args:
        question_data: Question dictionary
        log_file: Optional log file path for baseline execution

    Returns dict with:
        - predicted: str (A/B/C/D)
        - correct: bool
        - time_seconds: float
        - input_tokens: int
        - output_tokens: int
        - cost_usd: float
        - api_calls: int (always 1)
    """
    logging.info(f"  Running BASELINE (gpt-4o-mini, single call)...")

    formatted_question, correct_answer = format_question_as_mcq(question_data)

    # If log file provided, write baseline execution details
    if log_file:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"BASELINE EVALUATION - {question_data['record_id']}\n")
            f.write(f"Model: {MODEL_NAME}\n")
            f.write(f"Domain: {question_data.get('domain', 'Unknown')}\n")
            f.write("="*80 + "\n\n")
            f.write("QUESTION:\n")
            f.write(formatted_question + "\n\n")
            f.write("="*80 + "\n")

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant with expertise in scientific research. Answer the multiple choice question by selecting A, B, C, or D. You must respond in JSON format with the structure: {\"Answer\": \"A or B or C or D\", \"Reasoning\": \"your reasoning\"}."},
                {"role": "user", "content": formatted_question}
            ],
            max_tokens=1000,
            temperature=0.0  # Deterministic
        )

        elapsed = time.time() - start_time

        # Extract response and usage
        response_text = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        # Calculate cost
        cost = (input_tokens * PRICE_PER_INPUT_TOKEN_MINI +
                output_tokens * PRICE_PER_OUTPUT_TOKEN_MINI)

        # Extract predicted answer (A, B, C, or D)
        predicted = extract_answer_from_text(response_text)

        # Log to file if provided
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\nAPI RESPONSE:\n")
                f.write(response_text + "\n\n")
                f.write("="*80 + "\n")
                f.write(f"TOKEN_USAGE: input={input_tokens}, output={output_tokens}, total={total_tokens}, cost=${cost:.6f}\n")
                f.write(f"PREDICTED_ANSWER: {predicted}\n")
                f.write(f"CORRECT_ANSWER: {correct_answer}\n")
                f.write(f"RESULT: {'CORRECT' if predicted == correct_answer else 'INCORRECT'}\n")
                f.write(f"TIME_ELAPSED: {elapsed:.2f} seconds\n")
                f.write("="*80 + "\n")

        return {
            'predicted': predicted,
            'correct': (predicted == correct_answer),
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


def run_pipeline3_ga(question_data: Dict, num_generations: int, log_file: Optional[Path] = None) -> Dict:
    """
    Run Pipeline3 genetic algorithm evaluation.

    Args:
        question_data: Question dictionary
        num_generations: Number of GA generations (3 or 5)
        log_file: Optional log file path

    Returns dict with:
        - predicted: str (A/B/C/D)
        - correct: bool
        - time_seconds: float
        - input_tokens: int (estimated)
        - output_tokens: int (estimated)
        - cost_usd: float
        - api_calls: int
        - generations: int
    """
    logging.info(f"  Running GA-{num_generations}gen (gpt-4o-mini, {num_generations} generations)...")

    formatted_question, correct_answer = format_question_as_mcq(question_data)

    # Override config for this run
    original_generations = config.NUM_GENERATIONS
    config.NUM_GENERATIONS = num_generations

    # Setup log file if provided
    if log_file:
        config.setup_logging(str(log_file))

    start_time = time.time()

    try:
        # Set environment variable and module variable to use correct model
        os.environ['OPENAI_MODEL'] = MODEL_NAME
        gpt4o.MODEL_NAME = MODEL_NAME

        # Initialize and run Pipeline3
        supervisor = SupervisorAgent(
            research_goal=formatted_question,
            mode="general"  # MCQ mode
        )

        results = supervisor.run_genetic_algorithm()

        elapsed = time.time() - start_time

        # Extract predicted answer from best hypothesis
        best_hypothesis = results.get('best_hypothesis', {})
        predicted = extract_answer_from_pipeline3_result(best_hypothesis)

        # Parse real token usage from log file
        token_usage = parse_token_usage_from_log(log_file)

        # If log parsing failed (old logs), fall back to estimates
        if token_usage['api_calls'] == 0:
            logging.warning(f"  No token usage found in log, using estimates")
            estimated_input_tokens = estimate_input_tokens(num_generations)
            estimated_output_tokens = estimate_output_tokens(num_generations)
            cost = (estimated_input_tokens * PRICE_PER_INPUT_TOKEN_MINI +
                    estimated_output_tokens * PRICE_PER_OUTPUT_TOKEN_MINI)
            api_calls = estimate_api_calls(num_generations)

            return {
                'predicted': predicted,
                'correct': (predicted == correct_answer),
                'time_seconds': elapsed,
                'input_tokens': estimated_input_tokens,
                'output_tokens': estimated_output_tokens,
                'cost_usd': cost,
                'api_calls': api_calls,
                'generations': num_generations,
                'best_hypothesis': best_hypothesis,
                'note': 'Token usage estimated (no log data)'
            }
        else:
            # Use real token counts from logs
            return {
                'predicted': predicted,
                'correct': (predicted == correct_answer),
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

    except AnswerExtractionError as e:
        # Answer extraction failed - this is a critical error that should stop processing
        logging.error(f"  GA-{num_generations}gen ANSWER EXTRACTION FAILED: {e}")
        raise  # Re-raise to stop execution - don't silently continue with UNKNOWN

    except Exception as e:
        logging.error(f"  GA-{num_generations}gen failed: {e}")
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
        config.NUM_GENERATIONS = original_generations


def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Extract JSON object from text, handling various formats:
    - Plain JSON: {"Answer": "A", "Reasoning": "..."}
    - Markdown code blocks: ```json\n{...}\n```
    - Text before JSON: "Here's the answer\n```json\n{...}\n```"
    - Multiple JSON blocks (returns the last one with Answer field)
    - Nested JSON structures
    """
    import re

    # Pattern to match JSON in markdown code blocks
    # (```json ... ``` or ``` ... ```)
    # Use non-greedy match but allow for nested braces
    json_block_pattern = r'```(?:json)?\s*(\{(?:[^{}]|\{[^{}]*\})*?\})\s*```'
    json_blocks = re.findall(
        json_block_pattern, text, re.DOTALL | re.IGNORECASE
    )

    if json_blocks:
        # Try to parse the last JSON block found
        for json_str in reversed(json_blocks):
            try:
                parsed = json.loads(json_str.strip())
                if isinstance(parsed, dict):
                    # Prefer JSON with Answer field
                    if 'Answer' in parsed or 'answer' in parsed or 'ANSWER' in parsed:
                        return parsed
                    # Otherwise return any dict (might be nested)
                    return parsed
            except json.JSONDecodeError:
                continue

    # Try to find JSON objects by looking for balanced braces
    # This is more robust for nested JSON
    brace_count = 0
    start_idx = -1
    json_candidates = []

    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                json_candidates.append((start_idx, i + 1))
                start_idx = -1

    # Try to parse JSON candidates, starting from the end
    # (most likely to be the answer)
    # First pass: look for JSON with Answer field
    for start, end in reversed(json_candidates):
        json_str = text[start:end]
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and (
                'Answer' in parsed or 'answer' in parsed or
                'ANSWER' in parsed
            ):
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Second pass: if no Answer field found, return any valid JSON dict
    # (might contain nested structures with Answer)
    for start, end in reversed(json_candidates):
        json_str = text[start:end]
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                # Recursively search nested structures
                def find_answer_in_dict(d):
                    if isinstance(d, dict):
                        if 'Answer' in d or 'answer' in d or 'ANSWER' in d:
                            return d
                        for v in d.values():
                            result = find_answer_in_dict(v)
                            if result:
                                return result
                    elif isinstance(d, list):
                        for item in d:
                            result = find_answer_in_dict(item)
                            if result:
                                return result
                    return None
                
                nested_answer = find_answer_in_dict(parsed)
                if nested_answer:
                    return nested_answer
        except json.JSONDecodeError:
            continue

    return None


def extract_answer_from_text(text: str) -> str:
    """
    Extract answer choice (A, B, C, D) from response text.
    First tries to parse JSON format, then falls back to regex patterns.
    """
    import re

    # First, try to extract JSON
    json_data = extract_json_from_text(text)
    if json_data:
        # Try different case variations of the Answer key
        answer = json_data.get('Answer') or json_data.get('answer') or json_data.get('ANSWER')
        if answer:
            # Extract just the letter (A, B, C, or D) if it's in a longer string
            answer_upper = str(answer).upper().strip()
            # Match A, B, C, or D (possibly with quotes or extra text)
            match = re.search(r'\b([A-D])\b', answer_upper)
            if match:
                return match.group(1)

    # Fallback to regex patterns if JSON parsing failed
    # Prioritized patterns (most specific first, search from end backwards)
    patterns = [
        # Bold markdown variants (highest priority)
        r'\*\*CORRECT OPTION:\s*([A-D])\*\*',                    # **Correct option: B**
        r'\*\*OPTION\s+([A-D])\*\*:',                            # **Option A**:
        r'\*\*([A-D])\)\s*[^*]*\*\*',                            # **B) option text**
        r'\*\*([A-D])\)',                                        # **B)**
        r'\*\*([A-D])\*\*',                                      # **B**

        # "Answer:" prefix patterns
        r'ANSWER:\s*([A-D])\)',                                  # Answer: B) text
        r'ANSWER:\s*([A-D])\b',                                  # Answer: B

        # "Correct" statements with various phrasings
        r'CORRECT\s+(?:ANSWER|CHOICE|OPTION)\s+IS\s*[:;\s]*([A-D])\)', # correct answer is B)
        r'CORRECT\s+(?:ANSWER|CHOICE|OPTION)\s+IS\s*[:;\s]*([A-D])\b', # correct answer is B
        r'(?:THE\s+)?CORRECT\s+(?:ANSWER|CHOICE|OPTION).*?([A-D])\)',   # the correct choice is B)

        # FINAL_ANSWER format (Pipeline3)
        r'FINAL_ANSWER:\s*([A-D])',

        # Plain answer statements
        r'(?:ANSWER|CHOICE)\s+IS\s*[:;\n\s]*\s*([A-D])\)',       # answer is B)
        r'(?:ANSWER|CHOICE)\s+IS\s*[:;\n\s]*\s*([A-D])\b',       # answer is B

        # Hence/Therefore statements
        r'(?:HENCE|THEREFORE|THUS),?\s+(?:THE\s+)?CORRECT\s+(?:ANSWER|CHOICE).*?([A-D])\)',

        # Selection statements
        r'(?:SELECT|CHOOSE)\s+([A-D])',
        r'OPTION\s+([A-D])',

        # Parenthetical (avoid matching option listings)
        r'[^\w]\(([A-D])\)',                                     # (B) but not at word start
        r'\s([A-D])\)',                                          # space + B)
    ]

    text_upper = text.upper()

    # Search all patterns from END to beginning (prefer final answer)
    for pattern in patterns:
        matches = list(re.finditer(pattern, text_upper, re.IGNORECASE | re.DOTALL))
        if matches:
            return matches[-1].group(1).upper()

    # Fallback: last standalone letter with word boundary
    # But be careful - avoid matching A in "Answer", "Analysis", etc.
    # Look for A/B/C/D that appears in answer-like contexts
    answer_context_patterns = [
        r'(?:option|choice|answer|select|correct)\s+([A-D])\b',  # "option A", "choice B"
        r'\b([A-D])\s*(?:is|are|would be|should be)\s+(?:the\s+)?(?:correct|right|answer)',  # "A is correct"
        r'\b([A-D])\s*\)',  # "A)" - option format
        r'\(\s*([A-D])\s*\)',  # "(A)" - parenthetical
    ]
    
    for pattern in answer_context_patterns:
        matches = list(re.finditer(pattern, text_upper, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).upper()
    
    # Last resort: look for standalone A/B/C/D but exclude common words
    # Exclude: Answer, Analysis, Because, Data, etc.
    exclude_words = r'(?:ANSWER|ANALYSIS|BECAUSE|DATA|CASE|CODE|DATE)'
    matches = list(re.finditer(
        r'\b([A-D])\b(?!\w)',  # A/B/C/D not followed by word char
        text_upper
    ))
    # Filter out matches that are part of excluded words
    valid_matches = [
        m for m in matches
        if not re.search(exclude_words, text_upper[max(0, m.start()-10):m.end()+10], re.IGNORECASE)
    ]
    if valid_matches:
        return valid_matches[-1].group(1)

    return 'UNKNOWN'


def extract_answer_from_pipeline3_result(hypothesis: Dict, raise_on_unknown: bool = True) -> str:
    """
    Extract answer from Pipeline3's best hypothesis.
    Searches comprehensively through all fields of the hypothesis.
    Prioritizes the direct final_answer field (set by pipeline).

    Args:
        hypothesis: The hypothesis dictionary from Pipeline3
        raise_on_unknown: If True, raises AnswerExtractionError when answer cannot be extracted.
                         If False, returns 'UNKNOWN' (legacy behavior).

    Returns:
        The extracted answer letter (A, B, C, or D)

    Raises:
        AnswerExtractionError: If raise_on_unknown=True and no answer could be extracted
    """
    if not hypothesis:
        if raise_on_unknown:
            raise AnswerExtractionError("Hypothesis is empty or None - cannot extract answer")
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

    # Add title and description
    title = hypothesis.get('title', '')
    description = hypothesis.get('description', '')
    text_fields.append(title)
    text_fields.append(description)

    # Add summary, rationale, hypothesis_statement
    for field in ['summary', 'rationale', 'hypothesis_statement']:
        value = hypothesis.get(field, '')
        if value:
            text_fields.append(str(value))

    # Add testability_notes
    testability = hypothesis.get('testability_notes', '')
    if testability:
        text_fields.append(testability)

    # Add reviews (list of strings)
    reviews = hypothesis.get('reviews', [])
    if isinstance(reviews, list):
        for review in reviews:
            if isinstance(review, str):
                text_fields.append(review)
    elif isinstance(reviews, str):
        text_fields.append(reviews)

    # Add evolution_justification
    evolution = hypothesis.get('evolution_justification', '')
    if evolution:
        text_fields.append(evolution)

    # Combine all text and search for answer
    combined_text = ' '.join(text_fields)

    # First try JSON extraction (most reliable)
    answer = extract_answer_from_text(combined_text)

    # If we got UNKNOWN, try a more aggressive search
    if answer == 'UNKNOWN':
        # Look for JSON blocks that might contain Answer field
        # Search for any JSON-like structure with Answer field
        # This pattern handles nested JSON better
        json_pattern = r'"Answer"\s*:\s*"([A-D])"'
        matches = re.findall(json_pattern, combined_text, re.IGNORECASE)
        if matches:
            # Take the last match (most recent answer)
            answer = matches[-1].upper()
            logging.debug(f"Found answer in JSON pattern: {answer}")

    # Handle extraction failure
    if answer == 'UNKNOWN':
        error_msg = (
            f"Could not extract answer from hypothesis.\n"
            f"final_answer field: {final_answer},\n"
            f"Title: {title[:100] if title else 'None'}...,\n"
            f"Description length: {len(description)},\n"
            f"Text preview: {combined_text[:200] if combined_text else 'None'}..."
        )
        if raise_on_unknown:
            raise AnswerExtractionError(error_msg)
        else:
            logging.warning(error_msg)
            return 'UNKNOWN'

    logging.debug(f"Extracted answer: {answer} from hypothesis {hypothesis.get('id', 'unknown')}")
    return answer


def estimate_input_tokens(num_generations: int) -> int:
    """Estimate input tokens for Pipeline3 based on generations."""
    # Rough estimate based on observed patterns:
    # - Initial generation: ~2000 tokens per hypothesis
    # - Reflection: ~1500 tokens per hypothesis per generation
    # - Evolution: ~1200 tokens per operation

    population_size = config.POPULATION_SIZE  # Usually 6
    base_tokens = population_size * 2000  # Initial generation

    # Per generation
    per_gen_tokens = (
        population_size * 1500 +  # Reflections
        4 * 1200  # Evolution operations (crossover + mutation)
    )

    return base_tokens + (num_generations * per_gen_tokens)


def estimate_output_tokens(num_generations: int) -> int:
    """Estimate output tokens for Pipeline3 based on generations."""
    # Rough estimate:
    # - Initial generation: ~1500 tokens per hypothesis
    # - Reflection: ~2000 tokens per hypothesis per generation
    # - Evolution: ~1500 tokens per operation

    population_size = config.POPULATION_SIZE
    base_tokens = population_size * 1500

    per_gen_tokens = (
        population_size * 2000 +  # Reflections
        4 * 1500  # Evolution operations
    )

    return base_tokens + (num_generations * per_gen_tokens)


def estimate_api_calls(num_generations: int) -> int:
    """Estimate number of API calls for Pipeline3."""
    # Rough estimate:
    # - Initial generation: 6 hypotheses + 6 reflections = 12
    # - Per generation: 6 reflections + 4 evolutions = 10

    initial_calls = config.POPULATION_SIZE * 2  # Generation + reflection
    per_gen_calls = config.POPULATION_SIZE + 4  # Reflections + evolutions

    return initial_calls + (num_generations * per_gen_calls)


def parse_token_usage_from_log(log_file: Path) -> Dict:
    """Parse real token usage from log file.

    Returns dict with:
        - input_tokens: int
        - output_tokens: int
        - total_tokens: int
        - cost_usd: float
        - api_calls: int
    """
    if not log_file.exists():
        return {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
            'cost_usd': 0.0,
            'api_calls': 0
        }

    import re

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern: TOKEN_USAGE: input=123, output=456, total=579, cost=$0.001234
    pattern = r'TOKEN_USAGE: input=(\d+), output=(\d+), total=(\d+), cost=\$(\d+\.\d+)'
    matches = re.findall(pattern, content)

    if not matches:
        # No token usage found (old logs without token tracking)
        return {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
            'cost_usd': 0.0,
            'api_calls': 0
        }

    # Sum up all API calls
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


def load_existing_results(results_file: Path) -> Dict:
    """Load existing results if available for resume."""
    if results_file.exists():
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load existing results: {e}")
    return None


def is_approach_complete(approach_result: Dict) -> bool:
    """Check if a single approach is completed successfully."""
    if not approach_result:
        return False
    # Check if the approach succeeded (has a valid predicted answer)
    if approach_result.get('predicted') in ['ERROR', 'UNKNOWN', None, '']:
        return False
    return True


def is_question_complete(question_result: Dict) -> bool:
    """Check if a question has all three approaches completed."""
    required_keys = ['baseline', 'ga_3gen', 'ga_5gen']
    for key in required_keys:
        if not is_approach_complete(question_result.get(key, {})):
            return False
    return True


def get_existing_question_result(results_data: Dict, record_id: str) -> Dict:
    """Get existing result for a question by record_id, or return None."""
    for q in results_data.get('questions', []):
        if q.get('record_id') == record_id:
            return q
    return None


def run_comprehensive_evaluation():
    """Main evaluation function."""
    setup_logging()

    logging.info("="*80)
    logging.info("COMPREHENSIVE GA EVALUATION - 30 GPQA Questions")
    logging.info(f"Model: {MODEL_NAME}")
    logging.info(f"Timestamp: {TIMESTAMP}")
    logging.info("="*80)

    # Load questions
    questions_file = RESULTS_DIR / "selected_30_questions.json"
    if not questions_file.exists():
        logging.error(f"Questions file not found: {questions_file}")
        logging.error("Please run scripts/select_25_questions.py first!")
        sys.exit(1)

    questions = load_30_questions(questions_file)
    logging.info(f"Loaded {len(questions)} questions\n")

    # Try to load existing results for resume
    existing_results = load_existing_results(RESULTS_FILE)

    if existing_results:
        logging.info(f"Found existing results file: {RESULTS_FILE}")
        results_data = existing_results
        completed_ids = {q['record_id'] for q in results_data['questions'] if is_question_complete(q)}
        logging.info(f"Already completed: {len(completed_ids)} questions")
        logging.info(f"Completed IDs: {sorted(completed_ids)}\n")
    else:
        logging.info("Starting fresh evaluation\n")
        # Initialize results structure
        results_data = {
            'metadata': {
                'timestamp': TIMESTAMP,
                'total_questions': len(questions),
                'base_model': MODEL_NAME,
                'pricing': {
                    'input_per_million': 0.15,
                    'output_per_million': 0.60
                }
            },
            'questions': [],
            'summary': {
                'baseline': {'accuracy': 0, 'total_time': 0, 'total_cost': 0, 'avg_time_per_q': 0},
                'ga_3gen': {'accuracy': 0, 'total_time': 0, 'total_cost': 0, 'avg_time_per_q': 0},
                'ga_5gen': {'accuracy': 0, 'total_time': 0, 'total_cost': 0, 'avg_time_per_q': 0}
            }
        }
        completed_ids = set()

    # Process each question
    for idx, question_data in enumerate(questions, 1):
        record_id = question_data['record_id']
        domain = question_data['domain']

        logging.info(f"\n{'='*80}")
        logging.info(f"Question {idx}/{len(questions)}: {record_id} ({domain})")
        logging.info(f"{'='*80}")

        # Skip if already fully completed
        if record_id in completed_ids:
            logging.info(f"  SKIPPING - Already completed all approaches")
            continue

        # Check if we have partial results for this question
        existing_result = get_existing_question_result(results_data, record_id)

        if existing_result:
            # Resume from incomplete question - check which approaches are done
            logging.info(f"  Found partial results - checking progress...")
            question_result = existing_result
        else:
            # New question - start fresh
            question_result = {
                'record_id': record_id,
                'domain': domain,
                'subdomain': question_data.get('subdomain', ''),
                'correct_answer': 'A',  # Always A in our format
                'baseline': {},
                'ga_3gen': {},
                'ga_5gen': {}
            }
            results_data['questions'].append(question_result)

        # 1. Run baseline (if not done)
        if is_approach_complete(question_result.get('baseline', {})):
            logging.info(f"  Baseline: ALREADY DONE - {question_result['baseline']['predicted']} ({'✓' if question_result['baseline']['correct'] else '✗'})")
        else:
            logging.info(f"  Running Baseline...")
            log_file_baseline = RESULTS_DIR / f"logs/baseline/{record_id}_{TIMESTAMP}.log"
            log_file_baseline.parent.mkdir(parents=True, exist_ok=True)
            baseline_result = run_baseline(question_data, log_file_baseline)
            question_result['baseline'] = baseline_result
            logging.info(f"  Baseline: {baseline_result['predicted']} ({'✓' if baseline_result['correct'] else '✗'}) - {baseline_result['time_seconds']:.1f}s - ${baseline_result['cost_usd']:.6f}")
            # Save after baseline
            save_results(results_data, RESULTS_FILE)

        # 2. Run GA with 3 generations (if not done)
        if is_approach_complete(question_result.get('ga_3gen', {})):
            logging.info(f"  GA-3gen: ALREADY DONE - {question_result['ga_3gen']['predicted']} ({'✓' if question_result['ga_3gen']['correct'] else '✗'})")
        else:
            logging.info(f"  Running GA-3gen...")
            log_file_3gen = RESULTS_DIR / f"logs/ga3gen/{record_id}_{TIMESTAMP}.log"
            log_file_3gen.parent.mkdir(parents=True, exist_ok=True)
            ga_3gen_result = run_pipeline3_ga(question_data, 3, log_file_3gen)
            question_result['ga_3gen'] = ga_3gen_result
            logging.info(f"  GA-3gen: {ga_3gen_result['predicted']} ({'✓' if ga_3gen_result['correct'] else '✗'}) - {ga_3gen_result['time_seconds']:.1f}s - ${ga_3gen_result['cost_usd']:.6f}")
            # Save after GA-3gen
            save_results(results_data, RESULTS_FILE)

        # 3. Run GA with 5 generations (if not done)
        if is_approach_complete(question_result.get('ga_5gen', {})):
            logging.info(f"  GA-5gen: ALREADY DONE - {question_result['ga_5gen']['predicted']} ({'✓' if question_result['ga_5gen']['correct'] else '✗'})")
        else:
            logging.info(f"  Running GA-5gen...")
            log_file_5gen = RESULTS_DIR / f"logs/ga5gen/{record_id}_{TIMESTAMP}.log"
            log_file_5gen.parent.mkdir(parents=True, exist_ok=True)
            ga_5gen_result = run_pipeline3_ga(question_data, 5, log_file_5gen)
            question_result['ga_5gen'] = ga_5gen_result
            logging.info(f"  GA-5gen: {ga_5gen_result['predicted']} ({'✓' if ga_5gen_result['correct'] else '✗'}) - {ga_5gen_result['time_seconds']:.1f}s - ${ga_5gen_result['cost_usd']:.6f}")
            # Save after GA-5gen
            save_results(results_data, RESULTS_FILE)

    # Calculate summary statistics
    for approach in ['baseline', 'ga_3gen', 'ga_5gen']:
        correct_count = sum(1 for q in results_data['questions'] if q[approach].get('correct', False))
        total_time = sum(q[approach].get('time_seconds', 0) for q in results_data['questions'])
        total_cost = sum(q[approach].get('cost_usd', 0) for q in results_data['questions'])

        results_data['summary'][approach] = {
            'accuracy': correct_count / len(questions),
            'total_time': total_time,
            'total_cost': total_cost,
            'avg_time_per_q': total_time / len(questions)
        }

    # Final save
    save_results(results_data, RESULTS_FILE)

    # Print final summary
    logging.info(f"\n{'='*80}")
    logging.info("EVALUATION COMPLETE")
    logging.info(f"{'='*80}")
    logging.info(f"\nResults Summary:")
    for approach, stats in results_data['summary'].items():
        logging.info(f"\n{approach.upper()}:")
        logging.info(f"  Accuracy: {stats['accuracy']:.1%} ({int(stats['accuracy']*len(questions))}/{len(questions)})")
        logging.info(f"  Total Time: {stats['total_time']:.1f}s ({stats['total_time']/3600:.2f}h)")
        logging.info(f"  Total Cost: ${stats['total_cost']:.4f}")
        logging.info(f"  Avg Time/Q: {stats['avg_time_per_q']:.1f}s")

    logging.info(f"\nDetailed results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    run_comprehensive_evaluation()
