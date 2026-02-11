#!/usr/bin/env python3
"""
GPQA Diamond Evaluation Script

This script evaluates the model's performance on GPQA Diamond dataset.
It uses a simpler single-call approach (vanilla baseline) rather than the full GA pipeline.

Usage:
    # Single question evaluation
    python scripts/run_gpqa_evaluation.py --question-index 0

    # Batch evaluation (10 questions starting from index 0)
    python scripts/run_gpqa_evaluation.py --batch-start 0 --batch-end 10

    # Full evaluation (all questions)
    python scripts/run_gpqa_evaluation.py --all

    # Specify model
    python scripts/run_gpqa_evaluation.py --all --model gpt-5-mini
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add pipeline to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "pipeline"))

from external_tools.llm_client import llm_generate, MODEL_NAME

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def load_gpqa_dataset() -> pd.DataFrame:
    """Load GPQA Diamond dataset."""
    data_path = project_root / "data" / "gpqa" / "gpqa_diamond.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"GPQA dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    logging.info(f"Loaded GPQA Diamond dataset: {len(df)} questions")
    return df


def generate_gpqa_prompt(question: str) -> str:
    """Generate prompt for a GPQA question."""
    return f"""You are an expert in science and reasoning. Carefully analyze the following question and provide the best answer.

QUESTION:
{question}

Please think through this step by step:
1. Identify what the question is asking
2. Consider the key concepts and principles involved
3. Evaluate each answer option
4. Select the best answer

After your analysis, you MUST end your response with:
FINAL_ANSWER: [A, B, C, or D]

Your analysis:
"""


def extract_answer(response: str) -> str:
    """Extract the final answer from the response."""
    # Look for FINAL_ANSWER pattern
    match = re.search(r'FINAL_ANSWER:\s*([A-D])', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback: look for standalone letter at the end
    lines = response.strip().split('\n')
    for line in reversed(lines[-5:]):  # Check last 5 lines
        line_clean = line.strip()
        if line_clean in ['A', 'B', 'C', 'D']:
            return line_clean
        # Check for patterns like "Answer: A" or "The answer is B"
        match = re.search(r'(?:answer|correct).*?([A-D])', line_clean, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return None


def evaluate_question(question: str, correct_answer: str, model: str = None) -> dict:
    """
    Evaluate a single GPQA question.
    
    Args:
        question: The question text
        correct_answer: The correct answer (A, B, C, or D)
        model: Optional model override
    
    Returns:
        Dictionary with evaluation results
    """
    prompt = generate_gpqa_prompt(question)
    
    try:
        response = llm_generate(prompt, max_tokens=2000, temperature=0.7, model=model)
        predicted_answer = extract_answer(response)
        
        is_correct = predicted_answer == correct_answer.upper() if predicted_answer else False
        
        return {
            "question": question[:200] + "..." if len(question) > 200 else question,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "full_response": response,
            "error": None
        }
    except Exception as e:
        logging.error(f"Error evaluating question: {e}")
        return {
            "question": question[:200] + "...",
            "correct_answer": correct_answer,
            "predicted_answer": None,
            "is_correct": False,
            "full_response": None,
            "error": str(e)
        }


def run_gpqa_evaluation(start_idx: int = 0, end_idx: int = None, 
                         model: str = None, output_dir: str = None) -> dict:
    """
    Run GPQA evaluation on a range of questions.
    
    Args:
        start_idx: Starting question index (0-based)
        end_idx: Ending question index (exclusive), or None for all
        model: Model to use (default: gpt-5-mini)
        output_dir: Output directory for results
    
    Returns:
        Dictionary with evaluation summary and results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load dataset
    df = load_gpqa_dataset()
    
    if end_idx is None:
        end_idx = len(df)
    
    # Validate indices
    start_idx = max(0, start_idx)
    end_idx = min(end_idx, len(df))
    
    if start_idx >= end_idx:
        raise ValueError(f"Invalid range: start={start_idx}, end={end_idx}")
    
    # Set up output directory
    if output_dir is None:
        output_dir = project_root / "output" / "gpqa"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to file
    log_file = output_dir / f"gpqa_eval_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logging.getLogger().addHandler(file_handler)
    
    model_used = model if model else MODEL_NAME
    logging.info(f"Starting GPQA evaluation: questions {start_idx} to {end_idx-1}")
    logging.info(f"Model: {model_used}")
    logging.info(f"Output directory: {output_dir}")
    
    results = []
    correct_count = 0
    
    try:
        for idx in range(start_idx, end_idx):
            row = df.iloc[idx]
            question = row['question']
            correct_answer = row['answer']
            
            logging.info(f"\n{'='*60}")
            logging.info(f"Question {idx + 1}/{end_idx} (index {idx})")
            logging.info(f"Correct answer: {correct_answer}")
            logging.info(f"{'='*60}")
            
            result = evaluate_question(question, correct_answer, model=model)
            result["question_index"] = idx
            results.append(result)
            
            if result["is_correct"]:
                correct_count += 1
                logging.info(f"✓ CORRECT - Predicted: {result['predicted_answer']}")
            else:
                logging.info(f"✗ WRONG - Predicted: {result['predicted_answer']}, Correct: {correct_answer}")
            
            # Log running accuracy
            accuracy = correct_count / len(results) * 100
            logging.info(f"Running accuracy: {correct_count}/{len(results)} ({accuracy:.1f}%)")
        
        # Compile summary
        total_questions = len(results)
        final_accuracy = correct_count / total_questions * 100 if total_questions > 0 else 0
        errors = sum(1 for r in results if r['error'] is not None)
        
        summary = {
            "timestamp": timestamp,
            "model": model_used,
            "question_range": f"{start_idx}-{end_idx-1}",
            "total_questions": total_questions,
            "correct": correct_count,
            "incorrect": total_questions - correct_count - errors,
            "errors": errors,
            "accuracy": round(final_accuracy, 2),
            "results": results
        }
        
        # Save results
        results_file = output_dir / f"gpqa_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"\n{'='*60}")
        logging.info("GPQA EVALUATION COMPLETE")
        logging.info(f"{'='*60}")
        logging.info(f"Total questions: {total_questions}")
        logging.info(f"Correct: {correct_count}")
        logging.info(f"Accuracy: {final_accuracy:.2f}%")
        logging.info(f"Results saved to: {results_file}")
        
        return summary
        
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model performance on GPQA Diamond dataset"
    )
    parser.add_argument(
        "--question-index",
        type=int,
        help="Evaluate a single question by index"
    )
    parser.add_argument(
        "--batch-start",
        type=int,
        default=0,
        help="Starting question index (default: 0)"
    )
    parser.add_argument(
        "--batch-end",
        type=int,
        default=None,
        help="Ending question index (exclusive)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all questions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    if args.question_index is not None:
        # Single question
        start_idx = args.question_index
        end_idx = args.question_index + 1
    elif args.all:
        # All questions
        start_idx = 0
        end_idx = None
    else:
        # Batch
        start_idx = args.batch_start
        end_idx = args.batch_end if args.batch_end else args.batch_start + 10
    
    summary = run_gpqa_evaluation(
        start_idx=start_idx,
        end_idx=end_idx,
        model=args.model,
        output_dir=args.output_dir
    )
    
    print(f"\n{'='*60}")
    print("GPQA EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {summary['model']}")
    print(f"Questions: {summary['question_range']}")
    print(f"Total: {summary['total_questions']}")
    print(f"Correct: {summary['correct']}")
    print(f"Accuracy: {summary['accuracy']}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
