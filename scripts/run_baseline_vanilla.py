#!/usr/bin/env python3
"""
Vanilla LLM + Literature Baseline for Drug Repurposing

This script implements the baseline comparison:
- Single LLM call (vs multi-agent structured generation)
- WITH literature search (same context as full pipeline)
- Same drug constraints (62 approved drugs)
- Same cancer type names (TCGA)
- Same output format for comparison

Usage:
    python scripts/run_baseline_vanilla.py "Acute Myeloid Leukemia"
    python scripts/run_baseline_vanilla.py --all
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add pipeline to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "pipeline"))

from external_tools.llm_client import llm_generate
from external_tools.web_search import search_literature
from prompts import APPROVED_DRUGS_DEPMAP

# Import TCGA cancer types
sys.path.insert(0, str(project_root / "scripts"))
from tcga_cancer_types import ACTUAL_CANCER_TYPES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def generate_search_queries(cancer_type: str) -> list:
    """Generate 3 literature search queries for a cancer type."""
    return [
        f"{cancer_type} drug repurposing therapeutic targets",
        f"{cancer_type} molecular pathways treatment resistance",
        f"{cancer_type} novel therapy mechanisms approved drugs"
    ]


def search_literature_for_cancer(cancer_type: str, num_papers: int = 5) -> tuple:
    """
    Perform literature search for a cancer type.

    Returns:
        Tuple of (all_papers, literature_context_string)
    """
    queries = generate_search_queries(cancer_type)
    all_papers = []
    literature_context = ""

    for i, query in enumerate(queries):
        logging.info(f"Literature search {i+1}/3: {query}")
        papers = search_literature(query, max_results=num_papers)

        # Filter duplicates
        new_papers = [p for p in papers if not any(
            existing['title'].lower() == p['title'].lower() for existing in all_papers
        )]

        if new_papers:
            all_papers.extend(new_papers)

            # Build context string
            literature_context += f"\n\n--- Search {i+1}: {query} ---\n\n"
            for j, paper in enumerate(new_papers[:3], 1):
                title = paper.get('title', 'Untitled')
                abstract = paper.get('abstract', '')[:300]
                literature_context += f"{j}. {title}\n   {abstract}...\n\n"

    logging.info(f"Found {len(all_papers)} papers for {cancer_type}")
    return all_papers, literature_context


def generate_vanilla_hypotheses(cancer_type: str, literature_context: str,
                                 num_hypotheses: int = 1) -> list:
    """
    Generate hypotheses using a single LLM call (vanilla baseline).

    This is the key difference from the full pipeline:
    - Single LLM call instead of multi-agent generation
    - No reflection/fitness scoring
    - No evolution/GA

    Default is 1 hypothesis since without a reviewer agent,
    multiple hypotheses from the same prompt are redundant.
    """
    drug_list = ", ".join(APPROVED_DRUGS_DEPMAP)

    if num_hypotheses == 1:
        # Single hypothesis prompt (simpler, more focused)
        prompt = f"""You are a drug repurposing expert. Given the literature context below about {cancer_type},
generate exactly 1 drug repurposing hypothesis.

LITERATURE CONTEXT:
{literature_context if literature_context else "No literature found. Use your general knowledge."}

CONSTRAINTS:
1. You MUST use ONLY drugs from this approved list: {drug_list}
2. Use the exact cancer type name: {cancer_type}
3. The hypothesis should propose a specific mechanism

Provide:
TITLE: [Concise title]
SUMMARY: [One sentence summary]
HYPOTHESIS_STATEMENT: [2-3 sentences explaining the mechanism]
RATIONALE: [1-2 paragraphs of scientific reasoning]
FINAL DRUG: [Drug from approved list]
CANCER TYPE: {cancer_type}
"""
    else:
        # Multiple hypotheses prompt (original format)
        prompt = f"""You are a drug repurposing expert. Given the literature context below about {cancer_type},
generate exactly {num_hypotheses} drug repurposing hypotheses.

LITERATURE CONTEXT:
{literature_context if literature_context else "No literature found. Use your general knowledge."}

CONSTRAINTS:
1. You MUST use ONLY drugs from this approved list: {drug_list}
2. Use the exact cancer type name: {cancer_type}
3. Each hypothesis should propose a specific mechanism

For EACH of the {num_hypotheses} hypotheses, provide:

HYPOTHESIS [N]:
TITLE: [Concise title]
SUMMARY: [One sentence summary]
HYPOTHESIS_STATEMENT: [2-3 sentences explaining the mechanism]
RATIONALE: [1-2 paragraphs of scientific reasoning]
FINAL DRUG: [Drug from approved list]
CANCER TYPE: {cancer_type}

Generate all {num_hypotheses} hypotheses now:
"""

    logging.info(f"Generating {num_hypotheses} vanilla hypothesis(es) for {cancer_type}")
    response = llm_generate(prompt, max_tokens=2000 if num_hypotheses == 1 else 6000, temperature=0.7)

    # Parse the response into individual hypotheses
    hypotheses = parse_vanilla_response(response, cancer_type, num_hypotheses)
    return hypotheses


def parse_vanilla_response(response: str, cancer_type: str, expected_count: int) -> list:
    """Parse vanilla LLM response into structured hypotheses."""
    import re
    import uuid

    hypotheses = []

    # For single hypothesis, parse directly without splitting
    if expected_count == 1:
        hyp = {
            "id": f"vanilla-001-{uuid.uuid4().hex[:8]}",
            "origin": "vanilla_baseline",
            "generation_method": "single_llm_call",
            "elo_score": 1200,
            "fitness_score": None,
            "cancer_type": cancer_type
        }

        # Extract fields from response directly
        title_match = re.search(r'TITLE:\s*([^\n]+)', response, re.IGNORECASE)
        summary_match = re.search(r'SUMMARY:\s*([^\n]+)', response, re.IGNORECASE)
        statement_match = re.search(r'HYPOTHESIS_STATEMENT:\s*(.+?)(?=RATIONALE:|FINAL DRUG:|$)',
                                     response, re.IGNORECASE | re.DOTALL)
        rationale_match = re.search(r'RATIONALE:\s*(.+?)(?=FINAL DRUG:|CANCER TYPE:|$)',
                                     response, re.IGNORECASE | re.DOTALL)
        drug_match = re.search(r'FINAL DRUG:\s*([^\n]+)', response, re.IGNORECASE)

        hyp['title'] = title_match.group(1).strip() if title_match else "Untitled Hypothesis"
        hyp['summary'] = summary_match.group(1).strip() if summary_match else ""
        hyp['hypothesis_statement'] = statement_match.group(1).strip() if statement_match else ""
        hyp['rationale'] = rationale_match.group(1).strip() if rationale_match else ""
        hyp['final_drug'] = drug_match.group(1).strip().upper() if drug_match else None

        # Validate drug is in approved list
        if hyp['final_drug']:
            hyp['drug_in_approved_list'] = hyp['final_drug'] in [d.upper() for d in APPROVED_DRUGS_DEPMAP]
        else:
            hyp['drug_in_approved_list'] = False

        hypotheses.append(hyp)
        return hypotheses

    # For multiple hypotheses, split by hypothesis markers
    pattern = r'HYPOTHESIS\s*\[?\d+\]?:?'
    parts = re.split(pattern, response, flags=re.IGNORECASE)

    for i, part in enumerate(parts[1:], 1):  # Skip first empty part
        if not part.strip():
            continue

        hyp = {
            "id": f"vanilla-{i:03d}-{uuid.uuid4().hex[:8]}",
            "origin": "vanilla_baseline",
            "generation_method": "single_llm_call",
            "elo_score": 1200,
            "fitness_score": None,
            "cancer_type": cancer_type
        }

        # Extract fields
        title_match = re.search(r'TITLE:\s*([^\n]+)', part, re.IGNORECASE)
        summary_match = re.search(r'SUMMARY:\s*([^\n]+)', part, re.IGNORECASE)
        statement_match = re.search(r'HYPOTHESIS_STATEMENT:\s*(.+?)(?=RATIONALE:|FINAL DRUG:|$)',
                                     part, re.IGNORECASE | re.DOTALL)
        rationale_match = re.search(r'RATIONALE:\s*(.+?)(?=FINAL DRUG:|$)',
                                     part, re.IGNORECASE | re.DOTALL)
        drug_match = re.search(r'FINAL DRUG:\s*([^\n]+)', part, re.IGNORECASE)

        hyp['title'] = title_match.group(1).strip() if title_match else f"Hypothesis {i}"
        hyp['summary'] = summary_match.group(1).strip() if summary_match else ""
        hyp['hypothesis_statement'] = statement_match.group(1).strip() if statement_match else ""
        hyp['rationale'] = rationale_match.group(1).strip() if rationale_match else ""
        hyp['final_drug'] = drug_match.group(1).strip().upper() if drug_match else None

        # Validate drug is in approved list
        if hyp['final_drug']:
            hyp['drug_in_approved_list'] = hyp['final_drug'] in [d.upper() for d in APPROVED_DRUGS_DEPMAP]
        else:
            hyp['drug_in_approved_list'] = False

        hypotheses.append(hyp)

    # Ensure we have the expected count
    if len(hypotheses) < expected_count:
        logging.warning(f"Parsed {len(hypotheses)} hypotheses, expected {expected_count}")

    return hypotheses


def run_vanilla_baseline(cancer_type: str, output_dir: str = None) -> dict:
    """
    Run the vanilla baseline for a single cancer type.

    Args:
        cancer_type: Full TCGA cancer type name
        output_dir: Output directory (default: output/baselines/vanilla/{cancer_type})

    Returns:
        Dictionary with results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up output directory
    if output_dir is None:
        cancer_dir = cancer_type.replace(" ", "_").replace(",", "")
        output_dir = project_root / "output" / "baselines" / "vanilla" / cancer_dir
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging to file
    log_file = output_dir / f"vanilla_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logging.getLogger().addHandler(file_handler)

    logging.info(f"Starting vanilla baseline for: {cancer_type}")
    logging.info(f"Output directory: {output_dir}")

    try:
        # Step 1: Literature search
        papers, literature_context = search_literature_for_cancer(cancer_type)

        # Step 2: Generate hypotheses with single LLM call
        hypotheses = generate_vanilla_hypotheses(cancer_type, literature_context)

        # Compile results
        results = {
            "baseline_type": "vanilla_llm_literature",
            "cancer_type": cancer_type,
            "timestamp": timestamp,
            "model": os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            "num_papers_found": len(papers),
            "literature_queries": generate_search_queries(cancer_type),
            "hypotheses": hypotheses,
            "num_hypotheses": len(hypotheses),
            "drug_compliance": sum(1 for h in hypotheses if h.get('drug_in_approved_list', False)),
            "drug_compliance_rate": sum(1 for h in hypotheses if h.get('drug_in_approved_list', False)) / max(len(hypotheses), 1)
        }

        # Save results
        results_file = output_dir / f"vanilla_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logging.info(f"Results saved to: {results_file}")
        logging.info(f"Generated {len(hypotheses)} hypotheses, {results['drug_compliance']}/{len(hypotheses)} drug compliant")

        return results

    except Exception as e:
        logging.error(f"Error running vanilla baseline: {e}")
        raise
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run vanilla LLM + literature baseline for drug repurposing"
    )
    parser.add_argument(
        "cancer_type",
        nargs="?",
        help="Cancer type name (e.g., 'Acute Myeloid Leukemia')"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run baseline for all 33 TCGA cancer types"
    )
    parser.add_argument(
        "--output-dir",
        help="Custom output directory"
    )

    args = parser.parse_args()

    if args.all:
        # Run for all cancer types
        results_summary = []
        for abbrev, full_name in ACTUAL_CANCER_TYPES.items():
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing: {full_name} ({abbrev})")
            logging.info(f"{'='*60}\n")

            try:
                result = run_vanilla_baseline(full_name)
                results_summary.append({
                    "cancer_type": full_name,
                    "abbreviation": abbrev,
                    "num_hypotheses": result['num_hypotheses'],
                    "drug_compliance_rate": result['drug_compliance_rate'],
                    "status": "success"
                })
            except Exception as e:
                logging.error(f"Failed for {full_name}: {e}")
                results_summary.append({
                    "cancer_type": full_name,
                    "abbreviation": abbrev,
                    "status": "failed",
                    "error": str(e)
                })

        # Save summary
        summary_file = project_root / "output" / "baselines" / "vanilla" / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)

        logging.info(f"\nBatch complete. Summary saved to: {summary_file}")

    elif args.cancer_type:
        run_vanilla_baseline(args.cancer_type, args.output_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
