#!/usr/bin/env python3
"""
T2D Drug Target Evaluation Script

This script:
1. Unmasks predicted gene IDs to reveal real gene names
2. Evaluates predictions against known T2D literature
3. Performs validation checks
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

def load_results(results_path: str) -> Dict[str, Any]:
    """Load pipeline results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)

def load_gene_mapping(mapping_path: str) -> Dict[str, str]:
    """Load gene mapping from JSON file."""
    with open(mapping_path, 'r') as f:
        return json.load(f)

def unmask_gene(masked_id: str, mapping: Dict[str, str]) -> str:
    """Convert masked gene ID to real gene name."""
    # The mapping might be stored as masked->real or real->masked
    # Check both directions
    if masked_id in mapping:
        return mapping[masked_id]

    # Try reverse lookup
    for real, masked in mapping.items():
        if masked == masked_id:
            return real

    return f"UNKNOWN ({masked_id})"

def search_gene_literature(gene_name: str) -> Dict[str, Any]:
    """Search PubMed for gene + T2D literature."""
    try:
        from external_tools.web_search import search_literature

        query = f"{gene_name} type 2 diabetes"
        papers = search_literature(query, max_results=5)

        return {
            "gene": gene_name,
            "query": query,
            "n_papers": len(papers) if papers else 0,
            "papers": papers[:3] if papers else []
        }
    except Exception as e:
        return {"gene": gene_name, "error": str(e), "n_papers": 0}

def get_gene_info_from_ncbi(gene_name: str) -> Dict[str, Any]:
    """Get gene information from NCBI."""
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET

    try:
        # Search for gene
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&term={urllib.parse.quote(gene_name)}[sym]+AND+human[organism]&retmode=xml"

        with urllib.request.urlopen(search_url, timeout=10) as response:
            search_xml = response.read().decode('utf-8')

        root = ET.fromstring(search_xml)
        id_list = root.find('.//IdList')

        if id_list is None or len(id_list) == 0:
            return {"gene": gene_name, "found": False}

        gene_id = id_list[0].text

        # Get gene summary
        summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gene&id={gene_id}&retmode=xml"

        with urllib.request.urlopen(summary_url, timeout=10) as response:
            summary_xml = response.read().decode('utf-8')

        summary_root = ET.fromstring(summary_xml)

        # Extract info
        description = ""
        summary = ""

        for item in summary_root.findall('.//Item'):
            name = item.get('Name', '')
            if name == 'Description':
                description = item.text or ""
            elif name == 'Summary':
                summary = item.text or ""

        return {
            "gene": gene_name,
            "found": True,
            "ncbi_id": gene_id,
            "description": description,
            "summary": summary[:500] if summary else ""
        }

    except Exception as e:
        return {"gene": gene_name, "found": False, "error": str(e)}

def evaluate_hypothesis(hypothesis: Dict[str, Any], real_gene: str) -> Dict[str, Any]:
    """Evaluate a single hypothesis."""

    evaluation = {
        "masked_gene": hypothesis.get("target_gene_masked", "UNKNOWN"),
        "real_gene": real_gene,
        "title": hypothesis.get("title", ""),
        "fitness_score": hypothesis.get("fitness_score", 0),
        "confidence": hypothesis.get("confidence_level", "UNKNOWN"),
        "mechanism": hypothesis.get("mechanism_hypothesis", "")[:200],
        "therapeutic_approach": hypothesis.get("therapeutic_approach", "")[:200],
    }

    # Get gene info
    print(f"\n  Fetching NCBI info for {real_gene}...")
    gene_info = get_gene_info_from_ncbi(real_gene)
    evaluation["gene_info"] = gene_info

    # Search literature
    print(f"  Searching literature for {real_gene} + T2D...")
    lit_search = search_gene_literature(real_gene)
    evaluation["literature"] = lit_search

    return evaluation

def print_evaluation_report(evaluations: List[Dict], output_path: str = None):
    """Print and optionally save evaluation report."""

    lines = []
    lines.append("=" * 80)
    lines.append("T2D DRUG TARGET EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append("")

    for i, eval_result in enumerate(evaluations, 1):
        lines.append(f"\n{'─' * 80}")
        lines.append(f"HYPOTHESIS #{i}")
        lines.append(f"{'─' * 80}")
        lines.append(f"Masked ID:    {eval_result['masked_gene']}")
        lines.append(f"REAL GENE:    {eval_result['real_gene']} ⭐")
        lines.append(f"Title:        {eval_result['title'][:70]}...")
        lines.append(f"Fitness:      {eval_result['fitness_score']}")
        lines.append(f"Confidence:   {eval_result['confidence']}")
        lines.append("")

        # Gene info
        gene_info = eval_result.get("gene_info", {})
        if gene_info.get("found"):
            lines.append(f"NCBI Gene ID: {gene_info.get('ncbi_id', 'N/A')}")
            lines.append(f"Description:  {gene_info.get('description', 'N/A')}")
            if gene_info.get("summary"):
                lines.append(f"Summary:      {gene_info['summary'][:300]}...")
        else:
            lines.append(f"NCBI:         Gene not found or error: {gene_info.get('error', 'Unknown')}")

        lines.append("")

        # Literature
        lit = eval_result.get("literature", {})
        lines.append(f"Literature Search: Found {lit.get('n_papers', 0)} papers for '{lit.get('query', 'N/A')}'")

        if lit.get("papers"):
            lines.append("  Top papers:")
            for j, paper in enumerate(lit["papers"][:3], 1):
                title = paper.get("title", "Unknown")[:60]
                lines.append(f"    {j}. {title}...")

        lines.append("")
        lines.append(f"Mechanism: {eval_result.get('mechanism', 'N/A')[:150]}...")
        lines.append(f"Approach:  {eval_result.get('therapeutic_approach', 'N/A')[:150]}...")

    # Summary
    lines.append(f"\n{'=' * 80}")
    lines.append("SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total hypotheses evaluated: {len(evaluations)}")

    if evaluations:
        best = max(evaluations, key=lambda x: x.get("fitness_score", 0))
        lines.append(f"Best hypothesis: {best['real_gene']} (fitness: {best['fitness_score']})")

        # Check literature support
        genes_with_lit = [e for e in evaluations if e.get("literature", {}).get("n_papers", 0) > 0]
        lines.append(f"Genes with T2D literature support: {len(genes_with_lit)}/{len(evaluations)}")

    lines.append("=" * 80)

    report = "\n".join(lines)
    print(report)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")

    return report

def find_latest_run(output_dir: str) -> str:
    """Find the most recent run directory."""
    output_path = Path(output_dir)

    if not output_path.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    # Find run directories
    run_dirs = [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("run_")]

    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in: {output_dir}")

    # Sort by name (which includes timestamp) and get latest
    latest = sorted(run_dirs)[-1]
    return str(latest)

def main():
    parser = argparse.ArgumentParser(description="Evaluate T2D drug target predictions")
    parser.add_argument("--run-dir", type=str, help="Path to specific run directory")
    parser.add_argument("--output-dir", type=str, default="output/t2d_target",
                       help="Output directory to find latest run")
    parser.add_argument("--mapping", type=str, help="Path to gene mapping JSON file")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top hypotheses to evaluate")

    args = parser.parse_args()

    # Find run directory
    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = find_latest_run(args.output_dir)

    print(f"Evaluating run: {run_dir}")

    # Find results file
    run_path = Path(run_dir)
    results_files = list(run_path.glob("pipeline_results_*.json"))

    if not results_files:
        print(f"ERROR: No results file found in {run_dir}")
        sys.exit(1)

    results_file = sorted(results_files)[-1]  # Latest
    print(f"Loading results from: {results_file}")

    results = load_results(str(results_file))

    # Find gene mapping
    mapping_file = args.mapping
    if not mapping_file:
        # Try to find in run directory or parent
        possible_paths = [
            run_path / "gene_mapping.json",
            run_path.parent / "gene_mapping.json",
            run_path.parent.parent / "gene_mapping.json",
            Path("output/t2d_target/gene_mapping.json"),
            Path("data/cache/gene_mapping.json"),
        ]

        for p in possible_paths:
            if p.exists():
                mapping_file = str(p)
                break

    if mapping_file and Path(mapping_file).exists():
        print(f"Loading gene mapping from: {mapping_file}")
        gene_mapping = load_gene_mapping(mapping_file)
    else:
        print("WARNING: Gene mapping file not found. Will show masked IDs only.")
        print("To save gene mapping, add this to your pipeline run:")
        print("  t2d_runner.save_gene_mapping('output/t2d_target/gene_mapping.json')")
        gene_mapping = {}

    # Get hypotheses
    final_population = results.get("final_population", [])
    all_hypotheses = results.get("all_hypotheses", final_population)

    # Sort by fitness and take top N
    sorted_hyps = sorted(all_hypotheses, key=lambda x: x.get("fitness_score", 0), reverse=True)
    top_hypotheses = sorted_hyps[:args.top_n]

    print(f"\nEvaluating top {len(top_hypotheses)} hypotheses...")

    # Evaluate each hypothesis
    evaluations = []
    for hyp in top_hypotheses:
        masked_gene = hyp.get("target_gene_masked", "UNKNOWN")
        real_gene = unmask_gene(masked_gene, gene_mapping) if gene_mapping else masked_gene

        print(f"\nEvaluating: {masked_gene} -> {real_gene}")
        eval_result = evaluate_hypothesis(hyp, real_gene)
        evaluations.append(eval_result)

    # Print report
    report_path = str(run_path / "evaluation_report.txt")
    print_evaluation_report(evaluations, report_path)

    # Also save as JSON
    json_path = str(run_path / "evaluation_results.json")
    with open(json_path, 'w') as f:
        json.dump(evaluations, f, indent=2, default=str)
    print(f"JSON results saved to: {json_path}")

if __name__ == "__main__":
    main()
