"""
Utility functions for loading and managing complete prompt files.

Used by lethal_genes_2 mode to process individual prompt files
from data/lethal_genes/individual_prompts/ directory.
"""

import os
import re
from typing import List, Tuple


def list_prompt_files(prompts_dir: str) -> List[str]:
    """
    Discover all prompt_*.txt files in the specified directory.

    Args:
        prompts_dir: Path to directory containing prompt files

    Returns:
        List of absolute file paths to prompt files, sorted by filename

    Example:
        >>> files = list_prompt_files("data/lethal_genes/individual_prompts")
        >>> print(len(files))
        19
    """
    if not os.path.exists(prompts_dir):
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    # Find all files matching prompt_*.txt pattern
    all_files = os.listdir(prompts_dir)
    prompt_files = [f for f in all_files if f.startswith("prompt_") and f.endswith(".txt")]

    # Sort by filename to ensure consistent ordering
    prompt_files.sort()

    # Return absolute paths
    return [os.path.join(prompts_dir, f) for f in prompt_files]


def load_prompt_file(filepath: str) -> str:
    """
    Read complete prompt text from file.

    Args:
        filepath: Absolute path to prompt file

    Returns:
        Complete prompt text as string

    Example:
        >>> prompt = load_prompt_file("data/lethal_genes/individual_prompts/prompt_01_KLF5_ARID1A.txt")
        >>> print(len(prompt))
        4821
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prompt file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()


def extract_gene_pair_name(filename: str) -> str:
    """
    Parse gene pair name from prompt filename for output folder naming.

    Args:
        filename: Prompt filename (e.g., "prompt_01_KLF5_ARID1A.txt")

    Returns:
        Gene pair name (e.g., "KLF5_ARID1A")

    Examples:
        >>> extract_gene_pair_name("prompt_01_KLF5_ARID1A.txt")
        'KLF5_ARID1A'
        >>> extract_gene_pair_name("prompt_05_WRN_microsatellite_instability_MSI.txt")
        'WRN_microsatellite_instability_MSI'
        >>> extract_gene_pair_name("/path/to/prompt_14_PELO-HBS1L_ribosomal_rescue_complex_and_SKI_mRNA_quality_control_complex.txt")
        'PELO-HBS1L_ribosomal_rescue_complex_and_SKI_mRNA_quality_control_complex'
    """
    # Extract just the filename if full path provided
    basename = os.path.basename(filename)

    # Pattern: prompt_XX_GENE_PAIR_NAME.txt
    # Extract everything between "prompt_XX_" and ".txt"
    match = re.match(r'prompt_\d+_(.+)\.txt$', basename)

    if not match:
        raise ValueError(f"Invalid prompt filename format: {filename}")

    return match.group(1)


def get_prompt_info(filepath: str) -> Tuple[str, str]:
    """
    Convenience function to get both gene pair name and prompt text.

    Args:
        filepath: Absolute path to prompt file

    Returns:
        Tuple of (gene_pair_name, prompt_text)

    Example:
        >>> gene_pair, prompt = get_prompt_info("data/lethal_genes/individual_prompts/prompt_01_KLF5_ARID1A.txt")
        >>> print(gene_pair)
        KLF5_ARID1A
        >>> print(len(prompt))
        4821
    """
    gene_pair = extract_gene_pair_name(filepath)
    prompt_text = load_prompt_file(filepath)
    return gene_pair, prompt_text
