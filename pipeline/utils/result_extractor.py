import os
import re
import logging
from typing import Dict, List, Any, Optional

# Optional pandas import
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("Warning: pandas not available. CSV functionality will be limited.")
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)

class DrugResultExtractor:
    """
    A simplified utility class to extract drug names and ratings from pipeline2 hypotheses.
    Focuses on the specific formats produced by the reflection agent:
    - "FINAL DRUG: [drug name]" for drug names
    - "FINAL RATING: [1-5]" for ratings (integers only, no fallback to ELO or review scores)
    """
    
    def __init__(self, output_path: Optional[str] = None):
        """
        Initialize the extractor with the path to output directory.
        
        Args:
            output_path: Path to the pipeline output directory. If None, 
                         will use the default 'output' directory in pipeline2.
        """
        if output_path is None:
            # Set default path relative to the pipeline2 directory
            self.output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
        else:
            self.output_path = output_path
        
        logger.info(f"DrugResultExtractor initialized with output path: {self.output_path}")
    
    def extract_from_pipeline_output(self, research_overview: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract drug information from pipeline output.
        
        Args:
            research_overview: The research overview dictionary returned by the pipeline
            
        Returns:
            List of dictionaries with drug information, each containing:
                - drug_name: Name of the proposed drug
                - rating: Rating/score associated with the drug
                - cancer_type: Cancer type from the research goal (if available)
                - hypothesis_id: ID of the hypothesis containing the drug
        """
        results = []
        cancer_type = self._extract_cancer_type_from_goal(research_overview.get("research_goal", ""))
        
        # Process top hypotheses
        if "top_hypotheses" in research_overview:
            top_hyps = research_overview.get("top_hypotheses", [])
            logger.info(f"Processing {len(top_hyps)} top hypotheses")
            
            for hyp in top_hyps:
                drug_info = self._extract_drug_info_from_hypothesis(hyp, cancer_type)
                if drug_info:
                    results.append(drug_info)
        
        # Process all hypotheses if available
        all_hyps = research_overview.get("all_hypotheses", [])
        if all_hyps:
            logger.info(f"Processing {len(all_hyps)} total hypotheses")
            
            # Track hypothesis IDs we've already processed from top hypotheses
            processed_ids = {r.get("hypothesis_id") for r in results if r.get("hypothesis_id")}
            
            for hyp in all_hyps:
                # Skip if we've already processed this hypothesis
                if hyp.get("id") in processed_ids:
                    continue
                    
                drug_info = self._extract_drug_info_from_hypothesis(hyp, cancer_type)
                if drug_info:
                    results.append(drug_info)
        
        logger.info(f"Extracted information for {len(results)} drugs")
        return results
    
    def _extract_cancer_type(self, text: str, is_research_goal: bool = False) -> str:
        """
        Unified method to extract cancer type from any text, whether a hypothesis or research goal.
        
        Args:
            text: The text to extract cancer type from
            is_research_goal: Whether this text is a research goal (affects extraction strategy)
            
        Returns:
            Extracted cancer type or empty string if none found
        """
        if not text:
            return ""
            
        # First, look for the explicit CANCER TYPE format in all texts
        cancer_match = re.search(r"CANCER TYPE:\s*([^.\n]+)", text, re.IGNORECASE)
        if cancer_match:
            return cancer_match.group(1).strip()
            
        # For research goals or if no explicit label was found, use pattern matching
        if is_research_goal or not cancer_match:
            # Pattern 1: Cancer mentioned in context of treatment
            cancer_pattern1 = r"(?:for|in|against|treating|treatment of|therapy for|management of)\s+([A-Za-z\s-]+(?:cancer|leukemia|sarcoma|lymphoma|carcinoma|melanoma|tumor|tumour|glioma|myeloma))"
            cancer_pattern2 = r"([A-Za-z\s-]+(?:cancer|leukemia|sarcoma|lymphoma|carcinoma|melanoma|tumor|tumour|glioma|myeloma))\s+(?:treatment|therapy|patients|cells|cell lines)"
            
            # Also look for disease abbreviations in parentheses
            abbrev_pattern = r"([A-Za-z\s-]+(?:cancer|leukemia|sarcoma|lymphoma|carcinoma|melanoma))\s+\(([A-Z]{2,5})\)"
            
            # Try the patterns
            for pattern in [cancer_pattern1, cancer_pattern2]:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
                    
            # Check for abbreviation patterns
            abbrev_match = re.search(abbrev_pattern, text, re.IGNORECASE)
            if abbrev_match:
                # If we find both full name and abbreviation, return the abbreviation
                return abbrev_match.group(2).strip()
            
            # Fallback to reference list of common cancer types
            cancer_types = [
                "acute myeloid leukemia", "AML",
                "triple-negative breast cancer", "TNBC",
                "glioblastoma multiforme", "GBM",
                "colorectal cancer", "CRC",
                "pancreatic ductal adenocarcinoma", "PDAC",
                "prostate cancer",
                "Ewing sarcoma",
                "non-small cell lung cancer", "NSCLC",
                "hepatocellular carcinoma", "HCC",
                "ovarian cancer",
                "chronic lymphocytic leukemia", "CLL",
                "melanoma",
                "Burkitt lymphoma",
                "renal cell carcinoma", 
                "medulloblastoma",
                "multiple myeloma",
                "squamous cell carcinoma"
            ]
            
            for cancer in cancer_types:
                if cancer.lower() in text.lower():
                    return cancer
                    
        # If we reach here, no cancer type was found
        return ""
    
    def _extract_cancer_type_from_goal(self, research_goal: str) -> str:
        """
        Extract cancer type from research goal.
        
        Args:
            research_goal: The research goal to extract cancer type from
            
        Returns:
            Extracted cancer type or empty string if none found
        """
        return self._extract_cancer_type(research_goal, is_research_goal=True)
    
    def _extract_drug_info_from_hypothesis(self, hypothesis: Dict[str, Any], cancer_type: str) -> Optional[Dict[str, Any]]:
        """
        Extract drug information from a single hypothesis.
        
        Args:
            hypothesis: A hypothesis dictionary
            cancer_type: Cancer type from the research goal
            
        Returns:
            Dictionary with drug information or None if no drug info found
        """
        if not hypothesis:
            return None
            
        # Extract title and description to search for drug and rating
        title = hypothesis.get("title", "")
        description = hypothesis.get("description", "")
        combined_text = f"{title}\n{description}"
        
        # Look for the review section if available
        reviews = hypothesis.get("reviews", [])
        if reviews and isinstance(reviews, list):
            for review in reviews:
                if isinstance(review, dict) and "review" in review:
                    combined_text += f"\n{review['review']}"
        
        # Check combined review if available
        combined_review = hypothesis.get("combined_review", "")
        if combined_review:
            combined_text += f"\n{combined_review}"
            
        # Extract drug name using the expected format from reflection agent
        drug_name = None
        drug_match = re.search(r"FINAL DRUG:\s*([A-Za-z0-9-]+)", combined_text, re.IGNORECASE)
        if drug_match:
            drug_name = drug_match.group(1).strip()
            
        # Extract cancer type from the hypothesis text, fall back to provided cancer_type
        specific_cancer_type = self._extract_cancer_type(combined_text)
        if not specific_cancer_type:
            specific_cancer_type = cancer_type
        
        # Extract rating using ONLY the expected format from reflection agent (1-5 integers)
        rating = None
        rating_match = re.search(r"FINAL RATING:\s*(\d+)", combined_text, re.IGNORECASE)
        if rating_match:
            try:
                rating = int(rating_match.group(1))
            except ValueError:
                pass
        
        # Do not use fallback ratings - only accept explicit FINAL RATING integers
        
        # Return the extracted information if we found a drug name
        if drug_name:
            return {
                "drug_name": drug_name,
                "rating": rating,
                "cancer_type": specific_cancer_type,
                "hypothesis_id": hypothesis.get("id"),
                "hypothesis_title": title
            }
        
        return None
    
    def extract_from_log_file(self, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract drug information by parsing the pipeline log file.
        
        Args:
            log_file: Path to the log file. If None, uses the default pipeline.log
            
        Returns:
            List of dictionaries with drug information
        """
        if log_file is None:
            log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                  'pipeline.log')
        
        if not os.path.exists(log_file):
            logger.error(f"Log file not found: {log_file}")
            return []
        
        logger.info(f"Extracting drug information from log file: {log_file}")
        results = []
        current_text = ""
        collecting = False
        cancer_type = ""
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    # Try to extract cancer type from research goal
                    if "research_goal" in line.lower() and not cancer_type:
                        cancer_type = self._extract_cancer_type_from_goal(line)
                        
                    # Start collecting once we see a hypothesis
                    if "HYPOTHESIS" in line or "hypothesis:" in line:
                        collecting = True
                        current_text = line
                    elif collecting:
                        current_text += line
                        
                        # Check if we've found a FINAL DRUG and FINAL RATING
                        if "FINAL RATING" in line:
                            # Extract drug and rating
                            drug_match = re.search(r"FINAL DRUG:\s*([A-Za-z0-9-]+)", current_text, re.IGNORECASE)
                            rating_match = re.search(r"FINAL RATING:\s*(\d+)", current_text, re.IGNORECASE)
                            
                            if drug_match:
                                drug_name = drug_match.group(1).strip()
                                rating = None
                                
                                if rating_match:
                                    try:
                                        rating = int(rating_match.group(1))
                                    except ValueError:
                                        pass
                                
                                # Use unified cancer type extraction from the current text
                                specific_cancer_type = self._extract_cancer_type(current_text)
                                if not specific_cancer_type:
                                    specific_cancer_type = cancer_type
                                
                                results.append({
                                    "drug_name": drug_name,
                                    "rating": rating,
                                    "cancer_type": specific_cancer_type,
                                    "hypothesis_id": None
                                })
                                
                            # Reset for next hypothesis
                            collecting = False
                            current_text = ""
        except Exception as e:
            logger.error(f"Error parsing log file: {e}")
        
        logger.info(f"Extracted information for {len(results)} drugs from log file")
        return results
    
    def save_results_to_csv(self, results: List[Dict[str, Any]], output_file: Optional[str] = None) -> str:
        """
        Save extracted drug results to a CSV file.
        
        Args:
            results: List of drug information dictionaries
            output_file: Path to the output CSV file. If None, creates a file in the output directory.
            
        Returns:
            Path to the saved CSV file
        """
        if not results:
            logger.warning("No results to save")
            return None
            
        if output_file is None:
            output_file = os.path.join(self.output_path, 'drug_results.csv')
            
        try:
            if PANDAS_AVAILABLE:
                # Create DataFrame and save to CSV
                df = pd.DataFrame(results)
                # Sort by rating (highest first)
                if 'rating' in df.columns:
                    df = df.sort_values(by='rating', ascending=False)
                df.to_csv(output_file, index=False)
            else:
                # Fallback CSV writing without pandas
                import csv
                if results:
                    # Sort by rating if available
                    if 'rating' in results[0]:
                        results = sorted(results, key=lambda x: x.get('rating', 0) or 0, reverse=True)
                    
                    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                        fieldnames = results[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(results)
            logger.info(f"Results saved to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error saving results to CSV: {e}")
            return None
    
    def extract_and_save(self, research_overview: Dict[str, Any] = None, 
                        log_file: Optional[str] = None,
                        output_file: Optional[str] = None) -> str:
        """
        Extract drug information and save to CSV in one step.
        
        Args:
            research_overview: The research overview dictionary returned by the pipeline
            log_file: Path to the log file if research_overview is not provided
            output_file: Path to the output CSV file
            
        Returns:
            Path to the saved CSV file
        """
        if research_overview:
            results = self.extract_from_pipeline_output(research_overview)
        elif log_file or os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                  'pipeline.log')):
            results = self.extract_from_log_file(log_file)
        else:
            logger.error("No research_overview or valid log_file provided")
            return None
            
        return self.save_results_to_csv(results, output_file)


def extract_drug_results(research_overview: Dict[str, Any] = None, 
                        log_file: Optional[str] = None,
                        output_file: Optional[str] = None,
                        output_path: Optional[str] = None) -> str:
    """
    Convenience function to extract drug results from pipeline output.
    
    Args:
        research_overview: The research overview dictionary from the pipeline
        log_file: Path to the log file if research_overview is not provided
        output_file: Path to the output CSV file
        output_path: Path to the output directory
        
    Returns:
        Path to the saved CSV file
    """
    extractor = DrugResultExtractor(output_path)
    return extractor.extract_and_save(research_overview, log_file, output_file) 