#!/usr/bin/env python3
"""
BixBench Data Loader - Extracts and processes data files from research capsule zip files.
"""

import os
import zipfile
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re

# Optional imports with fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("pandas not available. CSV parsing will be limited.")

logger = logging.getLogger(__name__)


class BixBenchDataLoader:
    """
    Loads and processes data files from BixBench research capsule zip files.
    
    Handles CSV, TSV, and text files to provide structured data context
    for questions that require specific data analysis results.
    """
    
    def __init__(self, bixbench_dataset_dir: str = "bixbench_dataset"):
        """
        Initialize the data loader.
        
        Args:
            bixbench_dataset_dir: Path to directory containing BixBench zip files
        """
        self.dataset_dir = Path(bixbench_dataset_dir)
        self.cache_dir = None
        self._capsule_cache = {}
        
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"BixBench dataset directory not found: {bixbench_dataset_dir}")
    
    def get_capsule_data(self, capsule_id: str) -> Dict[str, Any]:
        """
        Extract and process all data files for a research capsule.
        
        Args:
            capsule_id: The capsule ID (e.g., 'bix-1' maps to UUID in zip filename)
            
        Returns:
            Dictionary containing processed data files and metadata
        """
        # Check cache first
        if capsule_id in self._capsule_cache:
            return self._capsule_cache[capsule_id]
        
        # Find the corresponding zip file
        zip_file = self._find_capsule_zip(capsule_id)
        if not zip_file:
            logger.warning(f"No zip file found for capsule {capsule_id}")
            return {}
        
        # Extract and process data files
        data_info = self._extract_capsule_data(zip_file)
        self._capsule_cache[capsule_id] = data_info
        
        return data_info
    
    def _find_capsule_zip(self, capsule_id: str) -> Optional[Path]:
        """Find the zip file corresponding to a capsule ID."""
        # First, get the UUID mapping from BixBench.csv
        uuid_mapping = self._get_capsule_uuid_mapping()
        
        if capsule_id in uuid_mapping:
            uuid = uuid_mapping[capsule_id]
            zip_pattern = f"CapsuleFolder-{uuid}.zip"
            zip_path = self.dataset_dir / zip_pattern
            
            if zip_path.exists():
                return zip_path
        
        logger.warning(f"Could not find zip file for capsule {capsule_id}")
        return None
    
    def _get_capsule_uuid_mapping(self) -> Dict[str, str]:
        """Extract capsule ID to UUID mapping from BixBench.csv."""
        bixbench_csv = self.dataset_dir / "BixBench.csv"
        if not bixbench_csv.exists():
            return {}
        
        mapping = {}
        try:
            if PANDAS_AVAILABLE:
                df = pd.read_csv(bixbench_csv)
                for _, row in df.iterrows():
                    uuid = row['uuid']
                    short_id = row['short_id']
                    mapping[short_id] = uuid
            else:
                # Fallback CSV parsing
                import csv
                with open(bixbench_csv, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        mapping[row['short_id']] = row['uuid']
        except Exception as e:
            logger.error(f"Error reading BixBench.csv: {e}")
        
        return mapping
    
    def _extract_capsule_data(self, zip_path: Path) -> Dict[str, Any]:
        """Extract and process all data files from a capsule zip."""
        data_info = {
            "files": {},
            "summary": "",
            "file_count": 0,
            "total_rows": 0,
            "available_data": []
        }
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Find data files (CSV, TSV, TXT)
                data_files = [
                    f for f in zip_ref.namelist() 
                    if any(ext in f.lower() for ext in ['.csv', '.tsv', '.txt']) 
                    and not f.endswith('/') 
                    and 'CapsuleData' in f
                ]
                
                if not data_files:
                    return data_info
                
                # Create temporary directory for extraction
                with tempfile.TemporaryDirectory() as temp_dir:
                    for file_path in data_files:
                        try:
                            # Extract individual file
                            zip_ref.extract(file_path, temp_dir)
                            extracted_path = Path(temp_dir) / file_path
                            
                            # Process the file
                            file_info = self._process_data_file(extracted_path)
                            
                            # Store with simplified name
                            file_name = Path(file_path).name
                            data_info["files"][file_name] = file_info
                            data_info["file_count"] += 1
                            data_info["total_rows"] += file_info.get("row_count", 0)
                            
                        except Exception as e:
                            logger.warning(f"Error processing {file_path}: {e}")
                
                # Generate summary
                data_info["summary"] = self._generate_data_summary(data_info["files"])
                data_info["available_data"] = list(data_info["files"].keys())
                
        except Exception as e:
            logger.error(f"Error extracting data from {zip_path}: {e}")
        
        return data_info
    
    def _process_data_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single data file and extract key information."""
        file_info = {
            "filename": file_path.name,
            "size_bytes": file_path.stat().st_size,
            "row_count": 0,
            "column_count": 0,
            "columns": [],
            "sample_data": [],
            "data_types": {},
            "summary_stats": {},
            "file_type": "unknown"
        }
        
        try:
            # Determine file type
            if file_path.suffix.lower() in ['.csv']:
                file_info["file_type"] = "csv"
                separator = ","
            elif file_path.suffix.lower() in ['.tsv', '.txt']:
                file_info["file_type"] = "tsv"
                separator = "\t"
            else:
                file_info["file_type"] = "text"
                separator = ","
            
            # Try to read with pandas if available
            if PANDAS_AVAILABLE and file_info["file_type"] in ["csv", "tsv"]:
                try:
                    df = pd.read_csv(file_path, sep=separator)  # No row limit - include all data
                    
                    file_info["row_count"] = len(df)
                    file_info["column_count"] = len(df.columns)
                    file_info["columns"] = df.columns.tolist()
                    
                    # Sample data (first 5 rows)
                    sample_df = df.head(5)
                    file_info["sample_data"] = sample_df.to_dict('records')
                    
                    # Data types
                    file_info["data_types"] = df.dtypes.astype(str).to_dict()
                    
                    # Basic summary statistics for numeric columns
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        stats = df[numeric_cols].describe()
                        file_info["summary_stats"] = stats.to_dict()
                        
                except Exception as e:
                    logger.warning(f"Pandas processing failed for {file_path}, using fallback: {e}")
                    file_info = self._process_file_fallback(file_path, separator)
            else:
                # Fallback processing
                file_info = self._process_file_fallback(file_path, separator)
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
        
        return file_info
    
    def _process_file_fallback(self, file_path: Path, separator: str = ",") -> Dict[str, Any]:
        """Fallback file processing without pandas."""
        file_info = {
            "filename": file_path.name,
            "size_bytes": file_path.stat().st_size,
            "row_count": 0,
            "column_count": 0,
            "columns": [],
            "sample_data": [],
            "file_type": "text"
        }
        
        try:
            import csv
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Try to detect if it's actually CSV/TSV
                sample = f.read(1024)
                f.seek(0)
                
                # Count separator occurrences to guess format
                comma_count = sample.count(',')
                tab_count = sample.count('\t')
                
                if tab_count > comma_count:
                    separator = '\t'
                    file_info["file_type"] = "tsv"
                elif comma_count > 0:
                    separator = ','
                    file_info["file_type"] = "csv"
                
                # Read with csv module
                if separator in [',', '\t']:
                    reader = csv.reader(f, delimiter=separator)
                    rows = []
                    
                    for i, row in enumerate(reader):
                        if i == 0:
                            file_info["columns"] = row
                            file_info["column_count"] = len(row)
                        elif i <= 5:  # Sample first 5 data rows
                            row_dict = {col: val for col, val in zip(file_info["columns"], row)}
                            file_info["sample_data"].append(row_dict)
                        
                        file_info["row_count"] = i
                        # No row limit - process all data
                else:
                    # Plain text file - just read first few lines
                    f.seek(0)
                    lines = f.readlines()[:10]
                    file_info["sample_data"] = [{"line": line.strip()} for line in lines]
                    file_info["row_count"] = len(lines)
                    
        except Exception as e:
            logger.warning(f"Fallback processing failed for {file_path}: {e}")
        
        return file_info
    
    def _generate_data_summary(self, files_info: Dict[str, Dict]) -> str:
        """Generate a human-readable summary of the data files."""
        if not files_info:
            return "No data files found."
        
        summary_parts = []
        summary_parts.append(f"Data contains {len(files_info)} file(s):")
        
        for filename, info in files_info.items():
            file_desc = f"- {filename}: {info.get('row_count', 0)} rows"
            if info.get('columns'):
                file_desc += f", {len(info.get('columns', []))} columns"
                if len(info['columns']) <= 10:
                    file_desc += f" ({', '.join(info['columns'])})"
            summary_parts.append(file_desc)
        
        return "\n".join(summary_parts)
    
    def _get_full_csv_content(self, capsule_id: str, filename: str, max_rows: Optional[int] = None) -> str:
        """
        Extract full CSV content from a zip file.
        
        Args:
            capsule_id: The capsule ID
            filename: The CSV filename to extract
            max_rows: Maximum number of rows to include (None = all rows)
            
        Returns:
            Full CSV content as string
        """
        zip_file = self._find_capsule_zip(capsule_id)
        if not zip_file:
            return ""
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Find the full path for this filename in the zip
                matching_files = [f for f in zip_ref.namelist() if f.endswith(filename) and 'CapsuleData' in f]
                
                if not matching_files:
                    return ""
                
                file_path = matching_files[0]
                
                # Read the file content directly
                with zip_ref.open(file_path) as f:
                    content_lines = []
                    line_count = 0
                    
                    for line in f:
                        # Decode bytes to string
                        decoded_line = line.decode('utf-8', errors='ignore').strip()
                        content_lines.append(decoded_line)
                        line_count += 1
                        
                        # Limit rows if specified
                        if max_rows and line_count >= max_rows:
                            break
                    
                    return "\n".join(content_lines)
                    
        except Exception as e:
            logger.warning(f"Error reading full content for {filename} in {capsule_id}: {e}")
            return ""
    
    def _get_truncated_csv_with_summary(self, capsule_id: str, filename: str, file_info: Dict, max_rows: int) -> str:
        """
        Get truncated CSV content with comprehensive summary.
        
        Args:
            capsule_id: The capsule ID
            filename: The CSV filename
            file_info: File information from data processing
            max_rows: Maximum number of data rows to include
            
        Returns:
            String with summary and truncated CSV content
        """
        # Get the full CSV content first to analyze it
        full_data = self._get_full_csv_content(capsule_id, filename, None)
        if not full_data:
            return ""
            
        lines = full_data.split('\n')
        if not lines:
            return ""
        
        # Analyze the data structure
        header_line = lines[0] if lines else ""
        data_lines = lines[1:] if len(lines) > 1 else []
        total_rows = len(data_lines)
        
        # Count columns
        if header_line:
            # Try to detect separator
            separators = ['\t', ',', '|', ';']
            separator = '\t'  # Default
            max_cols = 0
            for sep in separators:
                cols = len(header_line.split(sep))
                if cols > max_cols:
                    max_cols = cols
                    separator = sep
            
            columns = header_line.split(separator)
            num_columns = len(columns)
        else:
            columns = []
            num_columns = 0
            separator = '\t'
        
        # Create summary
        summary_parts = []
        summary_parts.append(f"[DATA SUMMARY]")
        summary_parts.append(f"Total rows: {total_rows:,}")
        summary_parts.append(f"Total columns: {num_columns}")
        summary_parts.append(f"File size: ~{len(full_data):,} characters")
        
        if columns and len(columns) <= 20:  # Show column names if reasonable number
            summary_parts.append(f"Columns: {', '.join(columns[:10])}")
            if len(columns) > 10:
                summary_parts.append(f"  ... and {len(columns)-10} more columns")
        elif num_columns > 20:
            summary_parts.append(f"Columns: {columns[0] if columns else 'Unknown'}, ... ({num_columns} total)")
        
        # Determine truncation strategy
        if total_rows <= max_rows:
            # Small file - include everything
            truncated_lines = [header_line] + data_lines
            summary_parts.append(f"[COMPLETE DATA - All {total_rows} rows shown]")
        else:
            # Large file - smart truncation: header + first rows + last rows  
            head_rows = max_rows // 2
            tail_rows = max_rows - head_rows
            
            truncated_lines = [header_line]
            truncated_lines.extend(data_lines[:head_rows])
            truncated_lines.append(f"... [{total_rows - max_rows:,} rows omitted] ...")
            if tail_rows > 0:
                truncated_lines.extend(data_lines[-tail_rows:])
            
            summary_parts.append(f"[TRUNCATED DATA - Showing first {head_rows} and last {tail_rows} rows of {total_rows:,} total]")
        
        # Sample data analysis for numeric columns
        if data_lines:
            sample_row = data_lines[0].split(separator)
            numeric_cols = []
            for i, value in enumerate(sample_row):
                try:
                    float(value)
                    if i < len(columns):
                        numeric_cols.append(columns[i])
                    else:
                        numeric_cols.append(f"Col{i+1}")
                except ValueError:
                    pass
            
            if numeric_cols:
                summary_parts.append(f"Numeric columns detected: {len(numeric_cols)} ({', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''})")
        
        # Combine summary and data
        summary = '\n'.join(summary_parts)
        truncated_data = '\n'.join(truncated_lines)
        
        return f"{summary}\n\n[DATA CONTENT]\n{truncated_data}"
    
    def format_data_for_prompt(self, capsule_id: str, max_rows_per_file: int = None) -> str:
        """
        Format CSV content with truncation and summaries for inclusion in agent prompts.
        New format: filename:\n[SUMMARY]\n[TRUNCATED CONTENT]\n\nfilename2:\n[SUMMARY]\n[TRUNCATED CONTENT]
        
        Args:
            capsule_id: The capsule ID to get data for
            max_rows_per_file: Maximum number of rows to include per file (default: 100)
            
        Returns:
            Truncated CSV content with summaries formatted for prompts
        """
        data_info = self.get_capsule_data(capsule_id)
        
        if not data_info or not data_info.get("files"):
            raise RuntimeError(f"No structured data files available for capsule {capsule_id}")
        
        # Default to reasonable truncation limit
        if max_rows_per_file is None:
            max_rows_per_file = 100
            
        file_contents = []
        
        for filename, file_info in data_info["files"].items():
            # Get truncated data content with summary
            truncated_content = self._get_truncated_csv_with_summary(capsule_id, filename, file_info, max_rows_per_file)
            
            if truncated_content:
                # New format: filename:\n[summary and truncated content]
                file_contents.append(f"{filename}:\n{truncated_content}")
            else:
                # Raise error if data extraction fails
                raise RuntimeError(f"Failed to extract data from {filename} in capsule {capsule_id}")
        
        # Join multiple files with double newline
        return "\n\n".join(file_contents)


def get_capsule_data_context(capsule_id: str, bixbench_dataset_dir: str = "bixbench_dataset") -> str:
    """
    Convenience function to get formatted data context for a capsule.
    
    Args:
        capsule_id: The capsule ID (e.g., 'bix-1')
        bixbench_dataset_dir: Path to BixBench dataset directory
        
    Returns:
        Formatted data context string for prompts
    """
    try:
        loader = BixBenchDataLoader(bixbench_dataset_dir)
        return loader.format_data_for_prompt(capsule_id)
    except Exception as e:
        logger.error(f"Error loading data context for {capsule_id}: {e}")
        return f"Error loading data for {capsule_id}: {e}"