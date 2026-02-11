"""
Configuration for Pipeline3 - Simplified Genetic Algorithm

This configuration is streamlined for the genetic algorithm approach,
removing complex task management and multi-agent coordination settings.
"""

import logging
import os
import time
from logging.handlers import RotatingFileHandler

# Load environment variables from .env file (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables directly.")

# Model provider config
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # Use OpenAI by default
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Model configuration
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "medium")  # GPT-5 reasoning: minimal, low, medium, high

# Genetic Algorithm Parameters
POPULATION_SIZE = int(os.getenv("POPULATION_SIZE", "6"))  # Initial population size
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "3"))  # Number of evolutionary generations
SELECTION_RATIO = float(os.getenv("SELECTION_RATIO", "0.5"))  # Fraction selected as parents
ELITISM_COUNT = int(os.getenv("ELITISM_COUNT", "2"))  # Top performers preserved
TOURNAMENT_SIZE = int(os.getenv("TOURNAMENT_SIZE", "4"))  # Tournament selection size

# Literature exploration settings
NUM_PAPERS_FOR_EXPLORATION = int(os.getenv("NUM_PAPERS_FOR_EXPLORATION", "3"))
LITERATURE_SEARCH_QUERIES = int(os.getenv("LITERATURE_SEARCH_QUERIES", "3"))

# Fitness evaluation settings - matching Pipeline2's 3-dimension scoring
FITNESS_CORRECTNESS_WEIGHT = float(os.getenv("FITNESS_CORRECTNESS_WEIGHT", "1.0"))
FITNESS_NOVELTY_WEIGHT = float(os.getenv("FITNESS_NOVELTY_WEIGHT", "1.3")) 
FITNESS_QUALITY_WEIGHT = float(os.getenv("FITNESS_QUALITY_WEIGHT", "1.5"))

# Elo rating system
DEFAULT_ELO_SCORE = int(os.getenv("DEFAULT_ELO_SCORE", "1200"))
ELO_K_FACTOR = int(os.getenv("ELO_K_FACTOR", "32"))

# Web search config
USE_WEB_SEARCH = (os.getenv("USE_WEB_SEARCH", "True").lower() == "true")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SEARCH_ENGINE = os.getenv("SEARCH_ENGINE", "tavily")

# PubMed integration
USE_PUBMED = (os.getenv("USE_PUBMED", "True").lower() == "true")
PUBMED_TOOL_LIMIT = int(os.getenv("PUBMED_TOOL_LIMIT", "10"))

# Output settings
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "pipeline/output")
FORMAT_OUTPUT_AS_PAPER = (os.getenv("FORMAT_OUTPUT_AS_PAPER", "False").lower() == "true")

# Safety settings
ENABLE_SAFETY_FILTERS = (os.getenv("ENABLE_SAFETY_FILTERS", "True").lower() == "true")

# Debug and logging
DEBUG_MODE = (os.getenv("DEBUG_MODE", "False").lower() == "true")
VERBOSE_LOGGING = (os.getenv("VERBOSE_LOGGING", "True").lower() == "true")
DETAILED_STATE_LOGGING = (os.getenv("DETAILED_STATE_LOGGING", "False").lower() == "true")

# Get module directory for file paths
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_absolute_path(relative_path):
    """Convert a relative path to an absolute path within the module directory."""
    return os.path.join(MODULE_DIR, relative_path)

# Setup output directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create timestamped log file
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(OUTPUT_DIR, f'pipeline_{TIMESTAMP}.log')

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colorized output for console logs"""
    
    # ANSI color codes
    GREY = "\033[37m"
    BLUE = "\033[34m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BOLD_RED = "\033[31;1m"
    GREEN = "\033[32m"
    RESET = "\033[0m"
    
    # Format strings for different log levels
    FORMATS = {
        logging.DEBUG: GREY + "%(asctime)s [%(levelname)s] %(name)s: %(message)s" + RESET,
        logging.INFO: "%(asctime)s [%(levelname)s] %(name)s: %(message)s" + RESET,
        logging.WARNING: YELLOW + "%(asctime)s [%(levelname)s] %(name)s: %(message)s" + RESET,
        logging.ERROR: RED + "%(asctime)s [%(levelname)s] %(name)s: %(message)s" + RESET,
        logging.CRITICAL: BOLD_RED + "%(asctime)s [%(levelname)s] %(name)s: %(message)s" + RESET,
    }
    
    def format(self, record):
        log_format = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logging(custom_log_file=None):
    """Setup logging configuration for Pipeline3"""
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Suppress verbose debug logs from httpcore/httpx but keep OpenAI INFO for retries
    logging.getLogger('openai').setLevel(logging.INFO)  # Keep INFO for retry messages
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('openai._base_client').setLevel(logging.INFO)  # Keep retry info

    # Create console handler with color formatting
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    root_logger.addHandler(console_handler)
    
    # Create file handler with detailed formatting
    file_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Use custom log file if provided, otherwise use default
    log_file_path = custom_log_file if custom_log_file else LOG_FILE
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(file_format)
    file_handler.setLevel(logging.INFO)

    # Force immediate flushing to file
    class FlushingFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    # Replace with flushing handler
    file_handler = FlushingFileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(file_format)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    
    # Create a dedicated logger for the system
    ga_logger = logging.getLogger('Pipeline3_GeneticAlgorithm')
    ga_logger.setLevel(logging.INFO)
    
    logging.info("=" * 80)
    logging.info(f"Pipeline3 Genetic Algorithm Logging initialized at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Log file: {log_file_path}")
    logging.info(f"Population size: {POPULATION_SIZE}")
    logging.info(f"Generations: {NUM_GENERATIONS}")
    logging.info(f"Selection ratio: {SELECTION_RATIO}")
    logging.info("=" * 80)
    
    return ga_logger

# Genetic Algorithm Strategy Configuration
CROSSOVER_STRATEGIES = ["combination", "inspiration"]
MUTATION_STRATEGIES = ["simplification", "out_of_box"]

# Validation settings
MIN_FITNESS_SCORE = 0.0
MAX_FITNESS_SCORE = 100.0
MIN_HYPOTHESIS_LENGTH = 50  # Minimum characters in hypothesis description
MAX_HYPOTHESIS_LENGTH = 2000  # Maximum characters in hypothesis description