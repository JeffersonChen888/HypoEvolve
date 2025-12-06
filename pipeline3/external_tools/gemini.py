"""
Gemini API Integration for Pipeline3

Provides interface to Google's Gemini 2.5 Pro model for hypothesis generation and evaluation.
"""

import logging
import os
import time
from typing import Optional

# Load environment variables from .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Initialize the Google Generative AI client
try:
    import google.generativeai as genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)
except ImportError:
    raise ImportError("Google Generative AI package not installed. Run: pip install google-generativeai")

# Model configuration
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")  # Default to Gemini 2.5 Pro

def gemini_generate(prompt: str, max_tokens: int = 2000, temperature: float = 0.7, model: Optional[str] = None) -> str:
    """Make an actual API call to Google's Gemini model to generate a response based on the given prompt.

    Args:
        prompt: The text prompt to send to the model
        max_tokens: Maximum number of tokens to generate in the response
        temperature: Controls randomness in the response (0.0-2.0)
        model: Model name to use (default: None, uses MODEL_NAME). Options: 'gemini-2.5-pro', 'gemini-2.0-flash'

    Returns:
        The generated text response from the model

    Raises:
        ValueError: If API key is not found
        Exception: For any API errors or other issues (pipeline will terminate)
    """

    if not os.getenv("GOOGLE_API_KEY"):
        error_msg = "ERROR: GOOGLE_API_KEY not found in environment variables"
        logging.error(error_msg)
        raise ValueError(error_msg)

    # Use provided model or fall back to default MODEL_NAME
    selected_model = model if model is not None else MODEL_NAME

    try:
        # Configure generation parameters
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # Create the model
        model_instance = genai.GenerativeModel(
            model_name=selected_model,
            generation_config=generation_config,
        )

        # Generate content
        response = model_instance.generate_content(prompt)

        # Extract the response text
        response_text = response.text

        # Calculate token usage (Gemini provides usage metadata)
        if hasattr(response, 'usage_metadata'):
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            total_tokens = response.usage_metadata.total_token_count
        else:
            # Fallback if usage metadata not available
            input_tokens = len(prompt.split()) * 1.3  # Rough estimate
            output_tokens = len(response_text.split()) * 1.3
            total_tokens = input_tokens + output_tokens

        # Calculate cost for Gemini 2.5 Pro
        # Gemini 2.5 Pro pricing: $1.25/1M input, $5.00/1M output (128K context)
        # Gemini 2.0 Flash pricing: $0.075/1M input, $0.30/1M output
        if 'flash' in selected_model.lower():
            cost = (input_tokens * 0.075 / 1_000_000) + (output_tokens * 0.30 / 1_000_000)
        else:  # gemini-2.5-pro
            cost = (input_tokens * 1.25 / 1_000_000) + (output_tokens * 5.00 / 1_000_000)

        # Log token usage and cost
        logging.info(f"TOKEN_USAGE: input={int(input_tokens)}, output={int(output_tokens)}, total={int(total_tokens)}, cost=${cost:.6f}")

        # Log the full response with clear separators
        logging.info(f"{selected_model.upper()} RESPONSE START" + "=" * 50)
        logging.info("\n%s", response_text)
        logging.info(f"{selected_model.upper()} RESPONSE END" + "=" * 50)

        return response_text

    except Exception as e:
        # Log the error and terminate
        error_message = f"{selected_model} API call failed: {str(e)}"
        logging.error(error_message)
        # Re-raise the exception to terminate the pipeline
        raise
