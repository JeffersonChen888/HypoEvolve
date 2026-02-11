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

# Import OpenAI library (client created on-demand in function)
try:
    from openai import OpenAI
except ImportError:
    raise ImportError("OpenAI package not installed. Run: pip install openai")

# Global client instance (lazy initialization)
_openai_client = None

def _get_openai_client():
    """Get or create OpenAI client with proper configuration."""
    global _openai_client

    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        _openai_client = OpenAI(
            api_key=api_key,
            timeout=300.0,  # 300 second timeout (5 minutes) for each request
            max_retries=20,  # Retry up to 20 times on connection errors (for unstable networks)
            http_client=None  # Use default httpx client with connection pooling
        )
        logging.info("OpenAI client initialized successfully")

    return _openai_client

# Model configuration
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Default to GPT-4o-mini
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "medium")  # Options: minimal, low, medium, high

def llm_generate(prompt: str, max_tokens: int = 2000, temperature: float = 0.7, model: Optional[str] = None, reasoning_effort: Optional[str] = None) -> str:
    """Make an actual API call to LLM model (OpenAI GPT or Google Gemini) to generate a response.

    Args:
        prompt: The text prompt to send to the model
        max_tokens: Maximum number of tokens to generate in the response
        temperature: Controls randomness in the response (0.0-2.0)
        model: Model name to use (default: None, uses MODEL_NAME).
               Options: 'gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'o3-mini', 'o3', 'gpt-4o', 'gemini-2.5-pro', 'gemini-2.0-flash'
        reasoning_effort: GPT-5 reasoning effort (default: medium). Options: 'minimal', 'low', 'medium', 'high'

    Returns:
        The generated text response from the model

    Raises:
        ValueError: If API key is not found
        Exception: For any API errors or other issues (pipeline will terminate)
    """

    # Use provided model or fall back to default MODEL_NAME
    selected_model = model if model is not None else MODEL_NAME

    # Route to Ollama if it's a local model (contains colon like qwen2.5:32b)
    if ":" in selected_model or selected_model.startswith("ollama/"):
        from .ollama_client import ollama_generate
        # Remove ollama/ prefix if present
        model_name = selected_model.replace("ollama/", "")
        return ollama_generate(prompt, max_tokens, temperature, model_name)

    # Route to Gemini if it's a Gemini model
    if selected_model.startswith("gemini"):
        from .gemini import gemini_generate
        return gemini_generate(prompt, max_tokens, temperature, selected_model)

    # Otherwise use OpenAI - get client (lazy initialization)
    try:
        client = _get_openai_client()
    except Exception as e:
        error_msg = f"Failed to initialize OpenAI client: {str(e)}"
        logging.error(error_msg)
        raise

    selected_reasoning_effort = reasoning_effort if reasoning_effort is not None else REASONING_EFFORT

    try:
        # Build API call parameters
        api_params = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant with expertise in scientific research."},
                {"role": "user", "content": prompt}
            ]
        }

        # o3 models use temperature=1 only, no token limit (let them think and respond fully)
        if selected_model.startswith("o3") or selected_model.startswith("o1"):
            # No max_completion_tokens - let model generate until natural completion
            # o-series models only support temperature=1 (default)
            api_params["temperature"] = 1.0
            logging.info(f"Using {selected_model} reasoning model (temperature fixed at 1.0, no token limit)")
        # GPT-5 models use temperature=1 only, no token limit (let them think and respond fully)
        elif selected_model.startswith("gpt-5"):
            # No max_completion_tokens - let model generate until natural completion
            # GPT-5 only supports temperature=1 (default)
            api_params["temperature"] = 1.0
            logging.info(f"Using {selected_model} model (temperature fixed at 1.0, no token limit)")
        else:
            # GPT-4o and earlier use max_tokens and support custom temperature
            api_params["max_tokens"] = max_tokens
            api_params["temperature"] = temperature

        response = client.chat.completions.create(**api_params)

        # Extract the response text
        choice = response.choices[0]
        message = choice.message
        response_text = message.content

        # Debug: Check if content is None or empty
        if not response_text:
            logging.warning(f"Empty response content received from {selected_model}")
            logging.warning(f"Message object: {message}")
            logging.warning(f"Choice finish_reason: {choice.finish_reason}")

            # Check for refusal
            if hasattr(message, 'refusal') and message.refusal:
                logging.error(f"API refused the request: {message.refusal}")
                raise ValueError(f"API refusal: {message.refusal}")

            # For reasoning models, check alternative content fields
            if hasattr(message, 'reasoning_content'):
                logging.info("Found reasoning_content field, using that instead")
                response_text = message.reasoning_content

        # Extract and log token usage
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        # Calculate cost based on model
        # o3-mini: $1.10/1M input, $4.40/1M output
        # GPT-5: $1.25/1M input, $10.00/1M output
        # GPT-5-mini: $0.25/1M input, $2.00/1M output
        # GPT-5-nano: $0.05/1M input, $0.40/1M output
        # GPT-4o: $2.50/1M input, $10.00/1M output
        # GPT-4o-mini: $0.15/1M input, $0.60/1M output
        if 'o3-mini' in selected_model or 'o1-mini' in selected_model:
            cost = (input_tokens * 1.10 / 1_000_000) + (output_tokens * 4.40 / 1_000_000)
        elif 'o3' in selected_model or 'o1' in selected_model:
            # o3 full model pricing (higher than mini)
            cost = (input_tokens * 10.00 / 1_000_000) + (output_tokens * 40.00 / 1_000_000)
        elif 'gpt-5' in selected_model:
            if 'nano' in selected_model:
                cost = (input_tokens * 0.05 / 1_000_000) + (output_tokens * 0.40 / 1_000_000)
            elif 'mini' in selected_model:
                cost = (input_tokens * 0.25 / 1_000_000) + (output_tokens * 2.00 / 1_000_000)
            else:
                cost = (input_tokens * 1.25 / 1_000_000) + (output_tokens * 10.00 / 1_000_000)
        elif 'gpt-4o' in selected_model:
            if 'mini' in selected_model:
                cost = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)
            else:
                cost = (input_tokens * 2.50 / 1_000_000) + (output_tokens * 10.00 / 1_000_000)
        else:
            # Default to o3-mini pricing
            cost = (input_tokens * 1.10 / 1_000_000) + (output_tokens * 4.40 / 1_000_000)

        # Log token usage and cost
        logging.info(f"TOKEN_USAGE: input={input_tokens}, output={output_tokens}, total={total_tokens}, cost=${cost:.6f}")

        # Log the full response with clear separators
        logging.info(f"{selected_model.upper()} RESPONSE START" + "=" * 50)
        logging.info("\n%s", response_text)
        logging.info(f"{selected_model.upper()} RESPONSE END" + "=" * 50)

        return response_text

    except Exception as e:
        # Specific error handling for different connection issues
        error_type = type(e).__name__

        if "timeout" in str(e).lower() or "timed out" in str(e).lower():
            error_message = f"{selected_model} API call timeout after 120s: {str(e)}"
            logging.error(error_message)
            logging.error("Suggestion: Check network connection or try again later")
        elif "connection" in str(e).lower() or "network" in str(e).lower():
            error_message = f"{selected_model} network connection error: {str(e)}"
            logging.error(error_message)
            logging.error("Suggestion: Verify internet connectivity and OpenAI API status")
        elif "rate limit" in str(e).lower() or "429" in str(e):
            error_message = f"{selected_model} rate limit exceeded: {str(e)}"
            logging.error(error_message)
            logging.error("Suggestion: Wait a moment and retry, or reduce request frequency")
        elif "authentication" in str(e).lower() or "401" in str(e):
            error_message = f"{selected_model} authentication failed: {str(e)}"
            logging.error(error_message)
            logging.error("Suggestion: Verify OPENAI_API_KEY is valid")
        else:
            error_message = f"{selected_model} API call failed ({error_type}): {str(e)}"
            logging.error(error_message)

        # Re-raise the exception to terminate the pipeline
        raise 