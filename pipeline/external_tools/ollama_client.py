import logging
import requests
import json
from typing import Optional

# Ollama API endpoint
OLLAMA_BASE_URL = "http://localhost:11434"

def ollama_generate(prompt: str, max_tokens: int = 2000, temperature: float = 0.7, model: str = "qwen2.5:32b") -> str:
    """Make an API call to local Ollama instance to generate a response.

    Args:
        prompt: The text prompt to send to the model
        max_tokens: Maximum number of tokens to generate in the response
        temperature: Controls randomness in the response (0.0-2.0)
        model: Model name to use (default: qwen2.5:32b)
               Available models: qwen2.5:32b, llama3.1:70b, etc.

    Returns:
        The generated text response from the model

    Raises:
        Exception: For any API errors or connection issues
    """

    logging.info(f"Using Ollama model: {model} (temperature={temperature}, no token limit)")

    # Ollama API endpoint
    url = f"{OLLAMA_BASE_URL}/api/generate"

    # Build request payload
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # Get complete response at once
        "options": {
            "temperature": temperature,
            # No num_predict - let model generate until natural completion
        }
    }

    try:
        # Make request to Ollama (no timeout - local models can be slow)
        response = requests.post(url, json=payload, timeout=None)
        response.raise_for_status()

        # Parse response
        result = response.json()
        response_text = result.get("response", "")

        # Log token usage if available
        if "eval_count" in result:
            eval_tokens = result.get("eval_count", 0)
            prompt_tokens = result.get("prompt_eval_count", 0)
            total_tokens = prompt_tokens + eval_tokens

            # No cost for local models
            logging.info(f"TOKEN_USAGE: input={prompt_tokens}, output={eval_tokens}, total={total_tokens}, cost=$0.000000 (local)")

        # Check if response is empty
        if not response_text:
            logging.error(f"Empty response received from Ollama model {model}")
            logging.error(f"Response JSON keys: {list(result.keys())}")
            logging.error(f"Done reason: {result.get('done_reason', 'unknown')}")

            # Diagnostic: check if model hit token limit while thinking
            if result.get('done_reason') == 'length' and 'thinking' in result:
                thinking_preview = result['thinking'][:500] if result['thinking'] else 'none'
                logging.error(f"Model hit token limit during reasoning phase. Thinking preview: {thinking_preview}...")
                raise ValueError(f"Ollama reasoning model exhausted tokens before producing response. Increase max_tokens.")

            raise ValueError(f"Ollama returned empty response")

        # Log the full response with clear separators
        logging.info(f"{model.upper()} RESPONSE START" + "=" * 50)
        logging.info("\n%s", response_text)
        logging.info(f"{model.upper()} RESPONSE END" + "=" * 50)

        return response_text

    except requests.exceptions.Timeout:
        error_msg = f"Ollama API timeout. Model {model} request was interrupted."
        logging.error(error_msg)
        raise Exception(error_msg)

    except requests.exceptions.ConnectionError as e:
        error_msg = f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Is Ollama running?"
        logging.error(error_msg)
        logging.error(f"Connection error details: {str(e)}")
        raise Exception(error_msg)

    except requests.exceptions.HTTPError as e:
        error_msg = f"Ollama API error: {response.status_code} - {response.text}"
        logging.error(error_msg)

        # Check for common issues
        if response.status_code == 404:
            logging.error(f"Model '{model}' not found. Run: ollama pull {model}")

        raise Exception(error_msg)

    except Exception as e:
        error_msg = f"Unexpected error calling Ollama: {type(e).__name__} - {str(e)}"
        logging.error(error_msg)
        raise Exception(error_msg)


def list_ollama_models() -> list:
    """List available models in local Ollama instance.

    Returns:
        List of model names
    """
    try:
        url = f"{OLLAMA_BASE_URL}/api/tags"
        response = requests.get(url, timeout=10.0)
        response.raise_for_status()

        result = response.json()
        models = [model["name"] for model in result.get("models", [])]

        return models

    except Exception as e:
        logging.warning(f"Could not list Ollama models: {str(e)}")
        return []
