#!/usr/bin/env python3
"""
Basic testing script for GPT API calls across different models.

This script tests API connectivity and basic functionality for:
- gpt-4o-mini
- gpt-4o
- o1-mini (if available)

Usage:
    python tests/test_gpt_api_calls.py
"""

import os
import sys
from pathlib import Path

# Add pipeline3 to path
pipeline3_dir = Path(__file__).parent.parent / "pipeline3"
sys.path.insert(0, str(pipeline3_dir))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import GPT generation function
from pipeline3.external_tools.gpt4o import gpt4o_generate


def test_model(model_name: str, test_prompt: str = "What is 2+2? Please answer briefly."):
    """Test a specific model with a simple prompt."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")
    
    try:
        print(f"Sending prompt: '{test_prompt}'")
        response = gpt4o_generate(
            prompt=test_prompt,
            max_tokens=200,
            temperature=0.7,
            model=model_name
        )
        
        if response:
            print(f"✓ Success! Response received ({len(response)} characters)")
            print(f"Response preview: {response[:200]}...")
            return True
        else:
            print("✗ Failed: Empty response received")
            return False
            
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {str(e)}")
        return False


def main():
    """Run tests for all models."""
    print("GPT API Testing Suite")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment variables")
        print("Please set your API key in a .env file or environment variable")
        return
    
    print(f"API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test models
    models_to_test = [
        "gpt-4o-mini",
        "gpt-4o",
        "o1-mini"
    ]
    
    results = {}
    for model in models_to_test:
        results[model] = test_model(model)
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    for model, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{model:20} {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")


if __name__ == "__main__":
    main()

