#!/usr/bin/env python3
"""
Specific test for o1-mini model (OpenAI's reasoning model).

Tests reasoning capabilities and the special responses API format.
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

from pipeline3.external_tools.gpt4o import gpt4o_generate


def test_reasoning_model():
    """Test o1-mini reasoning capabilities."""
    prompt = """Solve this step by step:
    
If a cell has a mutation in gene A, it survives. If it has a mutation in gene B, it also survives.
But if it has mutations in both gene A and gene B simultaneously, the cell dies.

What is this phenomenon called? Explain your reasoning."""
    
    print("Test 1: Reasoning model")
    print(f"Prompt length: {len(prompt)} characters")
    
    try:
        response = gpt4o_generate(
            prompt=prompt,
            max_tokens=300,
            temperature=0.7,
            model="o1-mini"
        )
        assert response, "Response should not be empty"
        print(f"✓ PASS: Generated {len(response)} characters")
        print(f"Response: {response[:300]}...")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        print("Note: o1-mini may not be available in all API accounts")
        return False


def test_scientific_reasoning():
    """Test scientific reasoning capabilities."""
    prompt = """Analyze this hypothesis:
    
"KDM5C and AURKA form a synthetic lethal pair in cancer cells."

What evidence would support this hypothesis? List 2-3 key points."""
    
    print(f"\nTest 2: Scientific reasoning")
    
    try:
        response = gpt4o_generate(
            prompt=prompt,
            max_tokens=200,
            temperature=0.7,
            model="o1-mini"
        )
        assert response, "Response should not be empty"
        print(f"✓ PASS: Generated {len(response)} characters")
        print(f"Response: {response}")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_reasoning_effort():
    """Test different reasoning effort levels (if supported)."""
    prompt = "What is synthetic lethality? Answer in one sentence."
    
    print(f"\nTest 3: Reasoning effort levels")
    
    # Note: reasoning_effort parameter may not be fully implemented
    # This test checks if the model responds at all
    try:
        response = gpt4o_generate(
            prompt=prompt,
            max_tokens=100,
            temperature=0.7,
            model="o1-mini",
            reasoning_effort="medium"
        )
        if response:
            print(f"✓ PASS: Model responded with reasoning effort parameter")
            print(f"Response: {response}")
            return True
        else:
            print("✗ FAIL: Empty response")
            return False
    except Exception as e:
        print(f"✗ FAIL: {e}")
        print("Note: reasoning_effort parameter may not be supported")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("o1-mini API Test Suite")
    print("=" * 60)
    print("Note: o1-mini uses OpenAI's reasoning API format")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found")
        return
    
    tests = [
        ("Reasoning Model", test_reasoning_model),
        ("Scientific Reasoning", test_scientific_reasoning),
        ("Reasoning Effort", test_reasoning_effort),
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{name:30} {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed < total:
        print("\nNote: Some tests may fail if o1-mini is not available")
        print("in your OpenAI API account or if there are API changes.")


if __name__ == "__main__":
    main()

