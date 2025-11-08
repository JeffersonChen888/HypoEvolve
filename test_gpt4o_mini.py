#!/usr/bin/env python3
"""
Specific test for gpt-4o-mini model.

Tests basic functionality, error handling, and response quality.
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


def test_basic_generation():
    """Test basic text generation."""
    prompt = "Explain synthetic lethality in one sentence."
    print(f"Test 1: Basic generation")
    print(f"Prompt: {prompt}")
    
    try:
        response = gpt4o_generate(
            prompt=prompt,
            max_tokens=100,
            temperature=0.7,
            model="gpt-4o-mini"
        )
        assert response, "Response should not be empty"
        assert len(response) > 10, "Response should have meaningful content"
        print(f"✓ PASS: Generated {len(response)} characters")
        print(f"Response: {response[:150]}...")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_scientific_question():
    """Test with a scientific question."""
    prompt = """What is the correct answer to this multiple choice question?
    
Question: Which of the following best describes synthetic lethality?
A) A phenomenon where two genes are both required for cell survival
B) A condition where disruption of either gene alone causes cell death
C) A situation where simultaneous disruption of two genes causes cell death, but individual disruption does not
D) A genetic interaction where one gene compensates for the loss of another

Please answer with just the letter (A, B, C, or D)."""
    
    print(f"\nTest 2: Scientific question")
    print(f"Prompt length: {len(prompt)} characters")
    
    try:
        response = gpt4o_generate(
            prompt=prompt,
            max_tokens=50,
            temperature=0.3,
            model="gpt-4o-mini"
        )
        assert response, "Response should not be empty"
        print(f"✓ PASS: Generated response")
        print(f"Response: {response}")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_different_temperatures():
    """Test with different temperature settings."""
    prompt = "Name a gene involved in cancer."
    
    print(f"\nTest 3: Different temperatures")
    
    temperatures = [0.0, 0.7, 1.0]
    results = []
    
    for temp in temperatures:
        try:
            response = gpt4o_generate(
                prompt=prompt,
                max_tokens=50,
                temperature=temp,
                model="gpt-4o-mini"
            )
            if response:
                print(f"✓ Temperature {temp}: {response[:50]}...")
                results.append(True)
            else:
                print(f"✗ Temperature {temp}: Empty response")
                results.append(False)
        except Exception as e:
            print(f"✗ Temperature {temp}: {e}")
            results.append(False)
    
    return all(results)


def main():
    """Run all tests."""
    print("=" * 60)
    print("gpt-4o-mini API Test Suite")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found")
        return
    
    tests = [
        ("Basic Generation", test_basic_generation),
        ("Scientific Question", test_scientific_question),
        ("Different Temperatures", test_different_temperatures),
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


if __name__ == "__main__":
    main()

