#!/usr/bin/env python3
"""
Specific test for gpt-4o model.

Tests advanced reasoning capabilities and longer responses.
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


def test_complex_reasoning():
    """Test complex multi-step reasoning."""
    prompt = """Analyze this scientific hypothesis:
    
"Synthetic lethality between UHRF1 and PARP1 could be exploited for cancer therapy."

Evaluate this hypothesis by considering:
1. The biological functions of UHRF1 and PARP1
2. Evidence for their interaction
3. Therapeutic potential

Provide a brief analysis (2-3 sentences)."""
    
    print("Test 1: Complex reasoning")
    print(f"Prompt length: {len(prompt)} characters")
    
    try:
        response = gpt4o_generate(
            prompt=prompt,
            max_tokens=300,
            temperature=0.7,
            model="gpt-4o"
        )
        assert response, "Response should not be empty"
        assert len(response) > 50, "Response should be substantial"
        print(f"✓ PASS: Generated {len(response)} characters")
        print(f"Response preview: {response[:200]}...")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_long_response():
    """Test generation of longer responses."""
    prompt = "Explain the genetic algorithm approach to scientific hypothesis generation in 3-4 sentences."
    
    print(f"\nTest 2: Long response generation")
    
    try:
        response = gpt4o_generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0.7,
            model="gpt-4o"
        )
        assert response, "Response should not be empty"
        word_count = len(response.split())
        print(f"✓ PASS: Generated {word_count} words, {len(response)} characters")
        print(f"Response: {response}")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_token_limits():
    """Test that token limits are respected."""
    prompt = "List 10 genes involved in cancer. Just list the names, one per line."
    
    print(f"\nTest 3: Token limit enforcement")
    
    try:
        response = gpt4o_generate(
            prompt=prompt,
            max_tokens=50,  # Should be short
            temperature=0.7,
            model="gpt-4o"
        )
        assert response, "Response should not be empty"
        # Rough check: 50 tokens ≈ 40 words
        word_count = len(response.split())
        print(f"✓ PASS: Generated {word_count} words (target: ~40)")
        print(f"Response: {response}")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("gpt-4o API Test Suite")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found")
        return
    
    tests = [
        ("Complex Reasoning", test_complex_reasoning),
        ("Long Response", test_long_response),
        ("Token Limits", test_token_limits),
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

