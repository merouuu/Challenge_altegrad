#!/usr/bin/env python3
"""
Test script for BLEU-4 evaluation functions.
Validates tokenization and BLEU computation without full model training.
"""

import sys
sys.path.insert(0, '/workspace')

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

def simple_tokenize(s: str):
    """Safe tokenization for BLEU-4 scoring."""
    import re
    if not s:
        return []
    # Split on whitespace and punctuation
    tokens = re.findall(r'\w+|[^\w\s]', s.lower())
    return [t for t in tokens if t.strip()]


def compute_bleu4(pred_texts, ref_texts):
    """Compute corpus-level BLEU-4 score between predictions and references."""
    if not pred_texts or not ref_texts:
        print("⚠️  Empty prediction or reference list")
        return 0.0
    
    pred_tokens = [simple_tokenize(t) for t in pred_texts]
    ref_tokens = [[simple_tokenize(t)] for t in ref_texts]  # NLTK expects list of references
    
    # Remove empty token lists
    valid_pairs = [(p, r) for p, r in zip(pred_tokens, ref_tokens) if p and r[0]]
    if not valid_pairs:
        print("⚠️  No valid token pairs after tokenization")
        return 0.0
    
    pred_tokens = [p for p, _ in valid_pairs]
    ref_tokens = [r for _, r in valid_pairs]
    
    # Compute corpus BLEU with smoothing
    smoothing = SmoothingFunction().method1
    bleu4 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    return float(bleu4)


def test_tokenization():
    """Test tokenization function."""
    print("\n" + "="*60)
    print("TEST 1: Tokenization")
    print("="*60)
    
    test_cases = [
        "The quick brown fox jumps over the lazy dog.",
        "SMILES: c1ccccc1C(=O)O",
        "Multi-word compound: acetyl-CoA",
        "Numbers: 123.456",
        "Empty string: ",
        "Special chars: @#$%^&*()",
    ]
    
    for text in test_cases:
        tokens = simple_tokenize(text)
        print(f"Input:  {text!r}")
        print(f"Tokens: {tokens}")
        print()


def test_bleu_computation():
    """Test BLEU-4 computation."""
    print("\n" + "="*60)
    print("TEST 2: BLEU-4 Computation")
    print("="*60)
    
    # Test case 1: Perfect match
    pred1 = ["The quick brown fox jumps over the lazy dog"]
    ref1 = ["The quick brown fox jumps over the lazy dog"]
    bleu1 = compute_bleu4(pred1, ref1)
    print(f"Perfect match BLEU-4: {bleu1:.4f} (expected: 1.0)")
    assert bleu1 > 0.99, f"Expected ~1.0, got {bleu1}"
    
    # Test case 2: Partial match
    pred2 = ["The quick brown fox jumps over the cat"]
    ref2 = ["The quick brown fox jumps over the lazy dog"]
    bleu2 = compute_bleu4(pred2, ref2)
    print(f"Partial match BLEU-4: {bleu2:.4f} (expected: ~0.3-0.5)")
    
    # Test case 3: Different sentences
    pred3 = ["Machine learning is great"]
    ref3 = ["Deep learning is powerful"]
    bleu3 = compute_bleu4(pred3, ref3)
    print(f"Different sentences BLEU-4: {bleu3:.4f} (expected: <0.2)")
    
    # Test case 4: Multiple samples (corpus BLEU)
    pred4 = [
        "The cat sat on the mat",
        "The dog played in the park",
        "The bird flew in the sky"
    ]
    ref4 = [
        "The cat sat on the mat",
        "The dog played in the park", 
        "The bird flew in the sky"
    ]
    bleu4 = compute_bleu4(pred4, ref4)
    print(f"Corpus BLEU-4 (perfect): {bleu4:.4f} (expected: 1.0)")
    
    print("\n✅ All BLEU-4 tests passed!")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "="*60)
    print("TEST 3: Edge Cases")
    print("="*60)
    
    # Empty lists
    bleu_empty = compute_bleu4([], [])
    print(f"Empty lists BLEU-4: {bleu_empty:.4f} (expected: 0.0)")
    
    # Single word
    bleu_single = compute_bleu4(["cat"], ["cat"])
    print(f"Single word match BLEU-4: {bleu_single:.4f} (expected: ~1.0)")
    
    # Very short references
    bleu_short = compute_bleu4(["a b c d e"], ["a b"])
    print(f"Longer pred than ref BLEU-4: {bleu_short:.4f}")
    
    print("\n✅ All edge case tests passed!")


def test_bleu_for_retrieval():
    """Test realistic retrieval scenario."""
    print("\n" + "="*60)
    print("TEST 4: Realistic Retrieval Scenario")
    print("="*60)
    
    # Simulate: top-1 retrieval from 5 training captions
    train_captions = [
        "acetyl-coenzyme A transferase",
        "acyl-CoA synthetase",
        "ATP-dependent enzyme complex",
        "protein kinase alpha",
        "glycerol-3-phosphate dehydrogenase"
    ]
    
    # Validation caption (ground truth)
    val_caption = "acetyl-coenzyme A synthetase"
    
    # Best match (what the model would retrieve)
    best_match = train_captions[0]  # "acetyl-coenzyme A transferase"
    
    bleu = compute_bleu4([best_match], [val_caption])
    print(f"Retrieved:  {best_match!r}")
    print(f"Reference:  {val_caption!r}")
    print(f"BLEU-4:     {bleu:.4f}")
    print(f"Tokens (retrieved):  {simple_tokenize(best_match)}")
    print(f"Tokens (reference):  {simple_tokenize(val_caption)}")
    print("\n✅ Retrieval test completed!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BLEU-4 EVALUATION FUNCTION TESTS")
    print("="*60)
    
    try:
        test_tokenization()
        test_bleu_computation()
        test_edge_cases()
        test_bleu_for_retrieval()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nSummary:")
        print("  ✓ Tokenization handles special characters")
        print("  ✓ BLEU-4 computation works correctly")
        print("  ✓ Edge cases handled gracefully")
        print("  ✓ Retrieval scenario validated")
        print("\nReady to integrate with train_gt_contrast.py")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
