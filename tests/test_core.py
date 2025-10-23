import pytest
import sys
import os
from pathlib import Path
import spacy

# Ensure the project root is on sys.path so tests can import the module directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import functions from the module
from bert_mlm_augmenter import (
    select_mask_indices,
    mask_text,
    compare_morphological_similarity,
    compute_semantic_similarity,
)


@pytest.fixture(scope="module")
def nlp():
    # Ensure the small model is available; tests will fail if not installed.
    return spacy.load("en_core_web_sm")


def test_select_mask_indices_basic(nlp):
    doc = nlp("The quick brown fox jumps over the lazy dog.")
    indices = select_mask_indices(doc, 0.5)
    # Should return a list of integers within token range
    assert isinstance(indices, list)
    for i in indices:
        assert isinstance(i, int)
        assert 0 <= i < len(doc)


def test_mask_text_preserves_whitespace(nlp):
    text = "The quick brown fox."
    # Use a tiny dummy tokenizer object with the attributes needed by mask_text
    class DummyTokenizer:
        mask_token = "[MASK]"
        mask_token_id = 103

    tokenizer = DummyTokenizer()
    masked_text, mask_indices, masked_words = mask_text(text, nlp, 0.5, tokenizer)
    assert isinstance(masked_text, str)
    assert isinstance(mask_indices, list)
    assert isinstance(masked_words, list)
    # number of masked words should equal number of indices
    assert len(mask_indices) == len(masked_words)


def test_compare_morphological_similarity_identical(nlp):
    s = "The quick brown fox jumps over the lazy dog."
    sim = compare_morphological_similarity(s, s, nlp)
    assert sim == 100.0


def test_compute_semantic_similarity_fallback(nlp):
    s1 = "The cat sat on the mat."
    s2 = "A dog lay on the rug."
    # This will use spaCy fallback if sentence-transformers isn't installed
    sim = compute_semantic_similarity(s1, s2, nlp)
    assert isinstance(sim, float)
    assert 0.0 <= sim <= 100.0
