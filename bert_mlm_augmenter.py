#!/usr/bin/env python3
"""
bert_mlm_augmenter.py

Simple CLI to iteratively mask words (excluding certain POS) and fill them using a BERT-style Masked Language Model.

Usage:
    python bert_mlm_augmenter.py --text "Your sentence here." --iterations 3 --mask_prob 0.2

This script uses spaCy for POS tagging and Hugging Face Transformers for MaskedLM.
"""

import argparse
import random
import re
from typing import List

import spacy
from transformers import BertForMaskedLM, BertTokenizerFast
import torch
import numpy as np

# POS tags to exclude from masking (coarse)
EXCLUDE_POS = {"VERB", "AUX", "ADP", "DET", "PRON", "PART", "INTJ"}


def load_models(bert_model_name: str = "bert-base-uncased"):
    tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    model = BertForMaskedLM.from_pretrained(bert_model_name)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return tokenizer, model


def select_mask_indices(doc, mask_prob: float):
    """Given a spaCy doc, return list of token indices to mask.
    Avoid tokens whose coarse POS is in EXCLUDE_POS or are punctuation/space.
    """
    candidates = []
    for i, token in enumerate(doc):
        if token.is_space or token.is_punct:
            continue
        if token.pos_ in EXCLUDE_POS:
            continue
        # skip very short tokens (like 's')
        if len(token.text.strip()) <= 1:
            continue
        candidates.append(i)
    num_to_mask = max(1, int(len(candidates) * mask_prob)) if candidates else 0
    return sorted(random.sample(candidates, num_to_mask)) if num_to_mask > 0 else []


def mask_text(text: str, nlp, mask_prob: float, tokenizer) -> (str, List[int], List[str]):
    """Return masked text, masked spaCy token indices, and masked original words.
    We build masked text at token level using spaCy tokens and then map to tokenizer tokens.
    """
    doc = nlp(text)
    mask_indices = select_mask_indices(doc, mask_prob)
    tokens = [t.text_with_ws for t in doc]

    # Create a text where selected tokens are replaced with [MASK]
    masked_tokens = []
    masked_words = []
    for i, t in enumerate(doc):
        if i in mask_indices:
            # preserve trailing whitespace
            ws = t.whitespace_ or ""
            masked_tokens.append(tokenizer.mask_token + ws)
            masked_words.append(t.text)
        else:
            masked_tokens.append(t.text_with_ws)

    masked_text = "".join(masked_tokens)
    return masked_text, mask_indices, masked_words


def fill_masks(masked_text: str, tokenizer, model) -> str:
    """Given masked_text (containing tokenizer.mask_token), run the model and replace masks with predictions.
    Uses top prediction for each mask.
    """
    inputs = tokenizer(masked_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model.to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_tokens = {}
    for i, mask_index in enumerate(mask_token_index):
        mask_logits = logits[0, mask_index, :]
        top_token = torch.topk(mask_logits, 1).indices.tolist()[0]
        predicted_tokens[int(mask_index)] = tokenizer.convert_ids_to_tokens(top_token)

    # reconstruct final text from tokens
    input_ids = inputs.input_ids[0].tolist()
    decoded_tokens = []
    for idx, id_ in enumerate(input_ids):
        if idx in predicted_tokens:
            tok = predicted_tokens[idx]
            # if token starts with '##' then attach without space
            if tok.startswith('##'):
                tok = tok[2:]
                if decoded_tokens:
                    decoded_tokens[-1] = decoded_tokens[-1] + tok
                else:
                    decoded_tokens.append(tok)
            else:
                decoded_tokens.append(tok)
        else:
            decoded_tokens.append(tokenizer.convert_ids_to_tokens(id_))

    # Detokenize simply by joining and replacing tokenization artifacts
    text = tokenizer.convert_tokens_to_string(decoded_tokens)
    # post-process spacing for punctuation
    text = text.replace(" ,", ",").replace(" .", ".").replace(" ' ", "'")
    return text


def compare_morphological_similarity(orig_text: str, gen_text: str, nlp) -> float:
    """Compare original and generated text morphologically and return a similarity percentage.

    Heuristic scoring per aligned token (non-space, non-punct):
      - lemma match: 1.0
      - same coarse POS: 0.6
      - any overlapping morphological feature: 0.3
      - else: 0.0

    Percentage = (sum(scores) / max_token_count) * 100, rounded to 2 decimals.
    """
    doc_o = nlp(orig_text)
    doc_g = nlp(gen_text)
    toks_o = [t for t in doc_o if not t.is_space and not t.is_punct]
    toks_g = [t for t in doc_g if not t.is_space and not t.is_punct]
    max_len = max(len(toks_o), len(toks_g))
    if max_len == 0:
        return 100.0

    total = 0.0
    for i in range(max_len):
        if i < len(toks_o):
            o = toks_o[i]
        else:
            o = None
        if i < len(toks_g):
            g = toks_g[i]
        else:
            g = None

        if o is None or g is None:
            # missing token in one of the sequences -> score 0 for this position
            continue

        # lemma match (case-insensitive)
        if o.lemma_.lower() == g.lemma_.lower():
            total += 1.0
            continue

        # same coarse POS
        if o.pos_ == g.pos_:
            total += 0.6
            continue

        # overlapping morphological features
        morph_o = set([m for m in str(o.morph).split("|") if m])
        morph_g = set([m for m in str(g.morph).split("|") if m])
        if morph_o & morph_g:
            total += 0.3

    percent = (total / max_len) * 100.0
    return round(percent, 2)


def compute_semantic_similarity(orig_text: str, gen_text: str, nlp, st_model_name: str = "all-MiniLM-L6-v2") -> float:
    """Compute semantic similarity between two sentences as a percentage.

    Primary: use SentenceTransformers (small model 'all-MiniLM-L6-v2') if installed.
    Fallback: use spaCy's `doc.similarity`.
    """
    # Try SentenceTransformers first (lazy import and model creation)
    try:
        from sentence_transformers import SentenceTransformer
        # cache model in globals to avoid reloading
        if "_st_model" not in globals() or globals().get("_st_model_name") != st_model_name:
            globals()['_st_model'] = SentenceTransformer(st_model_name)
            globals()['_st_model_name'] = st_model_name
        model = globals()['_st_model']
        emb = model.encode([orig_text, gen_text], convert_to_numpy=True)
        v0, v1 = emb[0], emb[1]
        # cosine similarity
        denom = (np.linalg.norm(v0) * np.linalg.norm(v1))
        if denom == 0:
            sim = 0.0
        else:
            sim = float(np.dot(v0, v1) / denom)
        return round(sim * 100.0, 2)
    except Exception:
        # Fallback to spaCy's similarity (may require model with vectors for best results)
        try:
            doc_o = nlp(orig_text)
            doc_g = nlp(gen_text)
            sim = doc_o.similarity(doc_g)
            return round(sim * 100.0, 2)
        except Exception:
            return 0.0


def iterate_text(text: str, nlp, tokenizer, model, iterations: int, mask_prob: float):
    results = []
    # Use the original input for every iteration (do not chain outputs)
    for i in range(iterations):
        masked, mask_indices, masked_words = mask_text(text, nlp, mask_prob, tokenizer)
        if not mask_indices:
            # nothing to mask -> stop
            morph_sim = compare_morphological_similarity(text, masked, nlp)
            sem_sim = compute_semantic_similarity(text, masked, nlp)
            results.append((masked, [], morph_sim, sem_sim))
            break
        filled = fill_masks(masked, tokenizer, model)
        # pair each masked index with the original word
        masked_info = list(zip(mask_indices, masked_words))
        morph_sim = compare_morphological_similarity(text, filled, nlp)
        sem_sim = compute_semantic_similarity(text, filled, nlp)
        results.append((filled, masked_info, morph_sim, sem_sim))
    return results


def main():
    parser = argparse.ArgumentParser(description="Iteratively mask and fill text using a BERT MaskedLM")
    parser.add_argument("--text", type=str, required=True, help="Input text (wrap in quotes)")
    parser.add_argument("--iterations", type=int, default=3, help="Number of mask-fill iterations")
    parser.add_argument("--mask_prob", type=float, default=0.15, help="Fraction of eligible words to mask each iteration (0-1)")
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", help="HuggingFace model name")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm", help="spaCy model name")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible masking")
    args = parser.parse_args()

    print(f"Loading spaCy model {args.spacy_model}...")
    nlp = spacy.load(args.spacy_model)
    print(f"Loading BERT model {args.bert_model}...")
    tokenizer, model = load_models(args.bert_model)

    if args.seed is not None:
        random.seed(args.seed)

    # Compute token counts for the input using the tokenizer
    try:
        token_strings = tokenizer.tokenize(args.text)
        token_ids_with_special = tokenizer.encode(args.text, add_special_tokens=True)
        token_count = len(token_strings)
        token_count_with_special = len(token_ids_with_special)
        model_max = getattr(tokenizer, 'model_max_length', None)
    except Exception:
        token_strings = args.text.split()
        token_count = len(token_strings)
        token_count_with_special = token_count
        model_max = None

    print(f"Input: {args.text}")
    if model_max:
        print(f"Token count: {token_count} (without special tokens), {token_count_with_special} (with special tokens). Model max tokens: {model_max}")
        if token_count_with_special > model_max:
            print("Warning: input token length (with special tokens) exceeds model max length and will be truncated by the tokenizer.")
    else:
        print(f"Token count (approx): {token_count}")

    results = iterate_text(args.text, nlp, tokenizer, model, args.iterations, args.mask_prob)

    for i, item in enumerate(results, start=1):
        # item is (generated_text, masked_info, morph_similarity, sem_similarity)
        res, meta, morph_sim, sem_sim = item
        print("\n--- Iteration {} ---".format(i))
        print(res)
        if not meta:
            print("masked token indices: none (no eligible tokens)")
        else:
            # meta is list of (spaCy_token_index, original_word)
            print("masked tokens (spaCy index:word):", ", ".join(f"{idx}:{word}" for idx, word in meta))
        print(f"morphological similarity to original: {morph_sim}%")
        print(f"semantic similarity to original: {sem_sim}%")


if __name__ == "__main__":
    main()
