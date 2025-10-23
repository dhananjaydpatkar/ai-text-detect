# BERT MLM Iterative Augmenter

This small project masks a subset of words in input text (excluding verbs, prepositions, common function words) and uses a BERT-style Masked Language Model to fill the masks. It repeats this for a configured number of iterations and prints outputs for each iteration.

Prerequisites
- Python 3.8+
- Install dependencies:

  python -m pip install -r requirements.txt

- Install a spaCy English model if not present:

  python -m spacy download en_core_web_sm

Detailed prerequisites and notes
- Recommended: create and activate a virtual environment before installing dependencies:

  python3 -m venv .venv
  source .venv/bin/activate

- The core Python packages used are:
  - transformers (Hugging Face) — for BERT masked LM
  - torch — PyTorch backend for model inference
  - spacy — tokenization and POS/morphology
  - numpy — numeric operations (used for semantic similarity)

- Optional but recommended for better semantic similarity:
  - sentence-transformers — for SBERT embeddings (model `all-MiniLM-L6-v2` used by default)
    Install with:

    python -m pip install sentence-transformers

Installing a larger spaCy model with vectors (optional)
- If you want better spaCy-based semantic similarity (fallback), install a model with word vectors:

  python -m spacy download en_core_web_md


Quick run

  python bert_mlm_augmenter.py --text "The quick brown fox jumps over the lazy dog." --iterations 3 --mask_prob 0.25

Example with seed and alternative spaCy model

  python bert_mlm_augmenter.py --text "The quick brown fox jumps over the lazy dog." --iterations 3 --mask_prob 0.25 --seed 42 --spacy_model en_core_web_md

How to run (step-by-step)
1. Create/activate virtualenv (optional but recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install requirements:

```bash
python -m pip install -r requirements.txt
```

3. Install spaCy model:

```bash
python -m spacy download en_core_web_sm
```

4. (Optional) Install sentence-transformers for better semantic similarity:

```bash
python -m pip install sentence-transformers
```

5. Run the script (see Quick run above).

Understanding the output
- For each iteration the script prints a block like:

  --- Iteration 1 ---
  The quick brown fox leaps over the lazy dog.
  masked tokens (spaCy index:word): 2:brown, 4:jumps
  morphological similarity to original: 83.33%
  semantic similarity to original: 78.45%

Explanation of fields
- Generated sentence: The text produced by filling the masked tokens with the BERT MLM predictions for that iteration.
- masked tokens (spaCy index:word): spaCy token indices (0-based) and the original words that were masked in that iteration. Use these to know which words were replaced by masks.
- morphological similarity to original: heuristic percent score comparing tokens' lemmas, POS and morphological features (higher means more morphologically similar to original).
- semantic similarity to original: cosine similarity between sentence embeddings (SBERT if installed, otherwise spaCy doc.similarity), reported as a percentage (higher means more semantically similar).

Testing
- Unit tests are provided under the `tests/` directory. To run them:

```bash
pytest -q
```

Troubleshooting
- If you see errors importing spaCy models, ensure you downloaded `en_core_web_sm` (or another model) and that your environment's Python matches the one where the model was installed.
- If semantic similarity appears low or is 0.0, install `sentence-transformers` for better embeddings.
- For large inputs, remember that BERT-style models have a maximum sequence length (typically 512 tokens); long inputs will be truncated by the tokenizer. Split long documents if needed.

CLI parameters (detailed)
-------------------------
Below are the available CLI parameters, their types, defaults, valid ranges, behavior, and examples.

- `--text` (string) — REQUIRED
  - Default: none
  - Description: The input sentence or text to be masked and processed. This is the original text used for masking on every iteration.
  - Notes: Keep inputs short enough to fit your model's max sequence length (typically 512 tokens for BERT-base). For longer documents, chunk before processing.
  - Example: `--text "The quick brown fox jumps over the lazy dog."`

- `--iterations` (int)
  - Default: `3`
  - Valid: integer >= 1
  - Description: Number of times to perform the mask → fill cycle. Each iteration uses the original input and samples a new random set of masks, producing independent outputs.
  - Example: `--iterations 5`
  - Edge case: `0` will produce no iterations; prefer to pass >=1.

- `--mask_prob` (float)
  - Default: `0.15`
  - Valid: `0.0` to `1.0`
  - Description: Fraction of eligible tokens to mask each iteration. Eligible tokens are determined after POS filtering (see EXCLUDE_POS below).
  - Behavior detail: The script computes `num_to_mask = max(1, int(len(candidates) * mask_prob))` if there are eligible tokens. This ensures at least one token is masked when candidates exist. For a true "zero masks" behavior you can set `mask_prob` to 0 and we can change code to allow zero masks explicitly.
  - Examples: `--mask_prob 0.2` masks ~20% of eligible tokens per iteration.

- `--bert_model` (string)
  - Default: `bert-base-uncased`
  - Description: Hugging Face model name for the masked-LM (must include a MaskedLM head). The model will be downloaded if not present locally.
  - Example: `--bert_model bert-large-uncased`

- `--spacy_model` (string)
  - Default: `en_core_web_sm`
  - Description: spaCy model used for tokenization, POS tagging, lemmas and morphological features. Use a model with vectors (e.g., `en_core_web_md`) for better spaCy-derived semantic similarity.
  - Example: `--spacy_model en_core_web_md`

- `--seed` (int)
  - Default: `None`
  - Description: Optional random seed to make mask selection deterministic. If set, `random.seed(seed)` is called. If you want different but reproducible masks per iteration, consider deriving per-iteration seeds (e.g., `seed + i`).
  - Example: `--seed 42`

Internal behavior notes
- EXCLUDE_POS (hard-coded): `{"VERB", "AUX", "ADP", "DET", "PRON", "PART", "INTJ"}`.
  - These coarse POS tags are excluded from masking (verbs, auxiliaries, prepositions/adpositions, determiners, pronouns, particles, interjections).
  - If you'd prefer to only mask nouns/adjectives, or to include verbs, we can add CLI flags to customize the included/excluded POS sets.

Masking and tokenization caveats
- The script selects spaCy tokens to mask and inserts a single `[MASK]` token in place of each selected token. However, the tokenizer may split a token into multiple wordpieces — e.g., `playing` -> `play` + `##ing`. Currently we insert one `[MASK]` token per spaCy token; improving fidelity would involve inserting the same number of masks as wordpieces (this can be added as an option).

Output fields explained
- Generated sentence: The reconstructed sentence after the model fills the masks.
- masked tokens (spaCy index:word): Lists the spaCy token indices and the original word that was masked (indices are 0-based).
- morphological similarity to original: A heuristic percent score (0–100%) computed from token lemmas / POS / morph features.
- semantic similarity to original: Cosine similarity between sentence embeddings (SBERT if `sentence-transformers` is installed, otherwise spaCy `Doc.similarity` fallback).

Suggested CLI additions (optional)
- `--chain` : make iterations chain (use output of iteration i as input to iteration i+1).
- `--per_iter_seed`: derive per-iteration deterministic seeds from base seed.
- `--pos_include` / `--pos_exclude`: customize POS filters from the CLI.
- `--mask_wordpiece_aware`: when enabled, insert multiple masks per token to match tokenizer wordpieces.

Notes
- The script excludes tokens whose spaCy coarse POS is one of VERB, AUX, ADP, DET, PRON, PART, INTJ from masking. This avoids masking common function words.
- Mask selection is random (seed not fixed). You can modify the script to set a fixed random seed for reproducible runs.
- The script uses the top single-token prediction for each [MASK]. It doesn't perform beam search or sampling across multiple possibilities.

Additional options
- --seed INTEGER: optional random seed to make mask selection reproducible across runs. Example:

  python bert_mlm_augmenter.py --text "A sample sentence" --iterations 2 --mask_prob 0.2 --seed 42

Input length limits
- The practical input length limit is determined by the tokenizer/model's maximum sequence length (typically 512 tokens for BERT-base). A long input will be truncated by the tokenizer. If you need to handle longer documents, split the input into chunks shorter than the model's max length and process them separately.

Future improvements
- Allow customizing POS include/exclude lists via CLI
- Add batch processing and caching for speed
- Use a different fill strategy (sampling, top-k, or beam search)

Semantic similarity
- The script computes a semantic similarity percentage between the original input and each generated sentence.
- If the `sentence-transformers` package is installed, the script uses a small SBERT model (`all-MiniLM-L6-v2`) for embeddings (recommended for best results). Otherwise it falls back to spaCy's `Doc.similarity` (which requires a spaCy model with vectors for better results).

To install SentenceTransformers:

  python -m pip install sentence-transformers
