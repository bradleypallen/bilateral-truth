# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

This is the [`bilateral-truth`](https://pypi.org/project/bilateral-truth/) Python package — a library implementing bilateral factuality evaluation with generalized truth values from the nine-valued bilattice NINE = ⟨V₃×V₃, ≤_t, ≤_k⟩. Each assertion is mapped to a pair ⟨u, v⟩ where u (verifiability) and v (refutability) are each ∈ {t, e, f}.

Based on ArXiv paper 2507.09751v2, *"BBL: A Bilateral Modal Logic for LLM Factuality Evaluation"*.

**Empirical evaluations, benchmark datasets, and experimental result files live in a separate repository, `tllm-2026-experiments`. They are intentionally not here.** Do not add result files, benchmark scripts, evaluation runs, or paper-related analysis to this repository.

## Architecture

### Core modules in `bilateral_truth/`

- `truth_values.py` — `TruthValueComponent` (t/e/f), `GeneralizedTruthValue` ⟨u,v⟩, `EpistemicPolicy` enum, `.project(policy)` method
- `assertions.py` — `Assertion` class, normalized representation for caching
- `zeta_function.py` — `zeta()` / `zeta_c()` implementing the mathematical ζ_c definition with persistent caching
- `llm_evaluators.py` — `LLMEvaluator` ABC, `OpenAIEvaluator`, `AnthropicEvaluator`, `MockLLMEvaluator`; verification + refutation prompts (Definition 3.4), evidence-based ternary, forced unilateral, confidence prompts
- `model_router.py` — `ModelRouter`, `OpenRouterEvaluator`; routes model names to providers
- `variation_generator.py` — `SituationVariationGenerator` (paraphrase generation for modal evaluation; uses Claude Opus by default)
- `cli.py` — command-line interface

### Important implementation details

- Use `GeneralizedTruthValue.undefined()` not `unknown()`; `TruthValueComponent.UNDEFINED` not `EMPTY`.
- Use `tv.project(policy)` not `tv.apply_policy(policy)`.
- The bilateral evaluator makes two **separate** calls per assertion: one for verification (VERIFIED / CANNOT VERIFY → u), one for refutation (REFUTED / CANNOT REFUTE → v).
- Majority voting (`samples > 1`) votes on each component independently.
- The ternary prompt is **evidence-based** ("supported by evidence / contradicted by evidence / insufficient evidence"), not confidence self-reporting.
- `zeta_c` uses a process-wide cache keyed by `(assertion, system_prompt, context)`. Different prompts/contexts produce different cached results.

## Setup

```bash
./setup_venv.sh && source venv/bin/activate
# or manually:
python3 -m venv venv && source venv/bin/activate
pip install -e .[all,dev]
```

API keys (if exercising live LLM evaluators) belong in `.env` in the repo root:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...
```

## Testing

```bash
python -m pytest                          # unit tests
python -m pytest -m integration           # integration tests (require API keys)
python -m pytest --cov=bilateral_truth    # with coverage
```

## What does *not* belong in this repo

- Benchmark result files (`*.json` with model results)
- Benchmark scripts (anything that runs an evaluator over a dataset)
- Standard datasets (TruthfulQA, SimpleQA, MMLU-Pro, FACTScore JSON files)
- Variant caches for modal evaluation
- LaTeX figures, paper drafts, slide decks
- Analysis or visualization scripts driven by paper findings

All of the above lives in `tllm-2026-experiments`.
