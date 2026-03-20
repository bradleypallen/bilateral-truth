# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **bilateral-truth**, a Python package for LLM factuality evaluation using generalized truth values from the nine-valued bilattice NINE = ⟨V₃×V₃, ≤_t, ≤_k⟩. Each assertion receives a bilateral truth value ⟨u,v⟩ where u=verifiability and v=refutability, each ∈ {t,e,f}. Based on ArXiv paper 2507.09751v2, "BBL: A Bilateral Modal Logic for LLM Factuality Evaluation."

## Environment Setup

```bash
./setup_venv.sh && source venv/bin/activate
# or manually:
python3 -m venv venv && source venv/bin/activate
pip install -e . && pip install -r requirements.txt
```

API keys are stored in `.env` in the root directory (OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY).

## Core Evaluation Framework

### Primary Evaluator: `proper_benchmark_evaluator.py`

The main evaluation script. Runs five epistemic approaches in a single pass per assertion, all using s₀ = source_question as context:

1. **bilateral** — VM(s₀, p) → ⟨u,v⟩ → projected under 3 epistemic policies (classical, paracomplete, paraconsistent)
2. **forced_unilateral** — forced binary TRUE/FALSE (no abstention)
3. **ternary** — evidence-based TRUE/FALSE/UNCERTAIN (abstains on UNCERTAIN)
4. **confidence @0.5** — numerical 0.0–1.0 thresholded

Key parameters:
- N=250 assertions, seed=42, balanced sampling (50% positive / 50% negative)
- bilateral_samples=3 (majority vote over 3 calls per assertion, matching paper methodology)
- Output: `results/{benchmark}_{model}_n3_proper_results.json`

```bash
cd evaluations
python proper_benchmark_evaluator.py \
  --model meta-llama/llama-4-maverick \
  --dataset standard_datasets/truthfulqa_complete.json \
  --samples 250
```

### Modal Evaluator: `modal_evaluator.py`

Evaluates the modal necessity operator [[□p]] = ⟨min uᵢ, max vᵢ⟩ over s₀ + 2 paraphrases (n=3 total situations). Does NOT include the null situation (no context). Computes:
- `src_bilateral_distribution` — VM(s₀, p) where s₀=source_question
- `modal_bilateral_distribution` — [[□p]] aggregated over s₀ + s₁ + s₂

Variants (paraphrases of s₀) must be pre-generated once per dataset using `pregenerate_variants.py`, cached in `variant_cache/`. Pre-generation uses Claude Opus 4.6 — one-time cost, shared across all evaluation models.

```bash
# Step 1: pre-generate variants (one-time per dataset)
python pregenerate_variants.py \
  --dataset standard_datasets/truthfulqa_complete.json \
  --n-variants 3 --samples 250

# Step 2: run modal evaluation using cached variants
python modal_evaluator.py \
  --model meta-llama/llama-4-maverick \
  --dataset standard_datasets/truthfulqa_complete.json \
  --n-variants 3 --samples 250 \
  --variant-cache variant_cache/truthfulqa_n3_variants.json
```

### Suite Launcher: `run_modal_suite.py`

Orchestrates pregeneration + all modal evaluation jobs. Use `--skip-pregen` if variant caches already exist.

```bash
python run_modal_suite.py --samples 250 --skip-pregen --models opus llama scout gemini deepseek qwen
```

## Current Model Lineup (March 2026)

Six models, skewed toward open-source, all via OpenRouter except Opus 4.1:

| Short | Model ID | Type |
|---|---|---|
| opus | claude-opus-4-1-20250805 | closed (Anthropic) |
| llama | meta-llama/llama-4-maverick | open |
| scout | meta-llama/llama-4-scout | open |
| gemini | google/gemini-2.5-flash | closed (Google) |
| deepseek | deepseek/deepseek-chat | open (DeepSeek-V3) |
| qwen | qwen/qwen-2.5-72b-instruct | open |

**Important:** GPT-4.1 and GPT-4.1-mini were dropped due to OpenAI quota issues and replaced with Llama 4 Scout and Qwen 2.5 72B. Do not reference GPT-4.1 results in new analysis — those result files exist but are not part of the current model lineup.

## Benchmarks

All datasets in `evaluations/standard_datasets/`:

| File | Benchmark name | Size |
|---|---|---|
| `truthfulqa_complete.json` | truthfulqa | ~1,580 assertions |
| `simpleqa_complete.json` | simpleqa | ~21,630 assertions |
| `mmlupro_complete.json` | mmlu-pro | ~110,225 assertions |
| `factscore_complete.json` | factscore | ~33,820 assertions |

Note: the file `mmlupro_complete.json` has benchmark name `mmlu-pro` in its metadata. Result files and variant caches use `mmlu-pro` as the benchmark identifier.

## Result File Naming

- Proper results: `results/{benchmark}_{model_safe}_n3_proper_results.json`
- Modal results: `results/{dataset_stem}_{model_safe}_n3_modal_results.json`
  - dataset_stem = stem of the dataset filename (e.g. `truthfulqa_complete`)
- Variant caches: `variant_cache/{benchmark}_n3_variants.json`
- model_safe = model ID with `/` and `:` replaced by `_`

## Visualization Scripts

All in `evaluations/`:

- `visualize_tv_distributions.py` — 4×6 grid of stacked bars showing VM(s₀,p) TV distributions
- `visualize_modal_distributions.py` — same grid for [[□p]] modal distributions
- `visualize_full_comparison.py` — F1-macro comparison across approaches, delta heatmaps
- `visualize_proper_pilot.py` — TruthfulQA pilot comparison (3-panel)

All outputs go to `results/` as both PDF and PNG.

## Epistemic Policies and D-Parameterization

The three epistemic policies correspond to different choices of the set of designated values D:

- **classical**: D = {⟨t,f⟩} — only clean verifications designated
- **paraconsistent**: D = {⟨t,f⟩, ⟨t,t⟩} — also designates contradictions
- **paracomplete**: D = {⟨t,f⟩, ⟨t,e⟩} — also designates partial verifications

The empirical comparison across policies is a study of how D-parameterization affects downstream performance. This is explicitly a theoretical contribution of the paper.

## Key Empirical Findings (March 2026)

- **Bilateral > forced binary** confirmed across benchmarks
- **Bilateral ≈ ternary (evidence-based)** — not bilateral > ternary as the paper draft claims for older models. This requires reframing in the paper.
- **FACTScore high ⟨f,f⟩** — models systematically cannot verify/refute biographical facts. This is a diagnostic finding (knowledge gap exposure), not a failure of bilateral evaluation. The framework surfaces epistemic structure that unilateral approaches obscure.
- **MMLU-Pro low classical coverage** — high abstention rate under D={⟨t,f⟩}; paraconsistent policy recovers significant coverage.
- **Gemini 2.5 Flash outlier** — uniquely high ⟨f,f⟩ on SimpleQA, hurts bilateral F1 there.

## Central Thesis (for paper framing)

The paper's claim is NOT "bilateral achieves higher accuracy." It is: **bilateral provides actionable information about LLM doxastic states that unilateral and confidence-based approaches cannot express.** The ⟨u,v⟩ pair distinguishes ignorance (⟨f,f⟩) from contradiction (⟨t,t⟩), asymmetric partial knowledge (⟨t,e⟩, ⟨e,f⟩), etc. — epistemic states with no representation in unilateral frameworks.

## Future Work (as discussed)

1. **Neighborhood semantics** — more principled formal basis for the modal accessibility structure
2. **Parameterization on D** — systematic study of how D-choice affects derivability and performance
3. **e as "off topic" (Paoli et al. 2025)** — reinterpreting e as domain-exclusion rather than evaluation failure; has proof-theoretic consequences for the sequent calculus

## Architecture Overview

### Core Package (`bilateral_truth/`)

- `truth_values.py` — `TruthValueComponent` (t/e/f), `GeneralizedTruthValue` ⟨u,v⟩, `EpistemicPolicy` enum, `.project(policy)` method
- `assertions.py` — `Assertion` class, normalized representation for caching
- `zeta_function.py` — `zeta()` / `zeta_c()` implementing the mathematical zeta_c definition
- `llm_evaluators.py` — `LLMEvaluator` ABC, `OpenAIEvaluator`, `AnthropicEvaluator`, `MockLLMEvaluator`; includes `_raw_complete`, evidence-based ternary prompt, N=3 majority voting
- `model_router.py` — `ModelRouter`, `OpenRouterEvaluator`; routes model names to providers
- `variation_generator.py` — `SituationVariationGenerator` using Claude Opus 4.6 for situation paraphrases

### Important Implementation Details

- Use `GeneralizedTruthValue.undefined()` not `unknown()`, `TruthValueComponent.UNDEFINED` not `EMPTY`
- Use `tv.project(policy)` not `tv.apply_policy(policy)`
- Dataset field is `assertion_text` not `statement`
- Ternary prompt is evidence-based ("supported by evidence / contradicted by evidence / insufficient evidence"), NOT confidence self-reporting
- Null situation (no context) is excluded from all current evaluations — always use s₀=source_question as context

## Testing

```bash
python -m pytest                          # unit tests
python -m pytest -m integration          # requires API keys
python -m pytest --cov=bilateral_truth   # with coverage
```
