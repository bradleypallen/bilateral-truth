# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **bilateral-truth**, a Python package for caching bilateral factuality evaluation using generalized truth values. The implementation is based on the mathematical zeta_c function described in ArXiv paper 2507.09751v2, which provides bilateral evaluation of assertions returning <u,v> truth values where u represents verifiability and v represents refutability.

## Development Commands

### Environment Setup
```bash
# Automated setup (creates venv and installs dependencies)
./setup_venv.sh
source venv/bin/activate

# Manual setup
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

### Testing
```bash
# Run full test suite (excludes integration tests by default)
python -m pytest

# Run specific test modules
python -m pytest tests/test_truth_values.py
python -m pytest tests/test_assertions.py
python -m pytest tests/test_zeta_function.py

# Run integration tests (requires API keys)
python -m pytest -m integration

# Run tests with coverage
python -m pytest --cov=bilateral_truth
```

### Running Benchmark Evaluations
```bash
# Generate/update benchmark datasets
cd evaluations
python data_generators/truthfulqa_generator.py
python data_generators/simpleqa_generator.py
python data_generators/mmlupro_generator.py
python data_generators/factscore_generator.py  # Requires downloading data first

# Run evaluations with different models
python generic_evaluator.py --dataset standard_datasets/truthfulqa_complete.json --model gpt-4 --samples 100
python generic_evaluator.py --dataset standard_datasets/factscore_complete.json --model claude-3-5-haiku-20241022 --samples 50

# View results
ls -la results/
```

### Code Quality
```bash
# Code formatting
black bilateral_truth/ tests/

# Linting
flake8 bilateral_truth/ tests/

# Type checking
mypy bilateral_truth/
```

### Package Installation
```bash
# Development installation
pip install -e .

# Test CLI after installation
bilateral-truth --help
bilateral-truth --model mock "The sky is blue"
```

### Evaluation Frameworks

The project includes a comprehensive benchmark evaluation infrastructure:

```bash
# Generic evaluator for all benchmarks
cd evaluations
python generic_evaluator.py --dataset standard_datasets/truthfulqa_complete.json --model gpt-4 --samples 100
python generic_evaluator.py --dataset standard_datasets/simpleqa_complete.json --model claude-3-5-haiku-20241022 --samples 50
python generic_evaluator.py --dataset standard_datasets/mmlupro_complete.json --model mock --samples 10
python generic_evaluator.py --dataset standard_datasets/factscore_complete.json --model gpt-4o-mini --samples 20
```

#### Available Benchmarks

1. **TruthfulQA** (870KB, ~800 assertions)
   - Tests factual accuracy and truthfulness
   - Categories: Misconceptions, Science, History, etc.
   
2. **SimpleQA** (14MB, ~4,000 assertions)  
   - Simple fact verification tasks
   - Binary true/false questions
   
3. **MMLU-Pro** (115MB, ~12,000 assertions)
   - Comprehensive knowledge evaluation
   - Multiple domains and difficulty levels
   
4. **FACTScore** (configurable size)
   - Atomic facts from LLM-generated biographies
   - Pre-generated facts with LLAMA+NP and ChatGPT labels
   - Sample data included; full dataset available from Google Drive

#### Benchmark Infrastructure Status

**✅ Completed:**
- Generic evaluation framework (`evaluations/generic_evaluator.py`)
- Standard data format for all benchmarks
- Data generators for all 4 benchmarks
- Checkpoint/recovery system for long evaluations
- Results analysis and reporting
- Sample FACTScore data (35 facts from 5 scientists)

**🔄 In Progress:**
- Running full evaluations on production models
- Downloading complete FACTScore dataset (~6,000 biography facts)

**📋 Next Steps:**
1. Download full FACTScore data from [Google Drive](https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I)
2. Run systematic evaluations across all models and benchmarks
3. Generate comparative analysis reports
4. Add more benchmarks (e.g., BIG-Bench, HellaSwag, GSM8K)
5. Implement cross-benchmark correlation analysis

## Architecture Overview

### Core Mathematical Framework
The package implements a bilateral factuality evaluation system where each assertion receives a generalized truth value <u,v>:
- **u (verifiability)**: Can the statement be verified as true? (t/e/f)
- **v (refutability)**: Can the statement be refuted as false? (t/e/f) 
- **Three-valued logic**: t (true), e (undefined), f (false)

### Module Structure

**`bilateral_truth/truth_values.py`**
- `TruthValueComponent` enum: Represents t, e, f components
- `GeneralizedTruthValue` class: Implements <u,v> truth value pairs
- Class methods: `.true()`, `.false()`, `.undefined()` for standard truth values

**`bilateral_truth/assertions.py`**
- `Assertion` class: Represents atomic formulas φ ∈ ℒ_AT
- Supports both natural language statements and predicate logic with arguments
- Normalized representation for consistent hashing and caching

**`bilateral_truth/zeta_function.py`**
- `ZetaCache` class: Persistent cache implementation for zeta_c function
- `zeta()` function: Base bilateral evaluation without caching
- `zeta_c()` function: Cached bilateral evaluation implementing the mathematical definition
- Global cache management: `clear_cache()`, `get_cache_size()`

**`bilateral_truth/llm_evaluators.py`**
- `LLMEvaluator` abstract base class for bilateral evaluation
- `OpenAIEvaluator`: Uses OpenAI GPT models for evaluation
- `AnthropicEvaluator`: Uses Anthropic Claude models for evaluation
- `MockLLMEvaluator`: Deterministic mock for testing/development
- Majority voting and sampling support for robust evaluation

**`bilateral_truth/model_router.py`**
- `ModelRouter` class: Routes model names to appropriate evaluators
- `OpenRouterEvaluator`: Supports 50+ models via OpenRouter API
- Pattern-based model detection and provider routing

### Key Design Patterns

**Caching Implementation**: The zeta_c function follows the mathematical definition:
```
zeta_c(φ) = {
  c(φ)   if φ ∈ dom(c)
  ζ(φ)   otherwise, and c := c ∪ {(φ, ζ(φ))}
}
```

**Bilateral Evaluation**: Each LLM evaluator performs two separate assessments:
1. Verifiability assessment (P+ function): Can this be verified as true?
2. Refutability assessment (P- function): Can this be refuted as false?

**Sampling and Majority Voting**: For robust evaluation, the system supports:
- Multiple samples per assertion
- Majority voting across samples
- Tiebreaking strategies (random, pessimistic, optimistic)

### API Key Configuration
Set environment variables for LLM providers:
```bash
export OPENAI_API_KEY='your-key'
export ANTHROPIC_API_KEY='your-key' 
export OPENROUTER_API_KEY='your-key'
```

### CLI Usage Patterns
```bash
# Single evaluations
bilateral-truth --model gpt-4 "Statement to evaluate"
bilateral-truth --model claude-sonnet-4-20250514 "Another statement"

# Robust evaluation with sampling
bilateral-truth --model gpt-4 --samples 5 "Statement requiring consensus"

# Mock evaluator for development
bilateral-truth --model mock "Test statement"

# Interactive mode
bilateral-truth --model mock --interactive
```

### Testing Strategy
- **Unit tests**: Test individual components in isolation using MockLLMEvaluator
- **Integration tests**: Test with real LLM APIs (marked with `@pytest.mark.integration`)
- **Sampling tests**: Verify majority voting and tiebreaking logic
- **Caching tests**: Ensure mathematical correctness of cache operations

### Important Implementation Details

**Truth Value Terminology**: Use `GeneralizedTruthValue.undefined()` not `unknown()` and `TruthValueComponent.UNDEFINED` not `EMPTY` - the mathematical framework uses "undefined" for the e component to align with the ArXiv paper terminology.

**Assertion Equality**: Assertions use normalized representations for consistent hashing, enabling proper cache behavior across equivalent formulations.

**LLM Prompt Design**: Evaluators use structured prompts that explicitly ask for separate verifiability and refutability assessments, following the bilateral evaluation framework from the research paper.

**Error Handling**: LLM evaluations gracefully handle API failures, network issues, and malformed responses by falling back to undefined values or continuing with fewer samples.

## Current Development Status (September 2025)

### Recently Completed
- ✅ Generic evaluation framework supporting multiple benchmarks
- ✅ Four benchmark datasets integrated (TruthfulQA, SimpleQA, MMLU-Pro, FACTScore)
- ✅ Standard data format for consistent evaluation across benchmarks
- ✅ FACTScore integration without pip package conflicts
- ✅ Checkpoint/recovery system for long-running evaluations

### Immediate Priorities
1. **Download FACTScore full dataset** - Get ~6,000 biography facts from [Google Drive](https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I)
2. **Run production evaluations** - Systematic testing across all models and benchmarks
3. **Generate analysis reports** - Comparative performance metrics and insights
4. **Add new benchmarks** - Consider BIG-Bench, HellaSwag, GSM8K for broader coverage

### Key Files for Benchmark Work
- `evaluations/generic_evaluator.py` - Main evaluation engine
- `evaluations/data_generators/` - Dataset conversion scripts
- `evaluations/standard_datasets/` - Converted benchmark data
- `evaluations/results/` - Evaluation outputs
- `evaluations/STANDARD_FORMAT.md` - Data format specification

### Notes
- API keys are stored in the .env file in the root directory
- Mock evaluator available for testing without API costs
- All benchmarks use the same evaluation pipeline for consistency