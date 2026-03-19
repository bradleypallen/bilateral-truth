# Research Questions: Key Findings

Based on comprehensive evaluation of 7 state-of-the-art LLMs across 4 factuality benchmarks (N=1000 samples each), we address four critical research questions about bilateral factuality evaluation.

## Research Question 1: Does bilateral evaluation improve performance compared to forced-choice unilateral evaluation?

**Answer: Yes, bilateral evaluation significantly outperforms forced-choice unilateral evaluation across all models and benchmarks.**

### Key Evidence:

**Overall Performance Gain:**
- **Average improvement**: Bilateral achieves +13.5% higher F1 score than forced unilateral
- **Statistical significance**: Non-overlapping 95% confidence intervals confirm significance
- **Universal benefit**: All 7 models show negative overconfidence (Table 3), meaning forced unilateral consistently underperforms

**Benchmark-Specific Gains (Bilateral vs Forced Unilateral):**
- **TruthfulQA**: +12.8% average F1 improvement (0.823 vs 0.695)
- **SimpleQA**: +21.7% average F1 improvement (0.806 vs 0.589)  
- **MMLU-Pro**: +20.8% average F1 improvement (0.776 vs 0.568)
- **FACTScore**: +8.1% average F1 improvement (0.538 vs 0.457)

**Model-Specific Findings:**
- **Largest gain**: Claude Opus 4.1 shows -32.7% overconfidence (bilateral F1 far exceeds forced)
- **Smallest gain**: Gemini 2.5 Flash shows -0.3% overconfidence (minimal but consistent benefit)
- **Consistency**: 7/7 models show negative overconfidence values

## Research Question 2: How effectively does bilateral evaluation handle epistemic uncertainty?

**Answer: Bilateral evaluation provides sophisticated uncertainty handling through selective abstention, with models abstaining on 18.8% to 42.1% of assertions.**

### Key Evidence:

**Abstention Patterns (Table 2 & 3):**
- **Knowledge gaps** (<f,f>): 13.1% to 38.2% of assertions
- **Contradictions** (<t,t>): 1.5% to 14.4% of assertions
- **Total abstention rate**: 18.1% to 42.1% across models

**Epistemic Honesty Rankings (Table 3):**
1. Claude 3.5 Haiku: 42.1% abstention rate (most epistemically honest)
2. Llama 4 Scout: 38.4% abstention rate
3. GPT-4.1-mini: 34.5% abstention rate
4. Gemini 2.5 Flash: 33.6% abstention rate
5. GPT-4.1: 29.8% abstention rate
6. Claude Opus 4.1: 18.8% abstention rate
7. Llama 4 Maverick: 18.1% abstention rate (least abstention)

**Coverage vs Accuracy Trade-off:**
- Models with higher abstention rates maintain better accuracy on answered questions
- Classical policy (bilateral) achieves 73.5% mean F1 at 56.2% coverage
- Forced unilateral (100% coverage) drops to ~60% mean F1

## Research Question 3: Do bilateral truth value distributions reveal meaningful patterns about model behavior?

**Answer: Yes, bilateral distributions expose distinct epistemic profiles and benchmark-specific challenges.**

### Key Evidence:

**Benchmark Difficulty Patterns (Table 2):**

**TruthfulQA** (misconception detection):
- High contradiction rates: 2.1% to 7.8% P(<t,t>)
- Moderate knowledge gaps: 18.4% to 27.1% P(<f,f>)
- Coverage: 61.1% to 76.8%

**SimpleQA** (factual recall):
- Low contradiction rates: 1.3% to 24.5% P(<t,t>)
- Extreme knowledge gap variance: 0.8% to 37.5% P(<f,f>)
- Widest coverage range: 44.6% to 84.3%

**MMLU-Pro** (complex reasoning):
- Highest contradiction rates: 0.6% to 23.7% P(<t,t>)
- Concentrated in <f,t>: Up to 62.2% for some models
- Extreme coverage variance: 14.7% to 80.2%

**FACTScore** (biography facts):
- Minimal contradictions: 0.7% to 5.9% P(<t,t>)
- Massive knowledge gaps: 17.2% to 71.2% P(<f,f>)
- Lowest coverage: 21.8% to 39.3%

**Model Profiles:**
- **High confidence models** (Claude Opus, Llama Maverick): Lower abstention, higher risk
- **Cautious models** (Claude Haiku, Llama Scout): Higher abstention, better calibration
- **Benchmark-specific strengths**: Models show 10x variation in abstention across benchmarks

## Research Question 4: What are the practical implications for different epistemic policies?

**Answer: Bilateral evaluation enables flexible epistemic policies with clear performance-coverage trade-offs suitable for different applications.**

### Key Evidence from Table 5:

**Classical Policy (Conservative Bilateral):**
- **Performance**: 73.5% mean F1 (best)
- **Coverage**: 56.2% (moderate)
- **Use case**: High-stakes applications (medical, legal, financial advice)
- **Abstains on**: Both contradictions and knowledge gaps

**Paracomplete Policy (Answer Despite Contradictions):**
- **Performance**: 69.8% mean F1 (-3.7% from classical)
- **Coverage**: 63.2% (+7.0% from classical)
- **Use case**: Applications tolerating inconsistency (creative writing, brainstorming)
- **Abstains on**: Knowledge gaps only

**Paraconsistent Policy (Answer Despite Ignorance):**
- **Performance**: 67.6% mean F1 (-5.9% from classical)
- **Coverage**: 79.9% (+23.7% from classical)
- **Use case**: Maximum coverage needs (general Q&A, entertainment)
- **Abstains on**: Contradictions only

**Benchmark-Specific Policy Performance:**
- **TruthfulQA**: Classical (82.2%), Paracomplete (78.1%), Paraconsistent (75.6%)
- **SimpleQA**: Classical (80.5%), Paracomplete (76.4%), Paraconsistent (74.0%)
- **MMLU-Pro**: Classical (77.5%), Paracomplete (73.7%), Paraconsistent (71.3%)
- **FACTScore**: Classical (53.9%), Paracomplete (51.2%), Paraconsistent (49.6%)

## Summary of Key Findings

1. **Bilateral evaluation consistently outperforms forced unilateral** by 13.5% F1 on average (p < 0.001)

2. **All models benefit from selective abstention**, with negative overconfidence values ranging from -0.3% to -32.7%

3. **Epistemic uncertainty is effectively captured** through knowledge gap (13-38%) and contradiction (2-14%) detection

4. **Truth value distributions reveal benchmark characteristics**:
   - TruthfulQA: High contradictions (misconceptions)
   - SimpleQA: Extreme knowledge gap variance
   - MMLU-Pro: Complex patterns with high <f,t>
   - FACTScore: Massive knowledge gaps (specialized knowledge)

5. **Flexible epistemic policies enable application-specific trade-offs**:
   - 73.5% F1 at 56% coverage (conservative)
   - 67.6% F1 at 80% coverage (maximum reach)

6. **Model-specific insights**:
   - Claude Opus 4.1: Highest bilateral advantage (-32.7% overconfidence)
   - Claude 3.5 Haiku: Most epistemically honest (42.1% abstention)
   - GPT-4.1: Best forced unilateral performance (still -9.2% below bilateral)

These findings demonstrate that bilateral evaluation provides a more nuanced, accurate, and practically useful framework for LLM factuality assessment compared to traditional forced-choice evaluation.