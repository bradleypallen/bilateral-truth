# Comprehensive Summary: Bilateral Truth Evaluation Findings

## Executive Summary

We evaluated 7 frontier language models across 4 major benchmarks (TruthfulQA, SimpleQA, MMLU-Pro, FACTScore) using the bilateral truth framework, which evaluates statements along two dimensions: verifiability (u) and refutability (v), producing truth values like <t,f> (classical true), <f,t> (classical false), <t,t> (contradiction), and <f,f> (knowledge gap). 

**Surprising finding**: Traditional unilateral evaluation (verification only) achieves 38.6% F1 with 89% coverage, and the bilateral framework only improves this by 4.9% with the best policy. The bilateral framework's main value is not accuracy improvement but **epistemic transparency** - revealing HOW models think, not just WHAT they answer.

## 1. Core Performance Metrics

### Overall Rankings (Conservative/Classical Policy)
1. **Gemini-2.5-Flash**: 70.1% accuracy, 40.2% F1, 66.4% coverage
2. **GPT-4.1**: 56.7% accuracy, 35.3% F1, 70.2% coverage  
3. **Claude-Opus-4.1**: 51.4% accuracy, 32.0% F1, 39.2% coverage
4. **GPT-4.1-mini**: 50.2% accuracy, 32.3% F1, 65.5% coverage
5. **Claude-3-5-Haiku**: 48.2% accuracy, 30.7% F1, 57.9% coverage
6. **Llama-4-Maverick**: 46.3% accuracy, 29.1% F1, 42.0% coverage
7. **Llama-4-Scout**: 39.7% accuracy, 26.4% F1, 52.1% coverage

### Key Finding: No Family Resemblance (p=0.52)
- Model families DO NOT share bilateral truth distributions
- Epistemic behavior is training-specific, not architecture-dependent
- Within-family distance (0.24) barely smaller than between-family (0.28)

## 2. Bilateral Truth Value Distributions

### Epistemic Profiles by Model

**High Epistemic Honesty (>35% <f,f> or <t,t>):**
- Claude-3-5-Haiku: 42.1% (38.1% <f,f>, 4.0% <t,t>)
- Llama-4-Scout: 38.4% (35.8% <f,f>, 2.6% <t,t>)
- GPT-4.1-mini: 34.5% (21.1% <f,f>, 13.4% <t,t>)

**High Contradiction Detection (<t,t>):**
- Gemini-2.5-Flash: 14.4% <t,t> (highest)
- GPT-4.1-mini: 13.4% <t,t>
- GPT-4.1: 8.3% <t,t>

**Low Contradiction Models:**
- Claude-Opus-4.1: 1.5% <t,t> (lowest)
- Llama-4-Scout: 2.6% <t,t>

### Critical Insight: Unidirectional Overconfidence
- **ALL overconfidence is <t,f> when false** (claiming false things are true)
- **ZERO <f,t> when true** (never claim true things are false)
- Models have systematic bias toward positive assertions

## 3. Category-Specific Performance

### Domain Specializations

**Business/Economics Excellence:**
- OpenAI GPT models: +137% advantage in business, +37.5% in economics
- Anthropic models: -137% disadvantage in business (surprising weakness)

**Technical/Scientific Excellence:**
- Anthropic: Physics (+40.5%), Chemistry (+18.2%), Engineering (+38.3%)
- Meta/Llama: Mathematics (+19.8%)
- Google: Weak across sciences (-40.5% physics)

**Humanities/Culture:**
- Anthropic: Music (+51.2%), Art (+21.2%), History (+50.1% MMLU-Pro)
- Meta/Llama: Sports (+32.8%), Politics (+37.4% SimpleQA)
- Google: Psychology (+63.8%)

**Disambiguation/Reasoning:**
- OpenAI: Confusion about places (+225%), people (+100%)
- Google: Logical falsehoods (+28.2%), identity reasoning (+167%)

### Categories with Highest Overconfidence (>50%)
- Logical Falsehood: 72.2% (Llama-Scout)
- Subjective questions: 66.7% (Claude-Haiku)
- Business (MMLU-Pro): 62.2% (Claude-Haiku)
- Education: 50-60% (multiple models)

### Categories with Highest Epistemic Honesty (>70%)
- Confusion about people: 88-100% honesty
- Indexical errors: 70-92% honesty
- Biography facts (FACTScore): 47-71% honesty

## 4. Epistemic Policy Comparison

### Policy Performance Rankings (Including Unilateral Baseline)

1. **Paracomplete** (Best Overall)
   - 69.0% accuracy, 40.5% F1, 81.9% coverage
   - Only 4.9% better than unilateral baseline
   - Treats <f,f> as false (epistemic conservatism)

2. **Unilateral (Verification Only)** (Strong Baseline)
   - 63.7% accuracy, 38.6% F1, 89.0% coverage
   - Traditional "Is this true?" evaluation
   - Surprisingly robust performance

3. **Conservative/Classical** (Too Restrictive)
   - 51.8% accuracy, 32.3% F1, 56.2% coverage
   - 16.4% WORSE than unilateral despite bilateral information
   - Leaves 44% of questions unanswered

4. **Paraconsistent** (Worst)
   - 47.6% accuracy, 30.7% F1, 66.2% coverage
   - 20.5% worse than unilateral
   - Accepting contradictions hurts performance

### Benchmark-Specific Policy Preferences vs Unilateral
- **FACTScore**: Unilateral ≈ Paracomplete (both 40.1% F1)
- **SimpleQA**: Small bilateral improvement (+6% for paracomplete)
- **MMLU-Pro**: Paracomplete beats unilateral (+11.6%)
- **TruthfulQA**: Minimal improvement (+3.3% for paracomplete)

## 5. Key Insights and Implications

### 1. Bilateral Value is Epistemic Transparency, Not Accuracy
- **Unilateral baseline achieves 89% coverage with 38.6% F1**
- **Bilateral only adds 4.9% F1 improvement**
- Main value: Distinguishing HOW models fail (contradiction vs uncertainty vs error)
- Critical for trust and interpretability, not raw performance

### 2. The Framework Reveals Four Hidden Epistemic States
- **Knowledge gaps (<f,f>)**: 13-38% of responses
- **Contradictions (<t,t>)**: 1.5-14.4% of responses  
- **Evaluation failures (<e,e>)**: Technical issues
- Unilateral evaluation conflates all these with true/false

### 3. Training Dominates Architecture
- No significant family resemblance (p=0.52)
- Individual model "epistemic personalities" emerge from training
- Challenges assumptions about model families

### 4. Models are Systematically Overconfident in One Direction
- Always claim false things are true (<t,f> errors only)
- NEVER claim true things are false (zero <f,t> errors)
- Suggests fundamental bias toward positive assertions
- Completely invisible in any form of unilateral evaluation

### 5. Conservative Policy is Counterproductive
- Despite having MORE information, performs 16.4% WORSE than unilateral
- Too restrictive: only 56% coverage vs 89% for unilateral
- Shows that more information doesn't always help if poorly utilized

### 6. Domain-Specific Model Selection Matters
- OpenAI for business/economics/disambiguation
- Anthropic for physics/engineering/music/art
- Meta/Llama for mathematics/sports
- Google for psychology (but weak overall)

## 6. Practical Recommendations

### For Production Systems
1. **Consider unilateral evaluation for simplicity**: 89% coverage, only 5% worse than best bilateral
2. **Use paracomplete only if epistemic nuance matters**: Small accuracy gain, big interpretability gain
3. **AVOID conservative policy**: Worse than simple unilateral despite more information
4. **Select models by domain**: Extreme specialization differences

### For Model Development
1. **Address unidirectional overconfidence**: Why do models NEVER say true things are false?
2. **Reduce contradiction rates**: 14.4% <t,t> in some models indicates confusion
3. **The refutation dimension barely helps accuracy**: Focus on improving verification
4. **Training matters more than architecture**: No family resemblance found

### For Evaluation Methodology
1. **Bilateral framework's value is diagnostic, not performative**: Use for understanding, not scoring
2. **Traditional evaluation is surprisingly robust**: Unilateral gets you 90% there
3. **Conservative bilateral policy is a trap**: More information can hurt if misused
4. **Track epistemic patterns for trust**: Know when models are guessing vs knowing

## 7. Connection to Propositional Interpretability

The bilateral framework operationalizes Chalmers' propositional interpretability by:
- Revealing **doxastic states** (belief-like patterns)
- Measuring **belief coherence** via contradiction rates
- Distinguishing **types of uncertainty**
- Showing belief formation is **training-dependent**

Models with lower <t,t> rates may be more amenable to propositional interpretation due to greater internal coherence.

## Conclusion

The bilateral truth evaluation framework reveals rich epistemic structures in language models, but **surprisingly, traditional unilateral evaluation captures most of the performance** (38.6% F1 with 89% coverage vs 40.5% F1 with 82% coverage for the best bilateral policy). 

The bilateral framework's true value lies not in accuracy improvements but in **epistemic transparency**:
- Distinguishing knowledge gaps (<f,f>) from falsehoods
- Detecting internal contradictions (<t,t>) 
- Revealing unidirectional overconfidence (only <t,f> errors, never <f,t>)
- Understanding when models are guessing vs knowing

Key findings:
1. **No family resemblance** (p=0.52) - training dominates architecture
2. **Conservative policy backfires** - more information used poorly is worse than less information
3. **Unilateral evaluation is robust** - the refutation dimension adds only 5% F1
4. **Models never claim true things are false** - fundamental positive bias

The bilateral framework is best used as a **diagnostic tool for model interpretability** rather than a replacement for traditional evaluation. It reveals HOW models think, not just WHAT they answer.