# Comparison: Bilateral Truth Evaluation vs HalluLens Benchmark

## Overview of Approaches

### Bilateral Truth Evaluation (Our Approach)
- **Philosophy**: Truth has multiple dimensions (verifiability/refutability)
- **Method**: Evaluates epistemic states through bilateral questions
- **Focus**: Understanding HOW models handle truth and uncertainty
- **Output**: Discrete epistemic states (<t,f>, <f,t>, <t,t>, <f,f>)

### HalluLens (Bang et al. 2024)
- **Philosophy**: Distinguish hallucination from factuality
- **Method**: Dynamic test generation to prevent data leakage
- **Focus**: Measuring consistency with training data vs absolute truth
- **Output**: Hallucination rates, refusal rates, precision/recall

## Fundamental Philosophical Difference

### Hallucination vs Factuality vs Epistemic States

**HalluLens Distinction**:
- **Hallucination**: Inconsistency with training data
- **Factuality**: Absolute correctness per verification sources
- Key insight: A model can be factually wrong without hallucinating (if trained on wrong data)

**Our Bilateral Framework**:
- **Verification**: Can this be verified as true?
- **Refutation**: Can this be refuted as false?
- Key insight: Truth states are multi-dimensional, not binary

**Critical Difference**: HalluLens separates data consistency from truth; we separate verification from refutation.

## Convergent Findings

### 1. Refusal/Uncertainty is Critical

**HalluLens Finding**:
- Tracks "false refusal rate" as key metric
- Better models know when to refuse
- GPT-4o and Llama-3.1-405B best at appropriate refusal

**Our Finding**:
- Knowledge gaps (<f,f>) represent 13-38% of responses
- But conservative policy (high refusal) performs WORSE
- Models have uncertainty capabilities but use them poorly

**Convergence**: Both identify that appropriate uncertainty expression is crucial but poorly optimized.

### 2. Model Size Isn't Everything

**HalluLens Finding**:
- Larger models generally better at recognizing limits
- But significant variation within size classes
- Model-specific behaviors matter

**Our Finding**:
- No family resemblance (p=0.52)
- Training dominates architecture
- Individual "epistemic personalities" emerge

**Convergence**: Both show that model behavior is more nuanced than size/architecture suggests.

### 3. Dynamic Evaluation Needed

**HalluLens Approach**:
- Dynamically generates test sets
- Prevents data leakage
- Tests "middle-frequency" knowledge

**Our Insight**:
- Different epistemic policies yield different results
- Static evaluation misses epistemic nuance
- Need multiple evaluation perspectives

**Convergence**: Both advocate for more sophisticated, dynamic evaluation approaches.

## Complementary Insights

### What Bilateral Truth Reveals That HalluLens Doesn't

1. **Contradiction Detection**:
   - We find 1.5-14.4% <t,t> rates
   - Models simultaneously verify AND refute
   - Reveals internal inconsistency invisible to HalluLens

2. **Unidirectional Overconfidence**:
   - Models ONLY make <t,f> errors, NEVER <f,t>
   - Systematic bias toward positive claims
   - Not captured by hallucination/refusal metrics

3. **Epistemic Policy Impact**:
   - Conservative: 32.3% F1, 56% coverage
   - Paracomplete: 40.5% F1, 82% coverage
   - Shows HOW to use uncertainty information

### What HalluLens Reveals That Bilateral Doesn't

1. **Training Data Consistency**:
   - Distinguishes "wrong but consistent" from hallucination
   - Tests non-existent entities explicitly
   - Reveals memorization vs generation

2. **Long-form Generation**:
   - Tests consistency across extended outputs
   - Precision/recall metrics for information density
   - Reveals degradation over length

3. **Dynamic Test Generation**:
   - Prevents benchmark gaming
   - Tests current knowledge
   - Avoids data leakage issues

## Methodological Comparison

### Evaluation Philosophy

**Bilateral Truth**:
```
For each statement:
1. Evaluate verifiability → u ∈ {t, f, e}
2. Evaluate refutability → v ∈ {t, f, e}
3. Combine to bilateral value <u,v>
4. Apply epistemic policy
```

**HalluLens**:
```
For each query:
1. Check if entity exists
2. Evaluate response type (answer/refuse/hallucinate)
3. Compare with ground truth
4. Calculate rates across categories
```

### Key Metrics

**Bilateral Truth**:
- Truth value distributions
- Contradiction rates (<t,t>)
- Knowledge gap rates (<f,f>)
- Policy-dependent F1/coverage

**HalluLens**:
- False refusal rate
- Hallucination rate
- Precision/Recall
- Entity-type specific rates

## Unified Framework Potential

### Combining Both Approaches

An ideal evaluation would:

1. **Test Epistemic States** (Bilateral):
   - Verification vs refutation
   - Contradiction detection
   - Knowledge gap identification

2. **Test Data Consistency** (HalluLens):
   - Training data alignment
   - Non-existent entity handling
   - Dynamic generation

3. **Cross-Analysis**:
   - Do hallucinations correlate with <t,t> contradictions?
   - Are false refusals related to <f,f> knowledge gaps?
   - Does bilateral profile predict HalluLens performance?

## Critical Insights

### Different Lenses on Same Problem

Both frameworks address model reliability but from different angles:

**HalluLens**: "Is the model making things up?"
**Bilateral**: "How does the model handle epistemic uncertainty?"

### The Refusal Paradox

**HalluLens Perspective**:
- Low false refusal rate is good
- Models should refuse when uncertain

**Our Finding**:
- Conservative policy (high refusal) performs worse
- Paracomplete (treating gaps as false) performs best

**Resolution**: The paradox reveals that **appropriate refusal** matters more than refusal rate. Models need to:
1. Refuse when truly uncertain (HalluLens goal)
2. Not over-refuse on answerable questions (our finding)

### Training Data vs Truth

**HalluLens Innovation**: Separating hallucination from factuality

**Our Complement**: Separating verification from refutation

**Together**: A complete picture emerges:
- **Hallucination**: Inconsistent with training
- **Factual Error**: Inconsistent with truth
- **Contradiction**: Both verifiable and refutable
- **Knowledge Gap**: Neither verifiable nor refutable

## Practical Integration

### Combined Evaluation Protocol

1. **Bilateral Assessment**:
   - Determine epistemic state
   - Identify contradictions
   - Measure knowledge gaps

2. **HalluLens Testing**:
   - Check training consistency
   - Test non-existent entities
   - Evaluate long-form coherence

3. **Correlation Analysis**:
   - <t,t> states → hallucination prone?
   - <f,f> states → appropriate refusals?
   - Epistemic profile → HalluLens performance?

## Conclusions

### Complementary Frameworks

**HalluLens** and **Bilateral Truth** are perfectly complementary:

- HalluLens: Tests WHAT models generate (consistency/accuracy)
- Bilateral: Tests HOW models reason (epistemic states)

### Unified Message

Both frameworks reveal that **current models have sophisticated capabilities but use them inappropriately**:

1. **Can express uncertainty but don't optimize it** (both findings)
2. **Have internal inconsistencies** (bilateral: contradictions, HalluLens: hallucinations)
3. **Need better evaluation paradigms** (both advocate for richer metrics)

### The Bigger Picture

Together, these frameworks suggest:

1. **Binary evaluation is insufficient** - Need multiple dimensions
2. **Refusal/uncertainty is critical but poorly calibrated** - Models over/under refuse
3. **Dynamic, multi-faceted evaluation essential** - Static benchmarks miss crucial patterns

The bilateral framework reveals the **epistemic mechanics** while HalluLens reveals the **behavioral outcomes**—together providing a complete picture of model reliability challenges.