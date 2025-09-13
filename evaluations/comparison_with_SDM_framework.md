# Comparison: Bilateral Truth Evaluation vs Semantic Divergence Metrics (SDM)

## Overview of Approaches

### Bilateral Truth Evaluation (Our Approach)
- **Framework**: Evaluates statements along verifiability (u) and refutability (v) dimensions
- **Output**: Truth values like <t,f>, <f,t>, <t,t>, <f,f>
- **Focus**: Epistemic states and how models handle uncertainty/contradiction
- **Method**: Direct evaluation via structured prompts to LLMs

### Semantic Divergence Metrics (Halperin 2024)
- **Framework**: Measures semantic alignment between prompts and responses
- **Output**: Continuous scores (SDM Hallucination Score, Semantic Exploration)
- **Focus**: Detecting hallucinations through semantic drift
- **Method**: Embedding clustering and divergence metrics (JS, KL, Wasserstein)

## Key Philosophical Differences

### 1. Epistemology vs Information Theory

**Bilateral Truth**:
- Grounded in philosophical epistemology (verification/refutation)
- Treats truth as multi-dimensional (can be verified AND refuted)
- Explicit handling of epistemic states (knowledge gaps, contradictions)

**SDM**:
- Grounded in information theory and statistics
- Treats hallucination as semantic divergence from prompt
- Implicit handling through continuous divergence scores

### 2. Discrete vs Continuous

**Bilateral Truth**:
- Discrete truth values (9 possible states)
- Clear categorical distinctions (true/false/contradiction/gap)
- Binary decisions on each dimension

**SDM**:
- Continuous scores (0-1 range)
- Gradient from faithful to hallucinatory
- Probabilistic interpretation

## Methodological Comparison

### Evaluation Process

**Bilateral Truth**:
1. Ask model: "Can this be verified as true?"
2. Ask model: "Can this be refuted as false?"
3. Combine into bilateral value
4. Apply epistemic policy

**SDM**:
1. Generate paraphrased prompts
2. Collect multiple responses
3. Create embeddings and cluster
4. Compute divergence metrics
5. Normalize by prompt complexity

### Computational Requirements

**Bilateral Truth**:
- 2 LLM calls per statement
- No embedding infrastructure needed
- Direct evaluation

**SDM**:
- Multiple paraphrases (10+ variations)
- Embedding model required
- Clustering algorithms
- More computationally intensive

## Findings Comparison

### On Hallucination Detection

**Bilateral Truth Findings**:
- Models show unidirectional overconfidence (only <t,f> errors)
- 1.5-14.4% contradiction rates (<t,t>)
- 13-38% knowledge gaps (<f,f>)
- Traditional unilateral evaluation captures 90% of performance

**SDM Findings**:
- Identified "confident hallucinations" with low semantic variability
- Semantic Entropy unreliable for hallucination detection
- Context-aware evaluation essential
- Different response modes (recall, interpretation, creative)

### On Model Behavior

**Bilateral Truth**:
- No family resemblance (p=0.52)
- Training dominates architecture
- Models never claim true things are false
- Epistemic patterns vary by benchmark

**SDM**:
- Models exhibit different "semantic exploration" patterns
- Response stability varies with prompt type
- Creative generation shows higher divergence
- Factual recall shows lower divergence

## Complementary Insights

### What Bilateral Truth Reveals that SDM Doesn't

1. **Epistemic States**: Explicit distinction between "don't know" (<f,f>) and "contradictory" (<t,t>)
2. **Directional Bias**: Discovery that models ONLY make <t,f> errors, never <f,t>
3. **Policy Impact**: How different epistemic policies affect performance
4. **Verification vs Refutation**: Asymmetry in reasoning capabilities

### What SDM Reveals that Bilateral Truth Doesn't

1. **Semantic Drift**: Gradual departure from prompt semantics
2. **Response Stability**: Consistency across paraphrases
3. **Creative vs Factual**: Distinguishing response modes
4. **Prompt Complexity**: Normalizing by input entropy

## Convergent Findings

Both approaches independently discovered:

1. **Context Matters**: 
   - Bilateral: Benchmark-specific patterns
   - SDM: Prompt-specific normalization needed

2. **Traditional Metrics Limited**:
   - Bilateral: Unilateral evaluation misses epistemic nuance
   - SDM: Semantic Entropy unreliable

3. **Multiple Dimensions Needed**:
   - Bilateral: Verification AND refutation
   - SDM: Multiple divergence metrics

4. **Interpretation > Detection**:
   - Bilateral: Framework reveals HOW models think
   - SDM: Context-aware interpretation more valuable than absolute scores

## Practical Integration

### Combined Framework Potential

A unified approach could:
1. Use bilateral evaluation for categorical epistemic states
2. Apply SDM for continuous hallucination scoring within states
3. Track semantic drift in <t,t> contradictions
4. Measure stability of <f,f> uncertainty responses

### Complementary Use Cases

**Use Bilateral Truth When**:
- Need clear epistemic categories
- Evaluating factual statements
- Understanding model uncertainty
- Implementing epistemic policies

**Use SDM When**:
- Detecting semantic drift
- Evaluating creative generation
- Measuring response stability
- Need continuous confidence scores

## Key Differences in Philosophy

### Bilateral Truth: Epistemic Realism
- Truth exists independently
- Models have belief-like states
- Contradictions are meaningful
- Uncertainty is epistemically significant

### SDM: Semantic Coherence
- Truth is alignment with prompt
- Models generate distributions
- Divergence indicates problems
- Stability indicates reliability

## Conclusions

### Strengths of Our Bilateral Approach
1. **Simpler**: Only 2 LLM calls, no embeddings needed
2. **Interpretable**: Clear epistemic categories
3. **Philosophical**: Grounded in epistemology
4. **Practical**: Unilateral baseline shows traditional methods work

### Strengths of SDM Approach
1. **Nuanced**: Continuous scores capture gradients
2. **Stable**: Multiple paraphrases reduce noise
3. **Mathematical**: Rigorous information-theoretic foundation
4. **Flexible**: Adapts to different prompt types

### Synthesis

Both frameworks reveal that **detecting hallucinations/falsehoods is less important than understanding the nature of model responses**. The bilateral framework excels at categorical epistemic analysis, while SDM excels at continuous semantic drift detection. Together, they suggest:

1. **Multiple evaluation dimensions are essential** (not just binary truth)
2. **Context dramatically affects evaluation** (benchmark/prompt specific)
3. **Traditional metrics miss crucial information** (both show this)
4. **Interpretability matters more than detection** (understanding > scoring)

The bilateral framework's discovery that models have unidirectional bias (only <t,f> errors) and SDM's discovery of "confident hallucinations" both point to systematic biases in how LLMs generate responses, suggesting deep architectural or training influences on truthfulness.