# Comparison: Bilateral Truth Evaluation vs "Why Language Models Hallucinate"

## Overview of Approaches

### Bilateral Truth Evaluation (Our Empirical Study)
- **Type**: Empirical evaluation framework
- **Method**: Evaluates verifiability and refutability dimensions
- **Focus**: Measuring epistemic states in practice
- **Output**: Discrete truth values and performance metrics

### Hallucination Theory (Kalai et al. 2024)
- **Type**: Theoretical/mathematical framework
- **Method**: Statistical learning theory analysis
- **Focus**: Explaining WHY hallucinations occur
- **Output**: Mathematical bounds and theoretical insights

## Fundamental Relationship

Their theoretical framework **explains** what our empirical results **demonstrate**.

### Their Theory Predicts What We Observe

**Kalai et al. Theoretical Prediction**:
- Models are incentivized to guess rather than abstain
- Binary evaluation rewards overconfidence
- Uncertainty expression is penalized

**Our Empirical Observation**:
- Models show unidirectional overconfidence (only <t,f> errors)
- Knowledge gaps (<f,f>) often inappropriate (13-38%)
- Conservative policy (requiring certainty) performs WORSE than guessing

## Convergent Core Finding

### The Evaluation Problem

**Their Theoretical Finding**:
"Existing evaluation benchmarks reward overconfident guessing"

**Our Empirical Finding**:
- Unilateral evaluation achieves 89% coverage by treating everything as true/false
- Conservative bilateral policy drops to 56% coverage and performs WORSE
- Models NEVER say true things are false (zero <f,t> errors)

**Together**: Both show that current evaluation systematically incentivizes hallucination over epistemic honesty.

## Complementary Insights

### What Their Theory Explains About Our Results

1. **Why Unidirectional Overconfidence Exists**:
   - **Their Theory**: Models trained to minimize error, not express uncertainty
   - **Our Finding**: Only <t,f> errors occur, never <f,t>
   - **Explanation**: Positive guessing minimizes expected loss

2. **Why Conservative Policy Fails**:
   - **Their Theory**: Abstention is penalized in binary evaluation
   - **Our Finding**: Conservative policy 16.4% worse than unilateral
   - **Explanation**: Being cautious is punished by current metrics

3. **Why Paracomplete Works Best**:
   - **Their Theory**: Some guessing is optimal given training incentives
   - **Our Finding**: Paracomplete (treating gaps as false) best at 40.5% F1
   - **Explanation**: Balances the incentive to guess with epistemic caution

### What Our Results Validate About Their Theory

1. **"Generative error rate ≳ 2 * IIV misclassification rate"**:
   - We observe high false positive rates (<t,f> errors)
   - Consistent with their prediction of amplified errors in generation

2. **Binary Grading Encourages Hallucination**:
   - Our unilateral baseline (binary) shows 89% coverage
   - Models forced to choose true/false even when uncertain
   - Validates their claim about binary evaluation problems

3. **Models Can't Express Appropriate Uncertainty**:
   - We find 13-38% knowledge gaps, but often inappropriate
   - Their theory: Models not trained to say "I don't know"
   - Our evidence: <f,f> patterns don't match true uncertainty

## Key Differences in Approach

### Theoretical vs Empirical

**Hallucination Theory**:
- Mathematical proofs and bounds
- Explains fundamental causes
- Predicts general behavior
- Abstract framework

**Bilateral Truth**:
- Concrete measurements
- Documents specific patterns
- Quantifies actual behavior
- Practical framework

### Complementary Perspectives

They explain **WHY** hallucinations occur; we show **HOW** they manifest:

**Their "Why"**:
1. Statistical inevitability in base models
2. Evaluation incentives
3. Training objectives

**Our "How"**:
1. Unidirectional overconfidence patterns
2. Contradiction rates (1.5-14.4%)
3. Knowledge gap distributions (13-38%)

## Unified Framework

### Combining Theory and Practice

Their theoretical framework + Our empirical measurements suggest:

1. **Hallucinations are Systematic, Not Random**:
   - **Theory**: Statistical pressures create predictable biases
   - **Evidence**: Unidirectional overconfidence, no <f,t> errors
   - **Implication**: Can be addressed through systematic changes

2. **Evaluation Reform is Critical**:
   - **Theory**: Binary grading causes hallucinations
   - **Evidence**: Conservative policy fails despite more information
   - **Solution**: Need evaluation that rewards appropriate uncertainty

3. **Models Need Uncertainty Training**:
   - **Theory**: Models not incentivized to express doubt
   - **Evidence**: <f,f> patterns often inappropriate
   - **Solution**: Explicit confidence targets in training

## Practical Implications

### Their Recommendations Validated by Our Data

**They Propose**: "Incorporating explicit confidence targets in benchmarks"

**Our Evidence Supporting This**:
- Bilateral framework distinguishes 4 types of uncertainty
- Different epistemic policies yield different results
- Models capable of expressing uncertainty (<f,f>) but don't use it appropriately

### Our Framework as Implementation

The bilateral truth framework could implement their theoretical recommendations:

1. **Non-Binary Evaluation**: Our <t,f>, <f,t>, <t,t>, <f,f> states
2. **Confidence Expression**: Knowledge gaps (<f,f>) as explicit uncertainty
3. **Penalty for Overconfidence**: Tracking <t,f> errors specifically

## Critical Synthesis

### The Complete Picture

**Theoretical Foundation** (Kalai et al.):
- Hallucinations are statistically inevitable
- Evaluation incentives drive overconfidence
- Binary grading is fundamentally flawed

**Empirical Evidence** (Our Study):
- Unidirectional overconfidence observed
- Conservative policies fail
- Models have epistemic states but use them poorly

**Unified Conclusion**:
Hallucinations aren't bugs but **features** created by current training and evaluation paradigms.

## Revolutionary Insight

### Their Theory + Our Data = New Understanding

1. **Models aren't lying; they're following incentives**:
   - Trained to guess rather than abstain
   - Evaluated on binary correctness
   - Naturally become overconfident

2. **The bilateral framework reveals the solution space**:
   - Shows models CAN express uncertainty
   - Documents current inappropriate patterns
   - Provides mechanism for better evaluation

3. **Reform must be systemic**:
   - Change evaluation metrics (their recommendation)
   - Use richer truth values (our contribution)
   - Retrain with uncertainty incentives

## Conclusion

The hallucination theory paper provides the **theoretical foundation** for what our bilateral truth evaluation **empirically demonstrates**:

1. **Current evaluation creates hallucinations** - Both theory and data confirm
2. **Models lack appropriate uncertainty** - Predicted theoretically, observed empirically
3. **Binary evaluation is the core problem** - Mathematical proof meets empirical evidence

Their work explains WHY our conservative policy fails (abstention is penalized) and WHY unidirectional overconfidence exists (positive guessing is rewarded). Our work provides the empirical patterns and practical framework to implement their theoretical recommendations.

Together, these papers suggest that **solving hallucinations requires fundamental changes to how we train and evaluate language models**, moving from binary correctness to rich epistemic evaluation that rewards appropriate uncertainty.