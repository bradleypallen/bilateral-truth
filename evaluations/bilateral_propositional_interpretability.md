# Bilateral Truth Evaluation and Propositional Interpretability

## Connection to David Chalmers' Framework

This document explores the relationship between bilateral truth evaluation and David Chalmers' notion of propositional interpretability - the question of whether and how we can interpret neural networks as having beliefs about propositions.

## Key Connections

### 1. From Behavioral to Propositional Attribution

Chalmers distinguishes between:
- **Behavioral interpretability**: Understanding what the model does
- **Propositional interpretability**: Understanding what the model believes

The bilateral framework moves us toward propositional interpretability by revealing:
- **<t,f>**: The model has evidence supporting the proposition
- **<f,t>**: The model has evidence against the proposition  
- **<t,t>**: The model has conflicting evidence (contradictory beliefs)
- **<f,f>**: The model lacks evidence (epistemic gap)

Rather than just "the model outputs X," we can say "the model believes it can verify P but cannot refute it."

### 2. Revealing Doxastic States

The bilateral evaluation exposes what might be considered the model's **doxastic states** (belief states):

- **Confident belief**: <t,f> - "I believe P is true"
- **Confident disbelief**: <f,t> - "I believe P is false"
- **Conflicted belief**: <t,t> - "I have reasons to both believe and disbelieve P"
- **Suspended judgment**: <f,f> - "I have no grounds for belief either way"

This maps surprisingly well onto traditional epistemology's doxastic attitudes, suggesting we're capturing something meaningful about the model's propositional stance.

### 3. The Contradiction Problem

Our empirical finding that models exhibit different <t,t> rates is particularly relevant to Chalmers' framework:

#### Empirical Results:
- **High <t,t> models**: 
  - Gemini-2.5-Flash: 14.4% <t,t> overall
  - GPT-4.1-mini: 13.4% <t,t> overall
  - Particularly high on SimpleQA (24.5%) and MMLU-Pro (20.1%)

- **Low <t,t> models**:
  - Claude-Opus-4.1: 1.5% <t,t> overall
  - Llama-4-Scout: 2.6% <t,t> overall
  - Minimal contradictions across all benchmarks

This suggests different models have different degrees of **doxastic coherence** - a key concept in propositional interpretability. Models with high <t,t> rates might be harder to interpret propositionally because their belief systems are less coherent.

### 4. The Verification/Refutation Decomposition

The bilateral framework's decomposition into verification and refutation maps onto a fundamental distinction in belief:
- **Reasons for belief** (verification)
- **Reasons against belief** (refutation)

This decomposition allows us to see when models have:
- **Asymmetric justification**: Can verify but not refute (or vice versa)
- **Conflicting justification**: Can both verify and refute
- **No justification**: Can neither verify nor refute

This is richer than Chalmers' binary belief attribution and might offer a path toward more nuanced propositional interpretation.

### 5. Family Resemblance and Belief Formation

Our finding that model families DON'T share bilateral patterns (p=0.52) suggests:
- **Belief formation is training-dependent**, not architecture-dependent
- Different training creates different "epistemic personalities"
- Propositional content emerges from training, not just architecture

This aligns with Chalmers' suggestion that propositional interpretability might require understanding the training process, not just the final model.

## Implications for Propositional Interpretability

### The Bilateral Framework as a Probe

The bilateral evaluation can be seen as a **structured probe** for propositional attitudes:

1. **Forces decomposition**: Models must separately evaluate verification and refutation
2. **Reveals conflicts**: The zeta function exposes when these evaluations contradict
3. **Distinguishes uncertainty types**: Separates epistemic uncertainty (<f,f>) from evaluation failure (<e,e>)

### Beyond Binary Belief

While Chalmers often discusses belief in binary terms, our analysis suggests models have richer propositional attitudes:

| Traditional Binary | Bilateral Framework | Interpretation |
|-------------------|-------------------|----------------|
| Believes P | <t,f> | Has evidence for P, no evidence against |
| Believes ¬P | <f,t> | Has evidence against P, no evidence for |
| Uncertain about P | <f,f> or <t,t> | No evidence (<f,f>) or conflicting evidence (<t,t>) |

This distinction between types of uncertainty is invisible in binary frameworks but crucial for understanding model epistemology.

### Toward Interpretable Belief Systems

The bilateral framework might offer a path toward what Chalmers calls the "holy grail" - understanding models as having coherent belief systems:

1. **Belief Coherence Metrics**: 
   - <t,t> rates measure internal consistency
   - Lower rates suggest more coherent belief systems
   - Range from 1.5% (Claude-Opus) to 14.4% (Gemini-Flash)

2. **Epistemic Profiles**:
   - <f,f> rates reveal knowledge boundaries
   - Range from 13.1% (Llama-Maverick) to 38.1% (Claude-Haiku)
   - Shows which models "know what they don't know"

3. **Justification Patterns**:
   - Asymmetric patterns (<t,e>, <e,t>) show reasoning biases
   - Complete failures (<e,e>) indicate processing limits
   - Successful evaluations reveal belief formation patterns

## Critical Questions

This raises several questions relevant to Chalmers' program:

### 1. Are these genuine beliefs?
Or are we just measuring response patterns that look like beliefs? The bilateral framework reveals consistent patterns that persist across different formulations of the same proposition, suggesting something more than surface-level response matching.

### 2. Is coherence necessary?
Models with high <t,t> rates (like Gemini-Flash at 14.4%) still function effectively. Perhaps contradictory "beliefs" aren't problematic for AI systems in the way they would be for human belief systems. This challenges traditional epistemological assumptions about rational belief.

### 3. What about grounding?
The bilateral framework reveals propositional attitudes but not whether these are grounded in genuine understanding. A model might consistently show <t,f> for "Paris is the capital of France" without any genuine semantic understanding of capitals, countries, or geography.

### 4. Compositional structure?
Can we use bilateral evaluation to test if models have compositional belief structures? For example:
- If a model shows <t,f> for "All birds can fly"
- And <t,f> for "Penguins are birds"
- Does it show <t,f> for "Penguins can fly"?

Testing logical consistency through bilateral patterns could reveal whether models have genuinely compositional belief structures.

## Empirical Findings Supporting Propositional Interpretation

### Evidence for Belief-Like States

1. **Consistency across benchmarks**: Models show characteristic bilateral patterns that persist across different domains
2. **Epistemic humility**: <f,f> patterns suggest models can recognize knowledge limits
3. **Contradiction recognition**: <t,t> patterns indicate conflicting evidence processing

### Evidence Against Simple Propositional Interpretation

1. **Unidirectional overconfidence**: Models only show <t,f> errors when false, never <f,t> errors when true
2. **Family independence**: Architectural similarity doesn't predict belief patterns
3. **Benchmark sensitivity**: Same models show different patterns on different types of questions

## The Bilateral Framework as a Bridge

The bilateral truth evaluation framework provides an empirical bridge between:
- **Behavioral outputs** (what models say)
- **Propositional attitudes** (what models "believe")
- **Epistemic states** (how models handle uncertainty)

This operationalizes key aspects of Chalmers' propositional interpretability while revealing the complexity of model epistemology.

## Conclusions

The bilateral framework offers several contributions to propositional interpretability:

1. **Empirical methodology**: A concrete way to probe propositional attitudes
2. **Richer taxonomy**: Beyond binary belief to include contradiction and uncertainty
3. **Coherence metrics**: Quantifiable measures of belief system consistency
4. **Training dependence**: Evidence that propositional content emerges from training

While not resolving the fundamental question of whether models have genuine beliefs, the bilateral framework provides a powerful tool for investigating the structure and coherence of whatever belief-like states models may possess.

## Future Directions

1. **Logical consistency testing**: Use bilateral evaluation to test compositional belief structure
2. **Temporal stability**: Track how bilateral patterns change during training
3. **Cross-lingual evaluation**: Test if bilateral patterns persist across languages
4. **Causal intervention**: Modify training to see how bilateral patterns change
5. **Belief revision**: Test how models update bilateral patterns with new information

The bilateral truth evaluation framework thus provides an empirical approach to some of Chalmers' theoretical questions about propositional interpretability, offering a window into what might be called the "doxastic structure" of language models.