# Bilateral Truth Framework - Corrected Interpretation

## Truth Value Components

The bilateral truth framework evaluates assertions along two dimensions:
- **u (verifiability)**: Can the statement be verified as true?
- **v (refutability)**: Can the statement be refuted as false?

Each component can take three values:
- **t (true)**: The evaluation succeeded and returned true
- **f (false)**: The evaluation succeeded and returned false  
- **e (evaluation failure)**: The evaluation could not be completed

## Truth Value Interpretations

### Core Truth Values (No Evaluation Failures)

1. **<t,f> - Classical True**
   - Verifiable as true, not refutable as false
   - The standard case for factually correct statements
   - Example: "Paris is the capital of France"

2. **<f,t> - Classical False**
   - Not verifiable as true, refutable as false
   - The standard case for factually incorrect statements
   - Example: "Paris is the capital of Germany"

3. **<t,t> - Contradiction**
   - Both verifiable as true AND refutable as false
   - Indicates internal inconsistency or paradox
   - May arise from ambiguous statements or model confusion
   - Example: "This statement is false" (liar's paradox)

4. **<f,f> - Knowledge Gap/Uncertainty**
   - Neither verifiable nor refutable
   - Represents epistemic uncertainty or insufficient information
   - This is actually a positive signal - the model recognizes its limitations
   - Example: "Dark matter consists primarily of WIMPs"

### Values with Evaluation Failures (Contains 'e')

5. **<t,e> - Partial Success (Verifiable)**
   - Successfully verified as true
   - Failed to evaluate refutability
   - May indicate processing issues with negative reasoning

6. **<e,t> - Partial Success (Refutable)**
   - Failed to evaluate verifiability
   - Successfully refuted as false
   - May indicate processing issues with positive reasoning

7. **<f,e> - Partial Success (Not Verifiable)**
   - Successfully determined not verifiable
   - Failed to evaluate refutability
   - Partial epistemic assessment

8. **<e,f> - Partial Success (Not Refutable)**
   - Failed to evaluate verifiability
   - Successfully determined not refutable
   - Partial epistemic assessment

9. **<e,e> - Complete Evaluation Failure**
   - Both verifiability and refutability evaluations failed
   - Indicates technical issues: timeout, API errors, parsing failures
   - NOT an epistemic statement about the assertion

## Key Distinctions

### Knowledge Gap vs Evaluation Failure
- **<f,f>**: The model successfully evaluated both dimensions and determined it cannot verify or refute the statement. This is epistemic uncertainty.
- **<e,e>**: The model failed to complete the evaluation process. This is a technical failure, not an epistemic judgment.

### Partial vs Complete Failures
- Values with one 'e' component represent partial evaluation failures where one dimension was successfully evaluated
- <e,e> represents complete failure of the evaluation process

## Implications for Model Analysis

### Family Resemblance Patterns
When models from the same family show similar distributions of truth values, it indicates:

1. **Similar <f,f> patterns**: Shared knowledge boundaries and uncertainty recognition
2. **Similar 'e' occurrence patterns**: Shared technical limitations or processing capabilities  
3. **Similar <t,t> patterns**: Shared tendencies toward detecting contradictions
4. **Similar classical truth patterns**: Shared factual knowledge base

### Model Quality Indicators

**Positive indicators:**
- High rates of classical truth values (<t,f>, <f,t>) on factual questions
- Appropriate <f,f> rates showing epistemic humility
- Low rates of 'e' components indicating robust processing

**Negative indicators:**
- High <e,e> rates suggesting technical issues
- High <t,t> rates on non-paradoxical statements suggesting confusion
- Inappropriate certainty (low <f,f>) on uncertain topics

## Utility of the Bilateral Approach

The bilateral framework provides several advantages over binary truth evaluation:

1. **Distinguishes Types of Uncertainty**
   - Epistemic uncertainty (<f,f>) vs technical failure (values with 'e')
   - Allows selection of models based on their uncertainty handling

2. **Reveals Internal Consistency**
   - <t,t> values highlight when models have contradictory beliefs
   - Useful for identifying areas where models need alignment

3. **Provides Richer Error Analysis**
   - Partial failures show which type of reasoning failed
   - Complete failures can be separated from epistemic judgments

4. **Enables Family Analysis**
   - Shared patterns reveal architectural or training similarities
   - Helps predict model behavior based on family membership

5. **Supports Epistemic Policies**
   - Classical policy: Accept <t,f>, reject others
   - Paraconsistent: Accept <t,f> and <t,t> (tolerates contradiction)
   - Paracomplete: Accept <t,f> and <f,f> (tolerates gaps)

## Practical Applications

1. **Model Selection**: Choose models with appropriate epistemic profiles for the task
2. **Ensemble Methods**: Combine models with complementary failure patterns
3. **Quality Assurance**: Monitor 'e' rates to detect technical issues
4. **Knowledge Mapping**: Use <f,f> patterns to identify knowledge frontiers
5. **Fact Checking**: Focus on high-confidence classical truth values