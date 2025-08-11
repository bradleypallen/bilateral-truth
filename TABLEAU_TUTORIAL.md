# Using bilateral-truth as an External Oracle in Tableau Reasoning

This tutorial demonstrates how to integrate the `bilateral-truth` package as an external oracle in tableau-based automated reasoning systems. The bilateral evaluation mechanism provides generalized truth values `<u,v>` that can enhance tableau construction by offering nuanced assessments of atomic formulas.

## Quick Start

### Installation and Setup

```bash
pip install bilateral-truth
export OPENAI_API_KEY='your-openai-key'  # Optional: for real LLM evaluation
export ANTHROPIC_API_KEY='your-anthropic-key'  # Optional: for Claude models
```

### Basic Usage as External Oracle

```python
from bilateral_truth import zeta_c, Assertion, create_llm_evaluator

# Create evaluator (mock for development, real LLM for production)
evaluator = create_llm_evaluator("mock")

# Define atomic formula
assertion = Assertion("The sky is blue")

# Get bilateral evaluation as external oracle
truth_value = zeta_c(assertion, evaluator.evaluate_bilateral)
print(f"Oracle result: {truth_value}")  # Output: <t,f> or similar
```

## Core API Components

### 1. Assertion Class - Representing Atomic Formulas

```python
from bilateral_truth import Assertion

# Natural language assertions
weather_claim = Assertion("It is raining in Seattle")
mathematical_fact = Assertion("2 + 2 = 4")

# Predicate logic with arguments
has_property = Assertion("HasColor", "sky", color="blue")
greater_than = Assertion("GreaterThan", 5, 3)

# Complex assertions with named arguments
database_query = Assertion("UserExists", user_id=12345, active=True)
```

### 2. Generalized Truth Values - Oracle Responses

```python
from bilateral_truth import GeneralizedTruthValue, TruthValueComponent

# Access components of oracle response
result = zeta_c(assertion, evaluator.evaluate_bilateral)
verifiability = result.u  # TruthValueComponent.TRUE/FALSE/UNDEFINED
refutability = result.v   # TruthValueComponent.TRUE/FALSE/UNDEFINED

# Standard truth value patterns
classical_true = GeneralizedTruthValue(TruthValueComponent.TRUE, TruthValueComponent.FALSE)    # <t,f>
classical_false = GeneralizedTruthValue(TruthValueComponent.FALSE, TruthValueComponent.TRUE)  # <f,t>
unknown = GeneralizedTruthValue(TruthValueComponent.UNDEFINED, TruthValueComponent.UNDEFINED) # <e,e>
```

### 3. Cached Evaluation Function - The Oracle Interface

```python
from bilateral_truth import zeta_c, clear_cache, get_cache_size

# Primary oracle function: ζ_c(φ) 
def query_oracle(formula):
    """Query bilateral truth oracle for atomic formula."""
    assertion = Assertion(formula)
    return zeta_c(assertion, evaluator.evaluate_bilateral)

# Cache management for tableau construction
def reset_oracle_cache():
    """Clear oracle cache for fresh tableau construction."""
    clear_cache()
    
def oracle_cache_stats():
    """Get oracle cache statistics."""
    return get_cache_size()
```

## Tableau Integration Patterns

### 1. Basic Oracle Integration

```python
class TableauOracleReasoner:
    def __init__(self, model_provider="mock"):
        """Initialize tableau reasoner with bilateral truth oracle."""
        self.evaluator = create_llm_evaluator(model_provider)
        self.oracle_queries = 0
    
    def query_bilateral_oracle(self, formula_str):
        """Query the bilateral truth oracle for an atomic formula."""
        assertion = Assertion(formula_str)
        self.oracle_queries += 1
        
        # Get cached bilateral evaluation
        result = zeta_c(assertion, self.evaluator.evaluate_bilateral)
        return result
    
    def should_expand_branch(self, atomic_formula):
        """Use oracle to decide tableau branch expansion."""
        oracle_result = self.query_bilateral_oracle(atomic_formula)
        
        # Classical interpretation for tableau decisions
        if oracle_result == GeneralizedTruthValue(TruthValueComponent.TRUE, TruthValueComponent.FALSE):
            return "accept_as_true"
        elif oracle_result == GeneralizedTruthValue(TruthValueComponent.FALSE, TruthValueComponent.TRUE):
            return "accept_as_false"
        else:
            return "requires_expansion"  # Non-classical cases need further reasoning
```

### 2. Robust Oracle with Sampling

```python
class RobustTableauOracle:
    def __init__(self, model_provider="gpt-4", samples=3):
        """Initialize robust oracle with majority voting."""
        self.evaluator = create_llm_evaluator(model_provider)
        self.samples = samples
    
    def evaluate_with_confidence(self, formula_str):
        """Get bilateral evaluation with confidence from multiple samples."""
        assertion = Assertion(formula_str)
        
        # Use sampling for robust evaluation
        result = zeta_c(
            assertion, 
            self.evaluator.evaluate_bilateral,
            samples=self.samples,
            tiebreak_strategy="random"
        )
        
        return result
    
    def classify_for_tableau(self, formula_str):
        """Classify atomic formula for tableau construction."""
        result = self.evaluate_with_confidence(formula_str)
        
        # Map to tableau-relevant categories
        if result.u == TruthValueComponent.TRUE and result.v == TruthValueComponent.FALSE:
            return {"classification": "definitely_true", "confidence": "high"}
        elif result.u == TruthValueComponent.FALSE and result.v == TruthValueComponent.TRUE:
            return {"classification": "definitely_false", "confidence": "high"}
        elif result.u == TruthValueComponent.TRUE and result.v == TruthValueComponent.TRUE:
            return {"classification": "contradictory", "confidence": "medium"}
        elif result.u == TruthValueComponent.FALSE and result.v == TruthValueComponent.FALSE:
            return {"classification": "neither_verifiable_nor_refutable", "confidence": "low"}
        else:
            return {"classification": "uncertain", "confidence": "low"}
```

### 3. Complete Tableau Example

```python
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class TableauNode:
    """Tableau node with bilateral oracle evaluations."""
    formulas: List[str]
    oracle_evaluations: Dict[str, GeneralizedTruthValue] = None
    is_closed: bool = False
    
    def __post_init__(self):
        if self.oracle_evaluations is None:
            self.oracle_evaluations = {}

class BilateralTableau:
    """Complete tableau reasoner using bilateral truth oracle."""
    
    def __init__(self, model_provider="mock"):
        self.oracle = TableauOracleReasoner(model_provider)
        self.proof_trace = []
    
    def prove_unsatisfiability(self, formula_set):
        """Prove unsatisfiability using tableau with oracle assistance."""
        root = TableauNode(formulas=list(formula_set))
        return self._expand_tableau([root])
    
    def _expand_tableau(self, open_nodes):
        """Expand tableau nodes using oracle for atomic formulas."""
        while open_nodes:
            current = open_nodes.pop(0)
            
            # Evaluate atomic formulas with oracle
            self._evaluate_atomics(current)
            
            # Check for closure
            if self._is_closed(current):
                current.is_closed = True
                continue
            
            # Expand complex formulas or declare open
            expanded_nodes = self._expand_complex_formulas(current)
            if expanded_nodes:
                open_nodes.extend(expanded_nodes)
            else:
                # Leaf node - check satisfiability using oracle
                if self._is_oracle_satisfiable(current):
                    return {"satisfiable": True, "model": current.oracle_evaluations}
        
        return {"satisfiable": False, "all_branches_closed": True}
    
    def _evaluate_atomics(self, node):
        """Use oracle to evaluate atomic formulas in node."""
        for formula in node.formulas:
            if self._is_atomic(formula) and formula not in node.oracle_evaluations:
                result = self.oracle.query_bilateral_oracle(formula)
                node.oracle_evaluations[formula] = result
                
                self.proof_trace.append({
                    "action": "oracle_query",
                    "formula": formula,
                    "result": str(result)
                })
    
    def _is_atomic(self, formula):
        """Check if formula is atomic (simplified)."""
        logical_operators = ['&', '|', '->', '~', '∃', '∀', '(', ')']
        return not any(op in formula for op in logical_operators)
    
    def _is_closed(self, node):
        """Check if node is closed using oracle evaluations."""
        # Look for classical contradictions in oracle results
        for formula, result in node.oracle_evaluations.items():
            # Check for explicit negation
            neg_formula = f"~{formula}" if not formula.startswith('~') else formula[1:]
            
            if neg_formula in node.oracle_evaluations:
                neg_result = node.oracle_evaluations[neg_formula]
                
                # Classical contradiction: <t,f> and <f,t>
                if (result.u == TruthValueComponent.TRUE and 
                    result.v == TruthValueComponent.FALSE and
                    neg_result.u == TruthValueComponent.FALSE and
                    neg_result.v == TruthValueComponent.TRUE):
                    
                    self.proof_trace.append({
                        "action": "closure",
                        "reason": f"Oracle contradiction: {formula} = {result}, {neg_formula} = {neg_result}"
                    })
                    return True
        
        return False
    
    def _is_oracle_satisfiable(self, node):
        """Check if node is satisfiable according to oracle."""
        # All atomic formulas must have definite oracle assignments
        for formula, result in node.oracle_evaluations.items():
            # Undefined results indicate insufficient information
            if (result.u == TruthValueComponent.UNDEFINED and 
                result.v == TruthValueComponent.UNDEFINED):
                return False
        
        # Check for internal contradictions
        contradictory_count = 0
        for result in node.oracle_evaluations.values():
            if (result.u == TruthValueComponent.TRUE and 
                result.v == TruthValueComponent.TRUE):
                contradictory_count += 1
        
        # Too many contradictions suggest unsatisfiability
        return contradictory_count <= len(node.oracle_evaluations) * 0.3
    
    def _expand_complex_formulas(self, node):
        """Expand complex formulas (simplified - implement your logical rules)."""
        # This would contain your tableau expansion rules
        # For demonstration, we assume all formulas are atomic
        return []

# Example usage
def demo_bilateral_tableau():
    """Demonstrate tableau reasoning with bilateral oracle."""
    tableau = BilateralTableau("mock")
    
    # Test unsatisfiability of contradictory statements
    formula_set = [
        "The sky is blue",
        "The sky is not blue",
        "2 + 2 = 4",
        "Water boils at 100°C"
    ]
    
    result = tableau.prove_unsatisfiability(formula_set)
    
    print("Tableau Result:", result)
    print("\nProof Trace:")
    for step in tableau.proof_trace:
        print(f"  {step}")
    
    return result
```

## Advanced Oracle Configurations

### Multiple Oracle Consensus

```python
class ConsensusOracle:
    """Oracle using multiple LLM providers for robust evaluation."""
    
    def __init__(self, providers=["openai", "anthropic"], consensus_threshold=0.6):
        self.evaluators = [create_llm_evaluator(p) for p in providers]
        self.threshold = consensus_threshold
    
    def evaluate_with_consensus(self, formula_str, samples_per_evaluator=1):
        """Get consensus evaluation across multiple oracles."""
        assertion = Assertion(formula_str)
        all_results = []
        
        for evaluator in self.evaluators:
            for _ in range(samples_per_evaluator):
                result = evaluator.evaluate_bilateral(assertion)
                all_results.append(result)
        
        # Find consensus
        from collections import Counter
        result_counts = Counter(all_results)
        
        if result_counts:
            most_common, count = result_counts.most_common(1)[0]
            confidence = count / len(all_results)
            
            if confidence >= self.threshold:
                return {"result": most_common, "confidence": confidence}
        
        # No consensus
        return {
            "result": GeneralizedTruthValue(TruthValueComponent.UNDEFINED, TruthValueComponent.UNDEFINED),
            "confidence": 0.0
        }
```

## Best Practices

### 1. Oracle Query Optimization
- Always use `zeta_c` (cached) instead of `zeta` for repeated queries
- Use mock evaluators during development for fast iteration
- Group related atomic formulas to minimize API calls

### 2. Error Handling
```python
def safe_oracle_query(formula_str, evaluator, fallback=None):
    """Safely query oracle with error handling."""
    try:
        assertion = Assertion(formula_str)
        return zeta_c(assertion, evaluator.evaluate_bilateral)
    except Exception as e:
        print(f"Oracle query failed: {e}")
        return fallback or GeneralizedTruthValue(
            TruthValueComponent.UNDEFINED, 
            TruthValueComponent.UNDEFINED
        )
```

### 3. Epistemic Policy Integration
```python
from bilateral_truth import EpistemicPolicy

def interpret_oracle_result(result, policy=EpistemicPolicy.CLASSICAL):
    """Interpret bilateral result for tableau decisions."""
    classical_value = result.project(policy)
    
    return {
        TruthValueComponent.TRUE: "tableau_true",
        TruthValueComponent.FALSE: "tableau_false", 
        TruthValueComponent.UNDEFINED: "tableau_unknown"
    }[classical_value]
```

### 4. Model Selection Guidelines
- **Development**: Use `"mock"` for deterministic, fast results
- **Mathematical reasoning**: Use `"openai"` with GPT-4
- **Factual evaluation**: Use `"anthropic"` with Claude models
- **Production**: Implement consensus with multiple providers

This tutorial provides a complete framework for integrating `bilateral-truth` as an external oracle in tableau-based reasoning systems. The key advantages include:

1. **Cached evaluations** via `zeta_c` for efficient repeated queries
2. **Generalized truth values** `<u,v>` providing richer information than classical true/false
3. **Multiple evaluation strategies** from deterministic mock to consensus-based real LLM evaluation
4. **Flexible integration patterns** that can adapt to different tableau implementations

The bilateral oracle can significantly enhance automated reasoning by providing nuanced assessments of atomic formulas that go beyond classical binary decisions, enabling more sophisticated tableau construction and proof strategies.