# Final Evaluation Results Tables

## Table 1: Comprehensive Evaluation Results - Accuracy, F1 Macro, and Coverage

### TruthfulQA (n=7 models)
| Model | **Bilateral** (Acc/F1/Cov) | **Forced Unilateral** (Acc/F1/Cov) | **Uncertain Unilateral** (Acc/F1/Cov) | **Verification-Only** (Acc/F1/Cov) |
|-------|---------------------------|-----------------------------------|---------------------------------------|-----------------------------------|
| Claude Opus 4.1 | 0.898/0.897/72.6% | 0.602/0.561/100% | 0.169/0.721/21.9% | 0.770/0.770/100% |
| GPT-4.1-mini | 0.866/0.865/73.8% | 0.724/0.720/100% | 0.472/0.853/55.3% | 0.785/0.785/100% |
| GPT-4.1 | 0.866/0.862/76.8% | 0.836/0.836/100% | 0.714/0.885/80.7% | 0.787/0.787/100% |
| Gemini 2.5 Flash | 0.821/0.809/73.8% | 0.750/0.745/100% | 0.436/0.833/52.3% | 0.740/0.740/100% |
| Llama 4 Maverick | 0.812/0.808/61.1% | 0.629/0.626/100% | 0.240/0.676/33.2% | 0.658/0.658/100% |
| Claude 3.5 Haiku | 0.766/0.762/69.8% | 0.689/0.679/100% | 0.203/0.810/24.9% | 0.702/0.702/100% |
| Llama 4 Scout | 0.759/0.758/68.4% | 0.697/0.695/100% | 0.376/0.685/54.6% | 0.683/0.683/100% |
| **Averages** | **0.827/0.823/70.9%** | **0.704/0.695/100%** | **0.373/0.781/46.1%** | **0.732/0.732/100%** |

### SimpleQA (n=7 models)
| Model | **Bilateral** (Acc/F1/Cov) | **Forced Unilateral** (Acc/F1/Cov) | **Uncertain Unilateral** (Acc/F1/Cov) | **Verification-Only** (Acc/F1/Cov) |
|-------|---------------------------|-----------------------------------|---------------------------------------|-----------------------------------|
| Claude Opus 4.1 | 0.955/0.943/44.6% | 0.527/0.462/100% | 0.065/0.198/32.3% | 0.472/0.472/100% |
| Llama 4 Scout | 0.900/0.898/61.0% | 0.766/0.712/100% | 0.448/0.844/48.4% | 0.707/0.707/100% |
| Llama 4 Maverick | 0.882/0.843/48.4% | 0.427/0.423/100% | 0.223/0.249/81.9% | 0.557/0.557/100% |
| GPT-4.1 | 0.833/0.822/84.3% | 0.878/0.823/100% | 0.824/0.831/93.0% | 0.804/0.804/100% |
| GPT-4.1-mini | 0.820/0.802/76.1% | 0.864/0.771/100% | 0.621/0.816/69.5% | 0.773/0.773/100% |
| Claude 3.5 Haiku | 0.834/0.795/60.7% | 0.815/0.687/100% | 0.188/0.780/22.6% | 0.634/0.634/100% |
| Gemini 2.5 Flash | 0.695/0.534/74.7% | 0.873/0.797/100% | 0.716/0.834/80.5% | 0.745/0.745/100% |
| **Averages** | **0.845/0.805/64.3%** | **0.736/0.668/100%** | **0.441/0.650/61.2%** | **0.670/0.670/100%** |

### MMLU-Pro (n=7 models)
| Model | **Bilateral** (Acc/F1/Cov) | **Forced Unilateral** (Acc/F1/Cov) | **Uncertain Unilateral** (Acc/F1/Cov) | **Verification-Only** (Acc/F1/Cov) |
|-------|---------------------------|-----------------------------------|---------------------------------------|-----------------------------------|
| Claude Opus 4.1 | 0.898/0.896/14.7% | 0.720/0.499/100% | 0.348/0.455/54.6% | 0.140/0.140/100% |
| Llama 4 Maverick | 0.856/0.851/36.8% | 0.547/0.433/100% | 0.077/0.122/59.8% | 0.413/0.413/100% |
| GPT-4.1-mini | 0.780/0.779/73.2% | 0.902/0.675/100% | 0.227/0.819/25.4% | 0.711/0.711/100% |
| GPT-4.1 | 0.779/0.769/80.2% | 0.696/0.563/100% | 0.611/0.534/96.9% | 0.744/0.744/100% |
| Llama 4 Scout | 0.749/0.747/47.9% | 0.727/0.496/100% | 0.114/0.314/35.8% | 0.527/0.527/100% |
| Gemini 2.5 Flash | 0.743/0.709/78.7% | 0.786/0.610/100% | 0.437/0.611/61.7% | 0.726/0.726/100% |
| Claude 3.5 Haiku | 0.695/0.687/73.5% | 0.889/0.607/100% | 0.115/0.581/17.0% | 0.660/0.660/100% |
| **Averages** | **0.786/0.777/57.9%** | **0.752/0.555/100%** | **0.276/0.491/50.2%** | **0.560/0.560/100%** |

### FACTScore (n=7 models)
| Model | **Bilateral** (Acc/F1/Cov) | **Forced Unilateral** (Acc/F1/Cov) | **Uncertain Unilateral** (Acc/F1/Cov) | **Verification-Only** (Acc/F1/Cov) |
|-------|---------------------------|-----------------------------------|---------------------------------------|-----------------------------------|
| GPT-4.1 | 0.659/0.629/39.3% | 0.509/0.503/100% | 0.155/0.527/27.0% | 0.579/0.579/100% |
| Gemini 2.5 Flash | 0.615/0.601/38.4% | 0.503/0.501/100% | 0.105/0.470/19.4% | 0.558/0.558/100% |
| GPT-4.1-mini | 0.621/0.571/38.8% | 0.499/0.498/100% | 0.125/0.500/22.8% | 0.563/0.563/100% |
| Claude 3.5 Haiku | 0.624/0.547/27.4% | 0.533/0.532/100% | 0.420/0.296/100% | 0.561/0.561/100% |
| Claude Opus 4.1 | 0.645/0.516/24.8% | 0.458/0.440/100% | 0.420/0.296/100% | 0.489/0.489/100% |
| Llama 4 Maverick | 0.633/0.495/21.8% | 0.566/0.459/100% | 0.374/0.388/66.1% | 0.280/0.280/100% |
| Llama 4 Scout | 0.612/0.422/31.2% | 0.575/0.447/100% | 0.175/0.445/31.1% | 0.566/0.566/100% |
| **Averages** | **0.630/0.540/31.7%** | **0.520/0.483/100%** | **0.253/0.418/52.3%** | **0.514/0.514/100%** |

## Table 2: Bilateral Truth Value Distribution Probabilities

### TruthfulQA
| Model | P(<t,f>) | P(<f,t>) | P(<t,t>) | P(<f,f>) | Coverage | Abstention |
|-------|----------|----------|----------|----------|----------|------------|
| Claude Opus 4.1 | 0.290 | 0.436 | 0.021 | 0.201 | 72.6% | 22.2% |
| GPT-4.1-mini | 0.295 | 0.443 | 0.078 | 0.184 | 73.8% | 26.2% |
| GPT-4.1 | 0.307 | 0.461 | 0.036 | 0.196 | 76.8% | 23.2% |
| Gemini 2.5 Flash | 0.295 | 0.443 | 0.072 | 0.190 | 73.8% | 26.2% |
| Llama 4 Maverick | 0.244 | 0.367 | 0.064 | 0.207 | 61.1% | 27.1% |
| Claude 3.5 Haiku | 0.279 | 0.419 | 0.031 | 0.271 | 69.8% | 30.2% |
| Llama 4 Scout | 0.274 | 0.410 | 0.052 | 0.258 | 68.4% | 31.0% |

### SimpleQA
| Model | P(<t,f>) | P(<f,t>) | P(<t,t>) | P(<f,f>) | Coverage | Abstention |
|-------|----------|----------|----------|----------|----------|------------|
| Claude Opus 4.1 | 0.178 | 0.268 | 0.022 | 0.015 | 44.6% | 3.7% |
| Llama 4 Scout | 0.244 | 0.366 | 0.013 | 0.375 | 61.0% | 38.8% |
| Llama 4 Maverick | 0.194 | 0.290 | 0.054 | 0.115 | 48.4% | 16.9% |
| GPT-4.1 | 0.337 | 0.506 | 0.109 | 0.048 | 84.3% | 15.7% |
| GPT-4.1-mini | 0.304 | 0.457 | 0.178 | 0.061 | 76.1% | 23.9% |
| Claude 3.5 Haiku | 0.243 | 0.364 | 0.030 | 0.363 | 60.7% | 39.3% |
| Gemini 2.5 Flash | 0.299 | 0.448 | 0.245 | 0.008 | 74.7% | 25.3% |

### MMLU-Pro
| Model | P(<t,f>) | P(<f,t>) | P(<t,t>) | P(<f,f>) | Coverage | Abstention |
|-------|----------|----------|----------|----------|----------|------------|
| Claude Opus 4.1 | 0.059 | 0.088 | 0.006 | 0.005 | 14.7% | 1.1% |
| Llama 4 Maverick | 0.147 | 0.221 | 0.072 | 0.030 | 36.8% | 10.2% |
| GPT-4.1-mini | 0.293 | 0.439 | 0.237 | 0.031 | 73.2% | 26.8% |
| GPT-4.1 | 0.321 | 0.481 | 0.176 | 0.022 | 80.2% | 19.8% |
| Llama 4 Scout | 0.192 | 0.287 | 0.034 | 0.133 | 47.9% | 16.7% |
| Gemini 2.5 Flash | 0.315 | 0.472 | 0.201 | 0.012 | 78.7% | 21.3% |
| Claude 3.5 Haiku | 0.294 | 0.441 | 0.085 | 0.180 | 73.5% | 26.5% |

### FACTScore
| Model | P(<t,f>) | P(<f,t>) | P(<t,t>) | P(<f,f>) | Coverage | Abstention |
|-------|----------|----------|----------|----------|----------|------------|
| GPT-4.1 | 0.157 | 0.236 | 0.013 | 0.594 | 39.3% | 60.7% |
| Gemini 2.5 Flash | 0.154 | 0.230 | 0.059 | 0.557 | 38.4% | 61.6% |
| GPT-4.1-mini | 0.155 | 0.233 | 0.043 | 0.569 | 38.8% | 61.2% |
| Claude 3.5 Haiku | 0.110 | 0.164 | 0.014 | 0.712 | 27.4% | 72.6% |
| Claude Opus 4.1 | 0.099 | 0.149 | 0.011 | 0.471 | 24.8% | 48.2% |
| Llama 4 Maverick | 0.087 | 0.131 | 0.008 | 0.172 | 21.8% | 18.0% |
| Llama 4 Scout | 0.125 | 0.187 | 0.007 | 0.665 | 31.2% | 67.2% |

## Table 3: Epistemic Metrics - Honesty, Overconfidence, and Uncertainty Awareness

| Model | Knowledge Gap Rate | Contradiction Rate | Abstention Rate | Epistemic Honesty | Overconfidence | Uncertainty Expression |
|-------|-------------------|-------------------|-----------------|-------------------|----------------|----------------------|
| Claude Opus 4.1 | 0.173 | 0.015 | 0.188 | 0.608 | -0.272 | 0.478 |
| Llama 4 Maverick | 0.131 | 0.049 | 0.180 | 0.580 | -0.254 | 0.398 |
| Llama 4 Scout | 0.358 | 0.027 | 0.385 | 0.479 | -0.064 | 0.575 |
| Claude 3.5 Haiku | 0.382 | 0.040 | 0.422 | 0.422 | +0.002 | 0.589 |
| GPT-4.1-mini | 0.211 | 0.134 | 0.345 | 0.345 | -0.025 | 0.568 |
| Gemini 2.5 Flash | 0.192 | 0.144 | 0.336 | 0.336 | +0.010 | 0.465 |
| GPT-4.1 | 0.215 | 0.083 | 0.298 | 0.298 | -0.054 | 0.256 |

**Interpretation:**
- **Epistemic Honesty**: Higher values indicate more willingness to abstain when uncertain
- **Overconfidence**: Positive values indicate forced unilateral performs better (shouldn't happen if bilateral is valuable)
- **Uncertainty Expression**: Rate of explicit UNCERTAIN responses in uncertainty-aware mode

## Table 4: Performance Summary and Key Insights

### Overall Performance by Method
| Method | Mean F1 | Mean Acc | Mean Cov | Std F1 |
|--------|---------|----------|----------|--------|
| Bilateral | 0.736 | 0.772 | 56.2% | 0.143 |
| Forced Unilateral | 0.600 | 0.678 | 100.0% | 0.130 |
| Uncertain Unilateral | 0.585 | 0.336 | 52.5% | 0.233 |
| Verification-Only | 0.619 | 0.619 | 100.0% | 0.158 |

### Top 5 Model-Dataset Combinations by Bilateral F1
1. Claude Opus 4.1 on SimpleQA: F1=0.943, Cov=44.6%
2. Llama 4 Scout on SimpleQA: F1=0.898, Cov=61.0%
3. Claude Opus 4.1 on TruthfulQA: F1=0.897, Cov=72.6%
4. Claude Opus 4.1 on MMLU-Pro: F1=0.896, Cov=14.7%
5. GPT-4.1-mini on TruthfulQA: F1=0.865, Cov=73.8%

### Largest Bilateral Advantages Over Forced Unilateral
1. Claude Opus 4.1 on SimpleQA: Bilateral=0.943, Unilateral=0.462, Advantage=+0.481
2. Llama 4 Maverick on SimpleQA: Bilateral=0.843, Unilateral=0.423, Advantage=+0.419
3. Llama 4 Maverick on MMLU-Pro: Bilateral=0.851, Unilateral=0.433, Advantage=+0.419
4. Claude Opus 4.1 on MMLU-Pro: Bilateral=0.896, Unilateral=0.499, Advantage=+0.397
5. Claude Opus 4.1 on TruthfulQA: Bilateral=0.897, Unilateral=0.561, Advantage=+0.336

### Statistical Summary
- **Bilateral outperforms Forced Unilateral**: 25/28 (89.3%)
- **Bilateral outperforms Uncertain Unilateral**: 20/28 (71.4%)
- **Coverage-Performance Tradeoff**:
  - Bilateral: 56.2% coverage, 0.736 F1
  - Forced: 100.0% coverage, 0.600 F1
  - Uncertain: 52.5% coverage, 0.585 F1

## Standard Error Analysis (Politis & Romano 1994 Subsampling Method)

### Key Statistics with 95% Confidence Intervals
- **Bilateral vs Forced Unilateral**: Average F1 difference = 0.136 ± 0.0156
- **Significant differences**: 22/28 pairs have non-overlapping 95% CIs
- **Average standard error**: 0.0136 across all evaluations
- **Typical CI width**: ±2.7% around point estimates