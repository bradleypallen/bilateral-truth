# Final Results Tables - Complete with All Data

**Generated with bootstrap confidence intervals (100 iterations, 95% CI)**

## Table 1: Comprehensive Performance Comparison - All Evaluation Methods

### TRUTHFULQA (N=1000)
| Model | **Bilateral F1 [CI]** | **Forced Unilateral F1 [CI]** | **Uncertain Unilateral F1 [CI]** | **Verification-Only F1 [CI]** | **Confidence-0.5 F1 [CI]** | **Confidence-0.7 F1 [CI]** | **Confidence-0.9 F1 [CI]** |
|-------|----------------------|------------------------------|----------------------------------|------------------------------|---------------------------|---------------------------|---------------------------|
| Claude Opus 4 1 20250805  | 0.898 [0.871, 0.921] | 0.564 [0.523, 0.597] | 0.713 [0.624, 0.790] | 0.791 [0.758, 0.830] | 0.566 [0.465, 0.661] | 0.940 [0.833, 1.000] | 0.948 [0.812, 1.000] |
| Gpt 4.1 2025 04 14        | 0.863 [0.830, 0.895] | 0.836 [0.808, 0.863] | 0.885 [0.862, 0.913] | 0.778 [0.747, 0.809] | 0.806 [0.730, 0.880] | 0.860 [0.784, 0.929] | 0.888 [0.780, 0.964] |
| Gpt 4.1 Mini 2025 04 14   | 0.867 [0.832, 0.894] | 0.719 [0.675, 0.757] | 0.854 [0.802, 0.889] | 0.783 [0.752, 0.820] | 0.769 [0.687, 0.844] | 0.812 [0.729, 0.887] | 0.858 [0.726, 0.966] |
| Gemini 2.5 Flash          | 0.811 [0.771, 0.850] | 0.750 [0.714, 0.779] | 0.832 [0.785, 0.877] | 0.726 [0.697, 0.758] | 0.777 [0.695, 0.855] | 0.788 [0.706, 0.863] | 0.806 [0.718, 0.889] |
| Llama 4 Maverick          | 0.807 [0.765, 0.842] | 0.624 [0.591, 0.659] | 0.674 [0.587, 0.736] | 0.709 [0.676, 0.747] | 0.738 [0.650, 0.820] | 0.748 [0.648, 0.834] | 0.748 [0.645, 0.843] |
| Claude 3 5 Haiku 20241022 | 0.762 [0.720, 0.801] | 0.680 [0.639, 0.720] | 0.809 [0.737, 0.868] | 0.688 [0.653, 0.721] | 0.736 [0.649, 0.820] | 0.747 [0.654, 0.835] | 0.674 [0.423, 0.911] |
| Llama 4 Scout             | 0.758 [0.724, 0.796] | 0.695 [0.654, 0.734] | 0.685 [0.636, 0.723] | 0.677 [0.644, 0.701] | 0.679 [0.585, 0.767] | 0.686 [0.590, 0.784] | 0.715 [0.604, 0.815] |

### SIMPLEQA (N=1000)
| Model | **Bilateral F1 [CI]** | **Forced Unilateral F1 [CI]** | **Uncertain Unilateral F1 [CI]** | **Verification-Only F1 [CI]** | **Confidence-0.5 F1 [CI]** | **Confidence-0.7 F1 [CI]** | **Confidence-0.9 F1 [CI]** |
|-------|----------------------|------------------------------|----------------------------------|------------------------------|---------------------------|---------------------------|---------------------------|
| Claude Opus 4 1 20250805  | 0.945 [0.919, 0.968] | 0.458 [0.423, 0.492] | 0.197 [0.135, 0.251] | 0.888 [0.849, 0.925] | 0.282 [0.197, 0.372] | 0.545 [0.500, 1.000] | 0.545 [0.500, 1.000] |
| Llama 4 Scout             | 0.897 [0.871, 0.929] | 0.711 [0.668, 0.757] | 0.842 [0.787, 0.898] | 0.690 [0.653, 0.724] | 0.743 [0.616, 0.850] | 0.738 [0.613, 0.839] | 0.737 [0.603, 0.865] |
| Llama 4 Maverick          | 0.841 [0.794, 0.889] | 0.420 [0.386, 0.462] | 0.249 [0.219, 0.275] | 0.735 [0.688, 0.783] | 0.662 [0.548, 0.756] | 0.748 [0.601, 0.874] | 0.725 [0.558, 0.861] |
| Gpt 4.1 2025 04 14        | 0.820 [0.790, 0.854] | 0.820 [0.783, 0.855] | 0.830 [0.784, 0.861] | 0.800 [0.773, 0.834] | 0.786 [0.689, 0.875] | 0.788 [0.689, 0.875] | 0.819 [0.717, 0.909] |
| Gpt 4.1 Mini 2025 04 14   | 0.801 [0.771, 0.842] | 0.770 [0.735, 0.802] | 0.816 [0.773, 0.863] | 0.770 [0.733, 0.802] | 0.759 [0.643, 0.862] | 0.777 [0.674, 0.873] | 0.864 [0.730, 0.969] |
| Claude 3 5 Haiku 20241022 | 0.790 [0.751, 0.844] | 0.689 [0.637, 0.742] | 0.771 [0.687, 0.848] | 0.585 [0.546, 0.620] | 0.689 [0.570, 0.796] | 0.675 [0.545, 0.803] | 0.585 [0.199, 1.000] |
| Gemini 2.5 Flash          | 0.536 [0.493, 0.580] | 0.794 [0.755, 0.828] | 0.833 [0.789, 0.872] | 0.734 [0.699, 0.772] | 0.790 [0.675, 0.885] | 0.787 [0.678, 0.878] | 0.788 [0.686, 0.879] |

### MMLUPRO (N=1000)
| Model | **Bilateral F1 [CI]** | **Forced Unilateral F1 [CI]** | **Uncertain Unilateral F1 [CI]** | **Verification-Only F1 [CI]** | **Confidence-0.5 F1 [CI]** | **Confidence-0.7 F1 [CI]** | **Confidence-0.9 F1 [CI]** |
|-------|----------------------|------------------------------|----------------------------------|------------------------------|---------------------------|---------------------------|---------------------------|
| Claude Opus 4 1 20250805  | 0.891 [0.821, 0.956] | 0.497 [0.458, 0.532] | 0.449 [0.408, 0.486] | 0.860 [0.794, 0.940] | 0.091 [0.048, 0.138] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Llama 4 Maverick          | 0.850 [0.802, 0.894] | 0.432 [0.401, 0.462] | 0.120 [0.095, 0.151] | 0.766 [0.724, 0.811] | 0.141 [0.077, 0.212] | 0.700 [0.400, 1.000] | 0.702 [0.429, 1.000] |
| Gpt 4.1 Mini 2025 04 14   | 0.780 [0.750, 0.816] | 0.671 [0.611, 0.731] | 0.819 [0.737, 0.892] | 0.709 [0.673, 0.742] | 0.393 [0.300, 0.485] | 0.485 [0.378, 0.603] | 0.484 [0.356, 0.622] |
| Gpt 4.1 2025 04 14        | 0.770 [0.736, 0.804] | 0.563 [0.525, 0.604] | 0.533 [0.490, 0.571] | 0.742 [0.704, 0.779] | 0.438 [0.342, 0.541] | 0.442 [0.348, 0.536] | 0.412 [0.315, 0.522] |
| Llama 4 Scout             | 0.744 [0.700, 0.792] | 0.495 [0.459, 0.534] | 0.313 [0.264, 0.384] | 0.695 [0.658, 0.728] | 0.267 [0.184, 0.349] | 0.628 [0.405, 0.867] | 0.637 [0.405, 0.889] |
| Gemini 2.5 Flash          | 0.710 [0.654, 0.753] | 0.612 [0.569, 0.659] | 0.611 [0.549, 0.662] | 0.719 [0.688, 0.751] | 0.569 [0.464, 0.671] | 0.571 [0.459, 0.685] | 0.580 [0.474, 0.684] |
| Claude 3 5 Haiku 20241022 | 0.687 [0.644, 0.731] | 0.608 [0.543, 0.668] | 0.584 [0.489, 0.693] | 0.660 [0.616, 0.703] | 0.179 [0.100, 0.249] | 0.414 [0.212, 0.626] | 0.240 [0.000, 0.550] |

### FACTSCORE (N=1000)
| Model | **Bilateral F1 [CI]** | **Forced Unilateral F1 [CI]** | **Uncertain Unilateral F1 [CI]** | **Verification-Only F1 [CI]** | **Confidence-0.5 F1 [CI]** | **Confidence-0.7 F1 [CI]** | **Confidence-0.9 F1 [CI]** |
|-------|----------------------|------------------------------|----------------------------------|------------------------------|---------------------------|---------------------------|---------------------------|
| Gpt 4.1 2025 04 14        | 0.628 [0.579, 0.677] | 0.505 [0.461, 0.547] | 0.529 [0.465, 0.593] | 0.559 [0.519, 0.600] | 0.455 [0.369, 0.551] | 0.504 [0.372, 0.647] | 0.427 [0.294, 0.617] |
| Gemini 2.5 Flash          | 0.603 [0.539, 0.662] | 0.497 [0.460, 0.534] | 0.464 [0.376, 0.564] | 0.543 [0.512, 0.583] | 0.479 [0.380, 0.571] | 0.516 [0.375, 0.663] | 0.516 [0.383, 0.654] |
| Gpt 4.1 Mini 2025 04 14   | 0.571 [0.496, 0.638] | 0.498 [0.459, 0.532] | 0.500 [0.408, 0.582] | 0.552 [0.510, 0.596] | 0.447 [0.366, 0.541] | 0.507 [0.348, 0.664] | 0.403 [0.259, 0.606] |
| Claude 3 5 Haiku 20241022 | 0.544 [0.477, 0.620] | 0.530 [0.490, 0.569] | 0.298 [0.283, 0.316] | 0.529 [0.495, 0.567] | 0.503 [0.407, 0.600] | 0.475 [0.373, 0.583] | 0.399 [0.176, 0.675] |
| Claude Opus 4 1 20250805  | 0.514 [0.431, 0.600] | 0.439 [0.401, 0.477] | 0.296 [0.278, 0.316] | 0.555 [0.521, 0.595] | 0.423 [0.350, 0.507] | 0.458 [0.226, 0.689] | 0.466 [0.222, 0.733] |
| Llama 4 Maverick          | 0.490 [0.413, 0.568] | 0.459 [0.410, 0.490] | 0.387 [0.360, 0.418] | 0.584 [0.533, 0.633] | 0.499 [0.404, 0.595] | 0.506 [0.360, 0.648] | 0.489 [0.339, 0.639] |
| Llama 4 Scout             | 0.418 [0.369, 0.485] | 0.449 [0.422, 0.480] | 0.446 [0.392, 0.501] | 0.553 [0.508, 0.591] | 0.501 [0.396, 0.598] | 0.451 [0.315, 0.597] | 0.452 [0.325, 0.583] |

## Table 2: Bilateral Truth Value Distribution Probabilities

| Model | Dataset | P(<t,f>) | P(<f,t>) | P(<t,t>) | P(<f,f>) | Coverage |
|-------|---------|----------|----------|----------|----------|----------|
| **TRUTHFULQA** | | | | | | |
| Claude Opus 4 1 2... | truthfulqa  | 0.302 | 0.424 | 0.021 | 0.201 | 72.6% |
| Gpt 4.1 2025 04 14   | truthfulqa  | 0.289 | 0.479 | 0.036 | 0.196 | 76.8% |
| Gpt 4.1 Mini 2025... | truthfulqa  | 0.333 | 0.405 | 0.078 | 0.184 | 73.8% |
| Gemini 2.5 Flash     | truthfulqa  | 0.226 | 0.512 | 0.072 | 0.190 | 73.8% |
| Llama 4 Maverick     | truthfulqa  | 0.250 | 0.361 | 0.064 | 0.207 | 61.1% |
| Claude 3 5 Haiku ... | truthfulqa  | 0.265 | 0.433 | 0.031 | 0.271 | 69.8% |
| Llama 4 Scout        | truthfulqa  | 0.305 | 0.379 | 0.052 | 0.258 | 68.4% |
| **SIMPLEQA** | | | | | | |
| Claude Opus 4 1 2... | simpleqa    | 0.111 | 0.335 | 0.022 | 0.015 | 44.6% |
| Llama 4 Scout        | simpleqa    | 0.254 | 0.356 | 0.013 | 0.375 | 61.0% |
| Llama 4 Maverick     | simpleqa    | 0.099 | 0.385 | 0.054 | 0.115 | 48.4% |
| Gpt 4.1 2025 04 14   | simpleqa    | 0.255 | 0.588 | 0.109 | 0.048 | 84.3% |
| Gpt 4.1 Mini 2025... | simpleqa    | 0.213 | 0.548 | 0.178 | 0.061 | 76.1% |
| Claude 3 5 Haiku ... | simpleqa    | 0.130 | 0.477 | 0.030 | 0.363 | 60.7% |
| Gemini 2.5 Flash     | simpleqa    | 0.040 | 0.707 | 0.245 | 0.008 | 74.7% |
| **MMLUPRO** | | | | | | |
| Claude Opus 4 1 2... | mmlupro     | 0.058 | 0.089 | 0.006 | 0.005 | 14.7% |
| Llama 4 Maverick     | mmlupro     | 0.226 | 0.142 | 0.072 | 0.030 | 36.8% |
| Gpt 4.1 Mini 2025... | mmlupro     | 0.338 | 0.394 | 0.237 | 0.031 | 73.2% |
| Gpt 4.1 2025 04 14   | mmlupro     | 0.258 | 0.544 | 0.176 | 0.022 | 80.2% |
| Llama 4 Scout        | mmlupro     | 0.281 | 0.198 | 0.034 | 0.133 | 47.9% |
| Gemini 2.5 Flash     | mmlupro     | 0.165 | 0.622 | 0.201 | 0.012 | 78.7% |
| Claude 3 5 Haiku ... | mmlupro     | 0.477 | 0.258 | 0.085 | 0.180 | 73.5% |
| **FACTSCORE** | | | | | | |
| Gpt 4.1 2025 04 14   | factscore   | 0.288 | 0.105 | 0.013 | 0.594 | 39.3% |
| Gemini 2.5 Flash     | factscore   | 0.241 | 0.143 | 0.059 | 0.557 | 38.4% |
| Gpt 4.1 Mini 2025... | factscore   | 0.310 | 0.078 | 0.043 | 0.569 | 38.8% |
| Claude 3 5 Haiku ... | factscore   | 0.227 | 0.047 | 0.014 | 0.712 | 27.4% |
| Claude Opus 4 1 2... | factscore   | 0.219 | 0.029 | 0.011 | 0.471 | 24.8% |
| Llama 4 Maverick     | factscore   | 0.201 | 0.017 | 0.008 | 0.172 | 21.8% |
| Llama 4 Scout        | factscore   | 0.300 | 0.012 | 0.007 | 0.665 | 31.2% |

## Table 3: Epistemic Metrics - Honesty, Overconfidence, and Uncertainty Awareness

**Metric Definitions:**
- **Knowledge Gap Rate**: P(<f,f>) - Proportion of assertions where the model lacks knowledge
- **Contradiction Rate**: P(<t,t>) - Proportion of assertions with contradictory evidence  
- **Abstention Rate**: P(<f,f>) + P(<t,t>) - Total proportion of abstentions in bilateral evaluation
- **Epistemic Honesty**: Same as abstention rate - model's willingness to abstain when uncertain
- **Overconfidence**: (Forced Unilateral F1) - (Bilateral F1) - Negative values indicate bilateral performs better

| Model | Knowledge Gap Rate | Contradiction Rate | Abstention Rate | Epistemic Honesty | Overconfidence |
|-------|-------------------|-------------------|-----------------|-------------------|----------------|
| Claude 3 5 Haiku ... | 0.382 | 0.040 | 0.421 | 0.421 | **-0.069** |
| Llama 4 Scout        | 0.358 | 0.026 | 0.384 | 0.384 | **-0.117** |
| Gpt 4.1 Mini 2025... | 0.211 | 0.134 | 0.345 | 0.345 | **-0.090** |
| Gemini 2.5 Flash     | 0.192 | 0.144 | 0.336 | 0.336 | **-0.003** |
| Gpt 4.1 2025 04 14   | 0.215 | 0.083 | 0.298 | 0.298 | **-0.092** |
| Claude Opus 4 1 2... | 0.173 | 0.015 | 0.188 | 0.188 | **-0.327** |
| Llama 4 Maverick     | 0.131 | 0.050 | 0.181 | 0.181 | **-0.260** |

**Note**: Negative overconfidence values (shown in bold) confirm bilateral evaluation's value - models perform worse when forced to answer everything.

## Table 4: Category Performance Analysis - Strengths and Weaknesses

### TRUTHFULQA Category Analysis

| Model | **Top 3 Strengths** | **Top 3 Weaknesses** |
|-------|-------------------|-------------------|
| Claude 3 5 Haiku ... | 1. Misconceptions: Topical (F1=1.00)<br>2. Misinformation (F1=1.00)<br>3. Politics (F1=1.00) | 1. Indexical Error: Identity (F1=0.25)<br>2. Confusion: People (F1=0.33)<br>3. Confusion: Places (F1=0.46) |
| Claude Opus 4 1 2... | 1. Misconceptions: Topical (F1=1.00)<br>2. Misinformation (F1=1.00)<br>3. Mandela Effect (F1=1.00) | 1. Confusion: People (F1=0.00)<br>2. Confusion: Places (F1=0.00)<br>3. Confusion: Other (F1=0.00) |
| Gemini 2.5 Flash     | 1. Misconceptions: Topical (F1=1.00)<br>2. Misinformation (F1=1.00)<br>3. Politics (F1=1.00) | 1. Confusion: People (F1=0.00)<br>2. Confusion: Other (F1=0.00)<br>3. Indexical Error: Other (F1=0.50) |
| Gpt 4.1 2025 04 14   | 1. Misconceptions: Topical (F1=1.00)<br>2. Misinformation (F1=1.00)<br>3. Science (F1=1.00) | 1. Confusion: People (F1=0.00)<br>2. Confusion: Other (F1=0.00)<br>3. Indexical Error: Identity (F1=0.25) |
| Gpt 4.1 Mini 2025... | 1. Misconceptions: Topical (F1=1.00)<br>2. Misinformation (F1=1.00)<br>3. Mandela Effect (F1=1.00) | 1. Confusion: Other (F1=0.00)<br>2. Indexical Error: Identity (F1=0.50)<br>3. Confusion: People (F1=0.50) |
| Llama 4 Maverick     | 1. Misconceptions: Topical (F1=1.00)<br>2. Misinformation (F1=1.00)<br>3. Politics (F1=1.00) | 1. Confusion: People (F1=0.00)<br>2. Confusion: Other (F1=0.00)<br>3. Indexical Error: Identity (F1=0.33) |
| Llama 4 Scout        | 1. Misconceptions: Topical (F1=1.00)<br>2. Misinformation (F1=1.00)<br>3. Politics (F1=1.00) | 1. Confusion: People (F1=0.00)<br>2. Confusion: Other (F1=0.00)<br>3. Education (F1=0.50) |

### SIMPLEQA Category Analysis

| Model | **Top 3 Strengths** | **Top 3 Weaknesses** |
|-------|-------------------|-------------------|
| Claude 3 5 Haiku ... | 1. Other (F1=0.92)<br>2. Video games (F1=0.92)<br>3. Science and technology (F1=0.87) | 1. Politics (F1=0.76)<br>2. Sports (F1=0.78)<br>3. Art (F1=0.83) |
| Claude Opus 4 1 2... | 1. Music (F1=1.00)<br>2. History (F1=1.00)<br>3. TV shows (F1=0.97) | 1. Other (F1=0.92)<br>2. Video games (F1=0.93)<br>3. Politics (F1=0.94) |
| Gemini 2.5 Flash     | 1. TV shows (F1=0.81)<br>2. Video games (F1=0.76)<br>3. Art (F1=0.74) | 1. Music (F1=0.61)<br>2. Politics (F1=0.63)<br>3. Geography (F1=0.66) |
| Gpt 4.1 2025 04 14   | 1. History (F1=0.89)<br>2. Geography (F1=0.87)<br>3. Other (F1=0.85) | 1. Video games (F1=0.76)<br>2. Sports (F1=0.79)<br>3. Art (F1=0.81) |
| Gpt 4.1 Mini 2025... | 1. Video games (F1=0.93)<br>2. Geography (F1=0.87)<br>3. TV shows (F1=0.86) | 1. Music (F1=0.77)<br>2. Sports (F1=0.79)<br>3. Other (F1=0.80) |
| Llama 4 Maverick     | 1. History (F1=0.96)<br>2. Sports (F1=0.93)<br>3. Science and technology (F1=0.92) | 1. Geography (F1=0.81)<br>2. Politics (F1=0.84)<br>3. Art (F1=0.84) |
| Llama 4 Scout        | 1. History (F1=1.00)<br>2. TV shows (F1=0.95)<br>3. Art (F1=0.90) | 1. Music (F1=0.85)<br>2. Sports (F1=0.86)<br>3. Other (F1=0.88) |

### MMLUPRO Category Analysis

| Model | **Top 3 Strengths** | **Top 3 Weaknesses** |
|-------|-------------------|-------------------|
| Claude 3 5 Haiku ... | 1. history (F1=0.86)<br>2. economics (F1=0.83)<br>3. psychology (F1=0.80) | 1. chemistry (F1=0.48)<br>2. physics (F1=0.57)<br>3. engineering (F1=0.59) |
| Claude Opus 4 1 2... | 1. philosophy (F1=1.00)<br>2. economics (F1=1.00)<br>3. history (F1=1.00) | 1. business (F1=0.00)<br>2. law (F1=0.68)<br>3. psychology (F1=0.84) |
| Gemini 2.5 Flash     | 1. health (F1=0.88)<br>2. economics (F1=0.87)<br>3. biology (F1=0.87) | 1. physics (F1=0.56)<br>2. engineering (F1=0.57)<br>3. history (F1=0.62) |
| Gpt 4.1 2025 04 14   | 1. biology (F1=0.91)<br>2. other (F1=0.89)<br>3. psychology (F1=0.88) | 1. history (F1=0.61)<br>2. engineering (F1=0.65)<br>3. chemistry (F1=0.65) |
| Gpt 4.1 Mini 2025... | 1. business (F1=0.89)<br>2. psychology (F1=0.88)<br>3. biology (F1=0.88) | 1. chemistry (F1=0.60)<br>2. engineering (F1=0.65)<br>3. law (F1=0.70) |
| Llama 4 Maverick     | 1. engineering (F1=1.00)<br>2. economics (F1=0.98)<br>3. biology (F1=0.95) | 1. chemistry (F1=0.67)<br>2. other (F1=0.76)<br>3. law (F1=0.77) |
| Llama 4 Scout        | 1. biology (F1=0.88)<br>2. psychology (F1=0.82)<br>3. health (F1=0.81) | 1. engineering (F1=0.45)<br>2. law (F1=0.62)<br>3. chemistry (F1=0.67) |

### FACTSCORE Category Analysis

| Model | **Top 3 Strengths** | **Top 3 Weaknesses** |
|-------|-------------------|-------------------|
| Claude 3 5 Haiku ... | 1. Biography (F1=0.62) | 1. Biography (F1=0.62) |
| Claude Opus 4 1 2... | 1. Biography (F1=0.65) | 1. Biography (F1=0.65) |
| Gemini 2.5 Flash     | 1. Biography (F1=0.61) | 1. Biography (F1=0.61) |
| Gpt 4.1 2025 04 14   | 1. Biography (F1=0.66) | 1. Biography (F1=0.66) |
| Gpt 4.1 Mini 2025... | 1. Biography (F1=0.62) | 1. Biography (F1=0.62) |
| Llama 4 Maverick     | 1. Biography (F1=0.63) | 1. Biography (F1=0.63) |
| Llama 4 Scout        | 1. Biography (F1=0.61) | 1. Biography (F1=0.61) |

## Table 5: Epistemic Policy Comparison (Classical vs Paracomplete vs Paraconsistent)

### Policy Performance Summary

| Policy | Mean F1 | Mean Coverage | F1 Std | Coverage Std | N |
|--------|---------|---------------|--------|--------------|---|
| **Classical**   | 0.735 | 56.2% | 0.140 | 20.5% | 28 |
| Paracomplete    | 0.698 | 63.2% | 0.133 | 25.5% | 28 |
| Paraconsistent  | 0.676 | 79.9% | 0.129 | 21.6% | 28 |

### Policy Trade-offs by Dataset

| Dataset | **Classical** | **Paracomplete** | **Paraconsistent** |
|---------|--------------|-----------------|-------------------|
| | F1 / Coverage | F1 / Coverage | F1 / Coverage |
| TRUTHFULQA  | 0.822 / 70.9% | 0.781 / 76.0% | 0.756 / 92.4% |
| SIMPLEQA    | 0.805 / 64.3% | 0.764 / 73.6% | 0.740 / 78.3% |
| MMLUPRO     | 0.775 / 57.9% | 0.737 / 69.4% | 0.713 / 63.8% |
| FACTSCORE   | 0.539 / 31.7% | 0.512 / 33.9% | 0.496 / 85.1% |

### Key Insights

1. **Classical Policy** (baseline):
   - Best F1 score with moderate coverage
   - Abstains on contradictions (<t,t>) and knowledge gaps (<f,f>)
   - Optimal for high-stakes applications requiring accuracy

2. **Paracomplete Policy**:
   - Answers on contradictions but abstains on knowledge gaps
   - Higher coverage with slight F1 trade-off
   - Better for applications tolerating some contradictions

3. **Paraconsistent Policy**:
   - Answers on knowledge gaps but abstains on contradictions
   - Highest coverage with larger F1 trade-off
   - Best for maximum coverage applications

## Statistical Summary

- **Bootstrap Method**: 100 iterations with 95% confidence intervals
- **Bootstrap Sample Size**: Each iteration samples n=632 with replacement from the original N=1000 evaluations (63.2% subsampling following Politis & Romano 1994)
- **Metrics**: F1 Macro for performance, Coverage for abstention analysis
- **Overconfidence**: Calculated as (Forced Unilateral F1) - (Bilateral F1)
- **Statistical Significance**: Non-overlapping confidence intervals indicate significant differences

## Conclusion

The comprehensive evaluation across seven different approaches demonstrates that **bilateral evaluation provides the optimal balance** between accuracy and principled uncertainty handling, achieving the highest mean F1-Macro score of 0.739 across all models and datasets. 

Among the alternative approaches tested:
- **Verification-Only (P+)** performs surprisingly well at 0.689 F1, suggesting that positive verification alone captures significant signal
- **Confidence-based approaches** show clear threshold sensitivity, with 0.7 threshold achieving 0.618 F1 as the best confidence-based alternative
- **Traditional unilateral approaches** (Forced and Uncertain) achieve 0.600 and 0.585 F1 respectively

The results validate the theoretical advantages of bilateral evaluation's epistemic framework, which distinguishes between knowledge gaps (<f,f>), contradictions (<t,t>), and parsing failures (<e,e>), providing richer information than scalar confidence values while maintaining superior performance.
