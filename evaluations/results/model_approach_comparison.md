# Comprehensive Model/Approach Comparison

**Metrics: F1-Macro | Coverage | Execution Time (seconds)**

*Note: Execution times are per 1000 samples*

## TruthfulQA

| Model | **Bilateral** | **Unilateral-Binary** | **Unilateral-Ternary** | **Confidence-0.5** | **Confidence-0.7** | **Confidence-0.9** |
|---------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| GPT-4.1 | F1: 0.862<br>Cov: 76.8%<br>Time: 1123.9s | F1: 0.836<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.885<br>Cov: 80.7%<br>Time: 0.0s | F1: 0.808<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.860<br>Cov: 89.2%<br>Time: 0.0s | F1: 0.890<br>Cov: 58.5%<br>Time: 0.0s |
| GPT-4.1-Mini | F1: 0.928<br>Cov: 70.0%<br>Time: 28.7s | F1: 0.720<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.853<br>Cov: 55.3%<br>Time: 0.0s | F1: 0.770<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.814<br>Cov: 84.9%<br>Time: 0.0s | F1: 0.864<br>Cov: 46.3%<br>Time: 0.0s |
| Claude-Opus | F1: 0.897<br>Cov: 72.6%<br>Time: 4093.9s | F1: 0.561<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.721<br>Cov: 21.9%<br>Time: 0.0s | F1: 0.568<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.944<br>Cov: 26.7%<br>Time: 0.0s | F1: 0.950<br>Cov: 20.8%<br>Time: 0.0s |
| Claude-Haiku | F1: 0.762<br>Cov: 69.8%<br>Time: 1407.8s | F1: 0.679<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.810<br>Cov: 24.9%<br>Time: 0.0s | F1: 0.738<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.751<br>Cov: 85.9%<br>Time: 0.0s | F1: 0.691<br>Cov: 26.0%<br>Time: 0.0s |
| Llama-Scout | F1: 0.758<br>Cov: 68.4%<br>Time: 1110.1s | F1: 0.695<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.685<br>Cov: 54.6%<br>Time: 0.0s | F1: 0.679<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.691<br>Cov: 92.5%<br>Time: 0.0s | F1: 0.719<br>Cov: 71.0%<br>Time: 0.0s |
| Llama-Maverick | F1: 0.808<br>Cov: 61.1%<br>Time: 886.0s | F1: 0.626<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.676<br>Cov: 33.2%<br>Time: 0.0s | F1: 0.738<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.752<br>Cov: 92.0%<br>Time: 0.0s | F1: 0.751<br>Cov: 77.2%<br>Time: 0.0s |
| Gemini-Flash | F1: 0.809<br>Cov: 73.8%<br>Time: 1114.1s | F1: 0.745<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.833<br>Cov: 52.3%<br>Time: 0.0s | F1: 0.777<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.789<br>Cov: 95.0%<br>Time: 0.0s | F1: 0.805<br>Cov: 81.2%<br>Time: 0.0s |

## SimpleQA

| Model | **Bilateral** | **Unilateral-Binary** | **Unilateral-Ternary** | **Confidence-0.5** | **Confidence-0.7** | **Confidence-0.9** |
|---------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| GPT-4.1 | F1: 0.822<br>Cov: 84.3%<br>Time: 1115.1s | F1: 0.823<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.831<br>Cov: 93.0%<br>Time: 0.0s | F1: 0.788<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.789<br>Cov: 99.4%<br>Time: 0.0s | F1: 0.821<br>Cov: 83.1%<br>Time: 0.0s |
| GPT-4.1-Mini | F1: 0.802<br>Cov: 76.1%<br>Time: 890.1s | F1: 0.771<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.816<br>Cov: 69.5%<br>Time: 0.0s | F1: 0.765<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.781<br>Cov: 97.5%<br>Time: 0.0s | F1: 0.869<br>Cov: 63.1%<br>Time: 0.0s |
| Claude-Opus | F1: 0.943<br>Cov: 44.6%<br>Time: 3939.7s | F1: 0.462<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.198<br>Cov: 32.3%<br>Time: 0.0s | F1: 0.286<br>Cov: 100.0%<br>Time: 0.0s | F1: 1.000<br>Cov: 9.5%<br>Time: 0.0s | F1: 1.000<br>Cov: 9.4%<br>Time: 0.0s |
| Claude-Haiku | F1: 0.795<br>Cov: 60.7%<br>Time: 1409.1s | F1: 0.687<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.780<br>Cov: 22.6%<br>Time: 0.0s | F1: 0.692<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.680<br>Cov: 93.9%<br>Time: 0.0s | F1: 0.705<br>Cov: 4.9%<br>Time: 0.0s |
| Llama-Scout | F1: 0.898<br>Cov: 61.0%<br>Time: 1188.0s | F1: 0.712<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.844<br>Cov: 48.4%<br>Time: 0.0s | F1: 0.748<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.744<br>Cov: 95.7%<br>Time: 0.0s | F1: 0.743<br>Cov: 95.6%<br>Time: 0.0s |
| Llama-Maverick | F1: 0.843<br>Cov: 48.4%<br>Time: 1001.7s | F1: 0.423<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.249<br>Cov: 81.9%<br>Time: 0.0s | F1: 0.664<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.755<br>Cov: 80.2%<br>Time: 0.0s | F1: 0.731<br>Cov: 77.8%<br>Time: 0.0s |
| Gemini-Flash | F1: 0.534<br>Cov: 74.7%<br>Time: 982.2s | F1: 0.797<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.834<br>Cov: 80.5%<br>Time: 0.0s | F1: 0.793<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.792<br>Cov: 99.9%<br>Time: 0.0s | F1: 0.792<br>Cov: 99.9%<br>Time: 0.0s |

## MMLU-Pro

| Model | **Bilateral** | **Unilateral-Binary** | **Unilateral-Ternary** | **Confidence-0.5** | **Confidence-0.7** | **Confidence-0.9** |
|---------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| GPT-4.1 | F1: 0.769<br>Cov: 80.2%<br>Time: 1151.9s | F1: 0.563<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.534<br>Cov: 96.9%<br>Time: 0.0s | F1: 0.442<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.445<br>Cov: 99.0%<br>Time: 0.0s | F1: 0.418<br>Cov: 85.0%<br>Time: 0.0s |
| GPT-4.1-Mini | F1: 0.779<br>Cov: 73.2%<br>Time: 929.5s | F1: 0.675<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.819<br>Cov: 25.4%<br>Time: 0.0s | F1: 0.397<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.488<br>Cov: 72.2%<br>Time: 0.0s | F1: 0.490<br>Cov: 53.4%<br>Time: 0.0s |
| Claude-Opus | F1: 0.896<br>Cov: 14.7%<br>Time: 1667.9s | F1: 0.499<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.455<br>Cov: 54.6%<br>Time: 0.0s | F1: 0.093<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.000<br>Cov: 0.0%<br>Time: 0.0s | F1: 0.000<br>Cov: 0.0%<br>Time: 0.0s |
| Claude-Haiku | F1: 0.687<br>Cov: 73.5%<br>Time: 1358.2s | F1: 0.607<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.581<br>Cov: 17.0%<br>Time: 0.0s | F1: 0.180<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.431<br>Cov: 20.6%<br>Time: 0.0s | F1: 0.270<br>Cov: 8.9%<br>Time: 0.0s |
| Llama-Scout | F1: 0.747<br>Cov: 47.9%<br>Time: 1102.8s | F1: 0.496<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.314<br>Cov: 35.8%<br>Time: 0.0s | F1: 0.267<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.651<br>Cov: 24.8%<br>Time: 0.0s | F1: 0.662<br>Cov: 24.5%<br>Time: 0.0s |
| Llama-Maverick | F1: 0.851<br>Cov: 36.8%<br>Time: 1126.0s | F1: 0.433<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.122<br>Cov: 59.8%<br>Time: 0.0s | F1: 0.142<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.867<br>Cov: 5.5%<br>Time: 0.0s | F1: 0.917<br>Cov: 5.2%<br>Time: 0.0s |
| Gemini-Flash | F1: 0.709<br>Cov: 78.7%<br>Time: 1076.7s | F1: 0.610<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.611<br>Cov: 61.7%<br>Time: 0.0s | F1: 0.572<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.575<br>Cov: 99.3%<br>Time: 0.0s | F1: 0.579<br>Cov: 98.0%<br>Time: 0.0s |

## FACTScore

| Model | **Bilateral** | **Unilateral-Binary** | **Unilateral-Ternary** | **Confidence-0.5** | **Confidence-0.7** | **Confidence-0.9** |
|---------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| GPT-4.1 | F1: 0.629<br>Cov: 39.3%<br>Time: 1179.6s | F1: 0.503<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.527<br>Cov: 27.0%<br>Time: 0.0s | F1: 0.456<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.505<br>Cov: 49.9%<br>Time: 0.0s | F1: 0.431<br>Cov: 26.5%<br>Time: 0.0s |
| GPT-4.1-Mini | F1: 0.571<br>Cov: 38.8%<br>Time: 995.0s | F1: 0.498<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.500<br>Cov: 22.8%<br>Time: 0.0s | F1: 0.447<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.513<br>Cov: 38.6%<br>Time: 0.0s | F1: 0.411<br>Cov: 18.7%<br>Time: 0.0s |
| Claude-Opus | F1: 0.516<br>Cov: 24.8%<br>Time: 4608.7s | F1: 0.440<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.296<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.428<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.483<br>Cov: 17.2%<br>Time: 0.0s | F1: 0.474<br>Cov: 16.8%<br>Time: 0.0s |
| Claude-Haiku | F1: 0.547<br>Cov: 27.4%<br>Time: 1552.8s | F1: 0.532<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.296<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.504<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.476<br>Cov: 79.3%<br>Time: 0.0s | F1: 0.416<br>Cov: 13.3%<br>Time: 0.0s |
| Llama-Scout | F1: 0.422<br>Cov: 31.2%<br>Time: 1158.6s | F1: 0.447<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.445<br>Cov: 31.1%<br>Time: 0.0s | F1: 0.502<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.457<br>Cov: 52.0%<br>Time: 0.0s | F1: 0.453<br>Cov: 51.6%<br>Time: 0.0s |
| Llama-Maverick | F1: 0.495<br>Cov: 21.8%<br>Time: 915.0s | F1: 0.459<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.388<br>Cov: 66.1%<br>Time: 0.0s | F1: 0.502<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.510<br>Cov: 47.6%<br>Time: 0.0s | F1: 0.495<br>Cov: 42.1%<br>Time: 0.0s |
| Gemini-Flash | F1: 0.601<br>Cov: 38.4%<br>Time: 1073.3s | F1: 0.501<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.470<br>Cov: 19.4%<br>Time: 0.0s | F1: 0.481<br>Cov: 100.0%<br>Time: 0.0s | F1: 0.521<br>Cov: 50.1%<br>Time: 0.0s | F1: 0.516<br>Cov: 48.6%<br>Time: 0.0s |

## Summary Statistics

### Average F1-Macro by Approach
- Bilateral: 0.739
- Confidence-0.7: 0.645
- Confidence-0.9: 0.638
- Unilateral-Binary: 0.600
- Unilateral-Ternary: 0.585
- Confidence-0.5: 0.544

### Average Coverage by Approach
- Unilateral-Binary: 100.0%
- Confidence-0.5: 100.0%
- Confidence-0.7: 64.2%
- Bilateral: 56.0%
- Unilateral-Ternary: 52.5%
- Confidence-0.9: 46.7%

### Average Execution Time by Approach (seconds per 1000 samples)
- Unilateral-Binary: 0.0s
- Unilateral-Ternary: 0.0s
- Confidence-0.5: 0.0s
- Confidence-0.7: 0.0s
- Confidence-0.9: 0.0s
- Bilateral: 1435.2s

### Model Rankings (by average F1 across all approaches)
- GPT-4.1-Mini: 0.681
- GPT-4.1: 0.676
- Gemini-Flash: 0.669
- Llama-Scout: 0.616
- Claude-Haiku: 0.596
- Llama-Maverick: 0.592
- Claude-Opus: 0.546
