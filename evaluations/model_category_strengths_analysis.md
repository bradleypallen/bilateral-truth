# Model-Specific Category Strengths Analysis

Based on the F1 family resemblance analysis, here are the category-specific strengths for each model family, showing where each excels relative to others:

## OpenAI Models (GPT-4.1, GPT-4.1-mini)

### Top Strengths (>20% advantage)
1. **Confusion: Places** (+225%): Exceptional at disambiguating similar place names
2. **Business** (+137%): Superior business knowledge and reasoning
3. **Indexical Error: Other** (+100%): Perfect handling of context-dependent references
4. **Confusion: People** (+100%): Excellent at distinguishing between similar people

### Strong Performance (10-20% advantage)
- **Economics** (+37.5%): Strong grasp of economic concepts
- **Law** (+22.7%): Good legal knowledge
- **Weather** (+19.5%): Accurate weather-related facts
- **Language** (+16.7%): Excellent linguistic knowledge
- **Sociology** (+15.8%): Good understanding of social concepts
- **Health** (+14.8%): Reliable medical information
- **Misconceptions** (+14.5%): Good at avoiding common misconceptions
- **Fiction** (+11.3%): Strong knowledge of fictional works
- **History** (+11.0%): Solid historical facts
- **Science** (+10.2%): Good scientific knowledge

### Weaknesses
- **Indexical Error: Identity** (-167%): Struggles with identity references
- **Math** (-19.8%): Weaker mathematical reasoning
- **Chemistry** (-18.2%): Less strong in chemistry
- **Statistics** (-17.1%): Statistical reasoning challenges
- **Advertising** (-15.5%): Less familiar with advertising concepts

## Anthropic Models (Claude Opus 4.1, Claude 3.5 Haiku)

### Top Strengths (>20% advantage)
1. **History (MMLU-Pro)** (+50.1%): Exceptional historical knowledge
2. **Music** (+51.2%): Outstanding music knowledge
3. **Physics** (+40.5%): Excellent physics understanding
4. **Engineering** (+38.3%): Strong engineering concepts
5. **Geography** (+34.9%): Superior geographical knowledge
6. **Other (SimpleQA)** (+30.1%): Broad general knowledge
7. **Science and technology** (+24.9%): Strong technical knowledge
8. **Misquotations** (+23.7%): Good at identifying misquotes
9. **Proverbs** (+24.2%): Strong knowledge of proverbs
10. **Video games** (+22.2%): Good gaming knowledge
11. **Distraction** (+20.7%): Resists misleading questions
12. **Other (MMLU-Pro)** (+20.4%): Broad academic knowledge
13. **Art** (+21.2%): Excellent art knowledge

### Strong Performance (10-20% advantage)
- **Superstitions** (+19.6%): Recognizes superstitious beliefs
- **Chemistry** (+18.2%): Strong chemistry knowledge
- **Indexical Error: Location** (+16.7%): Good with location references
- **Statistics** (+17.1%): Strong statistical reasoning
- **Advertising** (+15.5%): Good advertising knowledge
- **Conspiracies** (+14.1%): Identifies conspiracy theories

### Major Weaknesses
- **Confusion: Places** (-225%): Very poor at place disambiguation
- **Business** (-137%): Surprisingly weak in business knowledge
- **History (TruthfulQA)** (-11.0%): Weaker on history misconceptions
- **Psychology (MMLU-Pro)** (-7.7%): Below average psychology

## Google Models (Gemini 2.5 Flash only - Gemini 2.5 Pro did not respond)

*Note: Data represents only Gemini 2.5 Flash as Gemini 2.5 Pro failed to complete evaluations*

### Top Strengths (>20% advantage)
1. **Indexical Error: Identity** (+167%): Best at identity references
2. **Psychology (TruthfulQA)** (+63.8%): Exceptional psychological knowledge
3. **Stereotypes** (+31.1%): Good at handling stereotype-related questions
4. **Logical Falsehood** (+28.2%): Excellent logical reasoning

### Strong Performance (10-20% advantage)
- **Nutrition** (+15.7%): Good nutritional knowledge
- **Advertising** (+15.5%): Solid advertising concepts
- **Education** (+11.1%): Educational knowledge
- **Finance** (+9.1%): Financial concepts
- **Health (MMLU-Pro)** (+6.1%): Medical information

### Major Weaknesses (>30% disadvantage)
- **Music** (-51.2%): Significant weakness in music
- **History (MMLU-Pro)** (-50.1%): Poor historical knowledge
- **History (SimpleQA)** (-42.3%): Historical facts weakness
- **Physics** (-40.5%): Weak physics understanding
- **Engineering** (-38.3%): Engineering challenges
- **Politics (SimpleQA)** (-37.4%): Political facts weakness
- **Geography** (-34.9%): Geographic knowledge gaps
- **Sports** (-32.8%): Sports knowledge deficit
- **Other (SimpleQA)** (-30.1%): General knowledge gaps

## Meta/Llama Models (Llama 4 Scout, Llama 4 Maverick)

### Top Strengths (>20% advantage)
1. **History (SimpleQA)** (+42.3%): Excellent at historical facts
2. **Politics (SimpleQA)** (+37.4%): Strong political knowledge
3. **Sports** (+32.8%): Best at sports-related facts
4. **Math** (+19.8%): Strong mathematical reasoning

### Strong Performance (10-20% advantage)
- **TV shows** (+15.7%): Good entertainment knowledge
- **Biology** (+7.4%): Solid biological knowledge

### Major Weaknesses (>30% disadvantage)
- **Psychology** (-63.8%): Significant psychology weakness
- **Economics** (-37.5%): Poor economic reasoning
- **Stereotypes** (-31.1%): Challenges with stereotype questions
- **Logical Falsehood** (-28.2%): Weak logical reasoning
- **Proverbs** (-24.2%): Poor proverb knowledge
- **Law** (-22.7%): Legal knowledge gaps

### Complete Failures (0% F1 score)
- **Confusion: Other** (-100%): Complete failure on confusion tasks
- **Confusion: People** (-100%): Cannot distinguish similar people
- **Politics (TruthfulQA)**: Perfect failure on political misconceptions
- **Subjective**: Complete failure on subjective questions
- **Misinformation**: Unable to identify misinformation
- **Misconceptions: Topical**: Fails on topical misconceptions

## Key Insights

### Domain Specialization Patterns

1. **Technical/Scientific Excellence**:
   - Anthropic: Physics (+40.5%), Chemistry (+18.2%), Engineering (+38.3%)
   - Meta/Llama: Mathematics (+19.8%), Biology (+7.4%)
   - OpenAI: Balanced but not exceptional

2. **Business/Economics Excellence**:
   - OpenAI: Business (+137%), Economics (+37.5%), Law (+22.7%)
   - Anthropic: Weak in business (-137%)
   - Meta/Llama: Weak in economics (-37.5%)

3. **Humanities/Culture Excellence**:
   - Anthropic: Music (+51.2%), Art (+21.2%), Geography (+34.9%)
   - Meta/Llama: Sports (+32.8%), TV shows (+15.7%)
   - Google: Psychology (+63.8%)

4. **Reasoning/Logic Excellence**:
   - Google: Logical Falsehood (+28.2%), Identity reasoning (+167%)
   - OpenAI: Disambiguation of places (+225%) and people (+100%)
   - Anthropic: Resisting distraction (+20.7%)

### Critical Gaps by Model Family

**OpenAI**: Mathematical reasoning, identity references
**Anthropic**: Business knowledge, place disambiguation
**Google**: Most factual domains (history, music, sports, physics)
**Meta/Llama**: Psychology, confusion tasks, identifying misinformation

### Complementary Pairings

Based on these strengths, optimal model pairings would be:

1. **OpenAI + Anthropic**: Business/economics expertise paired with physics/engineering
2. **OpenAI + Meta/Llama**: Disambiguation skills complement sports/politics knowledge
3. **Anthropic + Google**: Technical knowledge paired with psychology/logic
4. **Avoid Google + Meta/Llama**: Too many overlapping weaknesses

### Task-Specific Recommendations

- **Business Analysis**: OpenAI models only
- **Scientific Research**: Anthropic for physics/chemistry, Meta/Llama for math
- **Historical Research**: Anthropic for MMLU-Pro, Meta/Llama for SimpleQA
- **Psychological Studies**: Google (Gemini 2.5 Flash)
- **Sports/Entertainment**: Meta/Llama
- **Music/Art**: Anthropic
- **Disambiguation Tasks**: OpenAI
- **Medical Information**: OpenAI for general health, Google for psychology