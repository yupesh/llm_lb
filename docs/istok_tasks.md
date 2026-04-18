# ISTOK Language Assessment Tasks

This document describes the 10 tasks contributed by the ISTOK research project
for benchmarking LLMs on automated writing assessment.

## Background

The ISTOK project uses **rubric-prompted LLM agents** to evaluate student
writing — zero-shot, with no training data. Each agent receives a detailed
scoring rubric (derived from an international language testing standard) and a
student essay, then returns a numeric score. This approach is benchmarked
against 25,000 human-rated EFCAMDAT texts and 17,307 ASAP essays.

These leaderboard tasks test a core capability: **can an LLM reliably score
student writing when given an explicit rubric?**

## Task Overview

| Task ID | Criterion | Scale | Labels | Samples | Gold Source |
|---------|-----------|-------|--------|---------|-------------|
| `essay_holistic_scoring` | Overall essay quality | 1–6 | 6 | 18 | ASAP human scores |
| `cefr_level_classification` | CEFR proficiency level | A2–C1 | 4 | 20 | EFCAMDAT certified levels |
| `essay_grammar_scoring` | Grammatical accuracy | 1–4 | 4 | 20 | EFCAMDAT CEFR proxy |
| `essay_coherence_scoring` | Coherence & cohesion | 1–4 | 4 | 20 | EFCAMDAT CEFR proxy |
| `essay_vocabulary_scoring` | Vocabulary range | 1–4 | 4 | 20 | EFCAMDAT CEFR proxy |

Each task has two variants: **full rubric** and **condensed rubric** (10 tasks
total). The condensed variants have the suffix `_condensed`.

## Full vs Condensed Rubrics

### Full rubric (5 tasks)

The `system_prompt` contains the **complete scoring rubric** as used in the
ISTOK production pipeline:

- Detailed band descriptors for every score level (derived from official
  testing standards such as the CEFR Companion Volume 2020, ASAP rubric, etc.)
- **Worked examples**: one scored example per band showing what a typical
  response looks like and how to justify the score
- Explicit output format instructions

**Pros:**
- Faithful to our actual research methodology
- Worked examples ground the model's understanding of each band
- Higher inter-rater agreement in our experiments (rubric + examples outperforms
  rubric alone by ~0.08 QWK on ASAP)

**Cons:**
- Long system prompts (1,000–2,000 tokens) — may disadvantage models with
  smaller context windows or those that degrade with long instructions
- Not a fair test of "intrinsic" scoring ability — the model is essentially
  doing few-shot classification disguised as zero-shot

### Condensed rubric (5 tasks, `_condensed` suffix)

The `system_prompt` contains a **stripped-down rubric**:

- One-line descriptor per band (no paragraph-level detail)
- **No worked examples**
- Same scoring scale and labels

**Pros:**
- Tests the model's ability to interpret scoring criteria without extensive
  scaffolding
- Short system prompts (~200 tokens) — fair across model sizes
- Better isolates language understanding from instruction-following

**Cons:**
- Lower expected accuracy (no exemplars to anchor the scale)
- More ambiguous — models may interpret band boundaries differently

### Why both?

The comparison between full and condensed variants reveals what drives scoring
performance:

1. **If full >> condensed**: the model relies on worked examples as implicit
   few-shot demonstrations. Scoring ability is primarily instruction-following.
2. **If full ≈ condensed**: the model has genuine rubric comprehension and can
   apply criteria from brief descriptions alone.
3. **Per-model gap analysis**: some models may benefit more from examples than
   others — this is a useful signal for practical deployment.

## Gold Label Sources

### Tasks with strong gold labels

| Task | Source | Quality |
|------|--------|---------|
| `essay_holistic_scoring` | ASAP 2.0 human scores (Kaggle) | High — double-scored by trained raters |
| `cefr_level_classification` | EFCAMDAT certified CEFR levels | High — placement test + teacher assignment |

### Tasks with proxy gold labels

| Task | Source | Quality |
|------|--------|---------|
| `essay_grammar_scoring` | EFCAMDAT CEFR level → numeric (A2=1, B1=2, B2=3, C1=4) | Medium — CEFR level is holistic, not grammar-specific |
| `essay_coherence_scoring` | Same mapping | Medium |
| `essay_vocabulary_scoring` | Same mapping | Medium |

For the criterion-level tasks (grammar, coherence, vocabulary), the expected
score is derived from the text's overall CEFR level. This is a reasonable
proxy: A2 writers typically have A2-level grammar, vocabulary, and coherence.
However, it is not perfect — a B2 writer may have C1 vocabulary but B1 grammar.

**Interpretation guidance:** For these three tasks, treat accuracy as a measure
of whether the model's per-criterion assessment aligns with the writer's
overall proficiency. Perfect accuracy is not expected; the benchmark value lies
in comparing models against each other on the same proxy labels.

## Sample Selection

### ASAP samples (18 essays)

- **Source**: ASAP 2.0 training set (17,307 essays, Kaggle)
- **Stratification**: 3 essays per score level (1–6)
- **Length filter**: 200–600 words preferred (readable, representative)
- **Random seed**: 42

### EFCAMDAT samples (20 essays)

- **Source**: EFCAMDAT balanced 25k subset (Cambridge Learner Corpus)
- **Stratification**: 5 essays per CEFR level (A2, B1, B2, C1)
- **Length filter**: 80–500 words preferred
- **Note**: C2 is not available in this dataset; labels cover A2–C1
- **Random seed**: 42

## Metric Choice

All tasks use **accuracy** as the primary metric with **macro_f1** as secondary.

- **Accuracy** is strict (off-by-one counts as wrong) but standard for
  classification benchmarks
- **macro_f1** accounts for class imbalance (score 6 essays are rare in ASAP)
- Quadratic Weighted Kappa (QWK) — the standard metric in AES research — is
  not available in the llm_lb metric set, so accuracy serves as a simpler proxy

## Reproducing Our Results

These tasks use the same rubrics deployed in the ISTOK research pipeline:

```
config/rubrics/asap/holistic_independent.txt    → essay_holistic_scoring
config/rubrics/cefr/overall_written_production.txt → cefr_level_classification
config/rubrics/cefr/grammatical_accuracy.txt    → essay_grammar_scoring
config/rubrics/cefr/coherence.txt               → essay_coherence_scoring
config/rubrics/cefr/vocabulary_range.txt        → essay_vocabulary_scoring
```

Our production results with `qwen3:14b` on the full ASAP corpus:
- Holistic scoring (13-agent ensemble): QWK = 0.709
- This exceeds the open-source LLM zero-shot baseline of QWK = 0.531

## Contributing Author

Tendai Chikake, ISTOK Research Project
