# Metrics reference

This document explains every metric the leaderboard supports, how to choose
one for your task, and how the LLM-as-Judge system works.

## Available metrics

| Metric | Type | Range | When to use |
|---|---|---|---|
| `accuracy` | classification | 0..1 | Labels are known, answer is one of a fixed set |
| `macro_f1` | classification | 0..1 | Like accuracy, but fair to imbalanced classes |
| `exact_match` | extractive | 0..1 | Short free-form answer that must match a reference (after normalisation) |
| `judge_score` | generative | 0..1 | Open-ended output scored by a second LLM (LLM-as-Judge) |

Every task declares a **primary** metric and optionally **secondary** ones in
`task.yaml`:

```yaml
metric:
  primary: accuracy          # used for ranking in the leaderboard
  secondary: [macro_f1]      # shown in the UI but not used for sorting
```

`tps` (tokens per second) and `p95_latency_ms` are always recorded
automatically and do not need to be listed in `metric`.

---

## Classification metrics: `accuracy`, `macro_f1`

Best for tasks where the expected answer is one of a known set of labels
(e.g. sentiment, topic classification).

- **`accuracy`** — fraction of correct predictions. Simple, intuitive, but
  can be misleading when classes are imbalanced.
- **`macro_f1`** — average F1 across all labels, giving equal weight to each
  class regardless of frequency. Requires `labels:` in `task.yaml`.

### Example task.yaml

```yaml
name: topic_classification
version: "1.0"
metric:
  primary: accuracy
  secondary: [macro_f1]
labels: [tech, sports, politics, health, entertainment]
llm_params:
  temperature: 0
  max_tokens: 8
  seed: 42
prompt_template: |
  Classify the topic of the following text.
  Categories: tech, sports, politics, health, entertainment
  Answer with exactly one category.

  Text: {prompt}

  Category:
```

### How prediction extraction works

The runner extracts the model's answer by:
1. If `labels:` is set — fuzzy match against the label list (case-insensitive,
   prefix-free, e.g. "Positive" matches "positive").
2. If `answer_regex:` is set — the first capture group of the regex.
3. Otherwise — the raw model output, stripped and lowercased.

---

## Extractive metric: `exact_match`

Best for factoid QA where the answer is a short span (a name, date, number).
Before comparing, both the prediction and the reference are normalised:
lowercase, strip articles ("a", "an", "the"), collapse whitespace, remove
punctuation. This follows the SQuAD evaluation convention.

### Example task.yaml

```yaml
name: extractive_qa
version: "1.0"
metric:
  primary: exact_match
llm_params:
  temperature: 0
  max_tokens: 32
  seed: 42
answer_regex: "^(.+)$"   # optional — useful when model pads its answer
prompt_template: |
  Answer the following question in as few words as possible.

  Question: {prompt}

  Answer:
```

---

## Generative metric: `judge_score` (LLM-as-Judge)

For open-ended tasks (summarisation, translation, creative writing) where
there is no single correct answer, a second LLM scores each candidate on a
fixed integer scale. The runner normalises scores into [0, 1] so they compose
with accuracy/F1 in the global compare view.

### How it works

1. The candidate model generates an answer for each sample.
2. The **judge model** receives the original prompt, the reference answer, and
   the candidate answer, then outputs a single integer score.
3. Scores are clamped to `[scale_min, scale_max]`, normalised to `[0, 1]`,
   and averaged across samples → `judge_score`.
4. If the judge output can't be parsed, that sample scores `scale_min` (so
   flaky judges hurt, not silently pass).

### Configuring the judge

Add a `judge:` block to `task.yaml`:

```yaml
name: summarization
version: "1.0"
metric:
  primary: judge_score
llm_params:
  temperature: 0
  max_tokens: 80
  seed: 42
system_prompt: |
  You write concise, factual one-sentence summaries. Output only the summary.
prompt_template: |
  Summarise the following paragraph in a single sentence.

  Paragraph:
  {prompt}

  Summary:
judge:
  model: gpt-4o-mini@openai          # model_id of a card in models/
  scale_min: 0
  scale_max: 5
  forbid_self_judge: true
  prompt: |
    You are a strict evaluator of one-sentence summaries.
    Score the candidate summary on faithfulness, completeness and concision
    on an integer scale 0..5, where:
      0 = unrelated or hallucinated
      1 = mostly wrong
      2 = partially correct, missing key facts
      3 = correct but verbose or missing minor facts
      4 = correct and concise, minor flaws
      5 = perfect, faithful and concise

    Original paragraph: {prompt}
    Reference summary:  {expected}
    Candidate summary:  {prediction}

    Reply with a single integer 0..5 and nothing else.
    Score:
```

### Key fields

| Field | Required | Description |
|---|---|---|
| `judge.model` | yes | `model_id` of a model card under `models/`. The judge model must have its own card — this is the **LLMasJ_URI** from the spec. It means the judge is a first-class artifact: pinned revision, visible in the models list, independently upgradable. |
| `judge.prompt` | yes | Prompt template with placeholders `{prompt}`, `{expected}`, `{prediction}`. Must instruct the judge to output a single integer. |
| `judge.scale_min` | no (default 0) | Minimum of the integer scale. |
| `judge.scale_max` | no (default 5) | Maximum of the integer scale. |
| `judge.forbid_self_judge` | no (default true) | When true, the runner refuses to score if the candidate model and the judge model have the same `model_id`. Prevents self-judge bias. |

### What is LLMasJ_URI?

In the original spec, `LLMasJ_URI` stands for **LLM-as-Judge URI** — the
standardised way to reference the judge model. In this implementation, the URI
is simply the `model_id` of the judge's model card (e.g.
`gpt-4o-mini@openai`). Because judges are normal model cards:

- They have a pinned `revision` and `endpoint_url`
- They appear in the models list on the site
- Swapping the judge is as easy as changing `judge.model` in the task YAML
  and bumping the task version

### Reproducibility

The runner forces `temperature=0` and `seed=0` on every judge call regardless
of the task's `llm_params`. The result JSON records:
- `judge_model_id` — which model was used
- `judge_model_revision` — its revision
- `judge_prompt_hash` — SHA-256 prefix of the judge prompt text

---

## Cost tracking

Cost is computed per run when the model card has both
`prompt_cost_per_1k_usd` and `completion_cost_per_1k_usd`. The total
`cost_usd` appears in the leaderboard UI alongside scores.

```yaml
# models/gpt-4o-mini@openai.yaml
prompt_cost_per_1k_usd: 0.00015
completion_cost_per_1k_usd: 0.0006
```

---

## Choosing a metric for your task

```
Is the expected answer one of a fixed label set?
  ├─ yes → accuracy (+ macro_f1 if classes are imbalanced)
  └─ no
       Is the answer a short exact span?
         ├─ yes → exact_match
         └─ no  → judge_score (configure a judge in task.yaml)
```
