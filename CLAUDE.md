# LLM Leaderboards — Specification for Claude Code

## 1. Overview

This project implements a GitHub-based Leaderboard system for evaluating Large Language Models (LLMs) on small, well-defined tasks.

Key principles:
- Everything is managed via GitHub (PR-driven workflow)
- Tasks are lightweight and reproducible
- Results are transparent and versioned
- Leaderboards are statically rendered via GitHub Pages

---

## 2. Core Concepts

### 2.1 Task
A task represents a small benchmark problem.

Characteristics:
- ~20 samples per task
- Fixed format
- Deterministic evaluation

Each task includes:
- Input prompts
- Expected outputs
- Evaluation logic

### 2.2 Model Entry
A model submission evaluated on a task.

Includes:
- Model metadata
- Runtime configuration
- Evaluation results

### 2.3 Leaderboard (LB)
A leaderboard ranks models per task based on:
- Accuracy (primary metric)
- TPS (tokens per second)


## 3. Task Specification

### task.yaml

```yaml
name: text_classification
version: 1.0
metric: accuracy
labels:
  - positive
  - negative

llm_params:
  temperature: 0
  max_tokens: 32
```

### Sample Format

```
{
  "id": "sample_001",
  "prompt": "Classify sentiment: I love this product",
  "expected": "positive"
}
```

---

## 4. Model Metadata

```
{
  "model_id": "gpt-4.1",
  "provider": "openai",
  "uri": "https://...",
  "gpu": {
    "count": 1,
    "type": "A100"
  }
}
```

---

## 5. Result Format

```
{
  "model_id": "gpt-4.1",
  "task": "text_classification",
  "task_version": "1.0",
  "accuracy": 0.85,
  "tps": 120.5,
  "samples": [
    {
      "id": "sample_001",
      "prediction": "positive",
      "correct": true,
      "latency_ms": 120
    }
  ]
}
```

---

## 6. Leaderboard Format

```
{
  "task": "text_classification",
  "version": "1.0",
  "ranking": [
    {
      "model_id": "gpt-4.1",
      "accuracy": 0.85,
      "tps": 120.5
    }
  ]
}
```

Sorting:
1. Accuracy (descending)
2. TPS (descending)

---

## 7. Workflow (GitHub PR-driven)

### Step-by-step

1. Task Owner:
   - Creates/updates task via PR
   - Adds samples and task.yaml

2. CI Validation:
   - Validate schemas
   - Check sample count
   - Ensure reproducibility

3. Admin Review:
   - Reviews PR
   - Runs evaluation locally or via CI

4. Scoring:
   - Admin runs evaluation:
     ```bash
     python utils/runner.py --task text_classification --model gpt-4.1
     ```
   - Generates result JSON

5. Update Leaderboard:
   - Update leaderboard.json
   - Commit to PR

6. Build Static Site:
   - Generate HTML from JSON

7. Merge:
   - Squash PR into main

---

Support:
- OpenAI
- HuggingFace
- Local inference endpoints

LLMasJ_URI:
- Standardized URI for model access

---


## 8. Best Practices

- Deterministic prompts
- Fixed temperature = 0
- Small datasets (fast CI)
- Explicit schemas
- Versioned tasks
- Reproducible runs

---

## 9. Open Questions

- Where to run evaluation (CI vs local)?
- How to handle API keys securely?
- Should contributors submit model results or only tasks?
- How to prevent cheating?

---

## 10. Expected Output from Claude Code

Claude Code should:
1. Propose detailed architecture
2. Refine repo structure
3. Generate initial implementation:
   - schemas
   - utils modules
   - CI workflows
4. Suggest UI implementation
5. Provide MVP plan

---

End of specification

