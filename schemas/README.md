# JSON Schemas

These files are **generated** from the pydantic models in `runner/src/llm_lb/models.py`.

To regenerate after changing the models:

```bash
cd runner
uv run llm-lb export-schemas --out-dir ../schemas
```

Do not edit the `*.schema.json` files by hand.
