"""Multi-turn dialog simulation for agent benchmarks.

Ported from DeepPavlov's `partial-mcp` (which itself wraps tau2-bench).
The "support agent" under test has access to a set of domain tools via
OpenAI function-calling; a "user agent" (the same model, different system
prompt) plays the customer with a scenario pulled from the sample.

The trace (full conversation + tool calls) is what the LLM judge scores.
"""
from .simulator import simulate_retail_dialog

__all__ = ["simulate_retail_dialog"]
