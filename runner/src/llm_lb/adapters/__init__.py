from .base import Completion, LLMClient, get_adapter  # noqa: F401

# Side-effect imports — each module registers itself with the adapter registry.
from . import dummy  # noqa: F401
from . import openai  # noqa: F401
from . import openai_compat  # noqa: F401
from . import hf  # noqa: F401
