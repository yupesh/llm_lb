"""Retail-domain dialog simulator.

One sample = one customer scenario from tau2-bench/retail. The model under
test plays both the support agent (with 15 retail tools via function-calling)
and the user agent (same model, different system prompt, no tools).

Flow:
  1. Reset retail DB to snapshot.
  2. User agent generates first message from scenario instructions.
  3. Loop up to `max_turns`:
       - Support agent replies (possibly calling tools; we execute tools
         against the DB and feed results back until it produces a final
         text message).
       - User agent replies. If the reply contains the end marker or the
         turn cap is reached, stop.
  4. Return the full conversation + tool-call trace as a JSON string; the
     judge scores it against the sample's expected `evaluation_criteria`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ...models import LLMParams
from .retail_data_model import get_db
from .retail_schemas import RETAIL_TOOL_SCHEMAS
from .retail_tools import RetailTools

log = logging.getLogger(__name__)

# Appended to the user-agent's system prompt. The original tau2 benchmark
# uses pydantic-ai structured output to signal end-of-conversation; since
# our chat-completions setup doesn't support that cleanly, we fall back on
# a text marker the user-agent can emit.
END_MARKER = "###END###"
USER_AGENT_END_INSTRUCTION = (
    "\n\nIf the customer service agent has fully satisfied your request, OR "
    "if you feel stuck and the agent cannot help further, OR if the agent has "
    f"transferred you to a human, end your reply with the marker {END_MARKER} "
    "on its own line. Do not use the marker otherwise."
)


def _call_retail_tool(retail: RetailTools, name: str, arguments: dict) -> str:
    """Invoke a tool by name. Returns a string to feed back as tool result.

    We JSON-serialise pydantic models and dicts; strings and exceptions are
    returned as-is / stringified. Exceptions become error messages (not
    raised) so the agent can recover like in the original tau2 setup.
    """
    method = getattr(retail, name, None)
    if method is None or name.startswith("_"):
        return f"Error: unknown tool {name!r}"
    try:
        result = method(**arguments)
    except TypeError as e:
        return f"Error: bad arguments for {name}: {e}"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:  # noqa: BLE001 — we want to surface any tool error to the agent
        return f"Error: {type(e).__name__}: {e}"
    # Serialise the result for the agent.
    if isinstance(result, str):
        return result
    try:
        # pydantic model
        return result.model_dump_json()
    except AttributeError:
        pass
    try:
        return json.dumps(result, default=str)
    except TypeError:
        return str(result)


def _extract_text(msg: dict) -> str:
    return (msg.get("content") or "").strip()


def simulate_retail_dialog(
    adapter,
    policy_prompt: str,
    user_agent_prompt: str,
    db_path: Path,
    sample_prompt_json: str,
    params: LLMParams,
    max_turns: int = 10,
) -> tuple[str, list[dict], int, int]:
    """Run one retail customer-support conversation.

    Returns:
      prediction — JSON string with keys `conversation` (list of
                   {role, content}) and `tool_calls` (list of
                   {name, arguments}). The judge scores this.
      trace      — list of tool calls (for potential F1 side-metrics).
      input_tok, output_tok — usage totals across all LLM calls.
    """
    if not hasattr(adapter, "chat_messages"):
        raise RuntimeError(
            "dialog_simulation requires an adapter with chat_messages() — "
            "use an openai or openai_compat model card"
        )

    retail = RetailTools(db=get_db(db_path), db_path=db_path)
    scenario = json.loads(sample_prompt_json)

    # ---- user-agent history --------------------------------------------------
    # The user-agent is given the scenario in its system prompt, then receives
    # a synthetic "agent greeting" as the first user-role message — this both
    # gives it something to react to and satisfies OpenAI-compatible proxies
    # (e.g. litellm) that reject message lists with no user-role messages.
    user_sys = user_agent_prompt + USER_AGENT_END_INSTRUCTION + "\n\n" + json.dumps(
        scenario, ensure_ascii=False, indent=2
    )
    agent_greeting = "Hi! How can I help you today?"
    user_messages: list[dict] = [
        {"role": "system", "content": user_sys},
        {"role": "user", "content": agent_greeting},
    ]

    # ---- support-agent history ----------------------------------------------
    support_messages: list[dict] = [{"role": "system", "content": policy_prompt}]

    conversation: list[dict] = []
    trace: list[dict] = []
    in_tok = 0
    out_tok = 0

    # First move: user-agent speaks.
    ua_resp = adapter.chat_messages(user_messages, params)
    in_tok += (ua_resp["usage"] or {}).get("prompt_tokens", 0) or 0
    out_tok += (ua_resp["usage"] or {}).get("completion_tokens", 0) or 0
    first_user_msg = _extract_text(ua_resp["message"]) or "Hi."
    user_messages.append({"role": "assistant", "content": first_user_msg})
    conversation.append({"role": "user", "content": first_user_msg})
    support_messages.append({"role": "user", "content": first_user_msg})

    for turn in range(max_turns):
        # Support agent, with tool-call loop.
        for _ in range(6):  # inner tool-call cap per turn
            sa_resp = adapter.chat_messages(
                support_messages, params, tools=RETAIL_TOOL_SCHEMAS
            )
            in_tok += (sa_resp["usage"] or {}).get("prompt_tokens", 0) or 0
            out_tok += (sa_resp["usage"] or {}).get("completion_tokens", 0) or 0
            msg = sa_resp["message"]
            # Keep the assistant message (including any tool_calls field) in
            # the support-history — providers require it for consistency. Drop
            # `reasoning_content`: vLLM reasoning parsers can produce massive
            # chain-of-thought blobs that, if echoed back each turn, blow past
            # the 40K context cap on multi-turn dialogs.
            slim_msg: dict = {"role": msg.get("role", "assistant"),
                              "content": msg.get("content") or ""}
            if msg.get("tool_calls"):
                slim_msg["tool_calls"] = msg["tool_calls"]
            support_messages.append(slim_msg)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                break
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}
                trace.append({"name": name, "arguments": args})
                result_str = _call_retail_tool(retail, name, args)
                support_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": result_str,
                    }
                )
        else:
            log.warning("tool-call loop exceeded inner cap on turn %d", turn)

        agent_text = _extract_text(msg)
        conversation.append({"role": "assistant", "content": agent_text})

        # Feed agent reply to user-agent as "user" role.
        user_messages.append({"role": "user", "content": agent_text or "(empty)"})
        ua_resp = adapter.chat_messages(user_messages, params)
        in_tok += (ua_resp["usage"] or {}).get("prompt_tokens", 0) or 0
        out_tok += (ua_resp["usage"] or {}).get("completion_tokens", 0) or 0
        ua_text = _extract_text(ua_resp["message"])
        user_messages.append({"role": "assistant", "content": ua_text})
        conversation.append({"role": "user", "content": ua_text})

        if END_MARKER in ua_text:
            break

        support_messages.append({"role": "user", "content": ua_text})

    prediction_obj = {"conversation": conversation, "tool_calls": trace}
    prediction = json.dumps(prediction_obj, ensure_ascii=False)
    return prediction, trace, in_tok, out_tok
