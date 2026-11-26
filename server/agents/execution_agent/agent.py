"""Execution Agent implementation."""

import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

from ...services.execution import get_execution_agent_logs
from ...services.execution.agent_memory import render_history
from ...utils.timezones import get_user_timezone_name, now_in_user_timezone
from ...logging_config import logger


# Load system prompt template from file
_prompt_path = Path(__file__).parent / "system_prompt.md"
if _prompt_path.exists():
    SYSTEM_PROMPT_TEMPLATE = _prompt_path.read_text(encoding="utf-8").strip()
else:
    # Placeholder template - you'll replace this with actual instructions
    SYSTEM_PROMPT_TEMPLATE = """You are an execution agent responsible for completing specific tasks using available tools.

Agent Name: {agent_name}
Purpose: {agent_purpose}

Instructions:
[TO BE FILLED IN BY USER]

You have access to Gmail tools to help complete your tasks. When given instructions:
1. Analyze what needs to be done
2. Use the appropriate tools to complete the task
3. Provide clear status updates on your actions

Be thorough, accurate, and efficient in your execution."""


class ExecutionAgent:
    """Manages state and history for an execution agent."""

    # Initialize execution agent with name, conversation limits, and log store access
    def __init__(
        self,
        name: str,
        conversation_limit: Optional[int] = None
    ):
        """
        Initialize an execution agent.

        Args:
            name: Human-readable agent name (e.g., 'conversation with keith')
            conversation_limit: Optional limit on past conversations to include (None = all)
        """
        self.name = name
        self.conversation_limit = conversation_limit
        self._log_store = get_execution_agent_logs()

    # Generate system prompt template with agent name and purpose derived from name
    def build_system_prompt(self) -> str:
        """Build the system prompt for this agent."""
        agent_purpose = f"Handle tasks related to: {self.name}"

        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            agent_name=self.name,
            agent_purpose=agent_purpose
        )
        # The prompt said times use "the user's automatically detected timezone" but never
        # said which one, or what time it is, so relative requests ("in 1 minute",
        # "tomorrow 9am") were guessed from log timestamps and could land hours off.
        return f"{prompt}\n\n{current_time_line()}"

    # Combine base system prompt with a bounded view of the agent's history
    def build_system_prompt_with_history(self) -> str:
        """
        Build system prompt including agent history.

        History is bounded: a rolling memory summary plus the most recent runs
        (see services/execution/agent_memory.py). ``conversation_limit``, when set,
        additionally caps how many recent raw runs are kept.

        Returns:
            System prompt with embedded history
        """
        base_prompt = self.build_system_prompt()
        history = render_history(self.name)
        if not history:
            return base_prompt

        if self.conversation_limit and self.conversation_limit > 0:
            history = _limit_recent_runs(history, self.conversation_limit)

        return f"{base_prompt}\n\n# Execution History\n\n{history}"

    # Format current instruction as user message for LLM consumption
    def build_messages_for_llm(self, current_instruction: str) -> List[Dict[str, str]]:
        """
        Build message array for LLM call.

        Args:
            current_instruction: Current instruction from interaction agent

        Returns:
            List of messages in OpenRouter format
        """
        return [
            {"role": "user", "content": current_instruction}
        ]

    # Log the agent's final response to the execution log store
    def record_response(self, response: str) -> None:
        """Record agent's response to the log."""
        self._log_store.record_agent_response(self.name, response)

    # Log tool invocation and results with truncated content for readability
    def record_tool_execution(self, tool_name: str, arguments: str, result: str) -> None:
        """Record tool execution details."""
        self._log_store.record_action(self.name, f"Calling {tool_name} with: {arguments[:200]}")
        # Results are truncated to keep the log small, but the identifiers an agent needs
        # later (thread/message/draft/trigger ids, email addresses) often sit past the cut.
        self._log_store.record_tool_response(self.name, tool_name, truncate_keeping_ids(result, 500))


def _limit_recent_runs(history: str, limit: int) -> str:
    """Keep the memory summary and only the last ``limit`` raw runs."""
    lines = history.split("\n")
    request_lines = [i for i, line in enumerate(lines) if line.startswith("<agent_request")]
    if len(request_lines) <= limit:
        return history
    cut_start, cut_end = request_lines[0], request_lines[-limit]
    return "\n".join(lines[:cut_start] + lines[cut_end:])


def current_time_line() -> str:
    now = now_in_user_timezone()
    return (f"Current time: {now:%Y-%m-%d %H:%M} ({get_user_timezone_name()}, UTC{now:%z}). "
            "Interpret relative times (\"in 1 minute\", \"tomorrow at 9am\") from this.")


_ID_KEY = re.compile(r"(^id$|_id$|Id$|^threadId$|^messageId$)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def _collect_ids(value, out: list) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (str, int)) and _ID_KEY.search(str(key)):
                out.append(f"{key}={item}")
            else:
                _collect_ids(item, out)
    elif isinstance(value, list):
        for item in value:
            _collect_ids(item, out)


def truncate_keeping_ids(text: str, limit: int) -> str:
    """Truncate ``text`` but append any ids/emails that only appeared after the cut."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    found: list = []
    try:
        _collect_ids(json.loads(text), found)
    except (ValueError, TypeError):
        pass
    found += _EMAIL.findall(text)
    missing = [item for item in dict.fromkeys(found) if item.split("=", 1)[-1] not in head]
    if not missing:
        return head + " …"
    return head + " … [ids beyond cut: " + ", ".join(missing[:40]) + "]"
