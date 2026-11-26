"""Execution Agent implementation."""

from pathlib import Path
from typing import List, Optional, Dict, Any

from ...services.execution import get_execution_agent_logs
from ...services.execution.agent_memory import render_history
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

        return SYSTEM_PROMPT_TEMPLATE.format(
            agent_name=self.name,
            agent_purpose=agent_purpose
        )

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
        # Record the tool response
        self._log_store.record_tool_response(self.name, tool_name, result[:500])


def _limit_recent_runs(history: str, limit: int) -> str:
    """Keep the memory summary and only the last ``limit`` raw runs."""
    lines = history.split("\n")
    request_lines = [i for i, line in enumerate(lines) if line.startswith("<agent_request")]
    if len(request_lines) <= limit:
        return history
    cut_start, cut_end = request_lines[0], request_lines[-limit]
    return "\n".join(lines[:cut_start] + lines[cut_end:])
