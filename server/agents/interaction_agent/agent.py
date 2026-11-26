"""Interaction agent helpers for prompt construction."""

from html import escape
from pathlib import Path
from typing import Dict, List, Set

from ...config import get_settings
from ...logging_config import logger
from ...services.execution import get_agent_roster
from ...services.execution.agent_ranker import mentioned_agent_keys, select_agents
from ...services.execution.roster import agent_key

_prompt_path = Path(__file__).parent / "system_prompt.md"
SYSTEM_PROMPT = _prompt_path.read_text(encoding="utf-8").strip()

# How much of the recent conversation feeds the agent-relevance query.
_QUERY_TRANSCRIPT_TAIL_CHARS = 1500


# Load and return the pre-defined system prompt from markdown file
def build_system_prompt() -> str:
    """Return the static system prompt for the interaction agent."""
    return SYSTEM_PROMPT


# Build structured message with conversation history, active agents, and current turn
def prepare_message_with_history(
    latest_text: str,
    transcript: str,
    message_type: str = "user",
) -> List[Dict[str, str]]:
    """Compose a message that bundles history, roster, and the latest turn."""
    sections: List[str] = []

    query = f"{latest_text}\n{transcript[-_QUERY_TRANSCRIPT_TAIL_CHARS:]}"
    sections.append(_render_conversation_history(transcript))
    sections.append(f"<active_agents>\n{_render_active_agents(query, pinned_text=latest_text)}\n</active_agents>")
    sections.append(_render_current_turn(latest_text, message_type))

    content = "\n\n".join(sections)
    return [{"role": "user", "content": content}]


# Format conversation transcript into XML tags for LLM context
def _render_conversation_history(transcript: str) -> str:
    history = transcript.strip()
    if not history:
        history = "None"
    return f"<conversation_history>\n{history}\n</conversation_history>"


def _trigger_pinned_keys() -> Set[str]:
    try:
        from ...services.triggers import get_trigger_service

        return {agent_key(name) for name in get_trigger_service().agents_with_active_triggers()}
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Could not load trigger-owning agents: {exc}")
        return set()


# Render the most relevant execution agents (not all of them) for this turn
def _render_active_agents(query: str = "", pinned_text: str = "") -> str:
    roster = get_agent_roster()
    roster.load()
    records = roster.records()

    if not records:
        return "None"

    pinned = _trigger_pinned_keys() | mentioned_agent_keys(records, pinned_text)
    selection = select_agents(
        records,
        query,
        k=get_settings().agent_roster_prompt_limit,
        pinned_keys=pinned,
    )

    rendered: List[str] = []
    for record in selection.shown:
        attrs = [f'name="{escape(record.name, quote=True)}"']
        last_used = record.last_used_dt
        if last_used is not None:
            attrs.append(f'last_used="{last_used.strftime("%Y-%m-%d")}"')
        if record.run_count:
            attrs.append(f'runs="{record.run_count}"')
        if record.key in selection.pinned_keys:
            attrs.append('pinned="true"')
        description = escape(record.description, quote=False)
        if description:
            rendered.append(f"<agent {' '.join(attrs)}>{description}</agent>")
        else:
            rendered.append(f"<agent {' '.join(attrs)} />")

    if selection.hidden_count:
        rendered.append(
            f'<more_agents count="{selection.hidden_count}">Older or less relevant agents are not listed. '
            "If this turn continues earlier work that is not shown above, call search_agents before "
            "creating a new agent.</more_agents>"
        )

    return "\n".join(rendered)


# Wrap the current message in appropriate XML tags based on sender type
def _render_current_turn(latest_text: str, message_type: str) -> str:
    tag = "new_agent_message" if message_type == "agent" else "new_user_message"
    body = latest_text.strip()
    return f"<{tag}>\n{body}\n</{tag}>"
