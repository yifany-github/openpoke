"""Tool definitions for interaction agent."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...logging_config import logger
from ...services.conversation import get_conversation_log
from ...services.execution import get_agent_roster
from ...services.execution.agent_ranker import search_agents as rank_search_agents
from ...services.execution.roster import AgentRecord
from ...services.timezone_store import get_timezone_store
from ..execution_agent.batch_manager import get_execution_batch_manager


@dataclass
class ToolResult:
    """Standardized payload returned by interaction-agent tools."""

    success: bool
    payload: Any = None
    user_message: Optional[str] = None
    recorded_reply: bool = False

# Tool schemas for OpenRouter
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "send_message_to_agent",
            "description": "Deliver instructions to a specific execution agent. Creates a new agent if the name doesn't exist in the roster, or reuses an existing one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Human-readable agent name describing its purpose (e.g., 'Vercel Job Offer', 'Email to Sharanjeet'). This name will be used to identify and potentially reuse the agent."
                    },
                    "instructions": {"type": "string", "description": "Instructions for the agent to execute."},
                    "force_new": {
                        "type": "boolean",
                        "description": "Set true only after a call was rejected with similar_agents_exist and you have decided this really is a separate thread of work.",
                    },
                },
                "required": ["agent_name", "instructions"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_timezone",
            "description": "Save the user's timezone when they tell you where they are or what timezone they're in. Reminders and schedules use it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "IANA timezone name, e.g. 'America/New_York', 'Europe/Berlin', 'Asia/Shanghai'."},
                },
                "required": ["timezone"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_agents",
            "description": "Search all existing execution agents (including ones not listed in <active_agents>) by keywords such as a person, company, or topic. Use it before creating a new agent when the request may continue earlier work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords describing the earlier work, e.g. 'Sam dinner' or 'lease renewal'."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message_to_user",
            "description": "Deliver a natural-language response directly to the user. Use this for updates, confirmations, or any assistant response the user should see immediately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Plain-text message that will be shown to the user and recorded in the conversation log.",
                    },
                },
                "required": ["message"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_draft",
            "description": "Record an email draft so the user can review the exact text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email for the draft.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject for the draft.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content (plain text).",
                    },
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait silently when a message is already in conversation history to avoid duplicating responses. Adds a <wait> log entry that is not visible to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why waiting (e.g., 'Message already sent', 'Draft already created').",
                    },
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
]

_EXECUTION_BATCH_MANAGER = get_execution_batch_manager()


def _agent_summary(record: AgentRecord) -> Dict[str, Any]:
    last_used = record.last_used_dt
    return {
        "agent_name": record.name,
        "description": record.description or None,
        "last_used": last_used.strftime("%Y-%m-%d") if last_used else None,
        "runs": record.run_count,
    }


# Create or reuse execution agent and dispatch instructions asynchronously
def send_message_to_agent(agent_name: str, instructions: str, force_new: bool = False) -> ToolResult:
    """Send instructions to an execution agent.

    Name resolution, in order:
    1. Same normalized key as an existing agent -> reuse it under its canonical name.
    2. Content-similar to existing agents and not ``force_new`` -> reject and return the
       candidates, so the model either reuses one or confirms a genuinely new agent.
    3. Otherwise create a new agent.
    """
    roster = get_agent_roster()
    roster.load()
    requested_name = agent_name
    existing = roster.get(agent_name)
    is_new = existing is None

    if existing is not None:
        agent_name = existing.name  # canonical spelling keeps logs/triggers consistent
    elif not force_new:
        similar = roster.find_similar(agent_name)
        if similar:
            logger.info(f"Agent '{agent_name}' rejected: similar agents exist")
            return ToolResult(
                success=False,
                payload={
                    "error": "similar_agents_exist",
                    "requested_agent_name": requested_name,
                    "similar_agents": [_agent_summary(r) for r in similar],
                    "hint": "If this continues one of these threads, call send_message_to_agent again with its exact agent_name. If it is genuinely separate work, call again with force_new=true.",
                },
            )

    if is_new:
        roster.add_agent(agent_name)
    roster.touch(agent_name)
    if is_new or not roster.get(agent_name).description:
        roster.update_description(agent_name, f"Started for: {instructions[:200]}")


    action = "Created" if is_new else "Reused"
    logger.info(f"{action} agent: {agent_name}")

    async def _execute_async() -> None:
        try:
            result = await _EXECUTION_BATCH_MANAGER.execute_agent(agent_name, instructions)
            status = "SUCCESS" if result.success else "FAILED"
            logger.info(f"Agent '{agent_name}' completed: {status}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Agent '{agent_name}' failed: {str(exc)}")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.error("No running event loop available for async execution")
        return ToolResult(success=False, payload={"error": "No event loop available"})

    loop.create_task(_execute_async())

    return ToolResult(
        success=True,
        payload={
            "status": "submitted",
            "agent_name": agent_name,
            "new_agent_created": is_new,
            **({"resolved_from": requested_name} if requested_name != agent_name else {}),
        },
    )


# Search the full agent registry (the prompt only lists the most relevant agents)
def search_agents(query: str) -> ToolResult:
    roster = get_agent_roster()
    roster.load()
    matches = rank_search_agents(roster.records(), query, limit=10)
    return ToolResult(
        success=True,
        payload={
            "query": query,
            "matches": [_agent_summary(r) for r in matches],
            "total_agents": len(roster.records()),
        },
    )


# Send immediate message to user and record in conversation history
def send_message_to_user(message: str) -> ToolResult:
    """Record a user-visible reply in the conversation log."""
    log = get_conversation_log()
    log.record_reply(message)

    return ToolResult(
        success=True,
        payload={"status": "delivered"},
        user_message=message,
        recorded_reply=True,
    )


# Format and record email draft for user review
def send_draft(
    to: str,
    subject: str,
    body: str,
) -> ToolResult:
    """Record a draft update in the conversation log for the interaction agent."""
    log = get_conversation_log()

    message = f"To: {to}\nSubject: {subject}\n\n{body}"

    log.record_reply(message)
    logger.info(f"Draft recorded for: {to}")

    return ToolResult(
        success=True,
        payload={
            "status": "draft_recorded",
            "to": to,
            "subject": subject,
        },
        recorded_reply=True,
    )


# Persist the user's timezone (previously only the browser could set it)
def set_timezone(timezone: str) -> ToolResult:
    try:
        get_timezone_store().set_timezone(timezone)
    except ValueError as exc:
        return ToolResult(success=False, payload={"error": str(exc), "hint": "Use an IANA name like 'America/New_York'."})
    return ToolResult(success=True, payload={"timezone": timezone, "status": "saved"})


# Record silent wait state to avoid duplicate responses
def wait(reason: str) -> ToolResult:
    """Wait silently and add a wait log entry that is not visible to the user."""
    log = get_conversation_log()
    
    # Record a dedicated wait entry so the UI knows to ignore it
    log.record_wait(reason)
    

    return ToolResult(
        success=True,
        payload={
            "status": "waiting",
            "reason": reason,
        },
        recorded_reply=True,
    )


# Return predefined tool schemas for LLM function calling
def get_tool_schemas():
    """Return OpenAI-compatible tool schemas."""
    return TOOL_SCHEMAS


# Route tool calls to appropriate handlers with argument validation and error handling
def handle_tool_call(name: str, arguments: Any) -> ToolResult:
    """Handle tool calls from interaction agent."""
    try:
        if isinstance(arguments, str):
            args = json.loads(arguments) if arguments.strip() else {}
        elif isinstance(arguments, dict):
            args = arguments
        else:
            return ToolResult(success=False, payload={"error": "Invalid arguments format"})

        if name == "send_message_to_agent":
            return send_message_to_agent(**args)
        if name == "send_message_to_user":
            return send_message_to_user(**args)
        if name == "search_agents":
            return search_agents(**args)
        if name == "send_draft":
            return send_draft(**args)
        if name == "set_timezone":
            return set_timezone(**args)
        if name == "wait":
            return wait(**args)

        logger.warning("unexpected tool", extra={"tool": name})
        return ToolResult(success=False, payload={"error": f"Unknown tool: {name}"})
    except json.JSONDecodeError:
        return ToolResult(success=False, payload={"error": "Invalid JSON"})
    except TypeError as exc:
        return ToolResult(success=False, payload={"error": f"Missing required arguments: {exc}"})
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("tool call failed", extra={"tool": name, "error": str(exc)})
        return ToolResult(success=False, payload={"error": "Failed to execute"})
