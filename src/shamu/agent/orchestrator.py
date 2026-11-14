"""Unified agent cognitive loop orchestrator service.

Extracts multi-turn tool calling, streaming chunk delta reassembly,
tool response formatting, deduplication, error dispatching, and loop termination
into an uncoupled domain service emitting pure TurnEvents.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Union

from shamu.gateway.router import ModelGateway
from shamu.memory.context_compactor import ContextCompactor
from shamu.protocols import PermissionDeniedError
from shamu.security.firewall import ExecutionFirewall
from shamu.types import ChatChunk, ChatMessage, Role, ToolCall

logger = logging.getLogger(__name__)

_SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_\-]{16,}", re.IGNORECASE),
    re.compile(r"bearer\s+[a-zA-Z0-9_\-\.]{16,}", re.IGNORECASE),
    re.compile(r"gh[pousr]_[a-zA-Z0-9]{36}", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[^\s'\"]+['\"]?", re.IGNORECASE),
]

_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:[\\/]Users[\\/][A-Za-z0-9_.\s-]+", re.IGNORECASE),
    re.compile(r"[\\/]\.ssh(?:[\\/][^\s'\"]*)?", re.IGNORECASE),
    re.compile(r"[\\/]\.aws(?:[\\/][^\s'\"]*)?", re.IGNORECASE),
    re.compile(r"[\\/]\.env[^\s'\"]*", re.IGNORECASE),
    re.compile(r"[\\/]id_[a-zA-Z0-9_]+", re.IGNORECASE),
]


def sanitize_error_message(err: Any) -> str:
    """Sanitizes an exception or error string to scrub credentials and sensitive local paths."""
    text = str(err)
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[REDACTED_SECRET]", text)
    for pat in _PATH_PATTERNS:
        text = pat.sub("[REDACTED_PATH]", text)
    return text


def is_complete_json_object(text: str) -> bool:
    """Checks whether the text represents a fully closed and valid JSON object."""
    if not text:
        return False
    s = text.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return False
    try:
        val = json.loads(s)
        return isinstance(val, dict)
    except Exception:
        return False


@dataclass
class TurnEvent:
    """Base class for all cognitive loop turn events."""
    pass


@dataclass
class TurnReset(TurnEvent):
    """Emitted when the model gateway signals a turn reset (e.g. failover retry)."""
    pass


@dataclass
class TokenChunk(TurnEvent):
    """Emitted when a streaming token or text chunk is received."""
    text: str


@dataclass
class ToolRequested(TurnEvent):
    """Emitted when a tool call has been requested by the LLM."""
    tool_call: ToolCall
    call_index: int = 0
    total_calls: int = 1


@dataclass
class ToolExecuted(TurnEvent):
    """Emitted after a tool call completes execution (or cached deduplicated result used)."""
    tool_call: ToolCall
    result: str
    is_cached: bool = False
    is_error: bool = False


@dataclass
class StatusUpdate(TurnEvent):
    """Status lifecycle event (e.g. 'Thinking...', 'Executing <tool>...', 'Finalizing response...')."""
    status: str


@dataclass
class TurnCompleted(TurnEvent):
    """Emitted when the multi-turn loop completes for this turn."""
    final_text: str
    tool_calls_executed: int
    turn_counter: int


class AgentOrchestrator:
    """Unified domain service executing multi-turn LLM turns,
    tool calling, streaming chunk reassembly, tool response formatting,
    deduplication, error handling, and termination conditions."""

    def __init__(
        self,
        gateway: ModelGateway,
        firewall: Optional[ExecutionFirewall] = None,
        compactor: Optional[ContextCompactor] = None,
        max_tool_turns: int = 5,
    ) -> None:
        self.gateway = gateway
        self.firewall = firewall
        self.compactor = compactor
        self.max_tool_turns = max_tool_turns
        self.last_compaction_error: Optional[str] = None

    async def run_turn(
        self,
        history: List[ChatMessage],
        session_id: Optional[str] = None,
        turn_id_prefix: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        event_callback: Optional[Callable[[TurnEvent], Union[None, Awaitable[None]]]] = None,
    ) -> AsyncIterator[TurnEvent]:
        """Executes a multi-turn conversational loop, yielding TurnEvent instances."""
        # 1. Compact history if ContextCompactor is wired
        self.last_compaction_error = None
        if self.compactor:
            try:
                compacted, _ = await self.compactor.compact(list(history))
                history[:] = list(compacted)
            except Exception as e:
                self.last_compaction_error = str(e)
                logger.warning("Context compaction failed during turn: %s", e)

        turn_counter = 0
        total_tool_calls_executed = 0
        executed_id_cache: Dict[str, str] = {}

        async def _emit(event: TurnEvent) -> None:
            if event_callback is not None:
                res = event_callback(event)
                if inspect.isawaitable(res):
                    await res

        while turn_counter < self.max_tool_turns:
            turn_counter += 1
            current_turn_id = f"{turn_id_prefix or 'turn'}-{len(history)}-{turn_counter}"
            active_tools = tools if tools is not None else (self.firewall.get_tool_definitions() if self.firewall else None)

            full_reply: List[str] = []
            completed_tool_calls: List[ToolCall] = []
            current_streaming_call: Optional[ToolCall] = None
            response_call_cache: Dict[str, str] = {}

            async for chunk in self.gateway.stream_chat(history, tools=active_tools):
                if chunk.reset_turn:
                    full_reply.clear()
                    completed_tool_calls.clear()
                    current_streaming_call = None
                    evt_reset = TurnReset()
                    await _emit(evt_reset)
                    yield evt_reset

                if chunk.text_delta:
                    full_reply.append(chunk.text_delta)
                    evt_chunk = TokenChunk(text=chunk.text_delta)
                    await _emit(evt_chunk)
                    yield evt_chunk

                if chunk.is_tool_call and chunk.tool_call_delta:
                    tc = chunk.tool_call_delta
                    if current_streaming_call is not None:
                        is_prev_complete = is_complete_json_object(current_streaming_call.arguments)
                        is_diff_id = bool(tc.id and current_streaming_call.id and tc.id != current_streaming_call.id)
                        is_diff_name = bool(tc.name and current_streaming_call.name and tc.name != current_streaming_call.name)
                        is_diff_distinct = is_prev_complete and not tc.arguments.startswith(current_streaming_call.arguments)

                        if is_diff_id or is_diff_name or is_diff_distinct or (is_prev_complete and tc.arguments == current_streaming_call.arguments):
                            completed_tool_calls.append(current_streaming_call)
                            current_streaming_call = tc.model_copy()
                        else:
                            if tc.id and not current_streaming_call.id:
                                current_streaming_call.id = tc.id
                            if tc.name and not current_streaming_call.name:
                                current_streaming_call.name = tc.name
                            current_streaming_call.arguments = tc.arguments
                    else:
                        current_streaming_call = tc.model_copy()

            if current_streaming_call is not None:
                completed_tool_calls.append(current_streaming_call)
                current_streaming_call = None

            reply_text = "".join(full_reply)

            if not completed_tool_calls:
                if reply_text:
                    history.append(ChatMessage(role=Role.ASSISTANT, content=reply_text))
                evt_done = TurnCompleted(
                    final_text=reply_text,
                    tool_calls_executed=total_tool_calls_executed,
                    turn_counter=turn_counter,
                )
                await _emit(evt_done)
                yield evt_done
                break

            # Record assistant message containing tool calls
            history.append(
                ChatMessage(
                    role=Role.ASSISTANT,
                    content=reply_text or None,
                    tool_calls=completed_tool_calls,
                )
            )

            # Process all completed tool calls strictly via ExecutionFirewall
            for idx, tc in enumerate(completed_tool_calls):
                if isinstance(tc.arguments, str):
                    raw_args = tc.arguments.strip()
                    try:
                        args_dict = json.loads(raw_args) if raw_args else {}
                    except Exception:
                        args_dict = raw_args
                elif isinstance(tc.arguments, dict):
                    args_dict = tc.arguments
                else:
                    args_dict = {}

                # Deduplication key combines ID (if available) or tool name + arguments
                if tc.id:
                    dedup_key = tc.id
                else:
                    try:
                        dedup_key = f"{tc.name}:{json.dumps(args_dict, sort_keys=True)}"
                    except Exception:
                        dedup_key = f"{tc.name}:{tc.arguments}"

                cached_result: Optional[str] = None
                if dedup_key in response_call_cache:
                    cached_result = response_call_cache[dedup_key]
                elif tc.id and tc.id in executed_id_cache:
                    cached_result = executed_id_cache[tc.id]

                req_evt = ToolRequested(tool_call=tc, call_index=idx, total_calls=len(completed_tool_calls))
                await _emit(req_evt)
                yield req_evt

                if cached_result is not None:
                    history.append(
                        ChatMessage(
                            role=Role.TOOL,
                            content=cached_result,
                            name=tc.name,
                            tool_call_id=tc.id or None,
                        )
                    )
                    exec_evt = ToolExecuted(tool_call=tc, result=cached_result, is_cached=True, is_error=False)
                    await _emit(exec_evt)
                    yield exec_evt
                    continue

                status_evt = StatusUpdate(status=f"Executing {tc.name}...")
                await _emit(status_evt)
                yield status_evt

                res_str = ""
                is_error = False
                try:
                    if not isinstance(args_dict, dict):
                        raise ValueError(f"Arguments for tool '{tc.name}' must be a JSON object.")

                    if self.firewall is not None:
                        exec_kwargs: Dict[str, Any] = {
                            "tool_name": tc.name,
                            "arguments": args_dict,
                        }
                        if session_id is not None:
                            exec_kwargs["session_id"] = session_id
                            exec_kwargs["turn_id"] = current_turn_id
                        tool_result = await self.firewall.execute_tool(**exec_kwargs)

                        if hasattr(tool_result, "stdout") and hasattr(tool_result, "stderr"):
                            res_str = tool_result.stdout
                            if tool_result.stderr:
                                res_str += f"\n[stderr]: {tool_result.stderr}"
                        else:
                            res_str = str(tool_result)
                    else:
                        raise RuntimeError(f"No execution firewall configured to run tool '{tc.name}'.")

                except PermissionDeniedError as pde:
                    is_error = True
                    safe_err = sanitize_error_message(pde)
                    res_str = f"Permission denied for tool '{tc.name}': {safe_err}"
                except Exception as err:
                    is_error = True
                    safe_err = sanitize_error_message(err)
                    res_str = f"Error executing tool '{tc.name}': {safe_err}"

                # Cache result for within-response deduplication and across-turn duplicate IDs
                response_call_cache[dedup_key] = res_str
                if tc.id:
                    executed_id_cache[tc.id] = res_str

                history.append(
                    ChatMessage(
                        role=Role.TOOL,
                        content=res_str,
                        name=tc.name,
                        tool_call_id=tc.id or None,
                    )
                )
                total_tool_calls_executed += 1

                exec_evt = ToolExecuted(tool_call=tc, result=res_str, is_cached=False, is_error=is_error)
                await _emit(exec_evt)
                yield exec_evt

            status_thinking = StatusUpdate(status="Thinking...")
            await _emit(status_thinking)
            yield status_thinking

            # SEC-01-C: If tool-turn limit reached, request one final assistant response with tools disabled
            if turn_counter >= self.max_tool_turns:
                finalizing_evt = StatusUpdate(status="Finalizing response...")
                await _emit(finalizing_evt)
                yield finalizing_evt

                final_reply = []
                try:
                    async for chunk in self.gateway.stream_chat(history, tools=None):
                        if chunk.text_delta:
                            final_reply.append(chunk.text_delta)
                            tok_evt = TokenChunk(text=chunk.text_delta)
                            await _emit(tok_evt)
                            yield tok_evt
                    final_text = "".join(final_reply)
                    if final_text:
                        history.append(ChatMessage(role=Role.ASSISTANT, content=final_text))
                    comp_evt = TurnCompleted(
                        final_text=final_text,
                        tool_calls_executed=total_tool_calls_executed,
                        turn_counter=turn_counter,
                    )
                    await _emit(comp_evt)
                    yield comp_evt
                except Exception as exc:
                    safe_exc = sanitize_error_message(exc)
                    err_msg = f"\n[Error: {safe_exc}]"
                    tok_evt = TokenChunk(text=err_msg)
                    await _emit(tok_evt)
                    yield tok_evt
                    comp_evt = TurnCompleted(
                        final_text=err_msg,
                        tool_calls_executed=total_tool_calls_executed,
                        turn_counter=turn_counter,
                    )
                    await _emit(comp_evt)
                    yield comp_evt
                break


# Alias for AgentOrchestrator
AgentLoop = AgentOrchestrator
