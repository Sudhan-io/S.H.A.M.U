"""Rolling conversation history context compactor with transactional SQLite archiving."""

import logging
import math
import time
import uuid
from typing import List, Optional, Tuple

from shamu.config import MemoryConfig
from shamu.protocols import MemoryStore
from shamu.types import ChatMessage, Role, MemoryRecord

logger = logging.getLogger(__name__)

# Average characters per BPE token in English/code (~3.8 chars/token)
CHARS_PER_TOKEN = 3.8
# Per-message overhead for chat template tokens (<|im_start|>role\n ... <|im_end|>)
MESSAGE_FRAMING_TOKENS = 4


def estimate_message_tokens(msg: ChatMessage) -> int:
    """Estimates the token count of a single ChatMessage conservatively."""
    tokens = MESSAGE_FRAMING_TOKENS
    if msg.content:
        tokens += int(math.ceil(len(msg.content) / CHARS_PER_TOKEN))

    if msg.tool_calls:
        for tc in msg.tool_calls:
            tokens += 6 + int(math.ceil(len(tc.name) / CHARS_PER_TOKEN))
            tokens += int(math.ceil(len(tc.arguments) / CHARS_PER_TOKEN))

    if msg.tool_call_id:
        tokens += 3

    return max(MESSAGE_FRAMING_TOKENS, tokens)


def estimate_history_tokens(messages: List[ChatMessage]) -> int:
    """Estimates total tokens for a sequence of conversation messages."""
    return sum(estimate_message_tokens(m) for m in messages)


def group_messages_into_turns(messages: List[ChatMessage]) -> List[List[ChatMessage]]:
    """Groups messages into atomic turns.

    An assistant message with tool calls and all subsequent tool messages that respond
    to it form a single atomic turn. Other messages form individual atomic turns.
    """
    turns: List[List[ChatMessage]] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        if msg.role == Role.ASSISTANT and msg.tool_calls:
            turn = [msg]
            i += 1
            while i < n and messages[i].role == Role.TOOL:
                turn.append(messages[i])
                i += 1
            turns.append(turn)
        else:
            turns.append([msg])
            i += 1
    return turns


class ContextCompactor:
    """Manages conversation context window bounding, preserving system instructions and latest user intent."""

    def __init__(self, store: Optional[MemoryStore], config: MemoryConfig) -> None:
        self.store = store
        self.config = config

    async def compact(
        self,
        messages: List[ChatMessage],
        budget_override: Optional[int] = None,
    ) -> Tuple[List[ChatMessage], int]:
        """Prunes oldest intermediate messages if history exceeds token budget.

        Returns:
            Tuple of (compacted_messages, count_of_pruned_and_archived_messages)
        """
        if not messages:
            return [], 0

        # Handle single-message input without duplication regardless of budget
        if len(messages) == 1:
            return list(messages), 0

        budget = budget_override if budget_override is not None else self.config.context_token_budget
        current_tokens = estimate_history_tokens(messages)
        if current_tokens <= budget:
            return list(messages), 0

        # Separate system instruction and content turns
        has_system = messages[0].role == Role.SYSTEM
        system_msg = messages[0] if has_system else None
        content_messages = messages[1:] if has_system else messages

        turns = group_messages_into_turns(content_messages)
        if len(turns) <= 1:
            # Only latest turn exists (along with optional system prompt); nothing intermediate to prune
            result = [system_msg] if system_msg is not None else []
            if turns:
                result.extend(turns[0])
            return result, 0

        middle_turns = list(turns[:-1])
        latest_turn = list(turns[-1])
        pruned_messages: List[ChatMessage] = []

        def build_candidate() -> List[ChatMessage]:
            cand = []
            if system_msg is not None:
                cand.append(system_msg)
            for t in middle_turns:
                cand.extend(t)
            cand.extend(latest_turn)
            return cand

        # Iteratively prune oldest middle turns atomically until within budget
        while middle_turns and estimate_history_tokens(build_candidate()) > budget:
            pruned_turn = middle_turns.pop(0)
            pruned_messages.extend(pruned_turn)

        # Transactionally archive pruned turns to SQLite in a single batch
        if self.store is not None and pruned_messages:
            archive_records: List[MemoryRecord] = []
            for p in pruned_messages:
                content_text = p.content.strip() if (p.content and p.content.strip()) else ""
                if p.tool_calls:
                    tc_desc = " ".join(f"{tc.name}({tc.arguments})" for tc in p.tool_calls)
                    if content_text:
                        content_text = f"{content_text} | tool_calls: {tc_desc}"
                    else:
                        content_text = f"tool_calls: {tc_desc}"

                if content_text:
                    rec_id = f"hist-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
                    meta: Dict[str, Any] = {"role": p.role.value}
                    if p.tool_calls:
                        meta["tool_calls"] = [tc.model_dump() for tc in p.tool_calls]
                    if p.tool_call_id:
                        meta["tool_call_id"] = p.tool_call_id
                    archive_records.append(
                        MemoryRecord(
                            id=rec_id,
                            content=f"[{p.role.value}] {content_text}",
                            source="conversation_archive",
                            category="conversation_history",
                            metadata=meta,
                            created_at=time.time(),
                        )
                    )
            if archive_records:
                try:
                    if hasattr(self.store, "store_batch"):
                        await self.store.store_batch(archive_records)
                    else:
                        for r in archive_records:
                            await self.store.store(r)
                except Exception as e:
                    logger.error(f"Failed to archive pruned messages to memory store: {e}")

        raw_compacted = build_candidate()
        # Ensure compacted list never contains an orphaned Role.TOOL message
        clean_compacted: List[ChatMessage] = []
        for m in raw_compacted:
            if m.role == Role.TOOL:
                if not clean_compacted:
                    continue
                prev = clean_compacted[-1]
                if (prev.role == Role.ASSISTANT and prev.tool_calls) or prev.role == Role.TOOL:
                    clean_compacted.append(m)
            else:
                clean_compacted.append(m)

        return clean_compacted, len(pruned_messages)

