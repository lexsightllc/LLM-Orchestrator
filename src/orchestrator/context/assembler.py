# SPDX-License-Identifier: MPL-2.0
"""Context assembly and management for LLM orchestration."""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, TypedDict, Union

import tiktoken
from pydantic import BaseModel, Field

# Type aliases
TokenCount = int
Message = Dict[str, str]  # Simplified for now; would use proper Message class in implementation

class ProviderCaps(TypedDict):
    max_tokens: int
    max_input_tokens: int
    max_output_tokens: int

@dataclass
class ContextBudget:
    """Budget allocation for different context components."""
    max_tokens_in: int = 8000
    bucket_pct: Dict[str, float] = field(
        default_factory=lambda: {"recent": 0.35, "retrieved": 0.45, "directives": 0.20}
    )
    
    def get_bucket_budget(self, bucket: str) -> int:
        """Calculate token budget for a specific bucket."""
        return int(self.max_tokens_in * self.bucket_pct.get(bucket, 0))

class ContextAssembler:
    """Assembles and manages context for LLM calls with token budgeting."""
    
    def __init__(self, tokenizer_name: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.dedupe_cache = set()
        
    async def build(
        self,
        messages: List[Message],
        provider_caps: ProviderCaps,
        budget: ContextBudget,
        recent_messages: Optional[List[Message]] = None,
        retrieved_context: Optional[List[Message]] = None,
        directives: Optional[List[Message]] = None,
    ) -> List[Message]:
        """
        Assemble context within token budget constraints.
        
        Args:
            messages: Current conversation messages
            provider_caps: Capabilities of the target provider
            budget: Token budget allocation
            recent_messages: Optional recent messages (if not using all messages)
            retrieved_context: Retrieved context from memory/vector store
            directives: System prompts and other directives
            
        Returns:
            List of messages fitting within token budget
        """
        recent_messages = recent_messages or messages[-6:]  # Default to last 6 messages
        retrieved_context = retrieved_context or []
        directives = directives or []
        
        # Process each bucket within its budget
        processed = {
            "recent": self._process_bucket(
                recent_messages, 
                budget.get_bucket_budget("recent")
            ),
            "retrieved": self._process_bucket(
                retrieved_context,
                budget.get_bucket_budget("retrieved")
            ),
            "directives": self._process_bucket(
                directives,
                budget.get_bucket_budget("directives")
            )
        }
        
        # Combine all buckets
        final_messages = []
        for bucket in ["directives", "retrieved", "recent"]:  # Order matters
            final_messages.extend(processed[bucket]["messages"])
        
        return self._prune_to_budget(final_messages, budget.max_tokens_in)
    
    def _process_bucket(
        self, 
        messages: List[Message], 
        token_budget: int
    ) -> Dict[str, Union[List[Message], int]]:
        """Process a bucket of messages to fit within token budget."""
        if not messages:
            return {"messages": [], "token_count": 0}
            
        # Deduplicate messages
        unique_msgs = self._deduplicate(messages)
        
        # Truncate if over budget
        current_tokens = self._count_tokens_messages(unique_msgs)
        if current_tokens <= token_budget:
            return {"messages": unique_msgs, "token_count": current_tokens}
            
        # If over budget, try to summarize
        return self._summarize_bucket(unique_msgs, token_budget)
    
    def _summarize_bucket(
        self, 
        messages: List[Message], 
        token_budget: int
    ) -> Dict[str, Union[List[Message], int]]:
        """Summarize messages to fit within token budget."""
        # Simple truncation for now; in practice would use a summarization model
        # with a fallback to taking the most recent messages
        summary_msg = {
            "role": "system",
            "content": "[Summary of previous context] " + 
                       " ".join(m.get("content", "")[:500] for m in messages)
        }
        
        # Ensure even the summary fits
        if self._count_tokens_message(summary_msg) > token_budget:
            # Last resort: truncate the content
            max_chars = token_budget * 4  # Rough estimate: 4 chars per token
            summary_msg["content"] = summary_msg["content"][:max_chars] + "..."
        
        return {"messages": [summary_msg], "token_count": self._count_tokens_message(summary_msg)}
    
    def _deduplicate(self, messages: List[Message]) -> List[Message]:
        """Remove duplicate or near-duplicate messages."""
        unique = []
        for msg in messages:
            msg_hash = self._hash_message(msg)
            if msg_hash not in self.dedupe_cache:
                self.dedupe_cache.add(msg_hash)
                unique.append(msg)
        return unique
    
    def _hash_message(self, message: Message) -> str:
        """Create a hash of a message for deduplication."""
        content = json.dumps(message, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _count_tokens_messages(self, messages: List[Message]) -> int:
        """Count tokens for a list of messages."""
        return sum(self._count_tokens_message(msg) for msg in messages)
    
    def _count_tokens_message(self, message: Message) -> int:
        """Count tokens in a single message."""
        content = message.get("content", "")
        return len(self.tokenizer.encode(content))
    
    def _prune_to_budget(
        self, 
        messages: List[Message], 
        max_tokens: int
    ) -> List[Message]:
        """Prune messages to fit within token budget, keeping most recent."""
        total_tokens = 0
        result = []
        
        # Process in reverse to keep most recent messages
        for msg in reversed(messages):
            msg_tokens = self._count_tokens_message(msg)
            if total_tokens + msg_tokens <= max_tokens:
                result.append(msg)
                total_tokens += msg_tokens
            else:
                break
                
        return list(reversed(result))  # Restore original order
