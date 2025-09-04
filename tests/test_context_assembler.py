import pytest
from unittest.mock import AsyncMock, MagicMock
from orchestrator.context.assembler import ContextAssembler, ContextBudget, ProviderCaps

@pytest.fixture
def assembler():
    return ContextAssembler()

@pytest.fixture
def sample_messages():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thanks for asking!"},
        {"role": "user", "content": "What's the weather like?"},
    ]

@pytest.fixture
def provider_caps():
    return {"max_tokens": 4000, "max_input_tokens": 4000, "max_output_tokens": 1000}

@pytest.mark.asyncio
async def test_build_within_budget(assembler, sample_messages, provider_caps):
    budget = ContextBudget(max_tokens_in=1000)
    result = await assembler.build(
        messages=sample_messages,
        provider_caps=provider_caps,
        budget=budget
    )
    assert isinstance(result, list)
    assert len(result) <= len(sample_messages)
    assert all(isinstance(msg, dict) for msg in result)

@pytest.mark.asyncio
async def test_build_with_retrieved_context(assembler, sample_messages, provider_caps):
    budget = ContextBudget(max_tokens_in=1000)
    retrieved = [
        {"role": "system", "content": "Retrieved fact 1"},
        {"role": "system", "content": "Retrieved fact 2"},
    ]
    result = await assembler.build(
        messages=sample_messages,
        provider_caps=provider_caps,
        budget=budget,
        retrieved_context=retrieved
    )
    assert any(msg.get("content") == "Retrieved fact 1" for msg in result)

@pytest.mark.asyncio
async def test_build_with_directives(assembler, sample_messages, provider_caps):
    budget = ContextBudget(max_tokens_in=1000)
    directives = [
        {"role": "system", "content": "[DIRECTIVE] Be concise in responses."},
    ]
    result = await assembler.build(
        messages=sample_messages,
        provider_caps=provider_caps,
        budget=budget,
        directives=directives
    )
    assert any("[DIRECTIVE]" in msg.get("content", "") for msg in result)

def test_deduplicate(assembler):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "Hello"},  # Duplicate
        {"role": "user", "content": "Hi"},
    ]
    unique = assembler._deduplicate(messages)
    assert len(unique) == 2  # One duplicate removed

def test_prune_to_budget(assembler):
    messages = [
        {"role": "system", "content": "A" * 100},
        {"role": "user", "content": "B" * 200},
        {"role": "assistant", "content": "C" * 300},
    ]
    # Rough estimate: 100 + 200 + 300 = 600 tokens (simplified for test)
    pruned = assembler._prune_to_budget(messages, max_tokens=400)
    assert len(pruned) == 2  # Should keep the two most recent messages

@pytest.mark.asyncio
async def test_build_with_large_context(assembler, provider_caps):
    # Create a large context that exceeds the budget
    large_context = [{"role": "system", "content": "X" * 5000}]
    budget = ContextBudget(max_tokens_in=1000)
    
    result = await assembler.build(
        messages=large_context,
        provider_caps=provider_caps,
        budget=budget
    )
    
    # Should be summarized/truncated to fit
    assert len(result) > 0
    assert len(result[0]["content"]) < 5000  # Should be truncated

@pytest.mark.asyncio
async def test_build_with_multiple_buckets(assembler, provider_caps):
    budget = ContextBudget(
        max_tokens_in=1000,
        bucket_pct={"recent": 0.4, "retrieved": 0.4, "directives": 0.2}
    )
    
    recent = [{"role": "user", "content": "Recent message"}]
    retrieved = [{"role": "system", "content": "Retrieved info"}]
    directives = [{"role": "system", "content": "[DIRECTIVE]"}]
    
    result = await assembler.build(
        messages=recent,
        provider_caps=provider_caps,
        budget=budget,
        recent_messages=recent,
        retrieved_context=retrieved,
        directives=directives
    )
    
    # Should include content from all buckets
    content = " ".join(msg["content"] for msg in result)
    assert "Recent message" in content
    assert "Retrieved info" in content
    assert "[DIRECTIVE]" in content
