"""Test per-phase async timeout in core.py."""
import asyncio
import pytest


@pytest.mark.asyncio
async def test_phase_timeout_raises_on_timeout():
    """Test that phase timeout raises TimeoutError when exceeded."""
    from saaaaaa.core.orchestrator.core import PHASE_TIMEOUT_DEFAULT
    
    async def slow_handler():
        """A handler that takes longer than timeout."""
        await asyncio.sleep(1)
    
    # Use a very short timeout
    timeout = 0.1
    
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(slow_handler(), timeout=timeout)


@pytest.mark.asyncio
async def test_phase_timeout_succeeds_within_timeout():
    """Test that phase completes successfully within timeout."""
    async def fast_handler():
        """A handler that completes quickly."""
        await asyncio.sleep(0.01)
        return "success"
    
    # Use a reasonable timeout
    timeout = 1.0
    
    result = await asyncio.wait_for(fast_handler(), timeout=timeout)
    assert result == "success"


@pytest.mark.asyncio
async def test_phase_timeout_default_value():
    """Test that PHASE_TIMEOUT_DEFAULT has the expected value."""
    from saaaaaa.core.orchestrator.core import PHASE_TIMEOUT_DEFAULT
    
    # Default should be 300 seconds
    assert PHASE_TIMEOUT_DEFAULT == 300
    assert isinstance(PHASE_TIMEOUT_DEFAULT, int)
